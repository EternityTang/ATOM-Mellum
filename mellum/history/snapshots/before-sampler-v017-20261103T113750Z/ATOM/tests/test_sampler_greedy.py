# SPDX-License-Identifier: MIT
"""Greedy ties must never enter a random sampling path, filters or no filters."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


@pytest.fixture
def sampler_class(monkeypatch):
    # Keep the regression CPU-only; no AITER kernel may execute for greedy.
    modules = {
        name: types.ModuleType(name)
        for name in ["aiter", "aiter.ops", "aiter.ops.sampling", "aiter.ops.triton",
                     "aiter.ops.triton.softmax", "aiter.ops.triton.topk"]
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Greedy unexpectedly called a random/native sampling kernel")

    modules["aiter"].mixed_sample_outer_exponential = forbidden
    modules["aiter.ops.triton.softmax"].softmax = forbidden
    modules["aiter.ops.triton.topk"].topk = forbidden
    modules["aiter"].ops = modules["aiter.ops"]
    modules["aiter.ops"].sampling = modules["aiter.ops.sampling"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/sampler.py"
    spec = importlib.util.spec_from_file_location("isolated_sampler_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sampler


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("filter_kind", ["none", "top_k", "top_p"])
def test_greedy_ties_choose_first_argmax_without_rng(sampler_class, dtype, batch, filter_kind):
    sampler = sampler_class()
    logits = torch.zeros(batch, 32, dtype=dtype)
    logits[:, [3, 9, 17]] = 5
    temperatures = torch.full((batch,), 1e-10)
    kwargs = {}
    if filter_kind == "top_k":
        kwargs["top_ks"] = torch.ones(batch, dtype=torch.int32)
    elif filter_kind == "top_p":
        kwargs["top_ps"] = torch.full((batch,), .9)
    before = torch.random.get_rng_state()
    result = sampler(logits, temperatures, all_greedy=True, **kwargs)
    assert torch.equal(result, torch.full((batch,), 3, dtype=torch.int32))
    assert result.dtype == torch.int32
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("filter_kind", ["none", "top_k", "top_p"])
@pytest.mark.parametrize("independent_noise", [False, True])
def test_nongreedy_dispatch_and_noise_flag_preserved(sampler_class, filter_kind, independent_noise):
    sampler = sampler_class()
    calls = []
    expected = torch.tensor([7], dtype=torch.int32)

    def temperature(logits, temperatures, needs_independent_noise=False):
        calls.append(("temperature", needs_independent_noise))
        return expected

    def filtered(logits, temperatures, top_ks, top_ps, all_greedy, needs_independent_noise=False):
        assert all_greedy is False
        calls.append(("filtered", needs_independent_noise))
        return expected

    sampler._temperature_sample = temperature
    sampler._topk_topp_sample = filtered
    kwargs = {}
    if filter_kind == "top_k":
        kwargs["top_ks"] = torch.tensor([5], dtype=torch.int32)
    elif filter_kind == "top_p":
        kwargs["top_ps"] = torch.tensor([.9])
    result = sampler(torch.zeros(1, 32), torch.ones(1), all_greedy=False,
                     needs_independent_noise=independent_noise, **kwargs)
    assert result is expected
    assert calls == [("temperature" if filter_kind == "none" else "filtered", independent_noise)]
