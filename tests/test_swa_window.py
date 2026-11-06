# SPDX-License-Identifier: MIT
"""SWA counts include the current token; AITER's left distance does not."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_method(name, namespace):
    tree = ast.parse((ROOT / "atom/model_ops/attention_mha.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PagedAttentionImpl")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace["ForwardContext"] = object
    exec(compile(module, "<actual attention frontend>", "exec"), namespace)
    return namespace[name]


def load_prefill(flash):
    # Execute the actual frontend method, stubbing only its GPU backend call.
    return load_method("prefill_attention", {"aiter": SimpleNamespace(flash_attn_varlen_func=flash)})


@pytest.mark.parametrize("window,length,queries", [
    (1, 2, 2), (2, 3, 1), (1024, 1024, 1024), (1024, 1025, 1025),
    (1024, 1025, 1), (1024, 1026, 3), (-1, 1026, 1026), (0, 1026, 1),
])
def test_prefill_visible_keys_follow_model_window(window, length, queries):
    def backend(query, key, value, **kwargs):
        assert kwargs["causal"] is True
        left, right, sinks = kwargs["window_size"]
        assert right in (0, -1) and sinks == 0
        # AITER's documented bottom-right causal alignment and inclusive left end.
        first_query = length - queries
        return [set(range(0 if left < 0 else max(0, q - left), q + 1))
                for q in [first_query, length - 1]]

    prefill = load_prefill(backend)
    metadata = SimpleNamespace(cu_seqlens_q=[0, queries], cu_seqlens_k=[0, length],
        max_seqlen_q=queries, max_seqlen_k=length, min_seqlen_q=0, dropout_p=0.)
    visible = prefill(SimpleNamespace(sliding_window=window, scale=.125, sinks=None),
        None, None, None, None, None, None, None, SimpleNamespace(attn_metadata=metadata))
    for q, actual in zip([length - queries, length - 1], visible):
        expected = {k for k in range(q + 1) if window <= 0 or k > q - window}
        assert actual == expected


@pytest.mark.parametrize("window", [None, -1, 0, 1, 2, 1024])
def test_plugin_window_arguments_have_same_count_semantics(window):
    tree = ast.parse((ROOT / "atom/plugin/attention_mha.py").read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "sliding_window" for t in n.targets)]
    assert len(assignments) == 2
    for assignment in assignments:
        # Check the actual two frontend argument conversions without importing vLLM.
        value = eval(compile(ast.Expression(assignment.value), "<plugin window argument>", "eval"),
                     {"self": SimpleNamespace(sliding_window=window)})
        expected = (window - 1, 0, 0) if window is not None and window > 0 else (-1, -1, 0)
        assert value == expected


@pytest.mark.parametrize("window,queries", [(1024, 1), (1024, 2), (1024, 3),
                                           (1024, 4), (-1, 1), (-1, 4)])
def test_gluon_decode_keeps_oldest_valid_key(monkeypatch, window, queries):
    import torch

    length = 1028
    captured = []

    def backend(*args, **kwargs):
        # The frozen Gluon kernel masks at L - kernel_window + query_index + 1.
        kernel_window = kwargs["sliding_window"]
        captured.extend([
            set(range(max(0, length - kernel_window + i + 1) if kernel_window > 0 else 0,
                      length - queries + i + 1))
            for i in range(queries)
        ])
        args[0].zero_()

    monkeypatch.setattr(torch.ops.aiter, "pa_decode_gluon", backend, raising=False)
    method = load_method("paged_attention_triton", {
        "torch": torch, "aiter": SimpleNamespace(dtypes=SimpleNamespace(fp8=None)),
        "get_recommended_splits": lambda *args: 1,
    })
    impl = SimpleNamespace(use_flash_layout=False, sliding_window=window,
                           kv_cache_dtype="bf16", scale=128 ** -.5, sinks=None)
    meta = SimpleNamespace(cu_seqlens_q=torch.tensor([0, queries]),
        context_lens=torch.tensor([length]), block_tables=torch.arange(65)[None],
        max_seqlen_q=queries)
    query = torch.zeros(queries, 32, 128, dtype=torch.bfloat16)
    key_cache = torch.zeros(65, 4, 16, 16, 8, dtype=torch.bfloat16)
    value_cache = torch.zeros(65, 4, 128, 16, dtype=torch.bfloat16)
    method(impl, query, None, None, key_cache, value_cache, None, None,
           SimpleNamespace(attn_metadata=meta))
    for i, actual in enumerate(captured):
        position = length - queries + i
        expected = {k for k in range(position + 1) if window <= 0 or k > position - window}
        assert actual == expected


@pytest.mark.parametrize("window", [-1, 1024])
@pytest.mark.parametrize("queries", [1, 2, 4])
def test_plugin_gluon_decode_argument_matches_frozen_kernel(window, queries):
    tree = ast.parse((ROOT / "atom/plugin/attention_mha.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "pa_decode_gluon"]
    assert len(calls) == 1
    value = next(k.value for k in calls[0].keywords if k.arg == "sliding_window")
    actual = eval(compile(ast.Expression(value), "<plugin decode window>", "eval"),
                  {"self": SimpleNamespace(sliding_window=window), "max_qlen": queries})
    assert actual == (window + queries if window > 0 else window)
