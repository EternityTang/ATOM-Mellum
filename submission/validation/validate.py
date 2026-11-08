"""Verify the four delivered entry points through native HTTP requests."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

import aiohttp

ROOT = Path(__file__).resolve().parent
REPO = Path("/workspace/branches/mellum-native-speed-final")
URL = "http://127.0.0.1:8940"
MODEL = "/data/models/Mellum2.1-12B-A2.5B-Thinking"


async def requests():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as client:
        async def one(index, tokens):
            body = dict(
                model=MODEL, prompt=f"Write a Python function that adds {index} to an integer.",
                temperature=0, max_tokens=tokens, ignore_eos=True, stream=False,
            )
            async with client.post(URL + "/v1/completions", json=body) as response:
                if response.status != 200:
                    raise RuntimeError(await response.text())
                result = await response.json()
            assert result["usage"]["completion_tokens"] == tokens
            return dict(request_id=result["id"], usage=result["usage"],
                        finish=result["choices"][0]["finish_reason"])
        single = await one(0, 64)
        batch = await asyncio.gather(*(one(i + 1, 8) for i in range(32)))
        return dict(single=single, concurrent32=batch)


def run(stage):
    flags = ("000", "100", "110", "111")[stage]
    out = ROOT / f"run{stage}"
    out.mkdir()
    env = dict(os.environ, CANDIDATE_GPU="0", SERVER_PORT="8940", ENGINE_PORT="8941",
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    command = ["bash", f"/workspace/run{stage}.sh"]
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(command, cwd="/tmp", env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 260
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"run{stage} exited {process.returncode}")
            try:
                with urllib.request.urlopen(URL + "/v1/models", timeout=2) as response:
                    if MODEL in response.read().decode():
                        break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError(command)
        entries = Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0")
        actual = {item.split(b"=", 1)[0].decode(): item.split(b"=", 1)[1].decode()
                  for item in entries if b"=" in item}
        expected = dict(zip(
            ("MELLUM_SPEED_GRAPH", "MELLUM_SPEED_EXPERT_FP8", "MELLUM_SPEED_QK_FUSION"), flags
        ))
        assert all(actual.get(key) == value for key, value in expected.items())
        assert actual["ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION"] == flags[2]
        result = asyncio.run(requests())
        log_text = (out / "server.log").read_text()
        assert f"'enforce_eager': {flags[0] == '0'}" in log_text
        assert ("'ptpc_fp8'" in log_text) == (flags[1] == "1")
        result.update(
            flags=flags, cwd="/tmp", command=command,
            effective_environment={key: actual.get(key) for key in
                                   (*expected, "NATIVE_PROFILE",
                                    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION",
                                    "MELLUM_FP8_MOE_MIN_TOKENS")},
        )
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        print("PASS", f"run{stage}.sh", flags, "33 HTTP requests", flush=True)
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.poll() is not None:
                break
            os.killpg(process.pid, sig)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                continue
        process.wait()


if __name__ == "__main__":
    tracked = [*REPO.glob("atom/**/*.py"), *REPO.glob("mellum/*.sh")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    for stage in range(4):
        run(stage)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in before.items())
    summaries = [json.loads((ROOT / f"run{stage}/summary.json").read_text()) for stage in range(4)]
    (ROOT / "summary.json").write_text(json.dumps(
        dict(passed=True, http_requests=132, source_unchanged=True, stages=summaries), indent=2
    ))
    (ROOT / "complete").touch()
