"""Small end-to-end compile/restart checks; GPU coverage is optional."""

import json
import importlib.util
import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("compile_threads", [1, 2])
def test_compile_and_fxgraph_restart(tmp_path, device, compile_threads):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cuda" and importlib.util.find_spec("triton_sqlite_patch") is None:
        pytest.skip("Install triton-sqlite-patch to test combined CUDA integration")
    script = """
import json, sys
import torch
from torch._dynamo.utils import counters
def fn(x):
    return torch.sin(x) + x * 2
x = torch.randn(256, device=sys.argv[1])
torch.testing.assert_close(torch.compile(fn)(x), fn(x))
print(json.dumps(dict(counters['inductor'])))
"""
    env = dict(
        os.environ,
        TORCHINDUCTOR_CACHE_BACKEND="sqlite",
        TRITON_CACHE_BACKEND="sqlite",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        TORCHINDUCTOR_COMPILE_THREADS=str(compile_threads),
        TORCHINDUCTOR_FX_GRAPH_CACHE="1",
    )
    env.pop("TRITON_CACHE_MANAGER", None)
    for run in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script, device],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        counts = json.loads(result.stdout.strip().splitlines()[-1])
        if run == 1:
            assert counts.get("fxgraph_cache_hit", 0) > 0, counts
    assert (tmp_path / "inductor/cache-v1.sqlite3").exists()
    if device == "cuda":
        assert (tmp_path / "triton/cache-v1.sqlite3").exists()
