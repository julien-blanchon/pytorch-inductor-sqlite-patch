import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from pytorch_inductor_sqlite_patch._hook import PatchFinder, patched_sources


@pytest.mark.parametrize("backend", [None, "file", "sqlite"])
def test_automatic_startup(tmp_path, backend):
    script = """
import os
from pathlib import Path
from torch._inductor.codecache import PyCodeCache
root = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
key, path = PyCodeCache.write('value = 42')
assert PyCodeCache.load_by_key_path(key, path).value == 42
assert (root / 'cache-v1.sqlite3').exists() == (os.environ['TORCHINDUCTOR_CACHE_BACKEND'] == 'sqlite')
"""
    env = dict(
        os.environ,
        TORCHINDUCTOR_CACHE_BACKEND=backend,
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path),
    )
    if backend is None:
        env.pop("TORCHINDUCTOR_CACHE_BACKEND", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=120)


def test_unsupported_version_is_rejected(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_BACKEND", "sqlite")
    with mock.patch(
        "pytorch_inductor_sqlite_patch._hook.metadata.distribution"
    ) as distribution:
        distribution.return_value.version = "2.13.0"
        with pytest.raises(ImportError, match="supports PyTorch 2.14.0"):
            PatchFinder().prepare()


def test_context_mismatch_leaves_source_untouched(tmp_path):
    path = tmp_path / "torch" / "_inductor" / "example.py"
    path.parent.mkdir(parents=True)
    path.write_text("unchanged\n")
    patch = "diff --git a/torch/_inductor/example.py b/torch/_inductor/example.py\n@@ -1 +1 @@\n-old\n+new\n"
    with pytest.raises(ImportError, match="Unsupported PyTorch source layout"):
        patched_sources(tmp_path, patch)
    assert path.read_text() == "unchanged\n"


def test_import_does_not_modify_installed_sources(tmp_path):
    import hashlib
    from importlib.metadata import distribution

    root = Path(distribution("torch").locate_file(""))
    paths = [
        root / "torch" / "_inductor" / name
        for name in ("codecache.py", "remote_cache.py", "output_code.py")
    ]
    before = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
    env = dict(
        os.environ,
        TORCHINDUCTOR_CACHE_BACKEND="sqlite",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path),
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from torch._inductor.codecache import PyCodeCache; PyCodeCache.load('x = 1')",
        ],
        env=env,
        check=True,
        timeout=120,
    )
    assert before == [hashlib.sha256(path.read_bytes()).digest() for path in paths]
    assert not (root / "torch/_inductor/runtime/sqlite_cache.py").exists()
