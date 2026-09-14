# TorchInductor SQLite patch

Install: `pip install git+https://github.com/julien-blanchon/pytorch-inductor-sqlite-patch.git`
Run Python normally; SQLite activates automatically. Supports PyTorch 2.14.0.
Set `TORCHINDUCTOR_CACHE_BACKEND=file` before starting Python to disable it.
Cache location follows `TORCHINDUCTOR_CACHE_DIR`; existing file caches are not migrated.
Use writable node-local cache storage and `TMPDIR`; live temporary inode use is not bounded.
Shared writable databases and full read-only compiler caches are unsupported.
Remove with `pip uninstall pytorch-inductor-sqlite-patch` and restart Python.
[Storage details and limitations](https://github.com/julien-blanchon/pytorch/blob/feat-sqlite-cache/docs/source/user_guide/torch_compiler/sqlite_cache.md).
