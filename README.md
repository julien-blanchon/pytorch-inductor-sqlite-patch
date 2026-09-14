# TorchInductor SQLite patch

## Problem

TorchInductor stores generated Python code and autotuning results as individual
files. These can exhaust inode quotas even when disk space remains, particularly
when compiler caches live on shared cluster filesystems such as GPFS or Lustre.

## Solution

This temporary patch stores those entries in SQLite, reducing persistent file
counts while preserving compilation reuse. Python source is materialized into
temporary files when a path is needed; native objects and other cache classes
keep their existing storage.

Supports **PyTorch 2.14.0**. Installation enables SQLite automatically in new
Python processes, without changing your training script or installed source files.

## Usage

Install into the environment containing PyTorch:

```sh
pip install git+https://github.com/julien-blanchon/pytorch-inductor-sqlite-patch.git
```

Choose writable node-local directories, then run normally:

```sh
export TORCHINDUCTOR_CACHE_DIR=/path/to/local/cache/inductor
export TMPDIR=/path/to/local/runtime  # Must already exist.
python train.py
```

The database is `inductor-cache-v1.sqlite3` inside the configured cache directory.
Without `TORCHINDUCTOR_CACHE_DIR`, the existing default location is used; ensure
it is local. Existing file-cache entries are not migrated.

- **Disable:** set `TORCHINDUCTOR_CACHE_BACKEND=file` before starting Python.
- **Remove:** `pip uninstall pytorch-inductor-sqlite-patch`, then restart Python.

Shared writable SQLite databases and fully read-only compiler caches are
unsupported. For multi-node reuse, stage a consistent snapshot into each node's
writable local cache. Live temporary inode use is not bounded.
See [storage, snapshot, and cleanup details](https://github.com/julien-blanchon/pytorch/blob/feat-sqlite-cache/docs/source/user_guide/torch_compiler/sqlite_cache.md).
