# TorchInductor SQLite cache patch

Install this temporary package and run your application normally. It automatically
enables SQLite persistence for TorchInductor autotune records (`.best_config`) and
generated `PyCodeCache` Python source. No launcher or application edits are needed.

```sh
pip install git+https://github.com/julien-blanchon/pytorch-inductor-sqlite-patch.git
python train.py
```

Supported: **PyTorch 2.14.0**, Python 3.10 or newer. Local version suffixes such as
`+cu130` are accepted. Unsupported versions, development builds, changed source
layouts, and an already-present upstream SQLite module fail clearly on import.
The package does not install or upgrade PyTorch for you.

This package defaults `TORCHINDUCTOR_CACHE_BACKEND` to `sqlite`. Set
`TORCHINDUCTOR_CACHE_BACKEND=file` **before starting Python** to opt out. The
proposed upstream feature instead defaults to files and requires explicit opt-in.

## Configuration and storage

```sh
export TORCHINDUCTOR_CACHE_DIR=/path/to/local/cache/inductor
export TMPDIR=/path/to/local/runtime
python train.py
```

The database is `TORCHINDUCTOR_CACHE_DIR/cache-v1.sqlite3`; the existing default
cache directory still applies. Autotune bytes are loaded directly. Python source
materializes under Python's temporary directory (`TMPDIR` on Unix), preserving
real paths for imports, tracebacks, inspection, debugging, subprocesses, and
FX-graph reconstruction. Files survive until normal process exit.

Active runtime inode use grows with sources touched; this patch does **not** bound
it. Abnormal exits may leave `torchinductor-sqlite-*` directories. Remove only
directories of stopped processes. Use temporary storage outside the constrained
persistent cache with sufficient inodes. Native objects, lock files, FX-graph
caches, and AOTAutograd caches retain filesystem storage. Total inode reduction
therefore depends on workload. Existing file entries are not migrated.

Use local storage. Sharing a database across hosts on NFS, Lustre, or similar
filesystems is unsupported. SQLite uses its rollback journal, transactions, and
a 30-second busy timeout. Connections close after each operation. There is no
TTL, LRU, size cap, or automatic corruption repair. Database errors are reported
without silently falling back to inode-heavy files. Stop all users before deleting
the database and any associated journal files to clear it; other cache files are
managed separately.

## How automatic activation works

A wheel-installed `.pth` startup hook registers a narrow import hook. On importing
PyTorch it checks the version and exact source contexts, then applies the proposed
upstream edits **in memory** to four Inductor modules and supplies one helper.
Installed PyTorch files are untouched. Fresh compiler subprocesses activate in
the same way. Patched modules bypass bytecode caching and expose modified source
through their loader and Python's line cache.

Normal Python `site` initialization is required (`python -S` disables `.pth`
processing). Configure the backend before interpreter startup; installing into a
running notebook requires a kernel restart. Use a regular `pip install .` when
developing: editable-install startup hooks are not supported. Other packages that
replace these same module loaders may be incompatible.

To remove, `pip uninstall pytorch-inductor-sqlite-patch` and restart Python.
There is no apply/revert step. After upstream adoption, uninstall this package and
set `TORCHINDUCTOR_CACHE_BACKEND=sqlite` explicitly for the upstream implementation.

## Development

```sh
pip install '.[test]'
pytest -s --tb=short tests
```

`upstream.patch` contains only the integration edits from the upstream feature
branch. `sqlite_cache.py.txt` is the matching upstream helper. Regenerate both
when changing the implementation and test each claimed supported release.
Upstream implementation: [feat-sqlite-cache](https://github.com/julien-blanchon/pytorch/tree/feat-sqlite-cache),
commit `f5a61698514`.

## Storage microbenchmark

Run `python benchmarks/cache.py --entries 5000 --directory /path/to/local/storage`.
For 5,000 Python sources plus 5,000 autotune records, one local run measured:

| Backend | Persistent objects | Write seconds | Restart-read seconds | Peak temporary objects |
| --- | ---: | ---: | ---: | ---: |
| File | 11,019 | 0.94 | 0.36 | 0 |
| SQLite | 1 | 6.44 | 2.38 | 6,020 |

Temporary objects returned to zero after normal process exit. Counts exclude the
benchmark's containing directories and do not include other Inductor cache types.
Timings exclude imports and kernel compilation; these are storage-only measurements,
not an end-to-end performance claim. Re-run on the intended local filesystem.
