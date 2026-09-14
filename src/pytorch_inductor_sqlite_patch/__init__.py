"""Automatic import integration for the temporary SQLite cache patch."""

import os


def _bootstrap():
    if os.environ.setdefault("TORCHINDUCTOR_CACHE_BACKEND", "sqlite") != "file":
        from ._hook import install

        install()
