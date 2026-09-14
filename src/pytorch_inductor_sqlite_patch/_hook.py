"""Apply the proposed upstream changes in memory without modifying PyTorch."""

import importlib.abc
import importlib.machinery
import importlib.util
import linecache
import os
import sys
from importlib import metadata, resources
from pathlib import Path

from packaging.version import Version


def patched_sources(root, patch):
    """Validate every context exactly once before installing any modified code."""
    results = {}
    target = None
    old, new = [], []

    def flush():
        if not old and not new:
            return
        before, after = "".join(old), "".join(new)
        source = results[target]
        if not before or source.count(before) != 1:
            raise ImportError(
                f"Unsupported PyTorch source layout in {target}; SQLite patch not activated"
            )
        results[target] = source.replace(before, after, 1)
        old.clear()
        new.clear()

    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            target = line.rstrip().split(" b/", 1)[1]
            if not target.startswith("torch/_inductor/") or ".." in Path(target).parts:
                raise ImportError(f"Invalid patch path: {target}")
            results[target] = (root / target).read_text()
        elif line.startswith("@@"):
            flush()
        elif line.startswith(("index ", "--- ", "+++ ")):
            continue
        elif line.startswith(" "):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith("-"):
            old.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
    flush()
    return results


class PatchedLoader(importlib.abc.InspectLoader):
    def __init__(self, source, filename):
        self.source = source
        self.filename = filename

    def get_source(self, fullname):
        return self.source

    def get_code(self, fullname):
        linecache.cache[self.filename] = (
            len(self.source),
            None,
            self.source.splitlines(keepends=True),
            self.filename,
        )
        return compile(self.source, self.filename, "exec", dont_inherit=True)


class PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.sources = None
        self.root = None

    def prepare(self):
        if os.environ.get("TORCHINDUCTOR_CACHE_BACKEND") != "sqlite":
            raise ImportError("TORCHINDUCTOR_CACHE_BACKEND must be 'file' or 'sqlite'")
        distribution = metadata.distribution("torch")
        installed = Version(distribution.version)
        if (
            installed.base_version != "2.14.0"
            or installed.is_prerelease
            or installed.is_devrelease
        ):
            raise ImportError(
                f"pytorch-inductor-sqlite-patch supports PyTorch 2.14.0; installed: {installed}"
            )
        self.root = Path(distribution.locate_file(""))
        package = resources.files("pytorch_inductor_sqlite_patch")
        helper = "torch/_inductor/runtime/sqlite_cache.py"
        if (self.root / helper).exists():
            raise ImportError(
                "An upstream or patched SQLite cache module already exists; uninstall the temporary patch"
            )
        sources = patched_sources(
            self.root, package.joinpath("upstream.patch").read_text()
        )
        sources[helper] = package.joinpath("sqlite_cache.py.txt").read_text()
        self.sources = {
            path[:-3].replace("/", "."): (source, str(self.root / path))
            for path, source in sources.items()
        }

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" and self.sources is None:
            if os.environ.get("TORCHINDUCTOR_CACHE_BACKEND") == "file":
                return None
            self.prepare()
        if self.sources is None or fullname not in self.sources:
            return None
        source, filename = self.sources[fullname]
        if fullname != "torch._inductor.runtime.sqlite_cache":
            original = importlib.machinery.PathFinder.find_spec(fullname, path)
            if (
                original is None
                or Path(original.origin).resolve() != Path(filename).resolve()
            ):
                raise ImportError(
                    f"Unexpected module origin for {fullname}; SQLite patch refused"
                )
        return importlib.util.spec_from_file_location(
            fullname, filename, loader=PatchedLoader(source, filename)
        )


def install():
    if not any(isinstance(finder, PatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, PatchFinder())
