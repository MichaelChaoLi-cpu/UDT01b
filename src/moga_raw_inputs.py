"""Read fixed Moga members using the study's original raw-relative names.

No raw copies or symlink trees are created. Acquisition writes require a separate
Moga import. Validation delegates to the installed Mili connection resolver.
"""
from __future__ import annotations
import fnmatch
import importlib.util
import os
from pathlib import Path, PurePosixPath

_CONTEXTS = {}

class _Context:
    def __init__(self, root):
        self.root = Path(root).resolve()
        helper = self.root / '.codex/skills/init-research-project/scripts/moga_connection.py'
        spec = importlib.util.spec_from_file_location('_study_moga_connection', helper)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        self.links, self.series = self.api.link_document(self.root)
        self.aliases = {}
        self.physical_aliases = {}
        self.entry_members = {}
        self.verified = {}
        for entry in self.links['datasets']:
            local = entry.get('local_source')
            if not local:
                continue
            for member in entry['members']:
                alias = member if local == '.' else local if len(entry['members']) == 1 else (PurePosixPath(local) / member).as_posix()
                uri = f"moga://{entry['id']}/{member}"
                physical = self.series / entry['data_path'] / member
                if alias in self.aliases:
                    raise ValueError('Ambiguous original raw name: ' + alias)
                self.aliases[alias] = (uri, physical)
                self.entry_members[alias] = (entry, member)
                self.physical_aliases[str(physical)] = uri
        self.directories = {''}
        for alias in self.aliases:
            self.directories.update(p.as_posix() if p.as_posix() != '.' else '' for p in PurePosixPath(alias).parents)

    def ensure_binding(self):
        self.api.binding(self.root)

    def physical(self, alias):
        self.ensure_binding()
        if alias not in self.aliases:
            raise FileNotFoundError('Raw member is not pinned in Moga: ' + alias)
        uri, path = self.aliases[alias]
        st = path.stat()
        signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if self.verified.get(alias) != signature:
            entry, member = self.entry_members[alias]
            selected, _, _, _ = self.api.dataset_files(self.series, {**entry, 'members': [member]})
            actual = selected[0][1]
            if actual != path:
                raise ValueError('Pinned source resolved to an unexpected path')
            self.verified[alias] = signature
        return path


def _context(root):
    key = str(Path(root).resolve())
    if key not in _CONTEXTS:
        _CONTEXTS[key] = _Context(key)
    return _CONTEXTS[key]


class PinnedRawPath(os.PathLike):
    """A read-only raw path; directory iteration uses exact pinned aliases."""
    def __init__(self, root, relative=''):
        self.root = Path(root).resolve()
        text = str(relative).replace('\\', '/')
        rel = PurePosixPath(text)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Raw path must remain within its original root')
        self.relative = '' if str(rel) == '.' else rel.as_posix()

    @property
    def logical(self): return self.root / 'data/raw' / self.relative
    @property
    def name(self): return self.logical.name
    @property
    def stem(self): return self.logical.stem
    @property
    def suffix(self): return self.logical.suffix
    @property
    def suffixes(self): return self.logical.suffixes
    @property
    def parts(self): return self.logical.parts
    @property
    def parent(self):
        if not self.relative: return self.logical.parent
        return PinnedRawPath(self.root, PurePosixPath(self.relative).parent)
    @property
    def parents(self): return self.logical.parents
    def __truediv__(self, other): return PinnedRawPath(self.root, PurePosixPath(self.relative) / str(other))
    def joinpath(self, *parts):
        result = self
        for part in parts: result = result / part
        return result
    def __fspath__(self): return str(_context(self.root).physical(self.relative))
    def __str__(self): return str(self.logical) if self.is_dir() else os.fspath(self)
    def __repr__(self): return f'PinnedRawPath({str(self.root)!r}, {self.relative!r})'
    def __lt__(self, other): return str(self.logical) < str(other.logical if isinstance(other, PinnedRawPath) else other)
    def __hash__(self): return hash(self.logical)
    def __eq__(self, other): return self.logical == (other.logical if isinstance(other, PinnedRawPath) else other)
    def exists(self):
        c = _context(self.root); c.ensure_binding()
        return self.relative in c.aliases or self.relative in c.directories
    def is_file(self): return self.relative in _context(self.root).aliases and self.exists()
    def is_dir(self): return self.relative in _context(self.root).directories and self.exists()
    def is_symlink(self): return False
    def is_absolute(self): return True
    def is_relative_to(self, other): return self.logical.is_relative_to(other)
    def relative_to(self, *other): return self.logical.relative_to(*other)
    def as_posix(self): return str(self)
    def resolve(self, strict=False): return _context(self.root).physical(self.relative)
    def absolute(self): return self
    def with_suffix(self, suffix): return PinnedRawPath(self.root, PurePosixPath(self.relative).with_suffix(suffix))
    def with_name(self, name): return PinnedRawPath(self.root, PurePosixPath(self.relative).with_name(name))
    def stat(self, *args, **kwargs): return _context(self.root).physical(self.relative).stat(*args, **kwargs)
    def open(self, mode='r', *args, **kwargs):
        if any(flag in mode for flag in 'wax+'):
            raise PermissionError('Pinned Moga sources are read-only; import new raw through Moga')
        return _context(self.root).physical(self.relative).open(mode, *args, **kwargs)
    def read_bytes(self):
        with self.open('rb') as f: return f.read()
    def read_text(self, encoding=None, errors=None):
        with self.open('r', encoding=encoding, errors=errors) as f: return f.read()
    def glob(self, pattern):
        c = _context(self.root); c.ensure_binding()
        prefix = self.relative + '/' if self.relative else ''
        candidates = set(c.aliases) | c.directories
        for alias in sorted(candidates):
            if not alias.startswith(prefix) or alias == self.relative: continue
            relative = alias[len(prefix):]
            # PurePath.match alone lets '*' cross the initial path boundary.
            if PurePosixPath(relative).match(pattern) and ('**' in pattern or len(PurePosixPath(relative).parts) == len(PurePosixPath(pattern).parts)):
                yield PinnedRawPath(self.root, alias)
    def rglob(self, pattern):
        c = _context(self.root); c.ensure_binding()
        prefix = self.relative + '/' if self.relative else ''
        for alias in sorted(set(c.aliases) | c.directories):
            if alias.startswith(prefix) and alias != self.relative:
                relative = alias[len(prefix):]
                if PurePosixPath(relative).match(pattern): yield PinnedRawPath(self.root, alias)
    def iterdir(self): return self.glob('*')
    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if exist_ok and self.is_dir(): return None
        raise PermissionError('New raw acquisition must use Moga import')
    def _deny(self, *args, **kwargs): raise PermissionError('Pinned Moga raw cannot be modified by a study reader')
    write_bytes = write_text = unlink = rename = replace = rmdir = touch = chmod = _deny


def raw_path(root, relative=''):
    return PinnedRawPath(root, relative)


def source_reference(path, root):
    if isinstance(path, PinnedRawPath):
        return _context(root).aliases[path.relative][0]
    p = Path(path)
    c = _context(root)
    if str(p) in c.physical_aliases:
        return c.physical_aliases[str(p)]
    return str(p.relative_to(root))


def acquisition_guard(root):
    if _context(root).links['migration']['status'] == 'complete':
        raise PermissionError('Local raw has migrated to Moga. Use moga-import-raw-data for new acquisition; this historical download script does not write pinned sources.')


def project_path(root, value):
    """Resolve raw aliases/URIs found in source manifests; preserve other paths."""
    if isinstance(value, PinnedRawPath): return value
    text = os.fspath(value)
    if text.startswith('moga://'):
        return _context(root).api.resolve_source(Path(root), text)
    if text == 'data/raw' or text.startswith('data/raw/'):
        return raw_path(root, text[len('data/raw'):].lstrip('/'))
    return Path(root) / value
