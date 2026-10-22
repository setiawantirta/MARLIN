"""Single point of truth for paths, thresholds and seeds.

Nothing in this package may hardcode a filesystem path. Every input is addressed
by its *logical name* (its filename) and resolved through :func:`path`, which
searches the data roots declared in ``config/config.yaml``. If an input is
absent, the raised :class:`MissingInput` carries the manifest row describing
where to obtain it, so the failure is actionable rather than a bare IOError.

Usage
-----
    from repro.config import CFG, path
    df = pd.read_parquet(path("cmnpd_curated.parquet"))
    seed_list = CFG["seeds"]["model_seeds"]
"""
from __future__ import annotations

import os
import sys
import csv
from pathlib import Path

__all__ = ["CFG", "path", "out_dir", "MissingInput", "MANIFEST", "have", "repo_root"]

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent


def repo_root() -> Path:
    """Directory holding README.md, config/, stages/ ..."""
    return REPO


# --------------------------------------------------------------------------- yaml
def _load_yaml(p: Path) -> dict:
    """Load config.yaml.

    PyYAML is used when available. The fallback parser handles the restricted
    subset this file is written in (two-space indentation, scalars, inline
    lists, comments) so that `--check` still works in a bare interpreter.
    """
    try:
        import yaml  # type: ignore
        with open(p) as fh:
            return yaml.safe_load(fh)
    except ImportError:
        pass

    def _scalar(tok: str):
        tok = tok.strip()
        if tok.startswith("[") and tok.endswith("]"):
            inner = tok[1:-1].strip()
            return [_scalar(x) for x in inner.split(",")] if inner else []
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            return tok[1:-1]
        low = tok.lower()
        if low in ("null", "none", "~", ""):
            return None
        if low == "true":
            return True
        if low == "false":
            return False
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            return tok

    root: dict = {}
    stack = [(-1, root, None, None)]
    with open(p) as fh:
        for raw in fh:
            line = raw.split("  #")[0].rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            # Block sequence item. Without this branch a key written in block form
            # parsed as a mapping whose keys were the literal strings "- value",
            # which produced nonsense data roots and no error at all - a silent
            # wrong answer, which is worse than a crash.
            if stripped.startswith("- "):
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                owner, okey = stack[-1][2], stack[-1][3]
                if okey is None:
                    raise ValueError(
                        f"{p}: sequence item {stripped!r} has no parent key")
                if not isinstance(owner.get(okey), list):
                    owner[okey] = []          # the empty mapping was really a sequence
                owner[okey].append(_scalar(stripped[2:]))
                continue

            key, _, val = stripped.partition(":")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            key = key.strip()
            if val.strip() == "":
                node: dict = {}
                parent[key] = node
                # carry (parent, key) so a following "- " line can convert this
                # placeholder mapping into the sequence it actually is
                stack.append((indent, node, parent, key))
            else:
                parent[key] = _scalar(val)
    return root


CONFIG_PATH = Path(os.environ.get("MARINE_REPRO_CONFIG", REPO / "config" / "config.yaml"))
CFG = _load_yaml(CONFIG_PATH)


# --------------------------------------------------------------------- manifest
def _load_manifest() -> dict:
    p = REPO / "config" / "inputs_manifest.tsv"
    rows = {}
    if p.exists():
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                rows[r["logical_name"]] = r
    return rows


MANIFEST = _load_manifest()


class MissingInput(FileNotFoundError):
    """Raised when a logical input cannot be found under any data root."""

    def __init__(self, name: str, roots, note: str | None = None):
        row = MANIFEST.get(name, {})
        lines = [
            f"input {name!r} was not found.",
            "searched: " + ", ".join(str(r) for r in roots),
        ]
        if note:
            lines.append("  " + note)
        if row:
            lines += [
                f"  produced by : stage {row.get('produced_by_stage', '?')}",
                f"  origin      : {row.get('origin', '?')}",
                f"  size        : {row.get('size_bytes', '?')} bytes",
                f"  sha256      : {row.get('sha256', '?')}",
                f"  ships inside: {row.get('ships_in_package', '?')}",
            ]
        else:
            lines.append("  (no manifest entry - this is an intermediate written by an earlier stage)")
        lines.append("See data/README.md for how to populate the data directory.")
        super().__init__("\n".join(lines))


def _roots():
    env = os.environ.get("MARINE_REPRO_DATA")
    roots = [Path(env)] if env else []
    for r in (CFG.get("paths", {}).get("data_roots") or ["data"]):
        # expanduser first: a root written as ~/... is not absolute, and without
        # this it would silently resolve against the package root and never match.
        # Environment variables are expanded too, so $HOME and %USERPROFILE%-style
        # roots work on a colleague's machine without editing the config.
        p = Path(os.path.expandvars(os.path.expanduser(str(r))))
        roots.append(p if p.is_absolute() else REPO / p)
    roots.append(REPO / "data" / "shipped")
    roots.append(Path(out_dir()))
    seen, uniq = set(), []
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            uniq.append(r)
    return uniq


def have(name: str) -> bool:
    try:
        path(name)
        return True
    except MissingInput:
        return False


def path(name: str) -> str:
    """Resolve a logical input name to an absolute path on this machine."""
    name = str(name)
    roots = _roots()
    for root in roots:
        direct = root / name
        # A root can exist and still be unreadable - a Drive folder the current user
        # has no rights to, a stale mount, a path behind a sandbox grant. Those raise
        # PermissionError or OSError from stat(), which used to abort the whole lookup
        # and so made one unreadable root hide every readable one behind it. An
        # unreadable root is skipped exactly like a non-existent one.
        try:
            if direct.exists():
                return str(direct)
        except (PermissionError, OSError):
            continue
    base = os.path.basename(name)
    hit_dirs = []
    for root in roots:
        try:
            if not root.is_dir():
                continue
            cands = list(root.rglob(base))
        except (PermissionError, OSError):
            continue
        for cand in cands:
            try:
                if cand.is_file():
                    return str(cand)
                if cand.is_dir():
                    hit_dirs.append(cand)
            except (PermissionError, OSError):
                continue
    if hit_dirs:
        # Found, but as a directory. Without this branch the caller got the generic
        # "no manifest entry - this is an intermediate written by an earlier stage",
        # which sends the reader looking for a stage that never produced it. This
        # resolver returns files only, by design: a stage should name the file it
        # reads so the file can live in any folder under any root.
        raise MissingInput(
            name, roots,
            note=("found as a DIRECTORY, not a file: "
                  + ", ".join(str(d) for d in hit_dirs[:3])
                  + ". This resolver returns files only. Ask for the file you actually "
                    "read, by its own basename, rather than for the folder holding it."))
    raise MissingInput(name, roots)


def out_dir(stage: str | None = None) -> str:
    """Output directory, creating it on demand.

    ``$MARINE_REPRO_OUTPUT`` overrides ``paths.output_dir``. On Colab or Kaggle the
    package itself is unpacked onto the ephemeral VM disk, so without this override
    every output - including the stage-09 per-epoch trajectories that Fig. S4 needs
    and the stage-10 per-configuration checkpoints - dies with the session. Point it
    at Drive or /kaggle/working and a dropped runtime resumes instead of restarting.
    """
    env_out = os.environ.get("MARINE_REPRO_OUTPUT")
    base = Path(env_out) if env_out else Path(CFG.get("paths", {}).get("output_dir", "outputs"))
    if not base.is_absolute():
        base = REPO / base
    p = base / stage if stage else base
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def smoke() -> bool:
    """True when the current process was started with --smoke."""
    return "--smoke" in sys.argv or os.environ.get("MARINE_REPRO_SMOKE") == "1"
