from pathlib import Path
from typing import Optional


def find_file(name: str, extra_paths=None) -> Optional[Path]:
    """Search common locations for a file name and return Path or None.

    Search order: current working dir, repository root, common data folders used in this project.
    """
    repo_root = Path.cwd()
    candidates = [
        repo_root / name,
        repo_root / 'FIFA-2026' / 'Week-1' / 'Data' / name,
        repo_root / 'FIFA-2026' / name,
        repo_root / name,
        repo_root / 'notebooks' / 'outputs' / name,
        repo_root / 'outputs' / name,
        repo_root / 'data' / name,
    ]
    if extra_paths:
        for p in extra_paths:
            candidates.append(Path(p) / name)

    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def find_outputs_dir() -> Optional[Path]:
    """Find a directory to place/read model outputs (prefers finalists subfolder).

    Returns absolute Path or None if not found.
    """
    repo_root = Path.cwd()
    candidates = [
        repo_root / 'outputs' / 'finalists',
        repo_root / 'notebooks' / 'outputs' / 'finalists',
        repo_root / 'FIFA-2026' / 'outputs' / 'finalists',
        repo_root / 'outputs',
        repo_root / 'notebooks' / 'outputs',
    ]
    for d in candidates:
        if d.exists():
            return d.resolve()
    # fallback: create standard outputs/finalists under repo root
    fallback = repo_root / 'outputs' / 'finalists'
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()
