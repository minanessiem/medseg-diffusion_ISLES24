"""Discover run directories inside an experiment-holder directory."""

from __future__ import annotations

from pathlib import Path
from typing import List


def discover_run_dirs(experiment_dir: Path) -> List[Path]:
    """
    Return direct child directories that look like Hydra training runs.

    A run directory is identified by the presence of ``.hydra/config.yaml``.
    Discovery is intentionally non-recursive for the first reporting component:
    the supplied directory is the experiment holder and its direct children are
    the participating runs.
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {experiment_dir}")

    run_dirs: List[Path] = []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / ".hydra" / "config.yaml").is_file():
            run_dirs.append(child)

    return sorted(run_dirs, key=lambda path: path.name)

