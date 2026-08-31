"""Configuration dataclass and loader for the CUBICS framework.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Resolve repo root relative to this file: src/cubics/config.py → repo root
_REPO_ROOT = Path(__file__).parent.parent.parent


@dataclass
class CubicsConfig:
    """Runtime configuration for the CUBICS framework.

    All path fields default to locations relative to the repository root so the
    package works out of the box after ``pip install -e .`` without any config
    file.  Override specific fields via a YAML file (see :func:`load_config`) or
    via argparse flags in individual experiment scripts.

    Args:
        repo_root: Absolute path to the repository root.
        data_dir: Directory containing dataset inputs (metadata CSV, configs,
            pre-computed results).
        metrics_csv: Path to the evaluation results CSV produced by run_evaluate.py.
        bdd_metadata_csv: Path to the BDD100K image metadata CSV.
        bdd_dataset_yaml: Path to the BDD100K dataset YAML used for training.
        model_path: Path to the trained YOLO weights file.
        output_dir: Directory where generated figures are written.
        sl_prior_weight: SL non-informative prior weight W (default 2.0 = Beta(1,1)).
        sl_context_prior_weight: Prior weight used when updating context opinions
            from observed situation frequencies during simulation.
        target_class: Default object class for per-class analysis (e.g. "person").
        simulation_num_runs: Number of Monte-Carlo iterations in run_simulation.
        simulation_seed: Random seed for reproducible simulation runs.
    """

    repo_root: Path = field(default_factory=lambda: _REPO_ROOT)

    # Paths — derived from repo_root by default
    data_dir: Path = field(default=None)          # type: ignore[assignment]
    metrics_csv: Path = field(default=None)        # type: ignore[assignment]
    bdd_metadata_csv: Path = field(default=None)   # type: ignore[assignment]
    bdd_dataset_yaml: Path = field(default=None)   # type: ignore[assignment]
    model_path: Path = field(default=None)         # type: ignore[assignment]
    output_dir: Path = field(default=None)         # type: ignore[assignment]

    # SL hyperparameters
    sl_prior_weight: float = 2.0
    sl_context_prior_weight: float = 2.0

    # Analysis defaults
    target_class: str = "person"
    simulation_num_runs: int = 5000
    simulation_seed: int = 0

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        if self.data_dir is None:
            self.data_dir = root / "data"
        if self.metrics_csv is None:
            self.metrics_csv = self.data_dir / "test_metrics_complete.csv"
        if self.bdd_metadata_csv is None:
            self.bdd_metadata_csv = self.data_dir / "bdd_metadata.csv"
        if self.bdd_dataset_yaml is None:
            self.bdd_dataset_yaml = self.data_dir / "bdd_100k.yaml"
        if self.model_path is None:
            self.model_path = root / "models" / "best.pt"
        if self.output_dir is None:
            self.output_dir = root / "outputs"


def default_config() -> CubicsConfig:
    """Return a :class:`CubicsConfig` with all paths resolved from the repo root.

    Returns:
        A fully-initialised config using built-in defaults.
    """
    return CubicsConfig()


def load_config(path: Path) -> CubicsConfig:
    """Load a :class:`CubicsConfig` from a YAML file, falling back to defaults.

    Keys present in the YAML override the dataclass defaults; absent keys keep
    their default values.  Path values in the YAML are resolved as absolute
    paths (``~`` expansion is applied).

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A :class:`CubicsConfig` with YAML overrides applied.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.

    Example:
        >>> cfg = load_config(Path("config.yaml"))
        >>> print(cfg.target_class)
        'person'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    path_fields = {
        "repo_root", "data_dir", "metrics_csv",
        "bdd_metadata_csv", "bdd_dataset_yaml", "model_path", "output_dir",
    }
    for key in path_fields:
        if key in data:
            data[key] = Path(data[key]).expanduser()

    cfg = CubicsConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg
