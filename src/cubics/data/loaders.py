"""Data loading utilities for the CUBICS framework.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de
"""


from __future__ import annotations

from pathlib import Path

import pandas as pd

from cubics.data.schemas import METRICS_CSV_REQUIRED_COLUMNS


def load_metrics_csv(path: Path) -> pd.DataFrame:
    """Load and validate the YOLO evaluation metrics CSV.

    Args:
        path: Path to ``test_metrics_complete.csv`` (or equivalent).

    Returns:
        DataFrame with all required columns present.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are absent from the file.

    Example:
        >>> df = load_metrics_csv(cfg.metrics_csv)
        >>> df.columns.tolist()
        ['Scenario', 'Weather', 'TimeOfDay', 'Class', ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {path}\n"
            "Run experiments/run_evaluate.py to generate it, or check cfg.metrics_csv."
        )
    df = pd.read_csv(path)
    missing = METRICS_CSV_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Metrics CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {df.columns.tolist()}"
        )
    return df


def load_bdd_metadata(path: Path) -> pd.DataFrame:
    """Load the BDD100K image metadata CSV.

    Args:
        path: Path to ``bdd_metadata.csv``.

    Returns:
        DataFrame with columns ``image_path``, ``split``, ``weather``,
        ``timeofday``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BDD metadata CSV not found: {path}")
    return pd.read_csv(path)
