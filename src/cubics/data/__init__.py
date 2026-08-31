"""CUBICS data loading utilities.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from cubics.data.loaders import load_bdd_metadata, load_metrics_csv
from cubics.data.schemas import METRICS_CSV_REQUIRED_COLUMNS, ScenarioRecord

__all__ = [
    "load_metrics_csv",
    "load_bdd_metadata",
    "ScenarioRecord",
    "METRICS_CSV_REQUIRED_COLUMNS",
]
