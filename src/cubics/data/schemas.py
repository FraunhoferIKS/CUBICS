"""Data schemas and column constants for the CUBICS evaluation CSV.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from typing import TypedDict, Union

METRICS_CSV_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "Scenario",
    "Class",
    "True_Positives",
    "False_Negatives",
    "Recall",
    "Precision",
    "mAP50-95",
})


class ScenarioRecord(TypedDict):
    """A single row from the YOLO evaluation metrics CSV.

    Produced by ``experiments/run_evaluate.py`` and consumed by the RQ2/RQ3
    analysis scripts.
    """

    Scenario: str
    Weather: str
    TimeOfDay: str
    Class: str
    Images_in_Subset: int
    Class_Instances_GT: int
    True_Positives: int
    False_Negatives: int
    False_Positives: Union[int, str]  # "N/A" when precision is 0
    mAP50_95: float
    Recall: float
    Precision: float
