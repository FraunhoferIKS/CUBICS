"""
Evaluate the YOLO model across BDD100K weather × time-of-day scenarios.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de


"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "ultralytics not installed. Run: pip install -e '.[perception]' from the repo root."
    )

try:
    from cubics import default_config, load_config
    from cubics.data import load_bdd_metadata
except ImportError:
    sys.exit("cubics package not found. Run: pip install -e . from the repo root.")


# YOLO class names matching the BDD100K label map used during training
_CLASS_NAMES = {
    0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign",
}


def _build_scenarios(test_df: pd.DataFrame) -> list[dict]:
    """Return the list of (weather, timeofday, combo) scenario dicts."""
    weathers = test_df["weather"].dropna().unique().tolist()
    times = test_df["timeofday"].dropna().unique().tolist()
    scenarios = []
    for w in weathers:
        scenarios.append({"name": f"{w.capitalize()} (All Times)", "weather": w, "timeofday": None})
    for t in times:
        scenarios.append({"name": f"{t.capitalize()} (All Weather)", "weather": None, "timeofday": t})
    for w in weathers:
        for t in times:
            scenarios.append({"name": f"{w.capitalize()} + {t.capitalize()}", "weather": w, "timeofday": t})
    return scenarios


def _count_gt_instances(subset: pd.DataFrame, class_ids: set[int]) -> dict[int, int]:
    """Count ground-truth instances per class by reading YOLO label files."""
    counts = {i: 0 for i in class_ids}
    for img_path in subset["image_path"]:
        label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            with open(label_path) as fh:
                for line in fh:
                    try:
                        cls_id = int(line.strip().split()[0])
                        if cls_id in counts:
                            counts[cls_id] += 1
                    except (IndexError, ValueError):
                        pass
    return counts


def _evaluate_scenario(
    model: YOLO,
    scenario: dict,
    test_df: pd.DataFrame,
    conf: float = 0.272,
) -> list[dict]:
    """Evaluate one scenario and return a list of per-class result dicts."""
    subset = test_df.copy()
    if scenario["weather"]:
        subset = subset[subset["weather"] == scenario["weather"]]
    if scenario["timeofday"]:
        subset = subset[subset["timeofday"] == scenario["timeofday"]]

    num_images = len(subset)
    if num_images == 0:
        print(f"  Skipping {scenario['name']} — 0 images found.")
        return []

    print(f"  Found {num_images} images.")

    class_ids = set(_CLASS_NAMES.keys())
    gt_counts = _count_gt_instances(subset, class_ids)

    # Write image list and YAML to a temp directory, cleaned up after val()
    with tempfile.TemporaryDirectory() as tmp:
        txt_path = Path(tmp) / "images.txt"
        yaml_path = Path(tmp) / "data.yaml"

        subset["image_path"].to_csv(txt_path, index=False, header=False)

        names_block = "\n".join(f"  {k}: {v}" for k, v in _CLASS_NAMES.items())
        yaml_path.write_text(
            f"path: /\n"
            f"train: {txt_path.as_posix()}\n"
            f"val: {txt_path.as_posix()}\n"
            f"names:\n{names_block}\n"
        )

        model.conf = conf
        metrics = model.val(data=str(yaml_path), conf = conf, plots=False, verbose=False)

    results = []
    for i, name in _CLASS_NAMES.items():
        gt = gt_counts[i]
        if gt == 0:
            continue

        if i in metrics.box.ap_class_index:
            result_idx = list(metrics.box.ap_class_index).index(i)
            map50_95 = metrics.box.maps[i]
            recall = metrics.box.r[result_idx]
            precision = metrics.box.p[result_idx]
        else:
            map50_95, recall, precision = 0.0, 0.0, 0.0

        tp = int(round(gt * recall))
        fn = gt - tp
        fp: int | str = int(round((tp / precision) - tp)) if precision > 0 else "N/A"

        results.append({
            "Scenario": scenario["name"],
            "Weather": scenario["weather"] or "All",
            "TimeOfDay": scenario["timeofday"] or "All",
            "Class": name,
            "Images_in_Subset": num_images,
            "Class_Instances_GT": gt,
            "True_Positives": tp,
            "False_Negatives": fn,
            "False_Positives": fp,
            "mAP50-95": round(map50_95, 4),
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
        })

    return results


def _append_overall_baseline(all_results: list[dict]) -> list[dict]:
    """Derive the Overall Test Baseline by summing across weather×time combos."""
    combo_rows = [r for r in all_results if "+" in r["Scenario"]]
    combo_df = pd.DataFrame(combo_rows)
    baselines = []

    for class_name, grp in combo_df.groupby("Class"):
        tp = int(grp["True_Positives"].sum())
        fn = int(grp["False_Negatives"].sum())
        gt = int(grp["Class_Instances_GT"].sum())
        valid_fps = [r for r in grp["False_Positives"] if r != "N/A"]
        fp: int | str = int(sum(valid_fps)) if valid_fps else "N/A"
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (isinstance(fp, int) and (tp + fp) > 0) else 0.0

        baselines.append({
            "Scenario": "Overall Test Baseline",
            "Weather": "All",
            "TimeOfDay": "All",
            "Class": class_name,
            "Images_in_Subset": int(grp["Images_in_Subset"].sum()),
            "Class_Instances_GT": gt,
            "True_Positives": tp,
            "False_Negatives": fn,
            "False_Positives": fp,
            "mAP50-95": 0.0,
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
        })

    return all_results + baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    p.add_argument("--model", type=Path, default=None, help="Path to .pt weights file.")
    p.add_argument("--metadata-csv", type=Path, default=None, help="Path to bdd_metadata.csv.")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Path to write results CSV (default: data/test_metrics_complete.csv).")
    p.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.272,
                   help="Confidence threshold for evaluation (default: 0.272, peak F1 from val run).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else default_config()

    model_path = args.model or cfg.model_path
    metadata_csv = args.metadata_csv or cfg.bdd_metadata_csv
    output_csv = args.output_csv or cfg.metrics_csv
    device = args.device if args.device is not None else 0

    metadata = load_bdd_metadata(metadata_csv)
    test_df = metadata[metadata["split"] == args.split]
    print(f"Loaded {len(test_df)} test images from {metadata_csv}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    model.to(f"cuda:{device}" if device >= 0 else "cpu")

    scenarios = _build_scenarios(test_df)
    print(f"Generated {len(scenarios)} scenarios to evaluate.")

    print(f"Confidence threshold: {args.conf}")
    all_results: list[dict] = []
    for scenario in scenarios:
        print(f"\n{'='*50}\nEvaluating: {scenario['name']}\n{'='*50}")
        all_results.extend(_evaluate_scenario(model, scenario, test_df, conf=args.conf))

    all_results = _append_overall_baseline(all_results)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. {len(final_df)} rows saved to {output_csv}")


if __name__ == "__main__":
    main()
