"""Train YOLOv12 on BDD100K.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "ultralytics not installed. Run: pip install -e '.[perception]' from the repo root."
    )

try:
    from cubics import default_config, load_config
except ImportError:
    sys.exit("cubics package not found. Run: pip install -e . from the repo root.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    p.add_argument("--data-yaml", type=Path, default=None,
                   help="BDD100K dataset YAML (default: data/bdd_100k.yaml).")
    p.add_argument("--model", type=str, default="yolo12l.pt",
                   help="Base model weights (default: yolo12l.pt — downloads from Ultralytics).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else default_config()

    epochs = args.epochs or 100
    imgsz = args.imgsz or 1024
    batch = args.batch or 4
    device = args.device if args.device is not None else 0
    project = args.project or "bdd_weather_yolo12l"
    run_name = args.run_name or "baseline_yolo12l"
    data_yaml = args.data_yaml or cfg.bdd_dataset_yaml

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on BDD100K — {epochs} epochs, imgsz={imgsz}, batch={batch}")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=run_name,
        save=True,
        exist_ok=True,
        device=device,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_pt}")

    print("Evaluating overall validation set...")
    metrics = model.val()
    print(f"Overall mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
