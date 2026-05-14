from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight YOLO baseline for PCB-AOI.")
    parser.add_argument("--data", default="data/yolo/data.yaml", type=Path)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--imgsz", default=320, type=int)
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--project", default="models/yolo_runs", type=Path)
    parser.add_argument("--name", default="pcb_aoi_baseline")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project.resolve()),
        name=args.name,
        device=device,
        workers=0,
        patience=5,
        exist_ok=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    summary = {
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    out = Path("reports/metrics/training_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
