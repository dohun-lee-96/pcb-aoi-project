from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model and export predictions.")
    parser.add_argument("--weights", default="models/yolo_runs/pcb_aoi_baseline/weights/best.pt", type=Path)
    parser.add_argument("--data", default="data/yolo/data.yaml", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", default=320, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        workers=0,
        project=str(Path("models/yolo_runs").resolve()),
        name=f"eval_{args.split}",
        exist_ok=True,
    )

    names = model.names
    metric_summary = {
        "split": args.split,
        "weights": str(args.weights),
        "device": str(device),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "class_names": names,
    }
    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation_summary.json").write_text(
        json.dumps(metric_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    image_dir = Path("data/yolo/images") / args.split
    prediction_rows = []
    for result in model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        project=str(Path("models/yolo_runs").resolve()),
        name=f"pred_{args.split}",
        exist_ok=True,
        verbose=False,
    ):
        image_path = Path(result.path)
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            prediction_rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "class_id": cls_id,
                    "class_name": names[cls_id],
                    "confidence": float(box.conf[0].item()),
                    "xmin": float(xyxy[0]),
                    "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]),
                    "ymax": float(xyxy[3]),
                }
            )
    pd.DataFrame(prediction_rows).to_csv(metrics_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    print(metric_summary)


if __name__ == "__main__":
    main()
