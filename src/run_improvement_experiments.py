from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from error_analysis import iou, load_ground_truth


def load_weights(default: Path) -> Path:
    summary_path = Path("reports/metrics/training_summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        weights = Path(summary["weights"])
        if weights.exists():
            return weights
    return default


def predict_rows(model: YOLO, image_dir: Path, imgsz: int, conf: float, device: str | int) -> list[dict]:
    rows: list[dict] = []
    names = model.names
    for result in model.predict(
        source=str(image_dir),
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
        verbose=False,
    ):
        image_path = Path(result.path)
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            rows.append(
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
    return rows


def summarize_errors(gt: pd.DataFrame, preds: pd.DataFrame, iou_threshold: float) -> dict:
    matched_gt: set[str] = set()
    matched_pred: set[int] = set()

    for pred_idx, pred in preds.iterrows():
        candidates = gt[(gt["image_id"] == pred["image_id"]) & (gt["class_id"] == int(pred["class_id"]))]
        best_iou = 0.0
        best_gt_id = None
        for _, gt_row in candidates.iterrows():
            score = iou(
                (pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"]),
                (gt_row["xmin"], gt_row["ymin"], gt_row["xmax"], gt_row["ymax"]),
            )
            if score > best_iou:
                best_iou = score
                best_gt_id = gt_row["gt_id"]
        if best_gt_id and best_iou >= iou_threshold and best_gt_id not in matched_gt:
            matched_gt.add(best_gt_id)
            matched_pred.add(pred_idx)

    false_positives = len([idx for idx in preds.index if idx not in matched_pred])
    false_negatives = len(gt) - len(matched_gt)
    recall = len(matched_gt) / len(gt) if len(gt) else 0.0
    precision = len(matched_pred) / len(preds) if len(preds) else 0.0
    return {
        "predicted_objects": int(len(preds)),
        "matches": int(len(matched_gt)),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "operating_precision": float(precision),
        "operating_recall": float(recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight improvement experiments for PCB-AOI.")
    parser.add_argument("--weights", default="models/yolo_runs/pcb_aoi_baseline/weights/best.pt", type=Path)
    parser.add_argument("--yolo-dir", default="data/yolo", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", nargs="+", default=[320, 640], type=int)
    parser.add_argument("--conf", nargs="+", default=[0.05, 0.10, 0.25, 0.40], type=float)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    weights = load_weights(args.weights)
    model = YOLO(str(weights))
    image_dir = args.yolo_dir / "images" / args.split
    gt = load_ground_truth(args.yolo_dir, args.split)
    metrics_dir = Path("reports/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows = []
    prediction_tables: dict[str, pd.DataFrame] = {}

    for imgsz in args.imgsz:
        for conf in args.conf:
            experiment_id = f"op_img{imgsz}_conf{conf:.2f}".replace(".", "p")
            preds = pd.DataFrame(predict_rows(model, image_dir, imgsz, conf, device))
            if preds.empty:
                preds = pd.DataFrame(
                    columns=["image_id", "image_path", "class_id", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"]
                )
            summary = summarize_errors(gt, preds, args.iou_threshold)
            summary.update(
                {
                    "experiment_id": experiment_id,
                    "weights": str(weights),
                    "split": args.split,
                    "imgsz": imgsz,
                    "conf": conf,
                    "iou_threshold": args.iou_threshold,
                    "device": str(device),
                }
            )
            experiment_rows.append(summary)
            prediction_tables[experiment_id] = preds

    results = pd.DataFrame(experiment_rows)
    results["fn_fp_score"] = results["false_negatives"] + 0.25 * results["false_positives"]
    results = results.sort_values(["fn_fp_score", "operating_recall", "operating_precision"], ascending=[True, False, False])
    best = results.iloc[0].to_dict()

    results.to_csv(metrics_dir / "improvement_experiments.csv", index=False, encoding="utf-8-sig")
    best_id = str(best["experiment_id"])
    prediction_tables[best_id].to_csv(metrics_dir / "predictions_recommended.csv", index=False, encoding="utf-8-sig")
    (metrics_dir / "recommended_experiment.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(results)
    print("recommended:", best)


if __name__ == "__main__":
    main()
