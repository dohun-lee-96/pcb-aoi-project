from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def xywhn_to_xyxy(row: pd.Series) -> tuple[float, float, float, float]:
    width = float(row["image_width"])
    height = float(row["image_height"])
    x = float(row["x_center"]) * width
    y = float(row["y_center"]) * height
    w = float(row["box_width"]) * width
    h = float(row["box_height"]) * height
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_ground_truth(yolo_dir: Path, split: str) -> pd.DataFrame:
    data_yaml = yaml.safe_load((yolo_dir / "data.yaml").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in data_yaml["names"].items()}
    rows = []
    for label_path in sorted((yolo_dir / "labels" / split).glob("*.txt")):
        image_path = next((yolo_dir / "images" / split).glob(f"{label_path.stem}.*"))
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            cls, x, y, w, h = line.split()
            rows.append(
                {
                    "gt_id": f"{label_path.stem}_{line_no}",
                    "image_id": label_path.stem,
                    "image_path": str(image_path),
                    "class_id": int(cls),
                    "class_name": names[int(cls)],
                    "x_center": float(x),
                    "y_center": float(y),
                    "box_width": float(w),
                    "box_height": float(h),
                    "image_width": width,
                    "image_height": height,
                }
            )
    gt = pd.DataFrame(rows)
    if not gt.empty:
        gt[["xmin", "ymin", "xmax", "ymax"]] = gt.apply(lambda row: pd.Series(xywhn_to_xyxy(row)), axis=1)
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Create FP/FN/localization error tables.")
    parser.add_argument("--yolo-dir", default="data/yolo", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", default="reports/metrics/predictions.csv", type=Path)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    args = parser.parse_args()

    metrics_dir = Path("reports/metrics")
    gt = load_ground_truth(args.yolo_dir, args.split)
    preds = pd.read_csv(args.predictions) if args.predictions.exists() else pd.DataFrame()

    matched_gt: set[str] = set()
    matched_pred: set[int] = set()
    matches = []

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
        if best_gt_id and best_iou >= args.iou_threshold and best_gt_id not in matched_gt:
            matched_gt.add(best_gt_id)
            matched_pred.add(pred_idx)
            matches.append({"prediction_index": int(pred_idx), "gt_id": best_gt_id, "iou": best_iou})

    false_positives = preds.loc[[idx for idx in preds.index if idx not in matched_pred]].copy()
    false_negatives = gt[~gt["gt_id"].isin(matched_gt)].copy()

    localization_rows = []
    for pred_idx, pred in preds.iterrows():
        if pred_idx in matched_pred:
            continue
        same_image = gt[gt["image_id"] == pred["image_id"]]
        best = 0.0
        for _, gt_row in same_image.iterrows():
            best = max(
                best,
                iou(
                    (pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"]),
                    (gt_row["xmin"], gt_row["ymin"], gt_row["xmax"], gt_row["ymax"]),
                ),
            )
        if 0 < best < args.iou_threshold:
            row = pred.to_dict()
            row["best_iou"] = best
            localization_rows.append(row)

    pd.DataFrame(matches).to_csv(metrics_dir / "matches.csv", index=False, encoding="utf-8-sig")
    false_positives.to_csv(metrics_dir / "false_positives.csv", index=False, encoding="utf-8-sig")
    false_negatives.to_csv(metrics_dir / "false_negatives.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(localization_rows).to_csv(metrics_dir / "localization_errors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "ground_truth_objects": int(len(gt)),
        "predicted_objects": int(len(preds)),
        "matches": int(len(matches)),
        "false_positives": int(len(false_positives)),
        "false_negatives": int(len(false_negatives)),
        "localization_errors": int(len(localization_rows)),
        "iou_threshold": args.iou_threshold,
    }
    (metrics_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
