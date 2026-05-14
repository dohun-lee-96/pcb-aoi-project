from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

from aoi_utils import copy_image, load_annotations, voc_to_yolo, write_json
from error_analysis import iou, load_ground_truth


def group_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original = df[df["split"] == "train_data"]
    for base_id, group in original.groupby("base_id"):
        classes = set(group["class_name"])
        if "Bad_qiaojiao" in classes and "Bad_podu" in classes:
            label = "both"
        elif "Bad_qiaojiao" in classes:
            label = "qiaojiao"
        else:
            label = "podu"
        rows.append({"base_id": base_id, "strata": label})
    return pd.DataFrame(rows).sort_values("base_id").reset_index(drop=True)


def write_yolo_records(records: list[dict], target_split: str, yolo_dir: Path, class_to_id: dict[str, int]) -> None:
    grouped = pd.DataFrame(records).groupby("image_id", sort=True)
    for _image_id, group in grouped:
        rows = group.to_dict("records")
        first = rows[0]
        src_image = Path(first["image_path"])
        dst_image = yolo_dir / "images" / target_split / f"{first['image_id']}{src_image.suffix.lower()}"
        dst_label = yolo_dir / "labels" / target_split / f"{first['image_id']}.txt"
        copy_image(src_image, dst_image)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with dst_label.open("w", encoding="utf-8") as f:
            for row in rows:
                class_id, x, y, w, h = voc_to_yolo(row, class_to_id)
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def build_fold_dataset(
    df: pd.DataFrame,
    fold_dir: Path,
    train_base_ids: set[str],
    val_base_ids: set[str],
    class_to_id: dict[str, int],
) -> None:
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_rows = df[(df["split"] == "train_data_augmentation") & (df["base_id"].isin(train_base_ids))]
    val_rows = df[(df["split"] == "train_data") & (df["base_id"].isin(val_base_ids))]
    write_yolo_records(train_rows.to_dict("records"), "train", fold_dir, class_to_id)
    write_yolo_records(val_rows.to_dict("records"), "val", fold_dir, class_to_id)

    data_yaml = {
        "path": str(fold_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for name, idx in class_to_id.items()},
    }
    (fold_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def predict_rows(model: YOLO, image_dir: Path, imgsz: int, conf: float, device: str | int) -> pd.DataFrame:
    names = model.names
    rows = []
    for result in model.predict(source=str(image_dir), imgsz=imgsz, conf=conf, device=device, save=False, verbose=False):
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
    return pd.DataFrame(rows)


def operating_metrics(gt: pd.DataFrame, preds: pd.DataFrame, iou_threshold: float) -> dict:
    if preds.empty:
        return {
            "predicted_objects": 0,
            "matches": 0,
            "false_positives": 0,
            "false_negatives": int(len(gt)),
            "operating_precision": 0.0,
            "operating_recall": 0.0,
        }

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

    false_positives = len(preds) - len(matched_pred)
    false_negatives = len(gt) - len(matched_gt)
    precision = len(matched_pred) / len(preds) if len(preds) else 0.0
    recall = len(matched_gt) / len(gt) if len(gt) else 0.0
    return {
        "predicted_objects": int(len(preds)),
        "matches": int(len(matched_gt)),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "operating_precision": float(precision),
        "operating_recall": float(recall),
    }


def paired_tests(cv_df: pd.DataFrame, baseline_id: str, metric: str) -> pd.DataFrame:
    rows = []
    baseline = cv_df[cv_df["candidate_id"] == baseline_id].sort_values("fold")[metric].to_numpy()
    for candidate_id, group in cv_df.groupby("candidate_id"):
        if candidate_id == baseline_id:
            continue
        values = group.sort_values("fold")[metric].to_numpy()
        try:
            stat, p_value = wilcoxon(values, baseline, zero_method="zsplit")
        except ValueError:
            stat, p_value = 0.0, 1.0
        rows.append({"candidate_id": candidate_id, "metric": metric, "wilcoxon_stat": stat, "p_value": p_value})
    test_df = pd.DataFrame(rows)
    if not test_df.empty:
        ordered = test_df.sort_values("p_value").reset_index()
        m = len(ordered)
        adjusted_values = []
        running_max = 0.0
        for rank, row in enumerate(ordered.itertuples(), start=1):
            adjusted = min(1.0, float(row.p_value) * (m - rank + 1))
            running_max = max(running_max, adjusted)
            adjusted_values.append(running_max)
        ordered["holm_p_value"] = adjusted_values
        ordered["significant_0_05"] = ordered["holm_p_value"] < 0.05
        test_df = ordered.set_index("index").sort_index()
    return test_df


def friedman_test(cv_df: pd.DataFrame, metric: str) -> dict:
    pivot = cv_df.pivot(index="fold", columns="candidate_id", values=metric)
    try:
        stat, p_value = friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
    except ValueError:
        stat, p_value = 0.0, 1.0
    return {"metric": metric, "friedman_stat": float(stat), "p_value": float(p_value), "candidates": list(pivot.columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-fold CV operating-point experiments for PCB-AOI.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--work-dir", default="data/cv", type=Path)
    parser.add_argument("--reports-dir", default="reports/metrics", type=Path)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--imgsz", default=320, type=int)
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--conf", nargs="+", default=[0.05, 0.10, 0.25, 0.40], type=float)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    rows = load_annotations(args.raw_dir, splits=["train_data", "train_data_augmentation"])
    df = pd.DataFrame(rows)
    df = df[df["image_path"].astype(bool)].copy()
    class_names = sorted(df["class_name"].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    groups = group_labels(df)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    cv_rows = []
    fold_rows = []
    prediction_frames = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(groups["base_id"], groups["strata"])):
        train_ids = set(groups.iloc[train_idx]["base_id"])
        val_ids = set(groups.iloc[val_idx]["base_id"])
        fold_dir = args.work_dir / f"fold_{fold}"
        build_fold_dataset(df, fold_dir, train_ids, val_ids, class_to_id)

        fold_rows.append(
            {
                "fold": fold,
                "train_groups": len(train_ids),
                "val_groups": len(val_ids),
                "val_strata": groups.iloc[val_idx]["strata"].value_counts().to_dict(),
            }
        )

        model = YOLO(args.model)
        train_result = model.train(
            data=str(fold_dir / "data.yaml"),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str((Path("models") / "cv_runs").resolve()),
            name=f"fold_{fold}_img{args.imgsz}_ep{args.epochs}",
            device=device,
            workers=0,
            patience=5,
            exist_ok=True,
            verbose=False,
        )
        trained = YOLO(str(Path(train_result.save_dir) / "weights" / "best.pt"))
        metrics = trained.val(
            data=str(fold_dir / "data.yaml"),
            split="val",
            imgsz=args.imgsz,
            conf=0.001,
            device=device,
            workers=0,
            project=str((Path("models") / "cv_runs").resolve()),
            name=f"fold_{fold}_metrics",
            exist_ok=True,
            verbose=False,
        )
        gt = load_ground_truth(fold_dir, "val")
        base_metrics = {
            "fold": fold,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision_mean": float(metrics.box.mp),
            "recall_mean": float(metrics.box.mr),
        }
        maps = list(metrics.box.maps)
        for class_id, class_name in enumerate(class_names):
            base_metrics[f"ap_{class_name}"] = float(maps[class_id]) if class_id < len(maps) else 0.0

        for conf in args.conf:
            candidate_id = f"img{args.imgsz}_conf{conf:.2f}".replace(".", "p")
            preds = predict_rows(trained, fold_dir / "images" / "val", args.imgsz, conf, device)
            op = operating_metrics(gt, preds, args.iou_threshold)
            row = {
                **base_metrics,
                **op,
                "candidate_id": candidate_id,
                "imgsz": args.imgsz,
                "conf": conf,
                "epochs": args.epochs,
                "device": str(device),
            }
            row["fn_fp_score"] = row["false_negatives"] + 0.25 * row["false_positives"]
            cv_rows.append(row)
            if not preds.empty:
                preds = preds.copy()
                preds["fold"] = fold
                preds["candidate_id"] = candidate_id
                prediction_frames.append(preds)

    cv_df = pd.DataFrame(cv_rows)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(args.reports_dir / "cv_operating_point_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(args.reports_dir / "cv_fold_summary.csv", index=False, encoding="utf-8-sig")
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            args.reports_dir / "cv_oof_predictions.csv", index=False, encoding="utf-8-sig"
        )

    summary = (
        cv_df.groupby("candidate_id")
        .agg(
            imgsz=("imgsz", "first"),
            conf=("conf", "first"),
            map50_mean=("map50", "mean"),
            map50_95_mean=("map50_95", "mean"),
            operating_precision_mean=("operating_precision", "mean"),
            operating_recall_mean=("operating_recall", "mean"),
            false_positives_mean=("false_positives", "mean"),
            false_negatives_mean=("false_negatives", "mean"),
            fn_fp_score_mean=("fn_fp_score", "mean"),
        )
        .reset_index()
        .sort_values(["fn_fp_score_mean", "operating_recall_mean"], ascending=[True, False])
    )
    summary.to_csv(args.reports_dir / "cv_operating_point_summary.csv", index=False, encoding="utf-8-sig")

    baseline_id = f"img{args.imgsz}_conf0p25"
    wilcoxon_df = paired_tests(cv_df, baseline_id, "fn_fp_score")
    wilcoxon_df.to_csv(args.reports_dir / "cv_wilcoxon_holm.csv", index=False, encoding="utf-8-sig")
    friedman = friedman_test(cv_df, "fn_fp_score")
    write_json(args.reports_dir / "cv_friedman.json", friedman)

    best = summary.iloc[0].to_dict()
    best.update({"selection_metric": "mean(false_negatives + 0.25 * false_positives)", "baseline_candidate": baseline_id})
    write_json(args.reports_dir / "cv_selected_candidate.json", best)

    print(summary)
    print("selected:", best)
    print("friedman:", friedman)


if __name__ == "__main__":
    main()
