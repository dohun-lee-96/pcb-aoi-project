from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from aoi_utils import copy_image, load_annotations, voc_to_yolo, write_json


def split_train_groups(base_ids: list[str], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(set(base_ids))
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val_ids = set(shuffled[:val_count])
    train_ids = set(shuffled[val_count:])
    return train_ids, val_ids


def write_yolo_sample(
    rows: list[dict],
    target_split: str,
    yolo_dir: Path,
    class_to_id: dict[str, int],
) -> dict:
    first = rows[0]
    src_image = Path(first["image_path"])
    image_name = f"{first['image_id']}{src_image.suffix.lower()}"
    image_dst = yolo_dir / "images" / target_split / image_name
    label_dst = yolo_dir / "labels" / target_split / f"{first['image_id']}.txt"

    copy_image(src_image, image_dst)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    with label_dst.open("w", encoding="utf-8") as f:
        for row in rows:
            class_id, x, y, w, h = voc_to_yolo(row, class_to_id)
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return {
        "target_split": target_split,
        "image_id": first["image_id"],
        "base_id": first["base_id"],
        "source_split": first["split"],
        "source_image": first["image_path"],
        "target_image": str(image_dst),
        "target_label": str(label_dst),
        "object_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PCB-AOI VOC XML files to YOLO format.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--yolo-dir", default="data/yolo", type=Path)
    parser.add_argument("--reports-dir", default="reports", type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    rows = load_annotations(args.raw_dir)
    df = pd.DataFrame(rows)
    df = df[df["image_path"].astype(bool)].copy()
    class_names = sorted(df["class_name"].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    original_train = df[df["split"] == "train_data"]
    train_group_ids, val_group_ids = split_train_groups(
        original_train["base_id"].unique().tolist(),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    manifest: list[dict] = []
    grouped = df.groupby(["split", "image_id"], sort=True)
    for (source_split, _image_id), group in grouped:
        records = group.to_dict("records")
        base_id = records[0]["base_id"]
        if source_split == "test_data":
            target_split = "test"
        elif source_split == "train_data_augmentation":
            if base_id in val_group_ids:
                continue
            target_split = "train"
        elif source_split == "train_data":
            target_split = "val" if base_id in val_group_ids else "train_original"
        else:
            continue

        if target_split == "train_original":
            # Keep original training files out of YOLO training to avoid duplicate originals.
            continue
        manifest.append(write_yolo_sample(records, target_split, args.yolo_dir, class_to_id))

    data_yaml = {
        "path": str(args.yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for name, idx in class_to_id.items()},
    }
    args.yolo_dir.mkdir(parents=True, exist_ok=True)
    (args.yolo_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    metrics_dir = args.reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(metrics_dir / "yolo_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(metrics_dir / "class_mapping.json", class_to_id)
    write_json(
        metrics_dir / "split_summary.json",
        {
            "class_mapping": class_to_id,
            "train_base_groups": len(train_group_ids),
            "val_base_groups": len(val_group_ids),
            "samples": {
                str(split): int(count)
                for split, count in pd.DataFrame(manifest)["target_split"].value_counts().items()
            },
        },
    )

    print(f"Created YOLO dataset at {args.yolo_dir}")
    print(data_yaml)


if __name__ == "__main__":
    main()
