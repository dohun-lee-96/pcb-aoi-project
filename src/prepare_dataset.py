from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from aoi_utils import load_annotations, write_json


def build_summary(df: pd.DataFrame) -> dict:
    image_level = df.drop_duplicates(["split", "image_id"])
    return {
        "total_images": int(image_level.shape[0]),
        "total_objects": int(df.shape[0]),
        "splits": {
            split: {
                "images": int(group.drop_duplicates("image_id").shape[0]),
                "objects": int(group.shape[0]),
            }
            for split, group in df.groupby("split")
        },
        "classes": {str(k): int(v) for k, v in Counter(df["class_name"]).items()},
        "image_size": {
            "width_values": sorted(int(v) for v in df["width"].unique()),
            "height_values": sorted(int(v) for v in df["height"].unique()),
        },
        "box_area_ratio": {
            "min": float(df["box_area_ratio"].min()),
            "median": float(df["box_area_ratio"].median()),
            "mean": float(df["box_area_ratio"].mean()),
            "max": float(df["box_area_ratio"].max()),
        },
    }


def save_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_counts = df["class_name"].value_counts()
    ax = class_counts.plot(kind="bar", title="Object Count by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Objects")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=160)
    plt.close()

    ax = df["box_area_ratio"].plot(kind="hist", bins=30, title="Bounding Box Area Ratio")
    ax.set_xlabel("Box area / image area")
    plt.tight_layout()
    plt.savefig(figures_dir / "box_area_ratio_distribution.png", dpi=160)
    plt.close()

    objects_per_image = df.groupby(["split", "image_id"]).size()
    ax = objects_per_image.plot(kind="hist", bins=20, title="Objects per Image")
    ax.set_xlabel("Object count")
    plt.tight_layout()
    plt.savefig(figures_dir / "objects_per_image_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA metadata for PCB-AOI.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--reports-dir", default="reports", type=Path)
    args = parser.parse_args()

    rows = load_annotations(args.raw_dir)
    if not rows:
        raise RuntimeError(f"No annotations found under {args.raw_dir}")

    df = pd.DataFrame(rows)
    metrics_dir = args.reports_dir / "metrics"
    figures_dir = args.reports_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(metrics_dir / "annotations.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(df)
    write_json(metrics_dir / "eda_summary.json", summary)
    save_plots(df, figures_dir)

    print("EDA summary saved:")
    print(summary)


if __name__ == "__main__":
    main()
