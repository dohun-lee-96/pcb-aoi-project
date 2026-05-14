from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


AUGMENT_SUFFIXES = ("_suofang", "_shuiping", "_shuzhi", "_180", "_270", "_90")


@dataclass(frozen=True)
class BoxRecord:
    split: str
    xml_path: str
    image_path: str
    image_id: str
    base_id: str
    filename: str
    width: int
    height: int
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def box_width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def box_height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def box_area(self) -> int:
        return self.box_width * self.box_height

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            {
                "box_width": self.box_width,
                "box_height": self.box_height,
                "box_area": self.box_area,
                "box_area_ratio": self.box_area / (self.width * self.height)
                if self.width and self.height
                else 0,
            }
        )
        return data


def base_image_id(image_id: str) -> str:
    for suffix in AUGMENT_SUFFIXES:
        if image_id.endswith(suffix):
            return image_id[: -len(suffix)]
    return image_id


def find_image_for_xml(xml_path: Path) -> Path | None:
    split_root = xml_path.parents[1]
    image_dir = split_root / "JPEGImages"
    stem = xml_path.stem
    for suffix in (".jpeg", ".jpg", ".png", ".bmp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_voc_xml(xml_path: Path, split: str) -> list[BoxRecord]:
    image_path = find_image_for_xml(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename") or f"{xml_path.stem}.jpeg"
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    image_id = xml_path.stem
    records: list[BoxRecord] = []

    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None or not class_name:
            continue
        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
        records.append(
            BoxRecord(
                split=split,
                xml_path=str(xml_path),
                image_path=str(image_path) if image_path else "",
                image_id=image_id,
                base_id=base_image_id(image_id),
                filename=filename,
                width=width,
                height=height,
                class_name=class_name,
                xmin=max(0, xmin),
                ymin=max(0, ymin),
                xmax=min(width, xmax) if width else xmax,
                ymax=min(height, ymax) if height else ymax,
            )
        )
    return records


def iter_xml_files(raw_dir: Path, splits: Iterable[str] | None = None) -> Iterable[tuple[str, Path]]:
    selected = list(splits or ("train_data", "train_data_augmentation", "test_data"))
    for split in selected:
        ann_dir = raw_dir / split / "Annotations"
        for xml_path in sorted(ann_dir.glob("*.xml")):
            yield split, xml_path


def load_annotations(raw_dir: Path, splits: Iterable[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for split, xml_path in iter_xml_files(raw_dir, splits):
        rows.extend(record.to_dict() for record in parse_voc_xml(xml_path, split))
    return rows


def voc_to_yolo(record: dict, class_to_id: dict[str, int]) -> tuple[int, float, float, float, float]:
    width = float(record["width"])
    height = float(record["height"])
    xmin = float(record["xmin"])
    ymin = float(record["ymin"])
    xmax = float(record["xmax"])
    ymax = float(record["ymax"])
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return (
        class_to_id[record["class_name"]],
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(box_width, 0.0), 1.0),
        min(max(box_height, 0.0), 1.0),
    )


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
