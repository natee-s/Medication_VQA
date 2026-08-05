from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_obb" / "best.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {0: "Medicine-Labels", 1: "patient_header"}


@dataclass(frozen=True)
class ObbLabel:
    class_id: int
    quad: np.ndarray
    confidence: float = 1.0


def safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem).strip("_") or "image"


def collect_images(source_dir: Path) -> list[Path]:
    ignored_parts = {
        "debug_yolo_local",
        "debug_pdpa",
        "json_overlay_sheets",
        "review_sheets",
        "__pycache__",
    }
    images = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        images.append(path)
    return sorted(images)


def get_yolo_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed in this Python environment") from exc
    return YOLO(str(model_path))


def extract_yolo_obb_labels(model, image_path: Path, conf: float, imgsz: int) -> list[ObbLabel]:
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    labels: list[ObbLabel] = []
    for result in results or []:
        obb = getattr(result, "obb", None)
        if obb is None or getattr(obb, "xyxyxyxy", None) is None:
            continue
        points = obb.xyxyxyxy.cpu().numpy()
        if points.ndim == 2 and points.shape[1] == 8:
            points = points.reshape(-1, 4, 2)
        if points.ndim != 3 or points.shape[1:] != (4, 2):
            continue
        confidences = obb.conf.cpu().numpy() if getattr(obb, "conf", None) is not None else np.ones(len(points))
        classes = obb.cls.cpu().numpy() if getattr(obb, "cls", None) is not None else np.zeros(len(points))
        for quad, confidence, class_id in zip(points, confidences, classes):
            labels.append(ObbLabel(int(class_id), quad.astype(np.float32), float(confidence)))
    return labels


def select_best_per_class(labels: list[ObbLabel]) -> list[ObbLabel]:
    selected: dict[int, ObbLabel] = {}
    for label in labels:
        if label.class_id not in CLASS_NAMES:
            continue
        current = selected.get(label.class_id)
        if current is None or label.confidence > current.confidence:
            selected[label.class_id] = label
    return [selected[class_id] for class_id in sorted(selected)]


def normalized_obb_line(label: ObbLabel, width: int, height: int) -> str:
    quad = label.quad.copy()
    quad[:, 0] = np.clip(quad[:, 0] / max(width, 1), 0.0, 1.0)
    quad[:, 1] = np.clip(quad[:, 1] / max(height, 1), 0.0, 1.0)
    values = " ".join(f"{value:.6f}" for value in quad.reshape(-1))
    return f"{label.class_id} {values}"


def write_yolo_obb_label_file(label_path: Path, labels: list[ObbLabel], width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(normalized_obb_line(label, width, height) for label in labels) + ("\n" if labels else ""),
        encoding="utf-8",
    )


def draw_preview(image: np.ndarray, labels: list[ObbLabel], output_path: Path) -> None:
    preview = image.copy()
    for label in labels:
        color = (0, 210, 0) if label.class_id == 0 else (0, 0, 255)
        name = CLASS_NAMES.get(label.class_id, str(label.class_id))
        quad = np.round(label.quad).astype(np.int32)
        cv2.polylines(preview, [quad], isClosed=True, color=color, thickness=4)
        x = int(np.min(quad[:, 0]))
        y = int(np.min(quad[:, 1]))
        cv2.putText(
            preview,
            f"{label.class_id}:{name} {label.confidence:.2f}",
            (max(8, x), max(30, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = homogeneous @ matrix.T
    transformed[:, 0] /= np.maximum(transformed[:, 2], 1e-6)
    transformed[:, 1] /= np.maximum(transformed[:, 2], 1e-6)
    return transformed[:, :2].astype(np.float32)


def transform_labels(labels: list[ObbLabel], matrix: np.ndarray, width: int, height: int) -> list[ObbLabel]:
    transformed_labels = []
    for label in labels:
        quad = transform_points(label.quad.astype(np.float32), matrix)
        if np.any(quad[:, 0] < -2) or np.any(quad[:, 0] > width + 2):
            continue
        if np.any(quad[:, 1] < -2) or np.any(quad[:, 1] > height + 2):
            continue
        quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
        transformed_labels.append(ObbLabel(label.class_id, quad, label.confidence))
    return transformed_labels


def random_geometry_matrix(width: int, height: int, rng: random.Random) -> np.ndarray:
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.92, 1.08)
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    affine[:, 2] += [rng.uniform(-0.04, 0.04) * width, rng.uniform(-0.04, 0.04) * height]
    affine3 = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)

    jitter = min(width, height) * rng.uniform(0.006, 0.025)
    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    dst = src + np.array(
        [
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
            [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
        ],
        dtype=np.float32,
    )
    perspective = cv2.getPerspectiveTransform(src, dst)
    return perspective @ affine3


def apply_photo_augmentation(image: np.ndarray, rng: random.Random) -> np.ndarray:
    output = image.astype(np.float32)
    alpha = rng.uniform(0.78, 1.28)
    beta = rng.uniform(-22.0, 22.0)
    output = output * alpha + beta

    gamma = rng.uniform(0.78, 1.28)
    output = np.power(np.clip(output / 255.0, 0.0, 1.0), gamma) * 255.0
    output = np.clip(output, 0, 255).astype(np.uint8)

    if rng.random() < 0.32:
        kernel_size = rng.choice([3, 5])
        output = cv2.GaussianBlur(output, (kernel_size, kernel_size), rng.uniform(0.2, 0.9))

    if rng.random() < 0.25:
        noise = rng.normalvariate(0, rng.uniform(2.0, 7.0))
        noise_image = np.random.default_rng(rng.randint(0, 999999)).normal(0, abs(noise), output.shape)
        output = np.clip(output.astype(np.float32) + noise_image, 0, 255).astype(np.uint8)

    if rng.random() < 0.22:
        overlay = output.copy()
        h, w = output.shape[:2]
        x0 = rng.randint(0, max(1, w - 1))
        y0 = rng.randint(0, max(1, h - 1))
        x1 = int(np.clip(x0 + rng.uniform(-0.3, 0.3) * w, 0, w - 1))
        y1 = int(np.clip(y0 + rng.uniform(-0.3, 0.3) * h, 0, h - 1))
        cv2.line(overlay, (x0, y0), (x1, y1), (255, 255, 255), rng.randint(18, 54), cv2.LINE_AA)
        output = cv2.addWeighted(overlay, rng.uniform(0.08, 0.18), output, 1.0 - rng.uniform(0.08, 0.18), 0)

    quality = rng.randint(58, 92)
    ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return output


def augment_image(image: np.ndarray, labels: list[ObbLabel], rng: random.Random) -> tuple[np.ndarray, list[ObbLabel]]:
    height, width = image.shape[:2]
    matrix = random_geometry_matrix(width, height, rng)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    transformed_labels = transform_labels(labels, matrix, width, height)
    if not any(label.class_id == 0 for label in transformed_labels):
        return image.copy(), []
    return apply_photo_augmentation(warped, rng), transformed_labels


def split_images(images: list[Path], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    shuffled = images[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = int(round(total * val_ratio))
    if train_count + val_count >= total and total >= 3:
        train_count = total - 2
        val_count = 1
    test_count = total - train_count - val_count
    if test_count < 0:
        test_count = 0
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def save_dataset_yaml(output_dir: Path) -> None:
    yaml_text = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: Medicine-Labels
  1: patient_header
"""
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def save_readme(output_dir: Path, args: argparse.Namespace, source_count: int, splits: dict[str, list[Path]]) -> None:
    readme = f"""# YOLO-OBB Augmented Dataset

This dataset was generated from images in:

```text
{Path(args.source).expanduser().resolve()}
```

Important:

- Labels were bootstrapped from the trained YOLO-OBB model, not manually redrawn.
- Review images in `preview/` before using this dataset for training.
- Green box/class 0 = Medicine-Labels
- Red box/class 1 = patient_header
- Only `train` images are augmented.
- `val` and `test` images are copied without augmentation.

Settings:

```text
source_images={source_count}
train={len(splits["train"])}
val={len(splits["val"])}
test={len(splits["test"])}
augment_per_train_image={args.augment_per_image}
confidence={args.conf}
imgsz={args.imgsz}
seed={args.seed}
```

Use this file for Ultralytics training:

```text
{output_dir / "data.yaml"}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def process_split(
    split_name: str,
    image_paths: list[Path],
    output_dir: Path,
    model,
    args: argparse.Namespace,
    report_rows: list[dict[str, str]],
) -> None:
    rng = random.Random(args.seed + sum(ord(char) for char in split_name))
    if args.limit:
        image_paths = image_paths[: args.limit]

    for index, source_path in enumerate(image_paths, start=1):
        if index == 1 or index % 10 == 0 or index == len(image_paths):
            print(f"  {split_name}: {index}/{len(image_paths)}")

        relative_tag = source_path.parent.name
        base_name = f"{relative_tag}_{safe_stem(source_path)}"
        if args.resume and example_is_complete(output_dir, split_name, base_name, args.augment_per_image):
            report_rows.append(
                {
                    "split": split_name,
                    "source": str(source_path),
                    "status": "resume_skipped_existing",
                }
            )
            continue

        image = cv2.imread(str(source_path))
        if image is None:
            report_rows.append({"split": split_name, "source": str(source_path), "status": "image_read_error"})
            if index % 10 == 0:
                write_report(output_dir, report_rows)
            continue

        detections = select_best_per_class(extract_yolo_obb_labels(model, source_path, args.conf, args.imgsz))
        has_label = any(label.class_id == 0 for label in detections)
        has_header = any(label.class_id == 1 for label in detections)
        if args.require_label and not has_label:
            report_rows.append(
                {
                    "split": split_name,
                    "source": str(source_path),
                    "status": "skipped_no_medicine_label",
                    "has_header": str(has_header),
                }
            )
            if index % 10 == 0:
                write_report(output_dir, report_rows)
            continue

        save_one_example(output_dir, split_name, base_name, image, detections)
        report_rows.append(
            {
                "split": split_name,
                "source": str(source_path),
                "status": "original_saved",
                "has_label": str(has_label),
                "has_header": str(has_header),
                "label_conf": f"{next((label.confidence for label in detections if label.class_id == 0), 0.0):.3f}",
                "header_conf": f"{next((label.confidence for label in detections if label.class_id == 1), 0.0):.3f}",
            }
        )

        if split_name != "train":
            continue

        for aug_index in range(1, args.augment_per_image + 1):
            augmented_image, augmented_labels = augment_image(image, detections, rng)
            if not augmented_labels:
                report_rows.append(
                    {
                        "split": split_name,
                        "source": str(source_path),
                        "status": f"augmentation_{aug_index}_skipped_label_out_of_frame",
                    }
                )
                continue
            save_one_example(
                output_dir,
                split_name,
                f"{base_name}_aug{aug_index:02d}",
                augmented_image,
                augmented_labels,
            )
            report_rows.append(
                {
                    "split": split_name,
                    "source": str(source_path),
                    "status": f"augmentation_{aug_index}_saved",
                    "has_label": "True",
                    "has_header": str(any(label.class_id == 1 for label in augmented_labels)),
                }
            )

        if index % 10 == 0:
            write_report(output_dir, report_rows)

def save_one_example(output_dir: Path, split_name: str, base_name: str, image: np.ndarray, labels: list[ObbLabel]) -> None:
    height, width = image.shape[:2]
    image_path = output_dir / "images" / split_name / f"{base_name}.jpg"
    label_path = output_dir / "labels" / split_name / f"{base_name}.txt"
    preview_path = output_dir / "preview" / split_name / f"{base_name}_preview.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    write_yolo_obb_label_file(label_path, labels, width, height)
    draw_preview(image, labels, preview_path)


def example_is_complete(output_dir: Path, split_name: str, base_name: str, augment_per_image: int) -> bool:
    required = [
        output_dir / "images" / split_name / f"{base_name}.jpg",
        output_dir / "labels" / split_name / f"{base_name}.txt",
        output_dir / "preview" / split_name / f"{base_name}_preview.jpg",
    ]
    if split_name == "train":
        for aug_index in range(1, augment_per_image + 1):
            required.extend(
                [
                    output_dir / "images" / split_name / f"{base_name}_aug{aug_index:02d}.jpg",
                    output_dir / "labels" / split_name / f"{base_name}_aug{aug_index:02d}.txt",
                    output_dir / "preview" / split_name / f"{base_name}_aug{aug_index:02d}_preview.jpg",
                ]
            )
    return all(path.exists() for path in required)


def write_report(output_dir: Path, rows: list[dict[str, str]]) -> None:
    report_path = output_dir / "augmentation_report.csv"
    fieldnames = [
        "split",
        "source",
        "status",
        "has_label",
        "has_header",
        "label_conf",
        "header_conf",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YOLO-OBB augmented dataset for Medicine-Labels and patient_header.",
    )
    parser.add_argument("--source", default=r"D:\4.Intern\2. Dataset\Labels", help="Source image dataset folder.")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "datasets" / "Labels_YOLO_OBB_Augmented"),
        help="Output YOLO-OBB dataset folder. The folder is recreated unless --keep-existing is used.",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO-OBB best.pt.")
    parser.add_argument("--augment-per-image", type=int, default=3, help="How many augmented train images per original.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N source images per split for a quick test.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete the output folder before writing.")
    parser.add_argument("--resume", action="store_true", help="Skip examples whose output image, label, and preview already exist.")
    parser.add_argument(
        "--allow-header-only",
        action="store_true",
        help="Keep images even when Medicine-Labels is not detected. Default skips them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    args.require_label = not args.allow_header_only

    if not source_dir.exists():
        print(f"ERROR: source folder not found: {source_dir}", file=sys.stderr)
        return 1
    if not model_path.exists():
        print(f"ERROR: YOLO model not found: {model_path}", file=sys.stderr)
        return 1
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source_dir)
    if not images:
        print(f"ERROR: no images found in {source_dir}", file=sys.stderr)
        return 1

    print(f"Source images: {len(images)}")
    splits = split_images(images, args.seed, args.train_ratio, args.val_ratio)
    print("Split:", ", ".join(f"{name}={len(paths)}" for name, paths in splits.items()))
    print(f"Loading model: {model_path}")
    model = get_yolo_model(model_path)

    report_rows: list[dict[str, str]] = []
    for split_name, split_images_list in splits.items():
        print(f"Processing {split_name}: {len(split_images_list)} images")
        process_split(split_name, split_images_list, output_dir, model, args, report_rows)

    save_dataset_yaml(output_dir)
    save_readme(output_dir, args, len(images), splits)
    write_report(output_dir, report_rows)
    saved_images = len(list((output_dir / "images").rglob("*.jpg")))
    saved_labels = len(list((output_dir / "labels").rglob("*.txt")))
    saved_previews = len(list((output_dir / "preview").rglob("*.jpg")))
    print("Done")
    print(f"Output: {output_dir}")
    print(f"Saved images: {saved_images}")
    print(f"Saved labels: {saved_labels}")
    print(f"Saved previews: {saved_previews}")
    print(f"Report: {output_dir / 'augmentation_report.csv'}")
    print(f"YAML: {output_dir / 'data.yaml'}")
    print(f"README: {output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
