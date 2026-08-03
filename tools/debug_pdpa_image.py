from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "debug-local-only")


@dataclass(frozen=True)
class PdpaDebugResult:
    normalized_path: Path
    yolo_overlay_path: Path | None
    rectified_path: Path
    safe_path: Path
    masking_mode: str


def _safe_stem(input_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", input_path.stem).strip("_")
    return stem or "label"


def _ensure_import_only_dependencies() -> None:
    try:
        import pytz  # noqa: F401
    except ModuleNotFoundError:
        import types

        fake_pytz = types.ModuleType("pytz")
        fake_pytz.timezone = lambda name: None
        sys.modules["pytz"] = fake_pytz


def _draw_yolo_obb_overlay(main_module, image_path: Path, output_path: Path) -> tuple[Path | None, str]:
    image = main_module.cv2.imread(str(image_path))
    if image is None:
        return None, "image_read_error"

    detections, message = main_module._predict_yolo_obb(str(image_path))
    if detections is None:
        return None, message

    overlay = image.copy()
    colors = {
        main_module.get_yolo_obb_label_class_id(): (0, 200, 0),
        main_module.get_yolo_obb_header_class_id(): (0, 0, 255),
    }
    names = {
        main_module.get_yolo_obb_label_class_id(): "Medicine-Labels",
        main_module.get_yolo_obb_header_class_id(): "patient_header",
    }

    for detection in detections:
        quad = detection["quad"].astype("int32")
        class_id = detection["class_id"]
        color = colors.get(class_id, (255, 120, 0))
        label = f"{names.get(class_id, class_id)} {detection['confidence']:.2f}"
        main_module.cv2.polylines(overlay, [quad], isClosed=True, color=color, thickness=4)
        x, y = int(quad[:, 0].min()), int(quad[:, 1].min())
        main_module.cv2.putText(
            overlay,
            label,
            (max(5, x), max(28, y - 8)),
            main_module.cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            3,
            main_module.cv2.LINE_AA,
        )

    if not detections:
        main_module.cv2.putText(
            overlay,
            "YOLO-OBB: no detections",
            (30, 50),
            main_module.cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            main_module.cv2.LINE_AA,
        )

    main_module.cv2.imwrite(str(output_path), overlay)
    return output_path, "OK"


def process_pdpa_debug_image(
    input_path: str | Path,
    output_dir: str | Path,
    run_qc: bool = True,
    use_yolo_obb: bool = True,
) -> PdpaDebugResult:
    _ensure_import_only_dependencies()
    import main

    source_path = Path(input_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input image not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Input path is not a file: {source_path}")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    output_stem = _safe_stem(source_path)
    normalized_path = output_path / f"{output_stem}_normalized.jpg"
    yolo_overlay_path = output_path / f"{output_stem}_yolo_overlay.jpg"
    rectified_path = output_path / f"{output_stem}_rectified.jpg"
    safe_path = output_path / f"{output_stem}_safe.jpg"

    with tempfile.TemporaryDirectory() as temp_dir:
        working_path = Path(temp_dir) / source_path.name
        shutil.copy2(source_path, working_path)

        if run_qc:
            prepare_ok, prepare_message = main.prepare_upload_image_for_qc(str(working_path))
            if not prepare_ok:
                raise RuntimeError(f"Image preparation failed: {prepare_message}")

            is_good, qc_message = main.check_image_quality(str(working_path))
            if not is_good:
                raise RuntimeError(f"QC failed: {qc_message}")

        normalize_ok, normalize_message = main.normalize_label_image_for_ai(str(working_path), str(normalized_path))
        if not normalize_ok:
            raise RuntimeError(f"Image preprocessing failed: {normalize_message}")

        saved_overlay_path = None
        if use_yolo_obb:
            saved_overlay_path, overlay_message = _draw_yolo_obb_overlay(main, normalized_path, yolo_overlay_path)
            if saved_overlay_path is None and overlay_message != "yolo_obb_disabled":
                print(f"YOLO overlay skipped: {overlay_message}")

        masking_mode = "opencv"
        pdpa_ok = False
        pdpa_message = "not_started"
        if use_yolo_obb:
            pdpa_ok, pdpa_message = main.create_yolo_obb_pdpa_safe_image(
                str(normalized_path),
                str(rectified_path),
                str(safe_path),
            )
            if pdpa_ok:
                masking_mode = "yolo_obb"
            elif pdpa_message != "yolo_obb_disabled":
                print(f"YOLO-OBB PDPA fallback to OpenCV: {pdpa_message}")

        if not pdpa_ok:
            rectify_ok, rectify_message = main.rectify_label_image_for_ai(
                str(normalized_path),
                str(rectified_path),
                use_yolo_obb=False,
            )
            if not rectify_ok:
                raise RuntimeError(f"Image rectification failed: {rectify_message}")

            pdpa_ok, pdpa_message = main.create_pdpa_safe_image(str(rectified_path), str(safe_path))

        if not pdpa_ok:
            raise RuntimeError(f"PDPA masking failed: {pdpa_message}")

    return PdpaDebugResult(
        normalized_path=normalized_path,
        yolo_overlay_path=saved_overlay_path,
        rectified_path=rectified_path,
        safe_path=safe_path,
        masking_mode=masking_mode,
    )


def main_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Create local normalized and PDPA-safe debug images from one medication label image.",
    )
    parser.add_argument("image_path", help="Path to the medication label image on this computer.")
    parser.add_argument(
        "--out",
        default="debug_pdpa",
        help="Output folder for debug images. Default: debug_pdpa",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip QC/Gatekeeper checks and run only preprocessing + PDPA masking.",
    )
    parser.add_argument(
        "--opencv-only",
        action="store_true",
        help="Skip YOLO-OBB and use only the older OpenCV fallback path.",
    )
    args = parser.parse_args()

    try:
        result = process_pdpa_debug_image(
            args.image_path,
            args.out,
            run_qc=not args.skip_qc,
            use_yolo_obb=not args.opencv_only,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("PDPA debug images created:")
    print(f"mode:       {result.masking_mode}")
    print(f"normalized: {result.normalized_path}")
    if result.yolo_overlay_path is not None:
        print(f"yolo:       {result.yolo_overlay_path}")
    print(f"rectified:  {result.rectified_path}")
    print(f"safe:       {result.safe_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
