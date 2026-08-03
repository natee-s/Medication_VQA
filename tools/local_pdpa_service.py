import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo_obb" / "best_round3.pt"

os.environ.setdefault("YOLO_OBB_ENABLED", "true")
os.environ.setdefault("YOLO_OBB_MODEL_PATH", str(MODEL_PATH))
os.environ.setdefault("YOLO_OBB_CONFIDENCE", "0.45")
os.environ.setdefault("YOLO_OBB_IMAGE_SIZE", "1024")
os.environ.setdefault("GEMINI_API_KEY", "local-pdpa-service-not-used")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import main  # noqa: E402


app = FastAPI(title="Medication VQA Local PDPA Masking Service")


def _check_token(received_token: str | None) -> None:
    expected_token = os.environ.get("LOCAL_PDPA_SERVICE_TOKEN", "").strip()
    if not expected_token:
        return
    if not received_token or received_token.strip() != expected_token:
        raise HTTPException(status_code=401, detail="unauthorized")


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _debug_dir() -> Path:
    return Path(os.environ.get("LOCAL_PDPA_DEBUG_DIR", str(PROJECT_ROOT / "test" / "local_pdpa_debug")))


def _debug_enabled() -> bool:
    return os.environ.get("SAVE_LOCAL_PDPA_DEBUG_IMAGES", "true").strip().lower() in ("1", "true", "yes", "on")


def _draw_yolo_overlay(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    detections, _ = main._predict_yolo_obb(str(image_path))
    if detections is None:
        return

    overlay = image.copy()
    for detection in detections:
        quad = np.round(detection["quad"]).astype(np.int32)
        color = (0, 180, 255)
        label = str(detection["class_id"])
        if detection["class_id"] == main.get_yolo_obb_label_class_id():
            color = (0, 220, 0)
            label = "Medicine-Labels"
        elif detection["class_id"] == main.get_yolo_obb_header_class_id():
            color = (0, 0, 255)
            label = "patient_header"

        cv2.polylines(overlay, [quad], True, color, 3)
        x, y = quad[0]
        cv2.putText(
            overlay,
            f"{label} {detection['confidence']:.2f}",
            (int(x), max(24, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), overlay)


def _save_compare_image(paths: list[Path], output_path: Path) -> None:
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        target_h = 520
        scale = target_h / image.shape[0]
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)
        images.append(resized)

    if not images:
        return

    gap = np.full((images[0].shape[0], 18, 3), 255, dtype=np.uint8)
    canvas = images[0]
    for image in images[1:]:
        canvas = np.hstack([canvas, gap, image])
    cv2.imwrite(str(output_path), canvas)


def _save_debug_images(work_dir: Path, request_id: str) -> None:
    if not _debug_enabled():
        return

    output_dir = _debug_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{request_id}"

    input_output = output_dir / f"{stem}_input_received.jpg"
    overlay_output = output_dir / f"{stem}_yolo_overlay.jpg"
    rectified_output = output_dir / f"{stem}_rectified.jpg"
    safe_output = output_dir / f"{stem}_safe.jpg"
    compare_output = output_dir / f"{stem}_compare.jpg"

    shutil.copyfile(work_dir / "input.jpg", input_output)
    if (work_dir / "rectified.jpg").exists():
        shutil.copyfile(work_dir / "rectified.jpg", rectified_output)
    if (work_dir / "safe.jpg").exists():
        shutil.copyfile(work_dir / "safe.jpg", safe_output)

    try:
        _draw_yolo_overlay(work_dir / "input.jpg", overlay_output)
        _save_compare_image(
            [input_output, overlay_output, rectified_output, safe_output],
            compare_output,
        )
    except Exception as e:
        print(f"Local PDPA debug image save skipped: {e}")


@app.get("/health")
def health(load_model: bool = False):
    model_path = Path(os.environ.get("YOLO_OBB_MODEL_PATH", str(MODEL_PATH)))
    response = {
        "ok": True,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "yolo_enabled": main.is_yolo_obb_enabled(),
    }
    if load_model:
        response["model_loaded"] = main.get_yolo_obb_model() is not None
    return response


@app.post("/mask")
async def mask_image(request: Request, x_pdpa_token: str | None = Header(default=None)):
    _check_token(x_pdpa_token)

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty_image_body")

    work_dir = Path(tempfile.mkdtemp(prefix="local_pdpa_"))
    input_path = work_dir / "input.jpg"
    rectified_path = work_dir / "rectified.jpg"
    safe_path = work_dir / "safe.jpg"
    input_path.write_bytes(image_bytes)

    ok, message = main.create_yolo_obb_pdpa_safe_image(
        str(input_path),
        str(rectified_path),
        str(safe_path),
    )
    if not ok:
        _cleanup(work_dir)
        return JSONResponse(
            status_code=422,
            content={"ok": False, "message": message},
        )

    _save_debug_images(work_dir, "mask")

    return FileResponse(
        str(safe_path),
        media_type="image/jpeg",
        filename="safe.jpg",
        headers={"X-PDPA-Masking-Message": message},
        background=BackgroundTask(_cleanup, work_dir),
    )
