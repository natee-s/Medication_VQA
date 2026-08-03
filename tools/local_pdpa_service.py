import os
import shutil
import sys
import tempfile
from pathlib import Path

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

    return FileResponse(
        str(safe_path),
        media_type="image/jpeg",
        filename="safe.jpg",
        headers={"X-PDPA-Masking-Message": message},
        background=BackgroundTask(_cleanup, work_dir),
    )
