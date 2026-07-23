# LIFF Camera Guideline MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LIFF camera page that helps users capture a medication label inside a fixed horizontal guide frame and uploads the cropped image to the backend.

**Architecture:** Serve a static LIFF camera web app from FastAPI. The browser opens the rear camera, draws a horizontal medication-label guide frame with a 25% header divider, crops the captured frame to `1344x1000`, initializes LIFF when `LIFF_ID` is configured, and posts raw JPEG bytes to a backend upload endpoint. The first backend MVP stores the upload and metadata in a temporary debug folder and returns JSON so mobile testing can verify the capture flow before connecting OCR/RAG.

**Tech Stack:** FastAPI, Starlette static files, browser `getUserMedia`, Canvas API, vanilla HTML/CSS/JavaScript.

## Global Constraints

- Do not change current PDPA masking behavior in this MVP.
- Use portrait mobile screen with a horizontal guide frame.
- Guide aspect ratio is `1.344`.
- Header divider is at `25%` from the top of the guide.
- Cropped upload output is `1344x1000`.
- Use rear camera by default: `facingMode: "environment"`.
- Keep backend upload endpoint testable without LINE/Gemini/Supabase credentials.

---

### Task 1: Backend LIFF Routes

**Files:**
- Modify: `main.py`
- Test: `tests/test_liff_camera.py`

**Interfaces:**
- Produces: `GET /liff/camera` returns the LIFF camera HTML.
- Produces: `GET /liff/config` returns `{"liff_id": "..."}` from the `LIFF_ID` environment variable.
- Produces: `POST /liff/upload-label` accepts raw JPEG or PNG bytes and returns JSON with `status`, `filename`, `size_bytes`.

- [ ] **Step 1: Write failing tests**

```python
from fastapi.testclient import TestClient


def test_liff_camera_page_is_served(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    import main

    client = TestClient(main.app)
    response = client.get("/liff/camera")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_liff_upload_label_accepts_jpeg(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("LIFF_UPLOAD_DEBUG_DIR", str(tmp_path))
    import main

    client = TestClient(main.app)
    response = client.post(
        "/liff/upload-label",
        content=b"\xff\xd8\xff\xd9",
        headers={"content-type": "image/jpeg"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["size_bytes"] == 4
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_liff_camera -v`

Expected: FAIL because `/liff/camera` and `/liff/upload-label` are not implemented.

- [ ] **Step 3: Implement minimal backend routes**

Add static file serving for `static/liff-camera`, return `index.html`, and write uploaded JPEG bytes to `LIFF_UPLOAD_DEBUG_DIR` or `/tmp/liff_uploads`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_liff_camera -v`

Expected: PASS.

### Task 2: Static LIFF Camera App

**Files:**
- Create: `static/liff-camera/index.html`
- Create: `static/liff-camera/style.css`
- Create: `static/liff-camera/app.js`

**Interfaces:**
- Consumes: `POST /liff/upload-label`.
- Produces: JPEG upload cropped to `1344x1000`.

- [ ] **Step 1: Add HTML shell**

Create a mobile-first page with video preview, guide overlay, capture button, retake button, and upload status.

- [ ] **Step 2: Add CSS guide frame**

Frame width is `92vw`, aspect ratio is `1.344 / 1`, max width fits mobile. Draw an inner divider at `25%` from top.

- [ ] **Step 3: Add JavaScript camera and crop**

Use `navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } } })`. On capture, compute the guide rectangle relative to the video element, draw that region to a `1344x1000` canvas, convert to JPEG blob, and upload.

- [ ] **Step 4: Manual mobile verification**

Open `https://<render-service-url>/liff/camera` on a phone. Confirm camera permission, guide frame, capture, preview, retake, and upload success.

### Task 3: LINE Setup Notes

**Files:**
- Create: `docs/liff-camera-setup.md`

**Interfaces:**
- Produces: beginner-friendly LIFF setup steps.

- [ ] **Step 1: Document LIFF channel setup**

Add steps for LINE Developers Console: create LIFF app, endpoint URL, add LIFF URL to Rich Menu, and test on mobile.

- [ ] **Step 2: Document environment variables**

Add optional `LIFF_UPLOAD_DEBUG_DIR` for local/backend upload debugging.

### Task 4: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_liff_camera -v`

- [ ] **Step 2: Run existing safety tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_language_core tests.test_pdpa tests.test_debug_pdpa_image tests.test_liff_camera -v`

- [ ] **Step 3: Compile check**

Run: `.\.venv\Scripts\python.exe -m py_compile main.py tools\debug_pdpa_image.py tests\test_liff_camera.py`
