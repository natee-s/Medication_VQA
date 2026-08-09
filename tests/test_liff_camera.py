import os
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import httpx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

if "pytz" not in sys.modules:
    fake_pytz = types.ModuleType("pytz")
    fake_pytz.timezone = lambda name: None
    sys.modules["pytz"] = fake_pytz


class LiffCameraTests(unittest.IsolatedAsyncioTestCase):
    async def test_liff_camera_page_is_served(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/liff/camera")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Medication Label Camera", response.text)

    async def test_liff_camera_css_hides_preview_panel_until_capture(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/static/liff-camera/style.css")

        self.assertEqual(response.status_code, 200)
        self.assertIn(".preview-panel[hidden]", response.text)
        self.assertIn("display: none", response.text)

    async def test_liff_camera_css_keeps_guide_between_header_and_controls(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/static/liff-camera/style.css")

        self.assertEqual(response.status_code, 200)
        css = response.text
        self.assertIn("grid-template-rows: auto minmax(0, 1fr) auto", css)
        self.assertIn("grid-row: 1", css)
        self.assertIn("grid-row: 2", css)
        self.assertIn("grid-row: 3", css)

    async def test_liff_camera_js_prevents_double_capture_and_allows_retake_after_upload(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/static/liff-camera/app.js")

        self.assertEqual(response.status_code, 200)
        script = response.text
        self.assertIn("let isCapturing = false;", script)
        self.assertIn("if (isCapturing)", script)
        self.assertIn("captureButton.disabled = true;", script)
        self.assertIn("retakeButton.disabled = false;", script)
        self.assertIn("result.processing_queued", script)
        self.assertIn('setStatusKey("status_upload_success")', script)

    async def test_liff_camera_js_uses_safe_rear_camera_fallbacks(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            script_response = await client.get("/static/liff-camera/app.js")
            css_response = await client.get("/static/liff-camera/style.css")

        self.assertEqual(script_response.status_code, 200)
        self.assertEqual(css_response.status_code, 200)
        script = script_response.text
        css = css_response.text
        self.assertIn("CAMERA_IDEAL_WIDTH", script)
        self.assertIn("requestCameraStreamWithFallbacks", script)
        self.assertIn("maybeSwitchToBetterRearCamera", script)
        self.assertIn("switchCamera", script)
        self.assertIn("CAMERA_DEVICE_STORAGE_KEY", script)
        self.assertIn("CAMERA_PREFERENCE_STORAGE_KEY", script)
        self.assertIn("getSavedCameraDeviceId", script)
        self.assertIn("getSavedCameraPreference", script)
        self.assertIn("saveCameraPreference", script)
        self.assertIn("requestCameraStreamByPreference", script)
        self.assertIn("findCameraIndexByPreference", script)
        self.assertIn("getCameraSwitchCandidates", script)
        self.assertIn("scoreCameraLabel", script)
        self.assertIn("optimizeCameraTrack", script)
        self.assertIn("normalizeCameraFocus", script)
        self.assertIn("cameraDebugMode", script)
        self.assertIn('facingMode: { ideal: "environment" }', script)
        self.assertIn("deviceId: { ideal: deviceId }", script)
        self.assertNotIn('facingMode: { exact: "environment" }', script)
        self.assertIn("object-fit: contain", css)

    async def test_liff_camera_uploads_masked_blob_without_showing_masked_preview(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/static/liff-camera/app.js")

        self.assertEqual(response.status_code, 200)
        script = response.text
        self.assertIn("const PDPA_MASK_RATIO = 0.25;", script)
        self.assertIn("function applyGuidelinePdpaMask(context)", script)
        self.assertIn("context.fillRect(0, 0, OUTPUT_WIDTH, maskHeight);", script)
        self.assertIn("function canvasToJpegBlob()", script)
        self.assertLess(script.index("const previewBlob = await canvasToJpegBlob();"), script.index("applyGuidelinePdpaMask(context);"))
        self.assertLess(script.index("applyGuidelinePdpaMask(context);"), script.index("const maskedBlob = await canvasToJpegBlob();"))
        self.assertIn("capturedBlob = maskedBlob;", script)
        self.assertIn("capturedPreview.src = URL.createObjectURL(previewBlob);", script)
        self.assertIn("stopCameraStream();", script)
        self.assertIn("async function retake()", script)
        self.assertIn("await startCamera();", script)
        self.assertIn("body: capturedBlob", script)

    async def test_liff_camera_page_has_processing_overlay_and_preview_instruction(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            html_response = await client.get("/liff/camera")
            css_response = await client.get("/static/liff-camera/style.css")
            script_response = await client.get("/static/liff-camera/app.js")

        self.assertEqual(html_response.status_code, 200)
        self.assertEqual(css_response.status_code, 200)
        self.assertEqual(script_response.status_code, 200)
        self.assertIn('id="processingOverlay"', html_response.text)
        self.assertIn('id="previewInstruction"', html_response.text)
        self.assertIn(".processing-overlay", css_response.text)
        self.assertIn(".camera-shell.capture-processing-mode", css_response.text)
        self.assertIn(".preview-instruction", css_response.text)
        self.assertIn(".camera-shell.preview-mode .preview-panel", css_response.text)
        self.assertIn(".camera-shell.preview-mode .controls", css_response.text)
        self.assertIn('id="switchCameraButton"', html_response.text)
        self.assertIn(".camera-switch-button", css_response.text)
        self.assertIn("setProcessingMode", script_response.text)
        self.assertIn("setCaptureProcessingMode", script_response.text)
        self.assertIn("loadLiffMessages", script_response.text)
        self.assertIn("applyTranslations", script_response.text)
        self.assertIn("cameraShell.classList.toggle", script_response.text)
        self.assertIn("processing_captured", script_response.text)
        self.assertIn('setProcessingMode(true, t("processing_captured"))', script_response.text)
        self.assertIn("status_camera_denied", script_response.text)

    async def test_liff_camera_js_closes_liff_window_after_successful_upload(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/static/liff-camera/app.js")

        self.assertEqual(response.status_code, 200)
        self.assertIn("closeLiffWindowSoon", response.text)
        self.assertIn("window.liff.closeWindow", response.text)

    async def test_liff_upload_label_accepts_jpeg(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["LIFF_UPLOAD_DEBUG_DIR"] = temp_dir
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/liff/upload-label",
                    content=b"\xff\xd8\xff\xd9",
                    headers={"content-type": "image/jpeg"},
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["size_bytes"], 4)
            self.assertTrue(body["filename"].endswith(".jpg"))
            saved_files = list(Path(temp_dir).glob("*.jpg"))
            self.assertEqual(len(saved_files), 1)
            self.assertEqual(saved_files[0].read_bytes(), b"\xff\xd8\xff\xd9")

    async def test_liff_config_returns_liff_id_from_environment(self):
        import main

        os.environ["LIFF_ID"] = "1234567890-test"
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/liff/config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["liff_id"], "1234567890-test")

    async def test_liff_messages_returns_selected_language_copy(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        with patch.object(main, "get_user_language", return_value="zh"):
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/liff/messages?line_user_id=U123456789")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["language"], "zh")
        self.assertEqual(body["messages"]["capture_button"], "拍照")
        self.assertEqual(body["messages"]["switch_camera_button"], "切换相机")
        self.assertEqual(body["messages"]["processing_captured"], "拍照完成，正在处理...")
        self.assertIn("status_upload_success", body["messages"])

    async def test_liff_messages_falls_back_to_thai_without_line_user_id(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/liff/messages")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["language"], "th")
        self.assertEqual(body["messages"]["capture_button"], "ถ่ายรูป")
        self.assertEqual(body["messages"]["processing_captured"], "ถ่ายรูปเสร็จแล้ว กำลังประมวลผล...")

    async def test_liff_upload_label_records_line_user_id_metadata(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["LIFF_UPLOAD_DEBUG_DIR"] = temp_dir
            transport = httpx.ASGITransport(app=main.app)
            with patch.object(main, "process_liff_uploaded_label_image", return_value=None):
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/liff/upload-label",
                        content=b"\xff\xd8\xff\xd9",
                        headers={
                            "content-type": "image/jpeg",
                            "x-line-user-id": "U123456789",
                        },
                    )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["line_user_id"], "U123456789")
            metadata_files = list(Path(temp_dir).glob("*.json"))
            self.assertEqual(len(metadata_files), 1)
            self.assertIn("U123456789", metadata_files[0].read_text(encoding="utf-8"))

    async def test_liff_upload_label_queues_chat_processing_when_line_user_id_is_present(self):
        import main

        queued_jobs = []

        def fake_process_liff_upload(line_user_id, image_path, upload_id):
            queued_jobs.append((line_user_id, Path(image_path).name, upload_id))

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["LIFF_UPLOAD_DEBUG_DIR"] = temp_dir
            transport = httpx.ASGITransport(app=main.app)
            with patch.object(main, "process_liff_uploaded_label_image", side_effect=fake_process_liff_upload, create=True):
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/liff/upload-label",
                        content=b"\xff\xd8\xff\xd9",
                        headers={
                            "content-type": "image/jpeg",
                            "x-line-user-id": "U123456789",
                        },
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["processing_queued"])
        self.assertEqual(len(queued_jobs), 1)
        self.assertEqual(queued_jobs[0][0], "U123456789")
        self.assertEqual(queued_jobs[0][1], body["filename"])
        self.assertTrue(queued_jobs[0][2].startswith("liff_"))

    def test_liff_processing_removes_uploaded_files_when_debug_storage_is_not_enabled(self):
        import main

        old_debug_dir = os.environ.pop("LIFF_UPLOAD_DEBUG_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / "upload.jpg"
                metadata_path = image_path.with_suffix(".json")
                image_path.write_bytes(b"\xff\xd8\xff\xd9")
                metadata_path.write_text("{}", encoding="utf-8")

                with (
                    patch.object(main, "start_line_loading_animation", return_value=None),
                    patch.object(main, "build_liff_label_result_message", return_value=main.TextSendMessage(text="ok")),
                    patch.object(main.line_bot_api, "push_message", return_value=None),
                ):
                    main.process_liff_uploaded_label_image("U123456789", str(image_path), "upload")

                self.assertFalse(image_path.exists())
                self.assertFalse(metadata_path.exists())
        finally:
            if old_debug_dir is not None:
                os.environ["LIFF_UPLOAD_DEBUG_DIR"] = old_debug_dir

    def test_liff_processing_uses_liff_specific_quality_gate(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "upload.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xd9")

            with (
                patch.object(main, "check_liff_image_quality", return_value=(False, "LIFF QC failed")) as liff_qc,
                patch.object(main, "check_image_quality", side_effect=AssertionError("LINE QC should not run")),
            ):
                result = main.build_liff_label_result_message("U123456789", str(image_path), "upload")

            self.assertEqual(result.text, "LIFF QC failed")
            liff_qc.assert_called_once_with(str(image_path))

    def test_liff_mask_debug_image_is_saved_to_test_folder(self):
        import main

        old_debug_dir = os.environ.get("LIFF_MASK_DEBUG_DIR")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                debug_dir = Path(temp_dir) / "test"
                os.environ["LIFF_MASK_DEBUG_DIR"] = str(debug_dir)
                source_path = Path(temp_dir) / "safe.jpg"
                source_path.write_bytes(b"masked-image")

                saved_path = main.save_liff_mask_debug_image(str(source_path), "upload/id:01")

                self.assertIsNotNone(saved_path)
                self.assertEqual(saved_path.parent, debug_dir)
                self.assertEqual(saved_path.name, "upload_id_01_safe.jpg")
                self.assertEqual(saved_path.read_bytes(), b"masked-image")
        finally:
            if old_debug_dir is None:
                os.environ.pop("LIFF_MASK_DEBUG_DIR", None)
            else:
                os.environ["LIFF_MASK_DEBUG_DIR"] = old_debug_dir

    async def test_liff_mask_debug_page_requires_token(self):
        import main

        old_token = os.environ.pop("LIFF_DEBUG_TOKEN", None)
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get("/debug/liff-masked-images")

            self.assertEqual(response.status_code, 404)
        finally:
            if old_token is not None:
                os.environ["LIFF_DEBUG_TOKEN"] = old_token

    async def test_liff_mask_debug_page_lists_masked_images_with_token(self):
        import main

        old_debug_dir = os.environ.get("LIFF_MASK_DEBUG_DIR")
        old_token = os.environ.get("LIFF_DEBUG_TOKEN")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                debug_dir = Path(temp_dir) / "test"
                debug_dir.mkdir()
                (debug_dir / "upload_001_safe.jpg").write_bytes(b"fake-image")
                os.environ["LIFF_MASK_DEBUG_DIR"] = str(debug_dir)
                os.environ["LIFF_DEBUG_TOKEN"] = "secret-token"

                transport = httpx.ASGITransport(app=main.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.get("/debug/liff-masked-images?token=secret-token")

                self.assertEqual(response.status_code, 200)
                self.assertIn("LIFF Masked Image Debug", response.text)
                self.assertIn("upload_001_safe.jpg", response.text)
                self.assertIn("/debug/liff-masked-images/upload_001_safe.jpg?token=secret-token", response.text)
        finally:
            if old_debug_dir is None:
                os.environ.pop("LIFF_MASK_DEBUG_DIR", None)
            else:
                os.environ["LIFF_MASK_DEBUG_DIR"] = old_debug_dir
            if old_token is None:
                os.environ.pop("LIFF_DEBUG_TOKEN", None)
            else:
                os.environ["LIFF_DEBUG_TOKEN"] = old_token

    async def test_liff_mask_debug_file_serves_masked_image_with_token(self):
        import main

        old_debug_dir = os.environ.get("LIFF_MASK_DEBUG_DIR")
        old_token = os.environ.get("LIFF_DEBUG_TOKEN")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                debug_dir = Path(temp_dir) / "test"
                debug_dir.mkdir()
                (debug_dir / "upload_001_safe.jpg").write_bytes(b"fake-image")
                os.environ["LIFF_MASK_DEBUG_DIR"] = str(debug_dir)
                os.environ["LIFF_DEBUG_TOKEN"] = "secret-token"

                transport = httpx.ASGITransport(app=main.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.get("/debug/liff-masked-images/upload_001_safe.jpg?token=secret-token")

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"fake-image")
                self.assertIn("image/jpeg", response.headers["content-type"])
        finally:
            if old_debug_dir is None:
                os.environ.pop("LIFF_MASK_DEBUG_DIR", None)
            else:
                os.environ["LIFF_MASK_DEBUG_DIR"] = old_debug_dir
            if old_token is None:
                os.environ.pop("LIFF_DEBUG_TOKEN", None)
            else:
                os.environ["LIFF_DEBUG_TOKEN"] = old_token

    def test_liff_guideline_pdpa_mask_uses_fixed_header_ratio(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "liff_input.jpg")
            output_path = str(Path(temp_dir) / "liff_safe.jpg")
            image = np.full((main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH, 3), 255, dtype=np.uint8)
            cv2.putText(image, "PRIVATE HEADER", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 7)
            cv2.putText(image, "MYOXAN", (70, main.PDPA_MASK_HEIGHT + 90), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 0), 8)
            cv2.imwrite(input_path, image)

            ok, message = main.create_liff_guideline_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT, :] < 20), 0.99)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 45:main.PDPA_MASK_HEIGHT + 155, :] < 20), 0.35)

    def test_liff_copy_verification_rejects_unmasked_uploads(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "unmasked.jpg")
            output_path = str(Path(temp_dir) / "safe.jpg")
            image = np.full((main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH, 3), 255, dtype=np.uint8)
            cv2.putText(image, "PRIVATE HEADER", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 7)
            cv2.imwrite(input_path, image)

            ok, message = main.copy_verified_liff_masked_image(input_path, output_path)

            self.assertFalse(ok)
            self.assertIn("liff_header_not_masked", message)
            self.assertFalse(Path(output_path).exists())

    def test_yolo_obb_rectification_warps_detected_quad_to_standard_label(self):
        import main

        class FakeObb:
            xyxyxyxy = np.array([[[30, 20], [370, 35], [350, 250], [45, 230]]], dtype=np.float32)
            conf = np.array([0.91], dtype=np.float32)
            cls = np.array([0], dtype=np.float32)

        class FakeResult:
            obb = FakeObb()

        class FakeModel:
            def predict(self, **kwargs):
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "upload.jpg")
            output_path = str(Path(temp_dir) / "rectified.jpg")
            image = np.full((280, 420, 3), 240, dtype=np.uint8)
            cv2.polylines(image, [FakeObb.xyxyxyxy[0].astype(np.int32)], True, (0, 0, 0), 3)
            cv2.putText(image, "MYOXAN", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
            cv2.imwrite(input_path, image)

            with patch.object(main, "get_yolo_obb_model", return_value=FakeModel()):
                ok, message = main.rectify_label_image_with_yolo_obb(input_path, output_path)

            self.assertTrue(ok, message)
            rectified = cv2.imread(output_path)
        self.assertIsNotNone(rectified)
        self.assertEqual(rectified.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))

    def test_yolo_obb_rectification_selects_label_class_not_header(self):
        import main

        class FakeObb:
            xyxyxyxy = np.array(
                [
                    [[60, 35], [360, 35], [360, 80], [60, 80]],
                    [[30, 20], [390, 20], [390, 280], [30, 280]],
                ],
                dtype=np.float32,
            )
            conf = np.array([0.99, 0.82], dtype=np.float32)
            cls = np.array([1, 0], dtype=np.float32)

        class FakeResult:
            obb = FakeObb()

        class FakeModel:
            def predict(self, **kwargs):
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "upload.jpg")
            output_path = str(Path(temp_dir) / "rectified.jpg")
            image = np.full((320, 440, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (30, 20), (390, 280), (0, 0, 0), 3)
            cv2.putText(image, "HEADER", (70, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(image, "MYOXAN", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
            cv2.imwrite(input_path, image)

            with patch.object(main, "get_yolo_obb_model", return_value=FakeModel()):
                ok, message = main.rectify_label_image_with_yolo_obb(input_path, output_path)

            self.assertTrue(ok, message)
            rectified = cv2.imread(output_path)
            self.assertIsNotNone(rectified)
            self.assertEqual(rectified.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            self.assertLess(np.mean(rectified[:120, :] < 20), 0.25)

    def test_yolo_obb_pdpa_masks_to_patient_header_bottom(self):
        import main

        class FakeObb:
            xyxyxyxy = np.array(
                [
                    [[20, 20], [380, 20], [380, 280], [20, 280]],
                    [[40, 30], [360, 30], [360, 80], [40, 80]],
                ],
                dtype=np.float32,
            )
            conf = np.array([0.95, 0.93], dtype=np.float32)
            cls = np.array([0, 1], dtype=np.float32)

        class FakeResult:
            obb = FakeObb()

        class FakeModel:
            def predict(self, **kwargs):
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "upload.jpg")
            rectified_path = str(Path(temp_dir) / "rectified.jpg")
            safe_path = str(Path(temp_dir) / "safe.jpg")
            image = np.full((300, 420, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (380, 280), (0, 0, 0), 2)
            cv2.putText(image, "PRIVATE", (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(image, "MYOXAN", (65, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
            cv2.imwrite(input_path, image)

            with patch.object(main, "get_yolo_obb_model", return_value=FakeModel()):
                ok, message = main.create_yolo_obb_pdpa_safe_image(input_path, rectified_path, safe_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(safe_path)
            self.assertIsNotNone(safe_image)
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            self.assertGreater(np.mean(gray[:220, :] < 20), 0.98)
            self.assertLess(np.mean(gray[340:520, :] < 20), 0.15)

    def test_yolo_obb_pdpa_falls_back_to_top_quarter_without_header(self):
        import main

        class FakeObb:
            xyxyxyxy = np.array([[[20, 20], [380, 20], [380, 280], [20, 280]]], dtype=np.float32)
            conf = np.array([0.95], dtype=np.float32)
            cls = np.array([0], dtype=np.float32)

        class FakeResult:
            obb = FakeObb()

        class FakeModel:
            def predict(self, **kwargs):
                return [FakeResult()]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "upload.jpg")
            rectified_path = str(Path(temp_dir) / "rectified.jpg")
            safe_path = str(Path(temp_dir) / "safe.jpg")
            image = np.full((300, 420, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (380, 280), (0, 0, 0), 2)
            cv2.putText(image, "MYOXAN", (65, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
            cv2.imwrite(input_path, image)

            with patch.object(main, "get_yolo_obb_model", return_value=FakeModel()):
                ok, message = main.create_yolo_obb_pdpa_safe_image(input_path, rectified_path, safe_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(safe_path)
            self.assertIsNotNone(safe_image)
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT - 5, :] < 20), 0.98)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 60:main.PDPA_MASK_HEIGHT + 220, :] < 20), 0.15)

    def test_upload_mask_debug_images_are_saved_to_test_folder(self):
        import main

        old_debug_dir = os.environ.get("UPLOAD_MASK_DEBUG_DIR")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                debug_dir = Path(temp_dir) / "test"
                os.environ["UPLOAD_MASK_DEBUG_DIR"] = str(debug_dir)
                rectified_path = Path(temp_dir) / "rectified.jpg"
                safe_path = Path(temp_dir) / "safe.jpg"
                rectified_path.write_bytes(b"rectified")
                safe_path.write_bytes(b"safe")

                saved_paths = main.save_upload_mask_debug_images(
                    str(rectified_path),
                    str(safe_path),
                    "message/id:01",
                )

                self.assertEqual(len(saved_paths), 2)
                self.assertEqual((debug_dir / "message_id_01_upload_rectified.jpg").read_bytes(), b"rectified")
                self.assertEqual((debug_dir / "message_id_01_upload_safe.jpg").read_bytes(), b"safe")
        finally:
            if old_debug_dir is None:
                os.environ.pop("UPLOAD_MASK_DEBUG_DIR", None)
            else:
                os.environ["UPLOAD_MASK_DEBUG_DIR"] = old_debug_dir

    def test_yolo_obb_is_disabled_on_render_unless_explicitly_enabled(self):
        import main

        old_render = os.environ.get("RENDER")
        old_enabled = os.environ.get("YOLO_OBB_ENABLED")
        try:
            os.environ["RENDER"] = "true"
            os.environ.pop("YOLO_OBB_ENABLED", None)
            self.assertFalse(main.is_yolo_obb_enabled())

            os.environ["YOLO_OBB_ENABLED"] = "true"
            self.assertTrue(main.is_yolo_obb_enabled())
        finally:
            if old_render is None:
                os.environ.pop("RENDER", None)
            else:
                os.environ["RENDER"] = old_render
            if old_enabled is None:
                os.environ.pop("YOLO_OBB_ENABLED", None)
            else:
                os.environ["YOLO_OBB_ENABLED"] = old_enabled

    def test_image_handler_sends_fallback_when_processing_raises(self):
        import main

        event = types.SimpleNamespace(
            source=types.SimpleNamespace(user_id="U123456789"),
            reply_token="reply-token",
        )

        with (
            patch.object(main, "get_user_language", return_value="th"),
            patch.object(main, "_handle_image_impl", side_effect=RuntimeError("boom")),
            patch.object(main, "reply_or_push_message") as reply_or_push,
        ):
            result = main.handle_image(event)

        self.assertIsNone(result)
        reply_or_push.assert_called_once()
        self.assertEqual(reply_or_push.call_args.args[1], "U123456789")
        self.assertEqual(reply_or_push.call_args.args[3].text, main.t("th", "image_processing_error"))

    def test_extract_label_ocr_and_match_uses_candidate_ranking(self):
        import main

        class FakeResponse:
            text = json.dumps(
                {
                    "error": None,
                    "trade_name": "MUCOLID 30MG.10 S.",
                    "generic_name": "AMBROXOL 30 mg",
                    "strength": "30 mg",
                    "dosage_frequency": "วันละ 3 ครั้ง",
                    "instruction_time": "หลังอาหาร เช้า-กลางวัน-เย็น",
                    "search_keyword": "AMBROXOL",
                    "search_candidates": ["AMBROXOL", "MUCOLID"],
                    "confidence": "high",
                }
            )

        db_row = {
            "trade_name": "MUCOLID 30MG.10 S.",
            "generic_name": "AMBROXOL 30 mg",
            "dosage_frequency": "วันละ 3 ครั้ง",
            "instruction_time": "หลังอาหาร เช้า-กลางวัน-เย็น",
        }

        with (
            patch.object(main.ai_client.models, "generate_content", return_value=FakeResponse()) as generate_content,
            patch.object(main, "search_medicine_candidates_in_db", return_value=(db_row, "AMBROXOL")) as search_candidates,
        ):
            result = main.extract_label_ocr_and_match(b"image-bytes", "th", source_label="unit-test")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["db_data"], db_row)
        self.assertEqual(result["matched_keyword"], "AMBROXOL")
        self.assertIn("config", generate_content.call_args.kwargs)
        search_candidates.assert_called_once()
        self.assertEqual(search_candidates.call_args.args[0], ["AMBROXOL", "MUCOLID"])
        self.assertEqual(search_candidates.call_args.args[1]["trade_name"], "MUCOLID 30MG.10 S.")

    def test_rectify_label_image_uses_yolo_obb_before_opencv_fallback(self):
        import main

        with patch.object(main, "rectify_label_image_with_yolo_obb", return_value=(True, "OK")) as yolo_rectify:
            ok, message = main.rectify_label_image_for_ai("input.jpg", "output.jpg")

        self.assertTrue(ok, message)
        yolo_rectify.assert_called_once_with("input.jpg", "output.jpg")

    def test_liff_processing_uses_verified_masked_upload_without_full_roi_pipeline(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "upload.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xd9")

            with (
                patch.object(main, "get_user_language", return_value="th"),
                patch.object(main, "check_liff_image_quality", return_value=(True, "OK")),
                patch.object(main, "normalize_label_image_for_ai", side_effect=AssertionError("LIFF should not normalize")),
                patch.object(main, "rectify_label_image_for_ai", side_effect=AssertionError("LIFF should not rectify")),
                patch.object(main, "create_pdpa_safe_image", side_effect=AssertionError("LIFF should not use generic PDPA mask")),
                patch.object(main, "create_liff_guideline_pdpa_safe_image", side_effect=AssertionError("LIFF should not mask again")),
                patch.object(main, "copy_verified_liff_masked_image", return_value=(False, "test_pdpa_failed")) as liff_verify,
            ):
                result = main.build_liff_label_result_message("U123456789", str(image_path), "upload")

            self.assertEqual(result.text, main.t("th", "pdpa_masking_failed"))
            liff_verify.assert_called_once()
            self.assertEqual(liff_verify.call_args.args[0], str(image_path))

    def test_ocr_candidates_prefer_generic_name_and_clean_dosage(self):
        import main

        candidates = main.extract_ocr_search_candidates(
            {
                "trade_name": "MYOXAN 50 MG 10'S",
                "generic_name": "TOLPERISONE 50 mg",
                "search_keyword": "TOLPERISONE 50 mg",
                "search_candidates": ["TOLPERISONE", "MYOXAN", "TOLI ERISONE"],
            }
        )

        self.assertEqual(candidates, ["TOLPERISONE", "MYOXAN", "TOLI ERISONE"])

    def test_fuzzy_medicine_search_tolerates_small_ocr_spacing_typo(self):
        import main

        class FakeResult:
            def __init__(self, data):
                self.data = data

        class FakeMedicineQuery:
            def __init__(self, rows):
                self.rows = rows
                self.exact_query = None

            def select(self, columns):
                return self

            def or_(self, query):
                self.exact_query = query
                return self

            def execute(self):
                if self.exact_query:
                    return FakeResult([])
                return FakeResult(self.rows)

        class FakeMedicineSupabase:
            def __init__(self, rows):
                self.rows = rows

            def table(self, name):
                self.assert_name = name
                return FakeMedicineQuery(self.rows)

        old_supabase = main.supabase
        main.supabase = FakeMedicineSupabase(
            [
                {"trade_name": "MYOXAN", "generic_name": "TOLPERISONE"},
                {"trade_name": "TYLENOL", "generic_name": "PARACETAMOL"},
            ]
        )
        try:
            db_data, matched_keyword = main.search_medicine_candidates_in_db(["TOLI ERISONE"])
        finally:
            main.supabase = old_supabase

        self.assertEqual(db_data["generic_name"], "TOLPERISONE")
        self.assertEqual(matched_keyword, "TOLI ERISONE")

    def test_variant_ranking_selects_matching_dosage_schedule(self):
        import main

        rows = [
            {
                "source_row_number": "337",
                "label_name": "1x3",
                "trade_name": "MYOXAN 50 MG 10'S",
                "generic_name": "TOLPERISONE 50 mg",
                "dosage_frequency": "ทานครั้งละ 1 เม็ด วันละ 3 ครั้ง",
                "instruction_time": "หลังอาหาร เช้า-กลางวัน-เย็น",
            },
            {
                "source_row_number": "338",
                "label_name": "1x2",
                "trade_name": "MYOXAN 50 MG 10'S",
                "generic_name": "TOLPERISONE 50 mg",
                "dosage_frequency": "ทานครั้งละ 1 เม็ด วันละ 2 ครั้ง",
                "instruction_time": "หลังอาหาร เช้า-เย็น",
            },
            {
                "source_row_number": "339",
                "label_name": "1x1",
                "trade_name": "MYOXAN 50 MG 10'S",
                "generic_name": "TOLPERISONE 50 mg",
                "dosage_frequency": "ทานครั้งละ 1 เม็ด วันละ 1 ครั้ง",
                "instruction_time": "หลังอาหาร เย็น",
            },
        ]

        ranked_rows = main.rank_medicine_rows(
            rows,
            ["TOLPERISONE", "MYOXAN"],
            {
                "trade_name": "MYOXAN",
                "generic_name": "TOLPERISONE",
                "strength": "50 mg",
                "dosage_frequency": "ทานครั้งละ 1 เม็ด วันละ 3 ครั้ง",
                "instruction_time": "หลังอาหาร เช้า-กลางวัน-เย็น",
            },
        )

        self.assertEqual(ranked_rows[0][0]["source_row_number"], "337")
        self.assertEqual(ranked_rows[0][0]["label_name"], "1x3")

    async def test_liff_upload_label_rejects_non_images(self):
        import main

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/liff/upload-label",
                content=b"not an image",
                headers={"content-type": "text/plain"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Only JPEG or PNG images are allowed")


if __name__ == "__main__":
    unittest.main()
