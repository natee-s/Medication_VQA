import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key")

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

    async def test_liff_upload_label_records_line_user_id_metadata(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["LIFF_UPLOAD_DEBUG_DIR"] = temp_dir
            transport = httpx.ASGITransport(app=main.app)
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
