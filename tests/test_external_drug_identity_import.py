import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "postgres" / "scripts" / "import_external_drug_identity_csv.py"

spec = importlib.util.spec_from_file_location("import_external_drug_identity_csv", SCRIPT_PATH)
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


class ExternalDrugIdentityImportTests(unittest.TestCase):
    def test_detect_columns_supports_common_thai_fda_headers(self):
        mapping = importer.detect_columns(
            ["เลขทะเบียนตำรับยา", "ชื่อผลิตภัณฑ์", "ตัวยาสำคัญ", "ผู้รับอนุญาต"],
            {},
        )

        self.assertEqual(mapping["fda_registration_no"], "เลขทะเบียนตำรับยา")
        self.assertEqual(mapping["trade_name"], "ชื่อผลิตภัณฑ์")
        self.assertEqual(mapping["generic_name"], "ตัวยาสำคัญ")
        self.assertEqual(mapping["manufacturer"], "ผู้รับอนุญาต")

    def test_detect_columns_supports_master_tmt_headers(self):
        mapping = importer.detect_columns(
            ["TPUCode", "ActiveIngredient", "Strength", "Dosageform", "TradeName", "Manufacturer", "Status"],
            {},
        )

        self.assertEqual(mapping["source_key"], "TPUCode")
        self.assertEqual(mapping["tmt_code"], "TPUCode")
        self.assertEqual(mapping["active_ingredient"], "ActiveIngredient")
        self.assertEqual(mapping["generic_name"], "ActiveIngredient")
        self.assertEqual(mapping["strength"], "Strength")
        self.assertEqual(mapping["dosage_form"], "Dosageform")
        self.assertEqual(mapping["trade_name"], "TradeName")
        self.assertEqual(mapping["manufacturer"], "Manufacturer")
        self.assertEqual(mapping["registration_status"], "Status")

    def test_explicit_mapping_overrides_auto_detection(self):
        mapping = importer.detect_columns(
            ["product", "generic", "custom_reg"],
            {"fda_registration_no": "custom_reg"},
        )

        self.assertEqual(mapping["fda_registration_no"], "custom_reg")
        self.assertEqual(mapping["generic_name"], "generic")

    def test_normalize_csv_builds_stable_source_key_from_registration_no(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "thai_fda.csv"
            output_path = Path(temp_dir) / "normalized.csv"
            source_path.write_text(
                "เลขทะเบียนตำรับยา,ชื่อผลิตภัณฑ์,ตัวยาสำคัญ\n"
                "1A 123/68,TESTDRUG,PARACETAMOL\n",
                encoding="utf-8",
            )

            row_count, mapping = importer.normalize_csv(
                source_path,
                output_path,
                source_name="ThaiFDA",
                source_version="test",
                source_url="",
                batch_id="batch_test",
                explicit_mapping={},
                encoding="utf-8-sig",
            )

            self.assertEqual(row_count, 1)
            self.assertEqual(mapping["fda_registration_no"], "เลขทะเบียนตำรับยา")
            normalized_text = output_path.read_text(encoding="utf-8")
            self.assertIn("1A 123/68", normalized_text)
            self.assertIn("TESTDRUG", normalized_text)


if __name__ == "__main__":
    unittest.main()
