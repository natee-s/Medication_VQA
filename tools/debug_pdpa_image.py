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
    safe_path: Path


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


def process_pdpa_debug_image(input_path: str | Path, output_dir: str | Path, run_qc: bool = True) -> PdpaDebugResult:
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
    safe_path = output_path / f"{output_stem}_safe.jpg"

    with tempfile.TemporaryDirectory() as temp_dir:
        working_path = Path(temp_dir) / source_path.name
        shutil.copy2(source_path, working_path)

        if run_qc:
            is_good, qc_message = main.check_image_quality(str(working_path))
            if not is_good:
                raise RuntimeError(f"QC failed: {qc_message}")

        normalize_ok, normalize_message = main.normalize_label_image_for_ai(str(working_path), str(normalized_path))
        if not normalize_ok:
            raise RuntimeError(f"Image preprocessing failed: {normalize_message}")

        pdpa_ok, pdpa_message = main.create_pdpa_safe_image(str(normalized_path), str(safe_path))
        if not pdpa_ok:
            raise RuntimeError(f"PDPA masking failed: {pdpa_message}")

    return PdpaDebugResult(normalized_path=normalized_path, safe_path=safe_path)


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
    args = parser.parse_args()

    try:
        result = process_pdpa_debug_image(args.image_path, args.out, run_qc=not args.skip_qc)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("PDPA debug images created:")
    print(f"normalized: {result.normalized_path}")
    print(f"safe:       {result.safe_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
