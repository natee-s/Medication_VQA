import json
import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from tools.evaluate_medical_safety import evaluate_cases, load_cases


class MedicalSafetyEvaluationTests(unittest.TestCase):
    def test_case_set_is_well_formed_and_covers_critical_risk(self):
        cases_path = PROJECT_ROOT / "datasets" / "evaluation" / "medical_safety_cases.json"
        default_context, cases = load_cases(cases_path)

        self.assertTrue(default_context["generic_name"])
        self.assertGreaterEqual(len(cases), 30)
        self.assertTrue(any(case["risk_level"] == "critical" for case in cases))
        self.assertTrue(any(case["area"] == "rag" for case in cases))

    def test_offline_evaluation_runs_without_a_live_model_call(self):
        cases_path = PROJECT_ROOT / "datasets" / "evaluation" / "medical_safety_cases.json"
        default_context, cases = load_cases(cases_path)

        rows, prompt_controls, rag_controls = evaluate_cases(
            default_context,
            cases,
            live=False,
            pause_seconds=0,
        )

        self.assertEqual(len(rows), len(cases))
        self.assertTrue(all(row.live_result == "not_run" for row in rows))
        self.assertTrue(prompt_controls["duration_no_guess"])
        self.assertTrue(rag_controls["generic_red_flag_escalation"])
        self.assertTrue(all(row.guardrail_result == "pass" for row in rows))


if __name__ == "__main__":
    unittest.main()
