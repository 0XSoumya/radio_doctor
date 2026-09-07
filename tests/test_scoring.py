import unittest
from src.scoring import (
    normalize_text,
    get_token_weight,
    weighted_levenshtein,
    compute_res_case,
    parse_report_sections
)

class TestScoring(unittest.TestCase):
    def test_exact_match_zero_res(self):
        report = (
            "FINDINGS:\n"
            "BONES: No acute fracture or focal osseous lesion.\n"
            "JOINTS: No dislocation.\n\n"
            "IMPRESSION:\n"
            "No acute osseous abnormality."
        )
        template = report
        res = compute_res_case(report, report, template)
        self.assertEqual(res["res"], 0.0)
        self.assertEqual(res["findings_f"], 0.0)
        self.assertEqual(res["impression_i"], 0.0)

    def test_normalization_rules(self):
        # Units
        self.assertEqual(normalize_text("10 millimeters"), ["10", "mm"])
        self.assertEqual(normalize_text("2.5centimeters"), ["2.5", "cm"])
        
        # Hyphens between letters removed
        self.assertEqual(normalize_text("osteo-arthritis"), ["osteoarthritis"])
        self.assertEqual(normalize_text("post-operative"), ["postoperative"])
        
        # Letter number boundaries tokenized
        self.assertEqual(normalize_text("L1-L2"), ["l", "1", "l", "2"])
        self.assertEqual(normalize_text("3mm"), ["3", "mm"])
        
        # List markers stripped
        self.assertEqual(normalize_text("1. No acute fracture."), ["no", "acute", "fracture"])
        self.assertEqual(normalize_text("- Normal alignment."), ["normal", "alignment"])
        self.assertEqual(normalize_text("• Pelvic ring intact."), ["pelvic", "ring", "intact"])
        
        # Leading +/- on numbers preserved
        self.assertEqual(normalize_text("+1.5 and -2.0"), ["+1.5", "and", "-2.0"])

    def test_token_weights(self):
        # Critical
        self.assertEqual(get_token_weight("no"), 4.0)
        self.assertEqual(get_token_weight("right"), 4.0)
        self.assertEqual(get_token_weight("mild"), 4.0)
        self.assertEqual(get_token_weight("acute"), 4.0)
        self.assertEqual(get_token_weight("3.5"), 4.0)
        self.assertEqual(get_token_weight("mm"), 4.0)
        
        # Function words
        self.assertEqual(get_token_weight("the"), 0.25)
        self.assertEqual(get_token_weight("and"), 0.25)
        self.assertEqual(get_token_weight("with"), 0.25)
        
        # Content words
        self.assertEqual(get_token_weight("fracture"), 2.0)
        self.assertEqual(get_token_weight("femur"), 2.0)
        self.assertEqual(get_token_weight("edema"), 2.0)

    def test_missing_field_penalty(self):
        template = "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\n\nIMPRESSION:\nNormal."
        reference = "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\n\nIMPRESSION:\nNormal."
        prediction_missing_joints = "FINDINGS:\nBONES: Normal.\n\nIMPRESSION:\nNormal."
        
        res = compute_res_case(prediction_missing_joints, reference, template)
        self.assertGreater(res["findings_f"], 0.0)
        self.assertGreater(res["res"], 0.0)

    def test_changed_field_weight(self):
        # When reference changes BONES from template, BONES has weight 3.0
        template = "FINDINGS:\nBONES: Normal.\nJOINTS: Normal.\n\nIMPRESSION:\nNormal."
        reference = "FINDINGS:\nBONES: Acute fracture.\nJOINTS: Normal.\n\nIMPRESSION:\nFracture."
        
        # If prediction leaves BONES as template (unchanged), distance on BONES is high
        prediction_template = template
        res = compute_res_case(prediction_template, reference, template)
        # BONES (changed, weight 3.0) has high edit distance
        # JOINTS (unchanged, weight 1.0) has 0.0 edit distance
        # F should reflect weighted average with weight 3 for BONES and 1 for JOINTS
        self.assertGreater(res["findings_f"], 0.5)

if __name__ == "__main__":
    unittest.main()
