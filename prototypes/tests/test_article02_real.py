import unittest

from prototypes.real_experiments.article02_binary_digits import run_real_experiment


class Article02RealExperimentTests(unittest.TestCase):
    def test_digit_experiment_reports_fp32_binary_and_exact_packed_inference(self):
        result = run_real_experiment(seeds=(7,), epochs=2, timing_repeats=1)

        self.assertEqual(result["dataset"]["source_id"], "UCI-80-digits")
        self.assertEqual(result["evidence_type"], "measured_public_dataset")
        self.assertEqual(len(result["runs"]), 1)
        run = result["runs"][0]
        self.assertTrue(run["packed_forward_matches_binary_torch"])
        self.assertLessEqual(run["packed_max_abs_logit_error"], 1e-5)
        self.assertGreater(run["test_rows"], 0)
        self.assertGreater(run["binary_hidden_packed_bytes"], 0)
        self.assertLess(run["binary_total_weight_bytes"], run["fp32_weight_bytes"])


if __name__ == "__main__":
    unittest.main()
