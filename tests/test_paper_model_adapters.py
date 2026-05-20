import os
import sys
import unittest
from unittest.mock import patch

from peptide_helper.models import SequenceAnalysis
from peptide_helper.nodes.agents import (
    _parse_model_payload,
    _run_configured_model,
    neuropeptide_node,
)
from peptide_helper.state import create_initial_state, merge_multi_results


class PaperModelAdapterTests(unittest.TestCase):
    def test_parse_model_payload_preserves_paper_metadata(self):
        spec = {
            "name": "ToxiPep",
            "task": "toxicity",
            "endpoint": "peptide_toxicity",
            "metric": "toxicity_probability",
            "unit": "",
            "weight": 1.45,
            "source": "Computational and Structural Biotechnology Journal 2025",
            "repo_url": "https://github.com/GGCL7/ToxiPep",
            "paper_url": "https://doi.org/10.1016/j.csbj.2025.05.039",
        }
        evidence = _parse_model_payload(spec, '{"label":"toxic","score":0.91}')

        self.assertEqual(evidence.name, "ToxiPep")
        self.assertEqual(evidence.endpoint, "peptide_toxicity")
        self.assertEqual(evidence.metric, "toxicity_probability")
        self.assertEqual(evidence.repo_url, "https://github.com/GGCL7/ToxiPep")
        self.assertEqual(evidence.paper_url, "https://doi.org/10.1016/j.csbj.2025.05.039")
        self.assertEqual(evidence.raw_output["label"], "toxic")
        self.assertAlmostEqual(evidence.score, 0.91)

    def test_unconfigured_command_returns_unavailable_evidence(self):
        spec = {
            "name": "MSKDNP",
            "task": "neuropeptide",
            "weight": 1.4,
            "source": "Briefings in Bioinformatics 2025",
            "command_env": "PEPTIDE_HELPER_TEST_MISSING_CMD",
        }
        with patch.dict(os.environ, {}, clear=True):
            evidence = _run_configured_model(spec, "ACDE")

        self.assertEqual(evidence.status, "unavailable")
        self.assertIn("未配置", evidence.error)

    def test_invalid_json_command_returns_unavailable_evidence(self):
        spec = {
            "name": "BrokenModel",
            "task": "toxicity",
            "weight": 1.0,
            "source": "test",
            "command_env": "PEPTIDE_HELPER_TEST_BROKEN_CMD",
        }
        command = f"{sys.executable} -c \"print('not json')\""
        with patch.dict(os.environ, {"PEPTIDE_HELPER_TEST_BROKEN_CMD": command}):
            evidence = _run_configured_model(spec, "ACDE")

        self.assertEqual(evidence.status, "unavailable")
        self.assertEqual(evidence.error, "模型命令未输出有效 JSON")

    def test_neuropeptide_node_handles_multiple_sequences(self):
        command = (
            f"{sys.executable} -c "
            "\"import json; print(json.dumps({'label':'neuropeptide','score':0.83}))\""
        )
        state = create_initial_state(sequences=["ACDE", "KLMN"], user_request="判断是不是神经肽")

        with patch.dict(os.environ, {"PEPTIDE_HELPER_MSKDNP_CMD": command}):
            result = neuropeptide_node(state)

        self.assertEqual(len(result["multi_results"]), 2)
        self.assertTrue(result["neuropeptide_res"].is_neuropeptide)
        self.assertEqual(result["multi_results"][0].neuropeptide_res.model_results[0].status, "available")

    def test_merge_multi_results_includes_neuropeptide_result(self):
        left = [SequenceAnalysis(sequence="ACDE")]
        state = create_initial_state(sequences=["ACDE"], user_request="神经肽")
        command = (
            f"{sys.executable} -c "
            "\"import json; print(json.dumps({'label':'neuropeptide','score':0.83}))\""
        )

        with patch.dict(os.environ, {"PEPTIDE_HELPER_MSKDNP_CMD": command}):
            right = neuropeptide_node(state)["multi_results"]

        merged = merge_multi_results(left, right)
        self.assertIsNotNone(merged[0].neuropeptide_res)


if __name__ == "__main__":
    unittest.main()
