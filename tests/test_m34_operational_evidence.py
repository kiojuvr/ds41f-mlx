"""M34 is fresh measured evidence, not an inheritance of M33/OFF soak claims."""
import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class M34Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q=json.loads((ROOT/'artifacts/m34/qualification.json').read_text())
        cls.soak=json.loads((ROOT/'artifacts/m34/soak.json').read_text())
    def test_gate(self):
        self.assertEqual(self.q['decision'],'OPERATIONALLY_QUALIFIED_BOUNDED_SINGLETON')
        self.assertTrue(all(self.q['checks'].values()))
        self.assertEqual(self.q['scope']['turns'],90)
        self.assertTrue(self.q['runtime_source_changed'])
        self.assertEqual(self.q['runtime_correction']['path'],'ds41f_mlx/runtime/mtp_lifecycle.py')
    def test_hashes(self):
        for path,sha in self.q['evidence_sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha)
        identity=self.q['identities']
        for path in ['ds41f_mlx/runtime/recipe_semantic_guard.py','ds41f_mlx/runtime/mtp_lifecycle.py','ds41f_mlx/prefill_fp8_mlx/handoff.py']:
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),self.q['current_runtime_sources'][path])
    def test_frontiers_and_terminal_ownership(self):
        for t in self.soak['turns']:
            with self.subTest(session=t['fresh_session'],turn=t['turn']):
                self.assertEqual(len(t['target_offsets']),40)
                self.assertEqual(set(t['target_offsets']+t['dspark_offsets']),{t['canonical_frontier']})
                self.assertTrue(t['queue_empty'] and t['retired_prediction'] and t['cache_ownership_preserved'])
                self.assertEqual(t['prompt_replay'],0)
                self.assertFalse(any(t['quiescence_counts'].values()))
                for owner in t['terminal_matches']:
                    # Exactly one canonical-token repair proves the completing
                    # response remained unforwarded until canonical emission.
                    self.assertEqual(t['quiescence']['counters']['target_forwards'],1)
                    self.assertEqual(t['quiescence']['counters']['dspark_appends'],1)
                    self.assertEqual(owner['ordinal'],t['generated']-1)
                    self.assertEqual(t['canonical_frontier'],t['prompt_frontier']+owner['ordinal']+1)
    def test_interruption_reentry(self):
        r=json.loads((ROOT/'artifacts/m34/recovery.json').read_text())
        self.assertTrue(r['turns'][8]['cancelled'])
        self.assertEqual(r['turns'][9]['suffix_start'],r['turns'][8]['canonical_frontier'])
        self.assertTrue(r['turns'][9]['prefix_preserving'])
        cases=json.loads((ROOT/'artifacts/m34/interruptions.json').read_text())['cases']
        self.assertEqual(sum(c['status']=='PASS_FAIL_CLOSED' for c in cases),3)
        self.assertTrue(all(c['status'] in ('PASS','PASS_FAIL_CLOSED') for c in cases))
    def test_controls_and_unsupported_scope(self):
        for r in self.q['controls']:
            self.assertGreater(r['speedup'],1.5)
            self.assertGreater(r['guard_vs_plain'],.9)
            self.assertGreaterEqual(r['output_tokens'],512)
        self.assertIn('MTP persistence/restore',self.q['unsupported'])
        self.assertIn('public MTP selector',self.q['unsupported'])
        self.assertEqual(self.q['replay'],0)
        self.assertEqual(self.q['repack'],0)
