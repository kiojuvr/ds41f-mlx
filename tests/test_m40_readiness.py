"""Historical M40 evaluation consistency, not current/release qualification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReadinessEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_commit = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--', 'artifacts/m40/readiness.json'],
            cwd=ROOT, text=True).strip()
        assert cls.source_commit

    def test_recorded_authority(self):
        subprocess.run([sys.executable, str(ROOT / 'tools/record_m40_readiness.py'), '--check'],
                       cwd=ROOT, check=True)

    def test_not_promotion(self):
        r = json.loads((ROOT / 'artifacts/m40/readiness.json').read_text())
        self.assertEqual(r['decision'], 'NOT_READY_PUBLIC_ADMISSION_AND_REPRODUCIBLE_DELIVERY')
        self.assertFalse(r['release_qualified'])
        self.assertFalse(r['architectural_blocker_found'])
        self.assertEqual(r['production_default_mtp'], 'OFF')
        self.assertEqual(r['profiles']['mtp-singleton-v1']['status'], 'proposed only; not selectable')
        self.assertEqual([x['rank'] for x in r['ranked_required_blockers']], [1, 2, 3, 4])
        manifest = json.loads(subprocess.check_output(
            ['git', 'show', f'{self.source_commit}:release/ds41f-release.json'], cwd=ROOT))
        self.assertEqual(manifest, r['dependency_delta']['off_manifest'])
        self.assertEqual(manifest['runtime']['mtp'], 'OFF')

    def test_no_runtime_or_release_change_since_m39(self):
        r = json.loads((ROOT / 'artifacts/m40/readiness.json').read_text())
        result = subprocess.run(['git', 'diff', '--exit-code', r['evaluation_base'], self.source_commit, '--',
            'ds41f_mlx', 'native', 'release', 'rust', 'Cargo.toml', 'Cargo.lock', 'pyproject.toml'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_candidate_provenance_not_off_inheritance(self):
        r = json.loads((ROOT / 'artifacts/m40/readiness.json').read_text())
        mtp = r['dependency_delta']['mtp_historical_required_identity']
        off = r['dependency_delta']['off_manifest']['dependencies']
        self.assertNotEqual(mtp['omlx_candidate'], off['omlx']['revision'])
        self.assertNotEqual(mtp['recipe_candidate'], off['deepseek_recipe']['revision'])
        for name, key in [('omlx', 'omlx_candidate'), ('recipe', 'recipe_candidate')]:
            # Observations missing on a future machine do not turn into a release PASS.
            observed = r['observed_candidates'][name]
            self.assertEqual(observed['revision'], mtp[key])
            self.assertEqual(observed['git_status_porcelain'], '')
        self.assertEqual(r['observed_candidates']['recipe_native']['sha256'], mtp['recipe_native_sha256'])
        self.assertIn('36', r['historical_evidence'])
        self.assertEqual(r['historical_evidence']['36']['decision'], 'BLOCKED_CANONICAL_PROTOCOL_REPRESENTABILITY')

    def test_contract_matches_document(self):
        doc = (ROOT / 'docs/archive/development/milestone-40-release-readiness.md').read_text()
        r = json.loads((ROOT / 'artifacts/m40/readiness.json').read_text())
        self.assertIn(r['decision'], doc)
        p = r['profiles']['mtp-singleton-v1']
        self.assertEqual(p['max_live_sessions'], 1)
        self.assertEqual(p['max_prompt_plus_response_tokens'], 8192)
        self.assertEqual(p['max_output_tokens'], 768)
        self.assertTrue(p['restart_to_switch'])
        self.assertIn('200K HTTP MTP', r['deferred'])
        for m in range(33, 40):
            path = ROOT / f'artifacts/m{m}/qualification.json'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             r['historical_evidence'][str(m)]['sha256'])


if __name__ == '__main__':
    unittest.main()
