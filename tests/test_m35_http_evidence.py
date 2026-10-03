"""Canonical M35 artifact gates, separate from live checkpoint workloads."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/m35'


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = json.loads((ART/'qualification.json').read_text())
        cls.http = json.loads((ART/'http.json').read_text())
        cls.cases = {c['name']:c for c in cls.http['cases']}

    def test_identity_and_sources(self):
        self.assertEqual(self.q['decision'], 'HTTP_SSE_QUALIFIED_BOUNDED_INTERNAL_SINGLETON')
        self.assertEqual(self.http['status'], 'PASS')
        self.assertEqual(self.q['identities']['status'], 'PASS')
        for path, digest in self.q['source_hashes'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest, path)
        for path, digest in self.q['evidence_hashes'].items():
            self.assertEqual(hashlib.sha256((ART/path).read_bytes()).hexdigest(), digest, path)

    def test_canonical_client_prefix_and_idle_authority(self):
        for c in self.http['cases']:
            r = c['runtime']
            received = c['transport'].get('received_events')
            if received is None:
                received = [json.loads(e['data']) for e in c['transport']['rows'] if e['data']!='[DONE]']
            self.assertEqual(received, r['protocol_events'][:len(received)], c['name'])
            self.assertEqual(r['client_acknowledged_ordinal'], 0)
            self.assertEqual(set(r['target_offsets']+r['dspark_offsets']), {r['canonical_frontier']})
            self.assertTrue(r['queue_empty'] and r['prediction_retired'])
            self.assertEqual(r['prompt_replay'], 0)
            self.assertEqual(r['full_cache_repack'], 0)
        self.assertEqual(self.http['counts']['replay'], 0)
        self.assertEqual(self.http['counts']['repack'], 0)
        self.assertEqual(self.http['counts']['fresh_target_allocations'], 11)

    def test_real_cancellation_and_terminal_recovery(self):
        for name in ['disconnect_long', 'rust_iterator_drop']:
            r = self.cases[name]['runtime']
            self.assertTrue(r['cancelled'])
            self.assertLess(r['generated'], 100)
            self.assertEqual(r['canonical_frontier'], 89)
        r = self.cases['undelivered_suffix']
        self.assertFalse(r['transport']['rows'])
        self.assertTrue(r['runtime']['canonical_generated'])
        self.assertEqual(self.cases['undelivered_suffix_continuation']['runtime']['canonical_frontier'], 89)
        terminal = self.cases['semantic_terminal_disconnect']
        self.assertTrue(terminal['runtime']['terminal_matches'])
        self.assertEqual(len(terminal['runtime']['response']['choices'][0]['message']['tool_calls']), 1)
        self.assertEqual(self.cases['terminal_tool_result_reentry']['runtime']['canonical_frontier'], 373)

    def test_transport_does_not_replace_acceptance_or_model_throughput(self):
        control = self.q['direct_same_request_control']
        self.assertTrue(control['identical_tokens'])
        self.assertGreater(control['http_model_phase_ratio'], .90)
        self.assertLess(control['http_model_phase_ratio'], 1.10)
        slow = next(p for p in self.q['performance'] if p['case']=='slow_sse')
        self.assertGreater(slow['max_observed_client_lag_input_watermarks'], 20)
        self.assertGreater(slow['client_wall_s'], slow['model_phase_s']*1.5)
        normal = self.cases['normal_sse']['runtime']
        stats = normal['mtp_stats']
        report = next(p for p in self.q['performance'] if p['case']=='normal_sse')
        self.assertEqual(report['acceptance'], stats['accepts']/sum(stats['depth_drafted']))

    def test_scope_and_defect_evidence_are_not_promoted(self):
        self.assertEqual(self.q['server']['admission'], 'explicit Python backend object injection; no public selector')
        self.assertEqual(self.q['server']['max_live_sessions'], 1)
        self.assertEqual(self.q['server']['max_response_tokens'], 768)
        self.assertEqual(len(self.q['admission']), 13)
        self.assertEqual(self.q['native_interruption_cases'], 16)
        self.assertIn('public/default/release MTP', self.q['unsupported'])
        old = json.loads((ART/'cancellation-checkpoint-defect.json').read_text())
        dropped = next(c for c in old['cases'] if c['name']=='disconnect_long')
        self.assertEqual(dropped['runtime']['generated'], 490)
        self.assertFalse(dropped['runtime']['cancelled'])
