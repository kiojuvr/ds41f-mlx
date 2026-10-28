"""Affected task-collector audit regressions using only this run's immutable receipts."""
import gzip,hashlib,json,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parent
class EvidenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.wire=[json.loads(x) for x in (OUT/'wire.jsonl').read_text().splitlines()]
  cls.trace=json.loads(gzip.decompress((OUT/'trace-00.json.gz').read_bytes()))
  cls.saved=json.loads((OUT/'persisted-00.json').read_text())
 def test_native_and_wire_events_match(self):
  self.assertEqual([r['event'] for r in self.wire if r['kind']=='sse_event'],self.trace['protocol_events'])
  self.assertEqual(self.trace['protocol_events'][-1],{'type':'message_stop'})
 def test_observer_error_after_terminal(self):
  terminal=next(i for i,r in enumerate(self.wire) if r['kind']=='native_message_stop')
  error=next(i for i,r in enumerate(self.wire) if r['kind']=='observer_read_error')
  self.assertGreater(error,terminal);self.assertTrue(self.wire[error]['message_stop_already_captured'])
  self.assertEqual(self.wire[error]['error_class'],'AbortError')
 def test_exact_request_budget_not_refunded(self):
  state=json.loads((OUT/'gate.jsonl').read_text().splitlines()[-1])['state']
  self.assertEqual((state['attempts'],state['http'],state['ordinary'],state['allocation']),(1,1,1,1200))
  self.assertEqual(state['effects'],[])
 def test_settlement_and_complete_pair(self):
  t=self.trace;self.assertEqual(t['authority_connection']['disposition'],'settled')
  self.assertEqual(t['target_offsets'],[623]*40);self.assertEqual(t['dspark_offsets'],[623]*3)
  self.assertTrue(t['cache_published'] and t['prediction_retired'] and t['queue_empty'])
  self.assertFalse(t['cancelled']);self.assertEqual(t['prompt_replay']+t['full_cache_repack'],0)
 def test_denied_tool_is_not_read_effect(self):
  results=[e for e in self.saved['events'] if e['type']=='tool/result']
  self.assertEqual(len(results),1);self.assertTrue(results[0]['data']['message']['isError'])
  self.assertFalse(any(r.get('path')=='/v1/files' for r in self.wire))
  self.assertEqual(self.saved['compaction'],[])
 def test_frozen_authorities_unchanged(self):
  for name,digest in json.loads((OUT/'frozen-inputs.json').read_text()).items():self.assertEqual(hashlib.sha256((OUT.parents[2]/name).read_bytes()).hexdigest(),digest)
if __name__=='__main__':unittest.main()
