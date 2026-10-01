import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from ds41f_mlx.prefill_fp8_mlx import (
    LoaderQualification,
    P7SchedulingError,
    RecordingDonor,
    RequestArena,
    SchedulingCoordinator,
    SweepCommandKind,
)
from ds41f_mlx.prefill_fp8_mlx.p6_append import DeferredPrefillAppend, P6AppendPlanner
from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel, FakeMx, FakeTensor, full_ready_cache


class FakeDiskEngramEmbedding:
    _p7_disk_engram = True

    def __init__(self):
        self.calls = []
        self._prefetched = None


class FakeEngram:
    def __init__(self):
        self.embed = FakeDiskEngramEmbedding()
        self.calls = []

    def __call__(self, h, ids, image_mask=None):
        self.calls.append(ids)
        return h


class FakeP7LM(FakeLanguageModel):
    def __init__(self):
        super().__init__()
        self._engram_prefetch = RecordingDonor()
        self._p7_enable_overlap = False
        self.layers[1].engram = FakeEngram()
        self.layers[14].engram = FakeEngram()

    def _hash(self, input_ids, prior_history, image_mask):
        self.hasher_calls.append(prior_history)
        return FakeTensor("hash", rows=input_ids.shape[1]), f"history_after_{prior_history}"


def p7_lm():
    return FakeP7LM()


class RecordingMx:
    def __init__(self):
        self.async_eval_calls = []
        self.eval_calls = []
        self.synchronize_calls = 0

    def async_eval(self, *values):
        self.async_eval_calls.append(values)

    def eval(self, *values):
        self.eval_calls.append(values)

    def synchronize(self):
        self.synchronize_calls += 1


def _assert_no_tensor_like(testcase, value, path='root'):
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_no_tensor_like(testcase, v, f'{path}.{k}')
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_no_tensor_like(testcase, v, f'{path}[{i}]')
        return
    testcase.assertFalse(hasattr(value, 'shape') and not isinstance(value, (str, bytes)), f'tensor-like retained at {path}: {value!r}')
    testcase.assertFalse(type(value).__name__ in {'Future', 'FakeTensor'}, f'model/future object retained at {path}: {value!r}')


class P7SchedulingTests(unittest.TestCase):
    def test_admission_full_resident_ssd_engram_only(self):
        lm = p7_lm()
        coord = SchedulingCoordinator(lm)
        self.assertTrue(coord.residency.admitted)
        lm._moe_offload_plan = object()
        with self.assertRaises(P7SchedulingError):
            SchedulingCoordinator(lm)
        lm = p7_lm()
        lm.layers[3].expert = SimpleNamespace(child=type('OffloadedExpert', (), {})())
        with self.assertRaises(P7SchedulingError):
            SchedulingCoordinator(lm)
        lm = p7_lm()
        lm.layers[1].engram = object()
        with self.assertRaises(P7SchedulingError):
            SchedulingCoordinator(lm)
        lm = p7_lm()
        lm.layers[1].engram.embed = object()
        with self.assertRaises(P7SchedulingError):
            SchedulingCoordinator(lm)
        with self.assertRaises(P7SchedulingError):
            SchedulingCoordinator(p7_lm(), qualification=LoaderQualification(moe_expert_offload_resident_fraction=0.5))

    def test_p6_custom_segments_emit_authority_prefetch_positions(self):
        plan = P6AppendPlanner().fixture('A_24577')
        cmds = plan.segments[0].commands
        kinds = [c.kind for c in cmds]
        self.assertLess(kinds.index(SweepCommandKind.PREFETCH_ENGRAM0), next(i for i,c in enumerate(cmds) if c.kind is SweepCommandKind.BEGIN_LAYER and c.layer == 0))
        self.assertLess(kinds.index(SweepCommandKind.PREFETCH_ENGRAM1), next(i for i,c in enumerate(cmds) if c.kind is SweepCommandKind.BEGIN_LAYER and c.layer == 2))
        self.assertFalse([c for c in cmds if c.layer is not None and c.layer >= 20 and c.kind is SweepCommandKind.SSD_READ_AHEAD])

    def test_chunk_exact_prefetch_pipeline_materialization_and_tensor_free_telemetry(self):
        lm = p7_lm()
        mx = RecordingMx()
        coord = SchedulingCoordinator(lm, mx=mx)
        cmds = P6AppendPlanner().plan(C=0, T=4096, deferral_enabled=False).segments[0].commands
        arena = RequestArena.from_plan(SimpleNamespace(count=4096, allocations=(), commands=cmds), token_ids=list(range(4096)), h_current=FakeTensor('h', rows=4096), h_next=FakeTensor('n', rows=4096), pre=FakeTensor('p', rows=4096), engram_hashes='hashes')
        def hash_slice(hashes, offset, rows, layer, language_model):
            return (layer, offset, rows)
        coord.set_command_stream(cmds)
        for c in cmds:
            coord.handle_command(c, arena, hash_slice)
            if c.kind is SweepCommandKind.ENCODE_ROWS and c.layer in (1, 14):
                ids = coord.before_engram_consumer(c, arena, 'h', 'pre', hash_slice)
                self.assertEqual(ids, (c.layer, c.offset, c.rows))
                coord.after_engram_consumer(c, arena, 'h2', 'pre', hash_slice)
        submits = [call for call in lm._engram_prefetch.calls if call[0] == 'submit']
        self.assertIs(submits[0][1], lm.layers[1].engram.embed)
        self.assertIs(submits[-1][1], lm.layers[14].engram.embed)
        submit_ids = [call[2] for call in submits]
        self.assertIn((1, 0, 4096), submit_ids)
        self.assertIn((14, 0, 4096), submit_ids)
        consumes = [e for e in coord.telemetry.events if e['event'] == 'engram_consume']
        self.assertTrue(all(e['logical_match'] and e['donor_issue_observed'] for e in consumes))
        for event in coord.telemetry.events:
            _assert_no_tensor_like(self, event)
        self.assertEqual(mx.eval_calls, [])
        self.assertEqual(mx.synchronize_calls, 0)
        self.assertTrue(mx.async_eval_calls)
        self.assertEqual(coord.read_ahead.ssd_read_ahead(3, coord.telemetry), 'ALREADY_READY')

    def test_source_only_close_drains_and_next_segment_reuses_model_donor(self):
        lm = p7_lm(); lm._p7_enable_overlap = True
        app = DeferredPrefillAppend.create(lm, full_ready_cache(0), list(range(24577)), committed_frontier=0, mx=FakeMx())
        app.begin()
        first_seg, second_seg = app.plan.segments[0], app.plan.segments[1]
        first = app._make_segment_execution(first_seg)
        first.runner.scheduling_coordinator.set_command_stream(first_seg.commands)
        first.runner.execute_command(first_seg.commands[0], first.arena)  # BEGIN_INVALIDATE
        first.runner.execute_command(first_seg.commands[1], first.arena)  # PREFETCH_ENGRAM0
        self.assertIsNotNone(lm._engram_prefetch._pending)
        first.runner.close()
        self.assertTrue(any(c[0] == 'drain' for c in lm._engram_prefetch.calls))
        self.assertFalse(hasattr(lm._engram_prefetch, 'closed'))
        self.assertIsNone(lm._engram_prefetch._pending)
        prior_submit_count = sum(1 for c in lm._engram_prefetch.calls if c[0] == 'submit')
        second = app._make_segment_execution(second_seg)
        second.runner.scheduling_coordinator.set_command_stream(second_seg.commands)
        second.runner.execute_command(second_seg.commands[0], second.arena)
        second.runner.execute_command(second_seg.commands[1], second.arena)
        self.assertIsNot(first.runner.scheduling_coordinator, second.runner.scheduling_coordinator)
        self.assertGreater(sum(1 for c in lm._engram_prefetch.calls if c[0] == 'submit'), prior_submit_count)
        self.assertTrue(any(e.get('event') == 'scheduling_revoked' for e in first.runner.scheduling_coordinator.telemetry.events))

    def test_revoke_drains_and_stale_cannot_submit_model_donor_not_closed(self):
        lm = p7_lm()
        coord = SchedulingCoordinator(lm)
        coord.revoke()
        self.assertEqual(lm._engram_prefetch.calls[-1][0], 'drain')
        self.assertFalse(hasattr(lm._engram_prefetch, 'closed'))
        with self.assertRaises(P7SchedulingError):
            coord.handle_command(P6AppendPlanner().plan(C=0, T=2048).segments[0].commands[0], None, lambda *a: None)

    def test_real_pinned_omlx_donor_exact_match_and_mismatch(self):
        try:
            from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding, EngramPrefetch
        except Exception as exc:
            self.skipTest(f'pinned oMLX unavailable: {exc}')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'tiny.tensor'
            raw = np.arange(32, dtype=np.uint16).reshape(16, 2)
            header = {'w': {'dtype': 'BF16', 'shape': [16, 2], 'data_offsets': [0, raw.nbytes]}}
            hb = json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q', len(hb)) + hb + raw.tobytes())
            embed = DiskEngramEmbedding(path, 'w', None)
            donor = EngramPrefetch()
            counts = {'background': 0, 'foreground': 0}
            original = embed._read_rows
            def wrapped(self_embed, host):
                if threading.current_thread().name.startswith('v41-engram'):
                    counts['background'] += 1
                else:
                    counts['foreground'] += 1
                return original(host)
            embed._read_rows = MethodType(wrapped, embed)
            try:
                donor.submit(embed, np.array([1, 2, 3], dtype=np.int64))
                self.assertIsNotNone(getattr(embed, '_prefetched', None))
                _ = embed(np.array([1, 2, 3], dtype=np.int64))
                donor.drain()
                self.assertEqual(counts, {'background': 1, 'foreground': 0})
                donor.submit(embed, np.array([4, 5, 6], dtype=np.int64))
                _ = embed(np.array([4, 5], dtype=np.int64))
                donor.drain()
                self.assertEqual(counts, {'background': 2, 'foreground': 1})
            finally:
                donor.close(); embed.close()


if __name__ == '__main__':
    unittest.main()
