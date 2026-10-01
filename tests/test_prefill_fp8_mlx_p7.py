import unittest
from types import SimpleNamespace

from ds41f_mlx.prefill_fp8_mlx import (
    LoaderQualification,
    P7SchedulingError,
    PublicationManager,
    RecordingDonor,
    RequestArena,
    SchedulingCoordinator,
    SweepCommandKind,
    OfficialFP8MLXBlockRunner,
)
from ds41f_mlx.prefill_fp8_mlx.p6_append import P6AppendPlanner, SegmentMode
from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel, FakeTensor


class DiskEngramEmbedding:
    _p7_disk_engram = True

    def __init__(self):
        self.calls = []
        self.prefetched_ids = None
        self.read_rows = 0
        self.prefetch_hits = 0

    def __call__(self, h, ids, image_mask=None):
        self.calls.append(ids)
        if self.prefetched_ids == ids:
            self.prefetch_hits += 1
        else:
            self.read_rows += 1
        return h


class FakeP7LM(FakeLanguageModel):
    def __init__(self):
        super().__init__()
        self._engram_prefetch = RecordingDonor()
        self._p7_enable_overlap = False
        self.layers[1].engram = DiskEngramEmbedding()
        self.layers[14].engram = DiskEngramEmbedding()


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
        with self.assertRaises(P7SchedulingError):
            SchedulingCoordinator(p7_lm(), qualification=LoaderQualification(moe_expert_offload_resident_fraction=0.5))

    def test_p6_custom_segments_emit_authority_prefetch_positions(self):
        plan = P6AppendPlanner().fixture('A_24577')
        cmds = plan.segments[0].commands
        kinds = [c.kind for c in cmds]
        self.assertLess(kinds.index(SweepCommandKind.PREFETCH_ENGRAM0), next(i for i,c in enumerate(cmds) if c.kind is SweepCommandKind.BEGIN_LAYER and c.layer == 0))
        self.assertLess(kinds.index(SweepCommandKind.PREFETCH_ENGRAM1), next(i for i,c in enumerate(cmds) if c.kind is SweepCommandKind.BEGIN_LAYER and c.layer == 2))
        self.assertFalse([c for c in cmds if c.layer is not None and c.layer >= 20 and c.kind is SweepCommandKind.SSD_READ_AHEAD])

    def test_chunk_exact_prefetch_pipeline_and_materialization(self):
        lm = p7_lm()
        mx = RecordingMx()
        coord = SchedulingCoordinator(lm, mx=mx)
        cmds = P6AppendPlanner().plan(C=0, T=4096, deferral_enabled=False).segments[0].commands
        arena = RequestArena.from_plan(SimpleNamespace(count=4096, allocations=(), commands=cmds), token_ids=list(range(4096)), h_current=FakeTensor('h', rows=4096), h_next=FakeTensor('n', rows=4096), pre=FakeTensor('p', rows=4096), engram_hashes='hashes')
        # Bypass full publication topology execution; drive only scheduling around real commands.
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
        submit_ids = [call[2] for call in submits]
        self.assertIn((1, 0, 4096), submit_ids)
        self.assertIn((14, 0, 4096), submit_ids)
        consumes = [e for e in coord.telemetry.events if e['event'] == 'engram_consume']
        self.assertTrue(all(e['exact_prefetch'] for e in consumes))
        self.assertEqual(mx.eval_calls, [])
        self.assertEqual(mx.synchronize_calls, 0)
        self.assertTrue(mx.async_eval_calls)
        self.assertEqual(coord.read_ahead.ssd_read_ahead(3, coord.telemetry), 'ALREADY_READY')

    def test_donor_exact_match_constraint_and_mismatch_regression(self):
        emb = DiskEngramEmbedding()
        emb.prefetched_ids = tuple(range(16))
        emb('h', tuple(range(8)))
        self.assertEqual((emb.prefetch_hits, emb.read_rows), (0, 1))
        emb.prefetched_ids = tuple(range(8))
        emb('h', tuple(range(8)))
        self.assertEqual((emb.prefetch_hits, emb.read_rows), (1, 1))

    def test_revoke_drains_and_stale_cannot_submit_model_donor_not_closed(self):
        lm = p7_lm()
        coord = SchedulingCoordinator(lm)
        coord.revoke()
        self.assertEqual(lm._engram_prefetch.calls[-1][0], 'drain')
        self.assertFalse(hasattr(lm._engram_prefetch, 'closed'))
        with self.assertRaises(P7SchedulingError):
            coord.handle_command(P6AppendPlanner().plan(C=0, T=2048).segments[0].commands[0], None, lambda *a: None)


if __name__ == '__main__':
    unittest.main()
