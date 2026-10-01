import os
import unittest
from unittest.mock import patch

from ds41f_mlx.prefill_fp8_mlx.p6_append import _segment_commands, _sweep_shell, SegmentMode
from ds41f_mlx.prefill_fp8_mlx.planner import SweepCommandKind, SweepPhase
from ds41f_mlx.prefill_fp8_mlx.tile_carry import (
    TileCarryError,
    TileCarryState,
    TileSpan,
    admit_tile_native,
    tile_native_enabled,
)


class FakeTensor:
    def __init__(self, name, rows, shape_tail=(4, 8), dtype="bf16", owned=False):
        self.name = name
        self.rows = int(rows)
        self.shape = (1, int(rows), *shape_tail)
        self.dtype = dtype
        self.owned = owned
        self.parts = [(name, 0, int(rows))]

    def __getitem__(self, item):
        row_slice = item[1]
        start = 0 if row_slice.start is None else int(row_slice.start)
        stop = self.rows if row_slice.stop is None else int(row_slice.stop)
        out = FakeTensor(f"{self.name}[{start}:{stop}]", stop - start, self.shape[2:], self.dtype)
        out.parts = [(self.name, start, stop)]
        return out

    def concat_rows(self, values):
        rows = sum(v.rows for v in values)
        out = FakeTensor("concat", rows, self.shape[2:], self.dtype)
        out.parts = []
        for v in values:
            out.parts.extend(v.parts)
        return out

    def copy(self):
        out = FakeTensor(f"copy({self.name})", self.rows, self.shape[2:], self.dtype, owned=True)
        out.parts = list(self.parts)
        return out


def p6_plan(count=16, mode=SegmentMode.FINAL_ENCODER_DECODER):
    return _sweep_shell(count, _segment_commands(seq=0, count=count, mode=mode), encoder_only=False)


class TileCarryTests(unittest.TestCase):
    def test_default_switch_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(tile_native_enabled())
        with patch.dict(os.environ, {"DS41F_P8_TILE_NATIVE_CARRY": "1"}):
            self.assertTrue(tile_native_enabled())

    def test_admission_requires_p6_source_lifecycle(self):
        ok, reason = admit_tile_native(p6_plan())
        self.assertTrue(ok, reason)

    def test_two_tile_transport_and_logical_swap(self):
        plan = p6_plan(count=16384)
        h = FakeTensor("h0", 16384)
        pre = FakeTensor("pre0", 16384, shape_tail=(4,), dtype="f32")
        state = TileCarryState.from_dense_initial(plan=plan, h_current=h, pre=pre, base_frontier=100)
        encodes = [c for c in plan.commands if c.kind is SweepCommandKind.ENCODE_ROWS and c.phase is SweepPhase.ENCODER and c.layer == 0]
        self.assertEqual([(c.offset, c.rows) for c in encodes], [(0, 8192), (8192, 8192)])
        for i, cmd in enumerate(encodes):
            h_in, pre_in, absolute_start = state.input_for(cmd, base_frontier=100)
            self.assertEqual(absolute_start, 100 + cmd.offset)
            state.bind_output(cmd, FakeTensor(f"h1_{i}", cmd.rows), FakeTensor(f"pre1_{i}", cmd.rows, shape_tail=(4,), dtype="f32"), base_frontier=100)
        state.logical_swap()
        h2, pre2, _ = state.input_for(encodes[1], base_frontier=100)
        self.assertEqual(h2.name, "h1_1")
        self.assertEqual(pre2.name, "pre1_1")
        self.assertEqual(state.telemetry.dense_carry_writes, 0)
        self.assertEqual(state.telemetry.logical_swaps, 1)

    def test_missing_tile_fail_closed(self):
        plan = p6_plan()
        state = TileCarryState.from_dense_initial(plan=plan, h_current=FakeTensor("h", 16), pre=FakeTensor("p", 16, shape_tail=(4,)))
        cmd = [c for c in plan.commands if c.kind is SweepCommandKind.ENCODE_ROWS and c.phase is SweepPhase.ENCODER][0]
        del state.current_h[cmd.offset]
        with self.assertRaises(TileCarryError):
            state.input_for(cmd, base_frontier=0)

    def test_gap_overlap_rejected_at_assembly(self):
        state = TileCarryState(count=16, current_h={0: TileSpan(FakeTensor("a", 8), 0, 8, "h"), 9: TileSpan(FakeTensor("b", 7), 9, 7, "h")}, pre={0: TileSpan(FakeTensor("p", 16, shape_tail=(4,)), 0, 16, "p")})
        with self.assertRaises(TileCarryError):
            state.assemble_full_source()

    def test_ordered_full_source_assembly_and_single_tile_noop(self):
        state = TileCarryState(
            count=16,
            current_h={8: TileSpan(FakeTensor("h8", 8), 8, 8, "h"), 0: TileSpan(FakeTensor("h0", 8), 0, 8, "h")},
            pre={0: TileSpan(FakeTensor("p0", 16, shape_tail=(4,)), 0, 16, "p")},
        )
        h_full, pre_full = state.assemble_full_source()
        self.assertEqual(h_full.rows, 16)
        self.assertEqual(h_full.parts[0][0], "h0")
        self.assertEqual(h_full.parts[1][0], "h8")
        self.assertEqual(pre_full.name, "p0")
        self.assertEqual(state.telemetry.full_source_assemblies, 1)

    def test_final_cone_same_tile_and_cross_tile(self):
        state = TileCarryState(
            count=16,
            current_h={0: TileSpan(FakeTensor("h0", 8), 0, 8, "h"), 8: TileSpan(FakeTensor("h8", 8), 8, 8, "h")},
            next_h={0: TileSpan(FakeTensor("n0", 8), 0, 8, "n"), 8: TileSpan(FakeTensor("n8", 8), 8, 8, "n")},
            pre={0: TileSpan(FakeTensor("p0", 8, shape_tail=(4,)), 0, 8, "p"), 8: TileSpan(FakeTensor("p8", 8, shape_tail=(4,)), 8, 8, "p")},
        )
        same = state.cone_value("current", 9, 3)
        self.assertEqual(same.rows, 3)
        cross = state.cone_value("pre", 6, 6)
        self.assertEqual(cross.rows, 6)
        self.assertEqual(state.telemetry.final_cone_minimal_concats, 1)

    def test_retire_releases_tile_parents(self):
        state = TileCarryState(count=1, current_h={0: TileSpan(FakeTensor("h", 1), 0, 1, "h")}, pre={0: TileSpan(FakeTensor("p", 1), 0, 1, "p")})
        state.retire()
        self.assertTrue(state.retired)
        self.assertFalse(state.current_h)
        self.assertFalse(state.pre)


if __name__ == "__main__":
    unittest.main()
