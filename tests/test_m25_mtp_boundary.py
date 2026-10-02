"""Fail closed before MLX, disk access or cache mutation for unsupported MTP."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from ds41f_mlx.runtime.kv_persistence import (
    M9PersistenceError, _require_mtp_off_artifact_model,
    save_m8_idle_state, restore_m8_idle_state,
)
from tools.run_m25_mtp_boundary_probe import boundary


class MTPBoundaryTests(unittest.TestCase):
    def test_idle_requires_all40_committed_offsets(self):
        def cache(n, offset):
            return [SimpleNamespace(size=lambda: offset) for _ in range(n)]
        self.assertTrue(boundary(cache(40, 3), [1, 2, 3])["coherent"])
        self.assertFalse(boundary(cache(40, 4), [1, 2, 3])["coherent"])
        self.assertFalse(boundary(cache(39, 3), [1, 2, 3])["coherent"])

    def test_mtp_save_restore_rejected_before_io(self):
        for preserved, active in ((True, False), (True, True), (False, True)):
            for wrapped in (False, True):
                lm = SimpleNamespace(_config=SimpleNamespace(preserve_mtp=preserved),
                                     _omlx_mtp_decode_enabled=active)
                model = SimpleNamespace(language_model=lm) if wrapped else lm
                with tempfile.TemporaryDirectory() as d:
                    root = Path(d)/"must-not-exist"
                    with self.assertRaisesRegex(M9PersistenceError, "MTP-OFF"):
                        save_m8_idle_state(artifact_root=root, model=model, live_cache=[],
                                           all_tokens=[1], checkpoint=root)
                    with self.assertRaisesRegex(M9PersistenceError, "MTP-OFF"):
                        restore_m8_idle_state(artifact_path=root, model=model, checkpoint=root)
                    self.assertFalse(root.exists())

    def test_base_policy_unchanged(self):
        _require_mtp_off_artifact_model(SimpleNamespace(
            _config=SimpleNamespace(preserve_mtp=False), _omlx_mtp_decode_enabled=False))


if __name__ == "__main__":
    unittest.main()
