import os
import unittest
from unittest.mock import patch

from ds41f_mlx.config import load_runtime_config, RuntimeConfig, validate_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_environment_overrides_machine_specific_paths(self):
        env = {
            "DS41F_CHECKPOINT": "/tmp/checkpoint",
            "DS41F_OMLX_PATH": "/tmp/omlx",
            "DS41F_RECIPE_PATH": "/tmp/recipe",
            "DS41F_KV_ROOT": "/tmp/kv",
            "DS41F_HOST": "127.0.0.2",
            "DS41F_PORT": "9000",
            "DS41F_MAX_LIVE_SESSIONS": "7",
            "DS41F_TRACE_HISTORY_LIMIT": "11",
            "DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_runtime_config()
        self.assertEqual(str(cfg.checkpoint_path), "/tmp/checkpoint")
        self.assertEqual(str(cfg.omlx_path), "/tmp/omlx")
        self.assertEqual(str(cfg.recipe_path), "/tmp/recipe")
        self.assertEqual(str(cfg.kv_root), "/tmp/kv")
        self.assertEqual(cfg.host, "127.0.0.2")
        self.assertEqual(cfg.port, 9000)
        self.assertEqual(cfg.max_live_sessions, 7)
        self.assertEqual(cfg.trace_history_limit, 11)
        self.assertTrue(cfg.enable_diagnostics)

    def test_validation_fails_missing_required_paths_without_fallback(self):
        cfg = RuntimeConfig(checkpoint_path="/definitely/missing/checkpoint", omlx_path="/definitely/missing/omlx", recipe_path="/definitely/missing/recipe")
        findings = validate_runtime_config(cfg)
        failed = {f["key"] for f in findings if f["status"] == "FAIL"}
        self.assertIn("checkpoint_path", failed)
        self.assertIn("omlx_path", failed)
        self.assertIn("recipe_path", failed)


if __name__ == "__main__":
    unittest.main()
