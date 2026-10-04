"""Extraction mechanics: Git authority, full manifests, drift and one-way policy."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('promotion', ROOT/'tools/promote_release.py')
promotion = importlib.util.module_from_spec(spec); spec.loader.exec_module(promotion)
from ds41f_mlx.projection import verify


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='m43-mechanics-')
        self.base = Path(self.tmp.name)
        self.source = self.base/'source'
        subprocess.run(['git','clone','--quiet','--shared',str(ROOT),str(self.source)],check=True)
        self.old = promotion.ROOT; promotion.ROOT = self.source

    def tearDown(self):
        promotion.ROOT = self.old; self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git','-C',str(self.source),*args],stderr=subprocess.DEVNULL)

    def commit(self):
        self.git('add','-A'); self.git('-c','user.name=M43 test','-c','user.email=m43@local','commit','-qm','negative fixture')

    def test_deterministic_complete_projection_and_reverse_patch(self):
        a,b = self.base/'a',self.base/'b'
        m = promotion.promote('HEAD','R1',a)
        n = promotion.promote('HEAD','R1',b)
        self.assertEqual(m,n)
        self.assertEqual((a/'release/promotion.json').read_bytes(),(b/'release/promotion.json').read_bytes())
        self.assertEqual(verify(a),m)
        (a/'ds41f_mlx/config.py').write_text('runtime-only patch\n')
        with self.assertRaisesRegex(ValueError,'runtime-only'): verify(a)
        with self.assertRaisesRegex(ValueError,'empty'): promotion.promote('HEAD','R1',a)

    def test_dirty_input_rejected(self):
        (self.source/'ds41f_mlx/config.py').write_text('uncommitted source')
        with self.assertRaisesRegex(ValueError,'clean HEAD'): promotion.promote('HEAD','R1',self.base/'a')

    def test_reference_identity_drift_rejected(self):
        p=self.source/'reference/R1/manifest.json'; p.write_bytes(p.read_bytes()+b'\n'); self.commit()
        with self.assertRaisesRegex(ValueError,'identity drift'): promotion.promote('HEAD','R1',self.base/'a')

    def test_required_material_missing_rejected(self):
        (self.source/'native/CMakeLists.txt').unlink(); self.commit()
        with self.assertRaisesRegex(ValueError,'required source'): promotion.promote('HEAD','R1',self.base/'a')

    def test_unexpected_runtime_source_rejected(self):
        a=self.base/'a'; promotion.promote('HEAD','R1',a)
        (a/'runtime_only.py').write_text('unsupported')
        with self.assertRaisesRegex(ValueError,'unexpected'): verify(a)

    def test_qualification_input_mismatch_rejected(self):
        a=self.base/'a'; m=promotion.promote('HEAD','R1',a)
        q=self.base/'q.json'
        q.write_text(json.dumps(dict(status='PASS',source_commit=m['source_commit'],
             source_tree_sha256='0'*64,reference_sha256=m['reference']['sha256'])))
        with self.assertRaisesRegex(ValueError,'qualification'): promotion.promote('HEAD','R1',self.base/'b',q)


if __name__ == '__main__': unittest.main()
