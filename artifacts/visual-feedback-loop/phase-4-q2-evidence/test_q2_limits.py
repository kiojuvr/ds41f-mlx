"""Model-free budget/retention checks; no browser, generation, or admitted effect."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]
def handoff():
 s=importlib.util.spec_from_file_location('handoff',ROOT/'tools/phase3_render_handoff.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_eighth_request_rejected_before_publication(tmp_path,monkeypatch):
 monkeypatch.setenv('PHASE3_MAX_RENDERS','7');stage=tmp_path/'.render-handoff';stage.mkdir()
 for i in range(7): (stage/f'request-{i:02}.json').write_text('{}')
 (tmp_path/'index.html').write_text('unused')
 with pytest.raises(ValueError,match='budget'): handoff().request(tmp_path,timeout=.01)
 assert not (stage/'request-07.json').exists()
def test_seventh_host_id_admitted_but_eighth_not(tmp_path,monkeypatch):
 monkeypatch.setenv('PHASE3_MAX_RENDERS','7');(tmp_path/'.render-handoff').mkdir();e=tmp_path/'e';e.mkdir();(tmp_path/'index.html').write_text('changed')
 m=handoff();r=m.serve_one(tmp_path,e,dict(id=6,source_sha256='not-current',source_bytes=0))
 assert 'revision changed' in r['error']
 r=m.serve_one(tmp_path,e,dict(id=7,source_sha256='not-current',source_bytes=0))
 assert r['status']=='ERROR' and 'revision changed' not in r['error']
def test_q2_final_flush_precedes_backend_destruction():
 s=(Path(__file__).parent/'runtime.py').read_text()
 assert s.index('archive.final(backend.serving_traces)')<s.index('asyncio.run(backend.shutdown_serving())')
