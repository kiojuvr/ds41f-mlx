"""Model-free focused evaluator regressions."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_shutdown_final_flush_captures_last_trace(tmp_path):
    m = module('phase3_trace_retention')
    archive = m.TraceArchive(tmp_path)
    traces = [dict(protocol_events=[dict(type='message_stop')], order=0)]
    archive.flush(traces)
    # Polling has ended; a settled request arrives just before shutdown.
    traces.append(dict(protocol_events=[dict(type='message_stop')], order=1))
    result = archive.final(traces)
    assert result['archived'] == result['final_in_memory'] == result['complete'] == 2
    import gzip
    assert [json.loads(gzip.decompress(p.read_bytes()))['order'] for p in sorted(tmp_path.glob('*.gz'))] == [0, 1]

def test_final_flush_precedes_backend_shutdown():
    source = (ROOT/'artifacts/visual-feedback-loop/phase-3-evidence/visual-loop/runtime.py').read_text()
    assert source.index('archive.final(backend.serving_traces)') < source.index('asyncio.run(backend.shutdown_serving())')

def test_incomplete_trace_is_explicit_failure(tmp_path):
    with pytest.raises(RuntimeError, match='incomplete'):
        module('phase3_trace_retention').TraceArchive(tmp_path).final([dict(protocol_events=[])])

def test_request_timeout_preserves_pending_and_identity(tmp_path):
    m = module('phase3_render_handoff')
    (tmp_path/'.render-handoff').mkdir()
    (tmp_path/'index.html').write_text('exact bytes')
    with pytest.raises(TimeoutError):
        m.request(tmp_path, timeout=.01)
    assert (tmp_path/'.render-handoff/pending').is_dir()
    record = json.loads((tmp_path/'.render-handoff/request-00.json').read_text())
    assert record['source_sha256'] == m.sha(b'exact bytes')
    with pytest.raises(FileExistsError):
        m.request(tmp_path, timeout=.01)

def test_handoff_publication_never_overwrites(tmp_path):
    m = module('phase3_render_handoff')
    p = tmp_path/'completion-00.json'
    m.atomic(p, dict(id=0, source='first'))
    with pytest.raises(FileExistsError):
        m.atomic(p, dict(id=0, source='different'))
    assert json.loads(p.read_text())['source'] == 'first'

def test_host_rejects_changed_revision(tmp_path):
    m = module('phase3_render_handoff')
    (tmp_path/'.render-handoff').mkdir()
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    (tmp_path/'index.html').write_text('changed')
    result = m.serve_one(tmp_path, evidence, dict(id=0, source_sha256=m.sha(b'original'), source_bytes=8))
    assert result['status'] == 'ERROR'
    assert 'revision changed' in result['error']
    assert not (tmp_path/'render-00.png').exists()
