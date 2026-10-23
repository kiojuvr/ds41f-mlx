"""Small controlled client safety checks; no model loading or generation."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('visual_loop', ROOT/'tools/accept_visual_feedback_loop.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def call(**kw):
    return dict(type='tool_use', id='opaque:Exact-ID', name='render', input={}) | kw


@pytest.mark.parametrize('bad', [call(id=''), call(id=None), call(input=[]), call(input={'x':1}),
    call(name='shell'), call(name='write_html',input={'html':'not html'})])
def test_bad_calls(bad):
    with pytest.raises(ValueError):
        v.validate_calls({'content':[bad]},set())


def test_duplicate_and_mismatched_ids():
    with pytest.raises(ValueError):
        v.validate_calls({'content':[call()]},{'opaque:Exact-ID'})
    with pytest.raises(ValueError):
        v.tool_result(call(),[],result_id='other')
    assert v.tool_result(call(),[])['content'][0]['tool_use_id']=='opaque:Exact-ID'


@pytest.mark.parametrize('html', ['<html><body><script>oops</script></body></html>',
    '<html><body onload="oops()">x</body></html>', '<html><body><style>@import "x";</style></body></html>',
    '<html><body><iframe src="file:///etc/passwd"></iframe></body></html>'])
def test_active_artifact_forbidden(html):
    with pytest.raises(ValueError):
        v.validate_html(html)


def test_body_budget():
    with pytest.raises(ValueError,match='16 MiB'):
        v.body_bytes({'x':'a'*v.LIMIT})
    v.validate_html(v.STARTER)


def test_unsettled_sse_never_authorizes_tools():
    c=v.NativeSettlement()
    c.feed(dict(type='message_start',message={'content':[]}))
    c.feed(dict(type='content_block_start',index=0,content_block=call()))
    c.feed(dict(type='content_block_stop',index=0))
    c.feed(dict(type='message_delta',delta={'stop_reason':'tool_use'}))
    with pytest.raises(ValueError,match='no message_stop'):
        c.finish()
    c.feed(dict(type='message_stop'))
    assert c.finish()['content']==[call()]


def test_malformed_completed_sse_arguments():
    c=v.NativeSettlement()
    for event in [dict(type='message_start',message={'content':[]}),
        dict(type='content_block_start',index=0,content_block=call()),
        dict(type='content_block_delta',index=0,delta={'type':'input_json_delta','partial_json':'{'}),
        dict(type='content_block_stop',index=0),dict(type='message_stop')]:
        c.feed(event)
    with pytest.raises(ValueError):
        c.finish()


def test_renderer_failure_and_timeout():
    with pytest.raises(RuntimeError,match='renderer failed'):
        v.bounded_process([sys.executable,'-c','raise RuntimeError("controlled")'],5)
    with pytest.raises(subprocess.TimeoutExpired):
        v.bounded_process([sys.executable,'-c','import time; time.sleep(60)'],.05)


def test_execution_process_needs_its_own_admission():
    # Isolated check: no checkpoint weights or model execution. Inspecting in a
    # different process cannot populate execution_identity's process-local bind.
    script = '''import json,runpy
from pathlib import Path
from ds41f_mlx.execution_identity import numerical_profile,MANIFEST
try:
    numerical_profile()
except RuntimeError as error:
    assert 'installed MTP admission' in str(error)
else:
    raise AssertionError('fresh process unexpectedly admitted')
v=runpy.run_path(SCRIPT)
a=json.loads(MANIFEST.read_text())
r=v['bind_execution_process'](a['checkpoint_variant'],a['provenance']['source_commit'])
assert numerical_profile()==a['numerics']==r['numerics']
print(json.dumps(r))
'''.replace('SCRIPT', repr(str(ROOT/'tools/accept_visual_feedback_loop.py')))
    result=subprocess.run([sys.executable,'-I','-c',script],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr


def test_blank_and_oversized_images(tmp_path):
    from PIL import Image
    spec=importlib.util.spec_from_file_location('renderer',ROOT/'tools/visual_feedback_renderer.py')
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    p=tmp_path/'blank.png'
    Image.new('RGB',(1280,720),'white').save(p)
    with pytest.raises(ValueError,match='blank'):
        renderer.check_png(p)
    from ds41f_mlx.runtime.multimodal import validate_encoded_image
    p=tmp_path/'oversized.png'
    Image.new('RGB',(2049,2049),'red').save(p)
    with pytest.raises(ValueError,match='pixel'):
        validate_encoded_image(p.read_bytes())
