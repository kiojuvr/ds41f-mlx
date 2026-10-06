"""Final CPU resource/metadata/failure regressions, not standalone GPU evidence."""
import io
import json

from PIL import Image
import pytest

from ds41f_mlx import web_tools as tools
from ds41f_mlx import web_binary_tools as binary
from ds41f_mlx.web_artifacts import ArtifactStore
from test_web_acquisition import Response, network, CounterTool, call


def test_content_length_completion_never_touches_closed_socket(monkeypatch):
    response=Response(b'data','text/plain');response.length=4
    connections=network(monkeypatch,response)
    original=response.read1
    def read(size):
        value=original(size);response.length-=len(value)
        if response.length==0:
            connections[0].sock.settimeout=lambda *a: pytest.fail('timeout on completed/closed socket')
        return value
    response.read1=read
    result=tools.acquire_public_resource('https://example.com/file',kind='text',max_bytes=5)
    assert result.data==b'data' and not result.truncated


def test_premature_content_length_is_acquisition_failure_not_complete_artifact(monkeypatch):
    response=Response(b'data','image/png');response.length=10
    original=response.read1
    def read(size):
        value=original(size);response.length-=len(value);return value
    response.read1=read
    network(monkeypatch,response)
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file',kind='image',max_bytes=20)
    assert error.value.code=='acquisition_failure'


def test_shutdown_eof_cannot_masquerade_as_success_after_deadline(monkeypatch):
    network(monkeypatch,Response(b'','image/png'))
    times=iter([0,0,0,0,0,46])
    monkeypatch.setattr(tools,'monotonic',lambda:next(times))
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file',kind='image',max_bytes=5)
    assert error.value.code=='resource_ceiling'


def test_mime_metadata_circuit_breaker_precedes_body_read(monkeypatch):
    response=Response(b'body','image/png;'+'x'*4096)
    network(monkeypatch,response)
    with pytest.raises(tools.ToolError) as error:
        tools.acquire_public_resource('https://example.com/file',kind='image',max_bytes=5)
    assert error.value.code=='resource_ceiling' and response.position==0


def test_elapsed_batch_breaker_defers_without_external_effect(monkeypatch):
    counter=CounterTool();times=iter([0,1,181])
    monkeypatch.setattr(tools,'monotonic',lambda:next(times))
    messages,_=tools.ToolRegistry([counter]).execute_calls([call('one'),call('two')])
    assert counter.effects==1
    assert json.loads(messages[1]['content'])['error_code']=='resource_deferred'
    assert 'NOT executed' in messages[1]['content']


@pytest.mark.parametrize('size',[(2049,2048)])
def test_vision_megapixel_rejection_retains_original_bytes(tmp_path,monkeypatch,size):
    stream=io.BytesIO();Image.new('RGB',size,'white').save(stream,format='PNG');data=stream.getvalue()
    store=ArtifactStore(tmp_path/'a.db')
    monkeypatch.setattr(binary,'acquire_public_resource',lambda url,**k:tools.AcquiredResource(url,url,'image/png',data,False,0))
    result=binary.FetchImageTool(store).run({'url':'https://example.com/large.png'})
    value=json.loads(result.content)
    assert value['error_code']=='vision_constraint_rejection'
    assert store.get(value['artifact_id'],'image').resource.data==data


def test_animated_png_is_vision_rejection_not_malformed(tmp_path,monkeypatch):
    stream=io.BytesIO();Image.new('RGB',(32,32),'red').save(stream,format='PNG',save_all=True,append_images=[Image.new('RGB',(32,32),'blue')],duration=50)
    data=stream.getvalue();store=ArtifactStore(tmp_path/'a.db')
    monkeypatch.setattr(binary,'acquire_public_resource',lambda url,**k:tools.AcquiredResource(url,url,'image/png',data,False,0))
    result=binary.FetchImageTool(store).run({'url':'https://example.com/animated.png'})
    assert json.loads(result.content)['error_code']=='vision_constraint_rejection'


def test_provider_json_and_text_urls_are_not_sliced():
    url='https://example.com/'+'a'*600
    for data in [json.dumps({'results':[{'url':url,'title':'Primary','snippet':'Evidence'}]}),f'Title: Primary\nURL: {url}\nEvidence']:
        result=tools._normalize_text_results(data,provider='test',max_results=1)
        assert result[0]['url']==url


def test_parser_diagnostics_not_exposed_in_tool_text(tmp_path):
    store=ArtifactStore(tmp_path/'a.db')
    artifact=store.put(tools.AcquiredResource('https://example.com/a.pdf','https://example.com/a.pdf','application/pdf',b'%PDF-broken',False,0),'pdf')
    result=binary.binary_failure(artifact,'pdf_parser_failure','SECRET PARSER INTERNAL STATE','fetch_pdf')
    assert 'SECRET' not in result.content and 'SECRET' not in result.display['error']
