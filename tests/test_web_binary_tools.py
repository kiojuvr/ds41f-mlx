"""CPU artifact/parser/security/insertion tests, not GPU/browser acceptance."""
import copy
import io
import json

from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject, TextStringObject
import pytest

from ds41f_mlx.web_tools import AcquiredResource, ToolRegistry, ToolError
from ds41f_mlx.web_artifacts import ArtifactStore
from ds41f_mlx import web_binary_tools as binary
from ds41f_mlx.web_budget import fit_tool_results
from ds41f_mlx.web_client import RuntimeHTTPError


def resource(data, kind):
    mime = {'pdf': 'application/pdf', 'image': 'image/png'}[kind]
    return AcquiredResource('https://example.com/source', 'https://example.com/final', mime, data, False, 0)


def image_bytes(format='PNG', size=(32, 32)):
    b = io.BytesIO(); Image.new('RGB', size, 'red').save(b, format=format); return b.getvalue()


def pdf_bytes(count=3, text=True, active=False, size=(400, 500)):
    writer = PdfWriter()
    for i in range(count):
        page = writer.add_blank_page(width=size[0], height=size[1])
        if text:
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
            stream = DecodedStreamObject(); stream.set_data(f'BT /F1 12 Tf 20 200 Td (Page {i+1} primary source text with sufficient readable content for auto mode.) Tj ET'.encode())
            page[NameObject('/Contents')] = writer._add_object(stream)
    if active:
        writer.add_js('app.alert("not executed")')
    b = io.BytesIO(); writer.write(b); return b.getvalue()


def acquire(monkeypatch, data, kind):
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs)); return resource(data, kind)
    monkeypatch.setattr(binary, 'acquire_public_resource', get)
    return calls


def payload(result):
    return json.loads(result.content[0]['text'] if isinstance(result.content, list) else result.content)


@pytest.fixture
def store(tmp_path): return ArtifactStore(tmp_path / 'artifacts.sqlite3')


def test_immutable_receipt_survives_process_store_reopen_and_detects_corruption(store):
    original = resource(image_bytes(), 'image')
    artifact = store.put(original, 'image')
    reopened = ArtifactStore(store.path)
    assert reopened.get(artifact.id, 'image').resource == original
    with reopened._connect() as db:
        db.execute('UPDATE artifacts SET data=? WHERE id=?', (b'changed', artifact.id))
    with pytest.raises(ToolError, match='integrity'): reopened.get(artifact.id, 'image')


def test_storage_ceiling_never_evicts_live_artifact(tmp_path):
    store = ArtifactStore(tmp_path / 'a.db', max_bytes=1024, max_count=1)
    kept = store.put(resource(b'12345', 'pdf'), 'pdf')
    with pytest.raises(ToolError) as error: store.put(resource(b'6', 'pdf'), 'pdf')
    assert error.value.code == 'artifact_resource_ceiling'
    assert store.get(kept.id, 'pdf').resource.data == b'12345'


def test_expired_artifact_never_reacquires(store, monkeypatch):
    a = store.put(resource(image_bytes(), 'image'), 'image')
    with store._connect() as db: db.execute('UPDATE artifacts SET expires=0')
    monkeypatch.setattr(binary, 'acquire_public_resource', lambda *a, **k: pytest.fail('reacquisition'))
    with pytest.raises(ToolError) as error: binary.FetchImageTool(store).run({'artifact_id': a.id})
    assert error.value.code == 'artifact_unavailable'
    store.cleanup()
    with store._connect() as db: assert db.execute('SELECT count(*) FROM artifacts').fetchone()[0] == 0


@pytest.mark.parametrize('format,mime', [('PNG','image/png'), ('JPEG','image/jpeg'), ('WEBP','image/webp')])
def test_original_image_becomes_actual_image_part_not_model_text(store, monkeypatch, format, mime):
    original = image_bytes(format)
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return AcquiredResource(url, url, mime, original, False, 0)
    monkeypatch.setattr(binary, 'acquire_public_resource', get)
    tool = binary.FetchImageTool(store)
    result = tool.run({'url':'https://example.com/image'})
    value = payload(result)
    assert len(result.content) == 2 and result.content[1]['type'] == 'image_url'
    assert 'base64' not in result.content[0]['text']
    import base64
    assert base64.b64decode(result.content[1]['image_url']['url'].split(',')[1]) == original
    again = binary.FetchImageTool(ArtifactStore(store.path)).run({'artifact_id': value['artifact_id']})
    assert again.content == result.content
    assert len(calls) == 1


@pytest.mark.parametrize('data,code', [(b'not png','malformed_image'), (image_bytes(size=(10,40)), 'vision_constraint_rejection')])
def test_image_failures_keep_completed_receipt(store, monkeypatch, data, code):
    calls = acquire(monkeypatch, data, 'image')
    result = binary.FetchImageTool(store).run({'url':'https://example.com/image'})
    value = payload(result)
    assert value['error_code'] == code
    assert store.get(value['artifact_id'], 'image').resource.data == data
    assert len(calls) == 1


def test_acquisition_storage_reserved_before_get(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / 'small.db', max_bytes=1024)
    monkeypatch.setattr(binary, 'acquire_public_resource', lambda *a, **k: pytest.fail('GET before storage reservation'))
    with pytest.raises(ToolError) as error:
        binary.FetchImageTool(store).run({'url': 'https://example.com/a.png'})
    assert error.value.code == 'artifact_resource_ceiling'


def test_text_pdf_metadata_selected_pages_and_retained_later_page(store, monkeypatch):
    calls = acquire(monkeypatch, pdf_bytes(count=17), 'pdf')
    tool = binary.FetchPDFTool(store)
    first = tool.run({'url':'https://example.com/paper.pdf', 'pages':[1,3], 'mode':'auto'})
    value = payload(first)
    assert 'error' not in value, value
    assert value['page_count'] == 17
    assert [p['page'] for p in value['pages']] == [1,3]
    assert 'Page 3' in value['pages'][1]['text']
    assert isinstance(first.content, str)
    later = binary.FetchPDFTool(ArtifactStore(store.path)).run({'artifact_id':value['artifact_id'], 'pages':[17], 'mode':'text'})
    assert 'Page 17' in payload(later)['pages'][0]['text']
    assert len(calls) == 1


@pytest.mark.parametrize('mode', ['auto','visual'])
def test_scanned_page_uses_vision_and_stable_local_raster(store, monkeypatch, mode):
    calls = acquire(monkeypatch, pdf_bytes(text=False), 'pdf')
    tool = binary.FetchPDFTool(store)
    first = tool.run({'url':'https://example.com/paper.pdf','mode':mode})
    value = payload(first)
    assert 'error' not in value, value
    assert isinstance(first.content, list)
    assert value['pages'][0]['interpretation'] == 'visual'
    assert value['pages'][0]['image_identity']['width'] == 600
    again = tool.run({'artifact_id': value['artifact_id'], 'mode':mode})
    assert again.content == first.content
    assert len(calls) == 1


@pytest.mark.parametrize('data,mode,code', [(b'%PDF-broken','text','pdf_parser_failure'),
    (pdf_bytes(active=True),'text','unsafe_pdf_capability'),
    (pdf_bytes(text=False,size=(14400,14400)),'visual','pdf_resource_ceiling')])
def test_pdf_failure_classes_retained_without_retry(store, monkeypatch, data, mode, code):
    calls = acquire(monkeypatch, data, 'pdf')
    result = binary.FetchPDFTool(store).run({'url':'https://example.com/a.pdf','mode':mode})
    value = payload(result)
    assert value['error_code'] == code, value
    assert store.get(value['artifact_id'], 'pdf').resource.data == data
    assert len(calls) == 1


def test_mixed_budget_changes_only_unobserved_page_text_never_images_or_history():
    image = binary.image_part(image_bytes(), 'image/png')
    content = [{'type':'text','text':json.dumps({'pages':[{'page':1,'text':'x'*100}]})},image]
    messages = [{'role':'tool','tool_call_id':'pdf','content':content}]
    history = [{'role':'user','content':[{'type':'text','text':'look'},copy.deepcopy(image)]}]
    base = {'messages':history, 'max_tokens':'auto'}
    original = copy.deepcopy((base,messages))
    class Runtime:
        bodies = []
        def request(self, method, path, body):
            self.bodies.append(copy.deepcopy(body))
            assert body['messages'][:-1] == history
            assert body['messages'][-1]['content'][1] == image
            text = json.loads(body['messages'][-1]['content'][0]['text'])['pages'][0]['text']
            if len(text) > 30:
                raise RuntimeHTTPError(400, json.dumps({'error':{'code':'context_capacity_exhausted'}}))
            return {'request_count':2}
    runtime = Runtime()
    out, _, _ = fit_tool_results(runtime,'session',2,base,messages,[{'tool':'fetch_pdf'}])
    assert len(json.loads(out[0]['content'][0]['text'])['pages'][0]['text']) == 30
    assert (base,messages) == original


def test_image_tool_part_passes_real_recipe_and_runtime_preparation(store, monkeypatch):
    acquire(monkeypatch,image_bytes(),'image')
    tool = binary.FetchImageTool(store)
    result = tool.run({'url':'https://example.com/a.png'})
    from ds41f_mlx.serving.server import prepare_request, load_v41_tokenizer
    messages = [{'role':'user','content':'Inspect the image.'},
        {'role':'assistant','content':'','tool_calls':[{'id':'a','type':'function','function':{'name':'fetch_image','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'a','content':result.content}]
    request = {'model':'deepseek-v4.1-flash','messages':messages,'tools':[tool.schema],'max_tokens':'auto'}
    prepared = prepare_request('chat_completions',json.dumps(request).encode(),tokenizer=load_v41_tokenizer())
    assert prepared.multimodal is not None
    assert prepared.multimodal.identities()[0]['sha256'] == payload(result)['sha256']


def test_retained_large_html_offset_and_no_old_excerpt_ceiling(store, monkeypatch):
    from ds41f_mlx.web_tools import FetchURLTool
    data = b'<html><body>' + b'x' * 90000 + b'</body></html>'
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return AcquiredResource(url, url, 'text/html; charset=utf-8', data, False, 0)
    monkeypatch.setattr(binary, 'acquire_public_resource', get)
    first = payload(FetchURLTool(store).run({'url':'https://example.com/page'}))
    assert first['acquired_bytes'] > 80000
    assert len(first['excerpt']) == 4096 and first['has_more']
    later = payload(FetchURLTool(ArtifactStore(store.path)).run({'artifact_id': first['artifact_id'], 'offset':4096, 'length':90000}))
    assert len(later['excerpt']) == 90000 - 4096
    assert not later['has_more'] and len(calls) == 1


def test_builtin_semantic_preflight_before_any_effect(store, monkeypatch):
    monkeypatch.setattr(binary, 'acquire_public_resource', lambda *a, **k: pytest.fail('external effect before complete semantic preflight'))
    registry = ToolRegistry([binary.FetchImageTool(store), binary.FetchPDFTool(store)])
    calls = [dict(id='first', function=dict(name='fetch_image',arguments=json.dumps({'url':'https://example.com/a.png'}))),
             dict(id='second',function=dict(name='fetch_pdf',arguments=json.dumps({'url':'https://example.com/a.pdf','pages':[False]})))]
    with pytest.raises(ToolError): registry.execute_calls(calls)


def test_legacy_eight_call_count_replaced_by_aggregate_bytes():
    from ds41f_mlx.web_tools import ToolResult
    class Tiny:
        name='tiny'; schema={}
        def run(self,args,context=None):return ToolResult('{}', {'tool':'tiny'})
    calls = [dict(id=str(i),function=dict(name='tiny',arguments='{}')) for i in range(10)]
    messages, displays = ToolRegistry([Tiny()]).execute_calls(calls)
    assert len(messages) == len(displays) == 10


def test_pdf_text_offset_and_cached_raster_do_not_reparse(store, monkeypatch):
    acquire(monkeypatch,pdf_bytes(),'pdf')
    tool = binary.FetchPDFTool(store)
    first = tool.run({'url':'https://example.com/paper.pdf','mode':'visual'})
    value = payload(first)
    monkeypatch.setattr(binary,'interpret_pdf',lambda *a, **k: pytest.fail('cached raster reparsed'))
    again = binary.FetchPDFTool(ArtifactStore(store.path)).run({'artifact_id':value['artifact_id'],'mode':'visual'})
    assert again.content == first.content
