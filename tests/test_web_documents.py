import base64
import io
import json
import pytest
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
from ds41f_mlx.web_artifacts import ArtifactStore
from ds41f_mlx.web_tools import AcquiredResource, ToolError, ToolRegistry
from ds41f_mlx.web_documents import DocumentPDFTool
from ds41f_mlx.serving.generation_policy import capability_contract


def pdf_bytes(count=1, long_text=None):
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')}))
    for number in range(1, count + 1):
        page = writer.add_blank_page(width=600, height=780)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        content = DecodedStreamObject()
        text = long_text if long_text is not None else f'Page {number}: retained document evidence needle{number}'
        size = 0.001 if long_text is not None else 18
        content.set_data(f'BT /F1 {size} Tf 40 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(content)
    stream = io.BytesIO(); writer.write(stream)
    return stream.getvalue()


class Vision:
    calls = 0
    fail = False
    def health(self):
        contract = capability_contract()
        return dict(status='ready', profile='standard-off', generation_policy=contract,
                    chat_defaults=dict(contract['defaults'], reasoning_effort='none', max_tokens='auto'))
    def request(self, method, path, request):
        self.calls += 1
        assert path == '/v1/chat/completions'
        assert len(request['messages']) == 1
        assert sum(p['type'] == 'image_url' for p in request['messages'][0]['content']) == 1
        assert request['max_tokens'] == 2048 and request['reasoning_effort'] == 'none'
        if self.fail: raise TimeoutError('unknown outcome')
        return dict(id='response-one', choices=[dict(message=dict(content='A derived observation needleVisual'), finish_reason='stop')])


def retained(tmp_path, count=1):
    store = ArtifactStore(tmp_path / 'a.db')
    original = pdf_bytes(count)
    artifact = store.put(AcquiredResource('upload:///evidence.pdf', 'upload:///evidence.pdf', 'application/pdf', original, False, 0), 'pdf')
    vision = Vision()
    return store, artifact, DocumentPDFTool(store, vision), vision


def test_acquire_is_text_only_and_additional_pages_never_reacquire(tmp_path, monkeypatch):
    from ds41f_mlx import web_binary_tools
    store = ArtifactStore(tmp_path / 'a.db')
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return AcquiredResource(url, url, 'application/pdf', pdf_bytes(6), False, 0)
    monkeypatch.setattr(web_binary_tools, 'acquire_public_resource', fetch)
    tool = DocumentPDFTool(store, Vision())
    result = tool.run(dict(action='acquire', url='https://example.com/doc.pdf', pages=[2]))
    assert isinstance(result.content, str) and 'image_url' not in result.content
    assert result.display['source_authority'] == 'original-pdf-bytes'
    identity = result.display['artifact_id']
    assert tool.run(dict(action='read', artifact_id=identity, pages=[6])).display['pages'][0]['page'] == 6
    assert calls == ['https://example.com/doc.pdf']
    with pytest.raises(ToolError): tool.validate(dict(action='read', url='https://example.com/doc.pdf'))


def test_long_search_read_noncontiguous_and_no_images(tmp_path):
    store, artifact, tool, _ = retained(tmp_path, 100)
    one = tool.run(dict(artifact_id=artifact.id, action='search', query='needle99')).display
    assert one['next_page'] == 33 and not one['matches']
    last = tool.run(dict(artifact_id=artifact.id, action='search', query='needle99', start_page=97)).display
    assert last['matches'][0]['page'] == 99 and last['matches'][0]['authority'] == 'parser-derived'
    result = tool.run(dict(artifact_id=artifact.id, action='read', pages=[99, 2, 50]))
    assert isinstance(result.content, str) and 'image_url' not in result.content
    assert [p['page'] for p in result.display['pages']] == [99, 2, 50]
    assert store.get(artifact.id, 'pdf').resource.data == pdf_bytes(100)


def test_observation_reuses_across_tool_instances(tmp_path):
    store, artifact, tool, vision = retained(tmp_path)
    result = tool.run(dict(artifact_id=artifact.id, action='observe'))
    assert isinstance(result.content, str) and 'image_url' not in result.content
    observation = result.display['pages'][0]['observation']
    assert observation['authority'] == 'model-derived-non-authoritative'
    assert observation['source_sha256'] == artifact.resource.sha256
    assert observation['state'] == 'completed'
    second = DocumentPDFTool(ArtifactStore(store.path), vision).run(dict(artifact_id=artifact.id, action='observe'))
    assert result.content == second.content and vision.calls == 1
    search = tool.run(dict(artifact_id=artifact.id, action='search', query='needleVisual')).display
    assert search['matches'][0]['authority'] == 'model-derived-non-authoritative'
    states = tool.run(dict(artifact_id=artifact.id, action='info')).display['pages']
    assert states[0]['observation_state'] == 'completed'
    assert not any(p.get('raster_retained') for p in states)  # no PNG per inspected page


def test_uncertain_and_process_death_never_retry(tmp_path):
    store, artifact, tool, vision = retained(tmp_path, 2)
    vision.fail = True
    args = dict(artifact_id=artifact.id, action='observe')
    assert tool.run(args).display['error_code'] == 'uncertain_document_observation'
    vision.fail = False
    assert tool.run(args).display['error_code'] == 'uncertain_document_observation'
    assert vision.calls == 1
    key = tool.observation_key(artifact, 2)
    assert store.reserve_interpretation(artifact.id, key, dict(state='reserved', page=2))
    assert not store.reserve_interpretation(artifact.id, key, dict(state='reserved', page=2))
    assert tool.run(dict(args, pages=[2])).display['error_code'] == 'uncertain_document_observation'
    assert vision.calls == 1


def test_policy_change_does_not_bypass_uncertain_page_fence(tmp_path):
    store, artifact, tool, vision = retained(tmp_path)
    old_key = 'pdf-page-observation-old:policy:' + artifact.resource.sha256 + ':1'
    assert store.reserve_interpretation(artifact.id, old_key, dict(page=1, state='reserved'))
    result = tool.run(dict(artifact_id=artifact.id, action='observe'))
    assert result.display['error_code'] == 'uncertain_document_observation'
    assert vision.calls == 0
    assert not store.reserve_interpretation(artifact.id, tool.observation_key(artifact, 1), dict(page=1, state='reserved'))


def test_read_offset_continuation_is_from_original_not_double_sliced(tmp_path):
    store = ArtifactStore(tmp_path / 'a.db')
    data = pdf_bytes(long_text='abcdef' * 5000)
    artifact = store.put(AcquiredResource('upload:///long.pdf', 'upload:///long.pdf', 'application/pdf', data, False, 0), 'pdf')
    tool = DocumentPDFTool(store, Vision())
    full = tool.text(artifact.id, [1])['pages'][0]['text']
    assert len(full) > 24000
    pieces, offset = [], 0
    while True:
        page = tool.run(dict(artifact_id=artifact.id, action='read', offset=offset)).display['pages'][0]
        pieces.append(page['text'])
        assert page['next_offset'] == offset + len(page['text']) <= len(full)
        if not page['has_more']:
            break
        assert page['context_truncated']
        offset = page['next_offset']
    assert ''.join(pieces) == full
    with pytest.raises(ToolError):
        tool.run(dict(artifact_id=artifact.id, action='read', offset=len(full) + 1))


def test_validation_and_expiry(tmp_path):
    store, artifact, tool, _ = retained(tmp_path)
    with pytest.raises(ToolError): tool.validate(dict(artifact_id=artifact.id, action='observe', pages=[1, 2]))
    with pytest.raises(ToolError): tool.validate(dict(artifact_id=artifact.id, action='search', query=''))
    with pytest.raises(ToolError): tool.validate(dict(artifact_id=artifact.id, action=['read']))
    store.delete(artifact.id)
    with pytest.raises(ToolError): tool.run(dict(artifact_id=artifact.id, action='info'))


def test_upload_receipt_and_ui_preview_not_model(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import ds41f_mlx.web as web
    store = ArtifactStore(tmp_path / 'a.db')
    vision = Vision()
    tool = DocumentPDFTool(store, vision)
    monkeypatch.setattr(web, 'registry_from_env', lambda: ToolRegistry([tool]))
    monkeypatch.setattr(web.RuntimeClient, 'health', lambda self: vision.health())
    with TestClient(web.create_app(), base_url='http://127.0.0.1') as client:
        identity = 'a' * 64
        data = pdf_bytes()
        value = dict(id=identity, name='source.pdf', data=base64.b64encode(data).decode())
        assert client.post('/api/documents', json=value).status_code == 200
        assert client.post('/api/documents', json=value).status_code == 400
        info = client.get('/api/documents/' + identity).json()
        assert info['filename'] == 'source.pdf' and info['page_count'] == 1
        assert info['pages'] == []  # discovery is not page extraction/model reading
        preview = client.get('/api/documents/' + identity + '/pages/1').json()
        assert preview['image'].startswith('data:image/png;base64,')
        assert preview['observation'] is None and vision.calls == 0
        assert not any(p.get('raster_retained') for p in store.document_page_states(identity))
        assert client.get('/api/documents/' + identity + '/original').content == data
        assert client.get('/api/documents/' + identity + '/pages/2').json()['error_code'] == 'pdf_page_selection'
        # Large static local PDFs are not limited by the public GET ceiling.
        writer = PdfWriter(clone_from=io.BytesIO(data))
        padding = DecodedStreamObject(); padding.set_data(b'x' * (9 * 1024 * 1024))
        writer.root_object[NameObject('/ApplicationPadding')] = writer._add_object(padding)
        stream = io.BytesIO(); writer.write(stream); large = stream.getvalue()
        assert len(large) > 8 * 1024 * 1024
        large_id = 'b' * 64
        assert client.post('/api/documents', json=dict(id=large_id, name='large.pdf', data=base64.b64encode(large).decode())).status_code == 200
        info = client.get('/api/documents/' + large_id).json()
        assert info['state'] == 'retained' and info['page_count'] == 1 and info['acquired_bytes'] == len(large)
        assert store.get(large_id, 'pdf').resource.data == large
