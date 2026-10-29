"""Processor-authoritative ordinary image spans, not a qualification constant."""
from hashlib import sha256
from io import BytesIO
import json
from types import SimpleNamespace as NS

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
import pytest

from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.model_execution import processing
from ds41f_mlx.runtime import multimodal as mm
from ds41f_mlx.serving.capacity import resolve_capacity, vision_capacity
from tools.qualify_image_token_envelope import fixture_image
from test_image_files import http, body, inline, ref, upload
from test_production_mtp import MX, pair
from ds41f_mlx.serving.paired_checkpoint import PairedCheckpointAuthority

CASES = [((1024,2048),1014), ((1582,1084),1016), ((3024,588),1024)]


def prepare(size, config=None):
    cfg = config or ModelConfig(vision_n_layers=1)
    data = fixture_image(size)
    return mm.prepare_multimodal([7,cfg.image_token_id,8,9], [NS(kind='bytes',data=data)],
                                 cfg, ordinary=True), data


@pytest.mark.parametrize('size,tokens', CASES)
def test_real_planner_packing_order_and_absolute_merge(size, tokens):
    inp, data = prepare(size)
    span, = inp.spans
    cfg = inp._config
    nh,nw,ph,pw = processing.plan_image_grid(*size,cfg)
    with Image.open(BytesIO(data)) as image:
        patches,h,w,kinds = processing.image_patch_array(image,cfg)
        # Independent packing expression from the official Processor's layout.
        image = ImageOps.pad(image.convert('RGB'), (pw,ph), color=(127,127,127))
        x = (np.asarray(image, dtype=np.float32)/255-.5)/.5
        expected = x.transpose(2,0,1).reshape(3,h,14,w,14).transpose(1,3,0,2,4).reshape(h*w,3,14,14)
    np.testing.assert_array_equal(patches,expected)
    assert span.length == tokens == nh*(nw+1)+2
    assert span.grid == (h,w) == (ph//14,pw//14)
    assert span.kinds == tuple(kinds) == tuple([0]+([1]*nw+[2])*nh+[3])
    assert kinds.count(1) == nh*nw and kinds.count(2) == nh
    assert inp.token_ids == (7,)+(cfg.image_token_id,)*tokens+(8,9)
    assert span.digest == sha256(data).hexdigest() and np.isfinite(patches).all()
    values = mx.arange(tokens*2).reshape(tokens,2)
    embeds = mm.ImageEmbeddings(((span.start,values),),cfg.image_token_id)
    merged = embeds.merge(mx.zeros((1,len(inp.token_ids),2)),0)
    np.testing.assert_array_equal(np.asarray(merged)[0,1:1+tokens],np.asarray(values))
    assert np.all(np.asarray(merged)[0,[0,-2,-1]] == 0)
    inp.release_sources()


def test_processor_resizes_over_envelope_source_and_config_remains_authority(tmp_path):
    cfg = ModelConfig(vision_n_layers=1)
    # Unresized grid would be 14*(73+1)+2 == 1038 positions.
    inp,_ = prepare((3066,588),cfg)
    assert len(processing.image_token_kinds(14,73)) == 1038
    assert inp.spans[0].length <= cfg.vision_max_n_token
    inp.release_sources()
    cfg.vision_max_n_token = 1000
    inp,_ = prepare((1582,1084),cfg)
    assert inp.spans[0].length <= 1000
    (tmp_path/'config.json').write_text(json.dumps(dict(max_position_embeddings=8192,
        vision_config=dict(num_hidden_layers=1,max_image_tokens=1000))))
    assert vision_capacity(tmp_path)['max_image_tokens'] == 1000
    inp.release_sources()


def test_over_envelope_planner_result_is_still_rejected(monkeypatch):
    # Negative structural-fault test only. No successful layout uses a patched planner.
    original = processing.image_token_kinds
    monkeypatch.setattr(processing,'image_token_kinds',lambda h,w: original(h,w)+[1]*9)
    with pytest.raises(ValueError,match='processor token envelope'):
        prepare((1582,1084))  # 1016+9 > the real configured 1024


@pytest.mark.parametrize('size,tokens', CASES)
def test_context_and_output_reservation_count_the_actual_span(tmp_path,size,tokens):
    inp,_ = prepare(size)
    prompt = len(inp.token_ids)
    assert prompt == tokens+3
    (tmp_path/'config.json').write_text(json.dumps(dict(max_position_embeddings=prompt+16)))
    p = NS(token_ids=inp.token_ids,multimodal=inp,inference_options=NS(max_tokens=16))
    p.conversation_request = NS(resolve_max_tokens=lambda *_: p.inference_options.max_tokens)
    assert resolve_capacity(p,checkpoint=tmp_path,ordinary=True)['max_tokens'] == 16
    p.inference_options.max_tokens = 17
    with pytest.raises(ValueError,match='requested output'):
        resolve_capacity(p,checkpoint=tmp_path,ordinary=True)
    cfg = ModelConfig(vision_n_layers=1,max_seq_len=prompt-1)
    with pytest.raises(ValueError,match='context envelope'):
        prepare(size,cfg)
    inp.release_sources()


@pytest.mark.parametrize('size,tokens', CASES[1:])
def test_paired_identity_and_partial_coverage_remain_fail_closed(size,tokens):
    inp,_ = prepare(size)
    C = len(inp.token_ids)-2
    authority = PairedCheckpointAuthority('identity',MX)
    saved = authority.capture(*pair(C)[:2],inp.token_ids[:C],inp.identities())
    authority.publish(saved)
    restored = authority.acquire(list(inp.token_ids),images=inp.identities())
    assert restored is not None and restored[2] == list(saved.tokens)
    with pytest.raises(ValueError,match='intersects'):
        inp.validate_prefix(inp.spans[0].start+tokens-1,inp.identities())
    changed = inp.identities()
    changed[0]['sha256'] = '0'*64
    with pytest.raises(ValueError,match='identity'):
        saved.restore('identity',inp.token_ids,MX,changed)
    assert authority.acquire(list(inp.token_ids),images=changed) is None
    authority.clear()
    inp.release_sources()


@pytest.mark.parametrize('size,tokens', CASES[1:])
def test_inline_file_and_nested_tool_results_share_exact_spans(http,size,tokens):
    client,_,prepared = http
    data = fixture_image(size)
    key = upload(client,data=data)['id']
    def request(image):
        b = body([])
        b['tools'] = [dict(name='read_image',input_schema={'type':'object'})]
        b['messages'] += [dict(role='assistant',content=[dict(type='tool_use',id='call-exact',name='read_image',input={})]),
            dict(role='user',content=[dict(type='tool_result',tool_use_id='call-exact',content=[image]),
                                     dict(type='text',text='Reply only OK.')])]
        return b
    for image in (inline(data),ref(key)):
        r = client.post('/v1/messages',json=request(image))
        assert r.status_code == 200,r.text
    a,b = prepared
    assert a.token_ids == b.token_ids and a.multimodal.identities() == b.multimodal.identities()
    assert a.multimodal.spans[0].length == tokens
    for p in prepared:
        p.multimodal.release_sources()
