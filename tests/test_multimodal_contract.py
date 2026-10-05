"""Affected input/lifecycle contracts; no full-model claims from these tests."""
import base64
from io import BytesIO
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from ds41f_mlx.model_execution.config import ModelConfig
from ds41f_mlx.runtime.multimodal import ImageEmbeddings, image_bytes, prepare_multimodal


def source(color='red', size=(400,300)):
    b=BytesIO(); Image.new('RGB',size,color).save(b,format='PNG')
    return SimpleNamespace(kind='data_url',data_url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode())


def config(): return ModelConfig(vision_n_layers=1)


def test_expanded_spans_and_identity_rejection():
    c=config(); ids=[1,c.image_token_id,2,c.image_token_id,3]
    inp=prepare_multimodal(ids,[source(),source('blue')],c)
    first,second=inp.spans
    assert first.start==1 and second.start==first.start+first.length+1
    assert inp.token_ids[first.start:first.start+first.length]==(c.image_token_id,)*first.length
    assert first.kinds[0]==0 and first.kinds[-1]==3
    assert first.kinds.count(2)==(first.grid[0]+2)//3
    frontier=second.start
    inp.validate_prefix(frontier,[first.identity()])
    with pytest.raises(ValueError,match='intersects'): inp.validate_prefix(2,[])
    changed=prepare_multimodal(ids,[source('green'),source('blue')],c)
    with pytest.raises(ValueError,match='identity'): changed.validate_prefix(frontier,[first.identity()])


@pytest.mark.parametrize('src',[
    SimpleNamespace(kind='url',url='http://localhost/secret'),
    SimpleNamespace(kind='bytes',data=b''),
    SimpleNamespace(kind='bytes',data=b'not an image'),
    SimpleNamespace(kind='data_url',data_url='data:image/png;base64,!!!!'),
    SimpleNamespace(kind='data_url',data_url='data:text/plain;base64,YQ=='),
    SimpleNamespace(kind='data_url',data_url='data:image/png,raw'),
])
def test_bad_images_fail_closed(src):
    with pytest.raises(ValueError): prepare_multimodal([1,config().image_token_id,2],[src],config())


def test_http_preprocessing_is_cpu_owned(monkeypatch):
    src=source()
    def forbidden(*a,**k): raise AssertionError('GPU allocation on input thread')
    monkeypatch.setattr(mx,'array',forbidden)
    inp=prepare_multimodal([1,config().image_token_id,2],[src],config())
    assert isinstance(inp.spans[0].patches,np.ndarray)


def test_counts_missing_terminal_and_disabled_tower():
    c=config()
    for ids,sources in [([1,2],[source()]),([1,c.image_token_id,2],[]),([1,c.image_token_id],[source()])]:
        with pytest.raises(ValueError): prepare_multimodal(ids,sources,c)
    with pytest.raises(ValueError): prepare_multimodal([1,c.image_token_id,2],[source()],ModelConfig())


def test_sparse_rows_absolute_slice_and_dtype():
    rows=ImageEmbeddings(((7,mx.arange(12).reshape(3,4).astype(mx.float32)),),129264)
    h=mx.zeros((1,4,4),mx.bfloat16)
    out=rows.merge(h,8)
    assert out.dtype==mx.bfloat16
    np.testing.assert_array_equal(np.asarray(out.astype(mx.float32))[0,:2],np.arange(4,12).reshape(2,4))
    np.testing.assert_array_equal(np.asarray(out.astype(mx.float32))[0,2:],0)


def test_committed_images_never_execute_tower(monkeypatch):
    inp=prepare_multimodal([1,config().image_token_id,2],[source()],config())
    monkeypatch.setattr('ds41f_mlx.runtime.resource_admission.resources_for',lambda _:SimpleNamespace(validate_binding=lambda _:None))
    def forbidden(*_): raise AssertionError('hidden image replay')
    model=SimpleNamespace(language_model=object(),vision=forbidden,aligner=forbidden,config=config())
    assert inp.encode_new(model,len(inp.token_ids)).rows==()


def test_pending_append_failure_cannot_advertise_idle(monkeypatch):
    from ds41f_mlx.runtime.continuation_session import M8LiveContinuationSession
    from ds41f_mlx.prefill_fp8_mlx import DeferredPrefillAppend
    def fail(): raise RuntimeError('injected pending mutation failure')
    monkeypatch.setattr(DeferredPrefillAppend,'create',lambda *a,**k:SimpleNamespace(execute_all=fail))
    sess=M8LiveContinuationSession(model=object(),live_cache=[object()],token_history=[1,2,3])
    with pytest.raises(RuntimeError,match='pending'): sess.begin_turn_from_suffix([4,5],max_tokens=8)
    assert sess.state=='closed' and sess.live_cache==[]
    with pytest.raises(Exception,match='closed'): sess.begin_turn_from_suffix([6],max_tokens=8)


def test_segment_setup_masks_engram_and_merges_before_hc():
    from ds41f_mlx.prefill_fp8_mlx.p6_append import _make_segment_tensors
    c=SimpleNamespace(hc_mult=4)
    observed=[]
    def hasher(ids,history,mask):
        observed.append(np.asarray(mask)); return None,history
    lm=SimpleNamespace(_config=c,embed=lambda ids:mx.zeros((1,ids.shape[1],4),mx.bfloat16),_hasher=hasher)
    rows=ImageEmbeddings(((100,mx.ones((2,4))),),129264)
    mask=mx.array([[True,True,False]])
    h,_,_,_,_,_=_make_segment_tensors(lm,mx,[129264,129264,7],None,mask,rows,100)
    np.testing.assert_array_equal(np.asarray(h.astype(mx.float32))[0,:2],1)
    np.testing.assert_array_equal(np.asarray(h.astype(mx.float32))[0,2],0)
    np.testing.assert_array_equal(observed[0],[[True,True,False]])


def test_block_mask_is_command_local_including_decoder_suffix(monkeypatch):
    from test_prefill_fp8_mlx_p1_p2 import FakeLanguageModel,FakeLayer,full_ready_cache
    from test_prefill_fp8_mlx_p6 import fake_p6_app
    calls=[]
    original=FakeLayer.__call__
    full=np.zeros((1,16384),dtype=bool); full[:,8190:8200]=True; full[:,-10:]=True
    def checked(self,h,pre,cache,shared,start,image_mask):
        rows=h.last_slice[1]
        assert image_mask.shape==(1,rows)
        np.testing.assert_array_equal(image_mask,full[:,start:start+rows])
        calls.append((start,rows))
        return original(self,h,pre,cache,shared,start,image_mask)
    monkeypatch.setattr(FakeLayer,'__call__',checked)
    app=fake_p6_app(FakeLanguageModel(),full_ready_cache(),list(range(16384)))
    app.begin()
    segment=app.plan.segments[0]
    execution=app._make_segment_execution(segment)
    execution.runner.image_mask=full
    execution.runner.execute_batch(segment.commands,execution.arena)
    assert (8192,8192) in calls and (16383,1) in calls


def test_vl_gate_uses_image_bias_only_on_image_rows():
    from ds41f_mlx.model_execution.language import Gate
    c=ModelConfig(vision_n_layers=1,dim=4,n_routed_experts=4,n_activated_experts=2,score_func='sigmoid',norm_topk_prob=True)
    gate=Gate(c)
    gate.weight=mx.array([[.2,-.1,.3,.4],[-.1,.3,.1,.2],[.1,.2,-.2,.3],[.4,.1,.2,-.1]])
    gate.bias=mx.array([0.,0.,3.,2.]);gate.bias_vl=mx.array([3.,2.,0.,0.])
    x=mx.array([[[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]]])
    mask=mx.array([[False,True,False]])
    ids,weights=gate(x,mask)
    scores=1/(1+np.exp(-np.asarray(x)@np.asarray(gate.weight).T/c.gate_temp))
    bias=np.where(np.asarray(mask)[...,None],np.asarray(gate.bias_vl),np.asarray(gate.bias))
    expected=np.argsort(-(scores+bias),axis=-1)[...,:2]
    values=np.take_along_axis(scores,expected,-1); values=values/values.sum(-1,keepdims=True)*c.route_scale
    np.testing.assert_array_equal(np.asarray(ids),expected)
    np.testing.assert_allclose(np.asarray(weights),values,rtol=1e-6)
    assert set(expected[0,0])=={2,3} and set(expected[0,1])=={0,1}


def test_encoder_failure_releases_request_lease(monkeypatch):
    import asyncio
    from ds41f_mlx.serving.deepseek_recipe_backend import DeepSeekRecipeRuntimeBackend,RecipePreparedRequest
    def fail(*_): raise RuntimeError('encoder failure before cache allocation')
    monkeypatch.setattr('ds41f_mlx.serving.deepseek_recipe_backend.DwarfStarMLXPrefillSession',lambda *a,**k:object())
    backend=DeepSeekRecipeRuntimeBackend()
    backend.load=lambda:None;backend.make_sampler=lambda _:None
    request=RecipePreparedRequest('chat_completions',SimpleNamespace(model=None,stream=True,inference_options=SimpleNamespace(max_tokens=32)),None,[1,129264,2],[object()],multimodal=SimpleNamespace(token_ids=(1,129264,2),encode_new=fail))
    async def run():
        with pytest.raises(RuntimeError,match='encoder failure'): await anext(backend.infer(request))
        assert not backend._lock.locked() and backend.active_generation_sessions==0
        assert backend.last_trace.cleanup_called
    try: asyncio.run(run())
    finally: backend.close()


def test_size_count_and_detail_boundaries():
    with pytest.raises(ValueError,match='1..4'): prepare_multimodal([1,2],[source()]*5,config())
    with pytest.raises(ValueError,match='aspect'): prepare_multimodal([1,129264,2],[source(size=(200,50))],config())
    src=source();src.detail='low'
    with pytest.raises(ValueError,match='low-detail'): prepare_multimodal([1,129264,2],[src],config())


def test_animated_image_rejected():
    b=BytesIO(); images=[Image.new('RGB',(50,50),c) for c in ('red','blue')]
    images[0].save(b,format='PNG',save_all=True,append_images=images[1:])
    with pytest.raises(ValueError,match='animated'):
        prepare_multimodal([1,config().image_token_id,2],[SimpleNamespace(kind='bytes',data=b.getvalue())],config())
