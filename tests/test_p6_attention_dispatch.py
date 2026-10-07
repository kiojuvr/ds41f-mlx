"""Unselected tool-only P6 candidate geometry/ownership, without a model load."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

mx=pytest.importorskip('mlx.core')
from tools.bench_target_prefill import _packed_suffix_attention


def fixture(length=128,old=127,dtype=None,heads=64,dim=512,window=128,owner='ds41f_mlx.model_execution.language',symbol=True,ratio=1):
    q=mx.zeros((1,length,heads,dim),dtype or mx.bfloat16)
    kv=mx.zeros((1,old+length,dim+dim//32),mx.uint8)
    pooled=mx.zeros((1,64,dim//2+dim//16),mx.uint8)
    idx=mx.arange(length)[:,None]+mx.arange(window)[None,:]
    idx=idx[None]
    ci=mx.zeros((1,length,16),mx.int32)
    sink=mx.zeros((heads,),mx.float32)
    native=Mock(side_effect=lambda q,*args:q)
    portable=Mock(side_effect=lambda q,*args:q)
    lang=SimpleNamespace(__name__=owner,glm_fast=SimpleNamespace(has_symbol=lambda name:symbol,deepseek_v41_packed_attention=native),packed_sparse_attention=portable)
    cfg=SimpleNamespace(n_heads=heads,head_dim=dim,window_size=window)
    out=_packed_suffix_attention(lang,mx,q,kv,pooled,idx,ci,sink,config=cfg,start=5000,old_len=old,ratio=ratio)
    assert out.shape==q.shape and out.dtype==q.dtype
    return native,portable,q,kv,idx


@pytest.mark.parametrize('length',[9,128,2414])
def test_native_actual_old_geometry_and_arguments(length):
    native,portable,q,kv,idx=fixture(length=length)
    native.assert_called_once(); portable.assert_not_called()
    args=native.call_args.args
    assert args[0].shape==(1,64,length,512)
    assert args[1].shape==(1,1,127+length,528)
    assert args[3].shape==(1,1,length,16) and args[3].dtype==mx.uint32
    assert args[5:]==(512**-0.5,5000,1,128)
    # C++/Metal derives local_offset from actual localL-qL. Every query's
    # ordered 128 key slots equal the explicit local indices, including self.
    offset=kv.shape[1]-q.shape[1]
    assert offset==127
    for t in (0,length//2,length-1):
        first=offset+t+1-128
        assert idx[0,t].tolist()==list(range(first,first+128))


@pytest.mark.parametrize('changes',[
    dict(length=1),dict(length=8),dict(old=128),dict(old=126),
    dict(dtype=mx.float32),dict(heads=8),dict(dim=256),dict(window=64),
    dict(owner='omlx.patches.deepseek_v41.language'),dict(symbol=False),
])
def test_preserve_unqualified_donor_mtp_portable_and_few_row_paths(changes):
    native,portable,*_=fixture(**changes)
    native.assert_not_called(); portable.assert_called_once()


def test_ratio_zero_uses_native_minimum_without_changing_other_inputs():
    native,portable,*_=fixture(ratio=0)
    native.assert_called_once(); portable.assert_not_called()
    assert native.call_args.args[7]==1
