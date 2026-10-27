"""Layout-bound metadata arithmetic + real authority controlled fixtures only.
No Model construction, checkpoint tensors or native completed KV is fabricated.
"""
import hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from conftest import installed_test_variant,admitted_test_numerics
from ds41f_mlx.serving import paired_checkpoint as pc
from ds41f_mlx.serving.production_sessions import ProductionSessions
from test_production_mtp import MX,pair
CHECKPOINT=Path('/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')

class Shape:
 def __init__(self,shape,itemsize):self.shape=shape;self.itemsize=itemsize;self.ndim=len(shape);self.nbytes=math.prod(shape)*itemsize

def test_layout_accounting():
 import mlx.core as mx
 from omlx.patches.deepseek_v41.config import ModelConfig
 from omlx.patches.deepseek_v41.quantization import pack_activation
 from omlx.patches.deepseek_v41 import growth
 cfg=ModelConfig.from_dict(json.loads((CHECKPOINT/'config.json').read_text()))
 assert cfg.n_layers==40 and cfg.head_dim==512 and cfg.window_size==128
 widths=[]
 for dim,bits,group,e4 in [(512,8,32,False),(512,4,16,True),(128,4,32,False)]:
  packed=pack_activation(mx.zeros((1,1,dim),mx.bfloat16),bits,group,e4);mx.eval(packed);assert packed.dtype==mx.uint8;widths.append(packed.shape[-1])
 assert widths==[528,288,68]
 def metadata(C):
  target=[]
  for i in range(40):
   r=cfg.compress_ratios[i] if i in cfg.kv_source_layers else 0
   slots=[Shape((1,),4),Shape((1,min(128,C),widths[0]),1),Shape((1,C//r if r else 0,widths[1]),1),Shape((1,C//r if r else 0,widths[2]),1),Shape((1,C%r if r>1 else 0,512),4 if r>1 else 2),Shape((1,C%r if r>1 else 0,512),4 if r>1 else 2),Shape((1,3 if i==0 else 0),8)]
   target.append(NS(cache=slots,compress_ratio=r,left_padding=Shape((1,),4),lengths=Shape((1,),4)))
  return NS(tokens=range(C),target=target,rings=[NS(keys=Shape((1,1,min(C,128),512),2)) for _ in range(3)])
 cases=[]
 for kind,reservation in [('ordinary',1200),('summary',2048)]:
  N=32768-reservation;end=N+reservation+5
  old=metadata(128);actual=metadata(end);logical=pc.pair_bytes(actual);projected=pc.projected_pair_bytes(old,end)
  assert projected>=logical
  growth_capacity=0;logical_cumulative=0
  per_source=[]
  for i in cfg.kv_source_layers:
   r=cfg.compress_ratios[i];rows=end//r+1;capacity=growth._capacity(rows,1)
   # Two alternating buffers in the sole active growth registry. A full
   # equivalent per snapshot is deliberately conservative: cloned snapshots
   # do not carry _ds41_buffers, but immutable backing may survive through views.
   amount=2*capacity*(widths[1]+widths[2]);growth_capacity+=amount
   logical_cumulative+=(end//r)*(widths[1]+widths[2])
   per_source.append(dict(layer=i,ratio=r,logical_rows=end//r,capacity_rows=capacity,two_buffers_bytes=amount))
  padded_pair=projected+growth_capacity-logical_cumulative
  retained=ProductionSessions.MAX_LIVE+2
  assert retained==18
  cases.append(dict(purpose=kind,input_positions=N,reservation=reservation,pinned_speculative_positions=5,end=end,logical_complete_pair_bytes=logical,existing_projected_pair_bytes=projected,calculated_padded_pair_capacity_bytes=padded_pair,per_source=per_source,retained_roots=retained,retained_logical_bytes=retained*projected,active_plus_two_pending_logical_bytes=3*projected,retained_active_two_capture_logical_bytes=(retained+3)*projected,conservative_padded_23_equivalents_bytes=23*padded_pair,conservative_python_token_metadata_bytes=23*(sys.getsizeof(())+end*(8+sys.getsizeof(2**63))),coexistence_components={'retained':18,'active_restored':1,'pending_prompt_and_final':2,'transient_restore_copy':1,'verify_predecessor_allowance':1}))
 # All-current projection <=3 small sources is below fresh-patch guard; these
 # ordinary pairs may exceed the separate 12MiB conditional discount ceiling.
 assert cases[0]['existing_projected_pair_bytes']>pc.PATCH_DISCOUNT_PAIR_BYTES
 srcs=['serving/paired_checkpoint.py','serving/production_mtp.py','serving/internal_mtp.py','runtime/multimodal.py']
 sources={s:hashlib.sha256((ROOT/'ds41f_mlx'/s).read_bytes()).hexdigest() for s in srcs}
 (OUT/'gate-c-layout.json').write_text(json.dumps(dict(kind='MODEL_FREE_PACKED_LAYOUT_CALCULATION_NOT_NATIVE_PEAK',weights_loaded=0,pass_=True,config_sha256=hashlib.sha256((CHECKPOINT/'config.json').read_bytes()).hexdigest(),target_layers=40,dspark_rings=3,packed_row_bytes=dict(window=528,compressed_kv=288,index=68),window_rows=128,ring_shape=[1,1,128,512],ring_dtype='bfloat16',compressor_tail_dtype='float32',kv_sources=list(cfg.kv_source_layers),calculated_three_image_payloads=dict(fresh_patches=4563,image_embedding_rows=552,bf16_embedding_payload_bytes=552*cfg.dim*2,float32_normalized_patch_payload_bytes=4563*cfg.vision_patch_size**2*3*4,not_vision_arena_or_native_peak=True),logical_conditional_limits=dict(pair=pc.PATCH_DISCOUNT_PAIR_BYTES,roots=pc.PATCH_DISCOUNT_ROOT_BYTES),cases=cases,sources=sources,limits='Logical/capacity arithmetic only. Model weights, Vision/prefill/verify logits/workspaces, allocator cache and process footprint separately monitored. No sum of overlapping host/MLX counters.'),indent=2)+'\n')

def test_actual_growth_buffers_at_acceptance_envelope():
 import mlx.core as mx
 from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
 from omlx.patches.deepseek_v41 import growth
 cache=DeepseekV41Cache(1);saved={}
 for slot,width in [(2,288),(3,68)]:cache[slot]=mx.zeros((1,32768,width),mx.uint8);saved[slot]=mx.array(cache[slot])
 mx.eval(*saved.values())
 end=32768
 for rows in [1,4]:
  for slot,width in [(2,288),(3,68)]:cache[slot]=growth.append(cache,slot,cache[slot],mx.ones((1,rows,width),mx.uint8),end)
  end+=rows;mx.eval(cache[2],cache[3])
 for slot in [2,3]:
  assert cache[slot].shape[1]==32773
  entry=cache._ds41_buffers[slot]
  assert entry['buffer'].shape[1]<=growth._capacity(32773,1)
  assert entry['idle'].shape[1]<=growth._capacity(32773,1)
  assert bool(mx.all(saved[slot]==0).item())

def test_real_authority_bounded_roots_restore_and_refusal():
 # Reuse lifecycle-only controlled cache arrays, never present them as native
 # values. Real authority, upstream hash/refcount/eviction and clone code run.
 auth=pc.PairedCheckpointAuthority('phase4-controlled',MX,retained_checkpoints=18)
 tokens=list(range(193));saved=auth.capture(*pair(193)[:2],tokens);auth.publish(saved)
 active=auth.acquire(tokens+[9,8]);assert active is not None
 prompt=auth.capture(*pair(194)[:2],tokens+[9]);final=auth.capture(*pair(195)[:2],tokens+[9,8])
 assert len(auth._payloads)==1  # captures pending, not discoverable
 auth.publish(prompt);auth.publish(final);assert len(auth._payloads)==3
 assert auth.acquire([999]+tokens[1:]+[9,8]) is None
 for i in range(40):
  snap=auth.capture(*pair(193)[:2],[1000+i]+tokens[1:]);auth.publish(snap)
  assert len(auth._payloads)<=18
 assert len(auth._payloads)==18
 assert auth.acquire(tokens+[9,8]) is None # normal LRU eviction, not manual clear
 before=dict(auth._payloads);last=next(iter(before.values()));lease=auth.retain(last.tokens)
 def refuse(c):raise ValueError('controlled pre-restore resource refusal')
 with pytest.raises(ValueError):auth.acquire_retained(lease,list(last.tokens)+[1,2],admit=refuse)
 assert auth._payloads==before
 assert auth.acquire_retained(lease,list(last.tokens)+[1,2]) is not None
 auth.release(lease)
 with pytest.raises(ValueError):auth.capture(pair(193)[0],pair(193)[1][:-1],tokens)
 auth.clear();assert not auth._payloads
