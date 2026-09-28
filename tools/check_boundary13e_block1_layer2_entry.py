#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-first-incremental-block1-layer2-entry-validation.json'
DOC=ROOT/'docs/first-incremental-block1-layer2-entry-validation.md'
ALLOWED={'tools/run_native_first_incremental_block1_layer2_entry_validation.py','tools/check_boundary13e_block1_layer2_entry.py','tools/run_m4_layer2_sparse_topology.py','tools/check_m4_layer2_sparse_topology.py','docs/first-incremental-block1-layer2-entry-validation.md','docs/implementation-plan.md','docs/milestone-4-base-decode-status.md','artifacts/native-first-incremental-block1-layer2-entry-validation.json'}
def dirty(): return {line[3:] for line in subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()}
def main():
 assert ART.exists() and DOC.exists()
 d=dirty()
 if not d:
  for tool in ['check_boundary13_decode_state_audit.py','check_boundary13a_prefill_state.py','check_boundary13b_incremental_ngram.py','check_boundary13c_incremental_window_kv.py','check_boundary13d_block0_engram1.py']:
   assert subprocess.run([sys.executable,str(ROOT/'tools'/tool)],cwd=ROOT).returncode==0, tool
 else: assert d <= ALLOWED, d
 r=json.loads(ART.read_text()); assert r['schema']=='ds41f.native-first-incremental-block1-layer2-entry-validation.v1'; assert r['ok'] and r['not_omlx_derived']; assert r['base_head']=='cd7791090cc3ec0684c1c419c232071e6b221d87'
 h=r['Boundary13d_handoff']; assert h['Block0_x_out'] and h['Block0_ffn_pre']; assert h['layer1_hash']=='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'; assert h['post_Engram1']
 b=r['Block1']; assert b['attention_input']['shape']==[1,1,5120]; assert b['q_rotary']['shape']==[1,1,64,512]; assert b['kv_rotary']['shape']==[1,1,512]; assert b['window_visible']['shape']==[1,3,512]; assert b['sparse_attn']['shape']==[1,1,64,512]; assert b['attention_output']['shape']==[1,1,5120]; assert b['x_out']['shape']==[1,1,4,5120]; assert b['ffn_pre']['shape']==[1,1,4]; assert b['moe_routing_indices']==[[382,172,228,372,340,343]]; assert b['selected_expert_set']==[172,228,340,343,372,382]
 l=r['Layer2_entry']; assert l['attention_input']['shape']==[1,1,5120]; assert l['rope']['ratio']==2 and l['rope']['original_seq_len']==65536 and l['rope']['theta']==160000.0; assert l['q_pre_rope']['shape']==[1,1,64,512]; assert l['q_rotary']['shape']==[1,1,64,512]; assert l['window_kv_norm']['shape']==[1,1,512]; assert l['kv_rotary']['shape']==[1,1,512]; assert l['window_visible']['shape']==[1,3,512]
 cp=l['compressor_partial']; assert cp['ratio']==2 and cp['start_pos']==2 and cp['written_slot']==0 and cp['valid_partial_slots']==[0]; assert cp['kv_state_slot0']['dtype']=='float32'; assert cp['score_state_slot0']['dtype']=='float32'; assert cp['future_first_read_position']==3 and cp['group_complete'] is False and cp['new_latent_produced'] is False
 ck=l['compress_kv_cache']; assert ck['before_digest']==ck['after_digest']; assert ck['new_cache_write'] is False and ck['compress_len']==1
 ix=l['indexer']; assert ix['query_rotary_absolute_position']==2; assert ix['fp4_contract']=='block32/E8M0'; assert ix['query_fp4_codes_digest'] and ix['query_e8m0_scales_digest']; assert ix['topk_compressed_local']['values']==[[[0]]] and ix['topk_omlx_ci']['values']==[[[0]]] and ix['topk_concat_kv']['values']==[[[3]]]; assert ix['new_key_publication'] is False and ix['prefill_topk_reused'] is False
 assert all(r['persistent_state_nonmutation'].values()); assert all(v is False for v in r['STOP_flags'].values()); assert all(r['gates'].values()), [k for k,v in r['gates'].items() if not v]
 doc=DOC.read_text();
 for p in ['Boundary13e','Block1 completion','layer2 Compressor partial state','Indexer lifecycle','compressed RoPE','block32/E8M0','coordinate','Stopped before layer2 sparse attention']: assert p in doc, p
 print(f'Boundary13e Block1->layer2 entry check PASS: {ART}'); return 0
if __name__=='__main__': raise SystemExit(main())
