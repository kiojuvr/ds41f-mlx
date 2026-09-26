#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-first-incremental-block1-layer2-entry-validation.json'
DOC=ROOT/'docs/first-incremental-block1-layer2-entry-validation.md'
ALLOWED={'tools/run_native_first_incremental_block1_layer2_entry_validation.py','tools/check_boundary13e_block1_layer2_entry.py','docs/first-incremental-block1-layer2-entry-validation.md','artifacts/native-first-incremental-block1-layer2-entry-validation.json'}
def dirty(): return {line[3:] for line in subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.splitlines()}
def main():
 assert ART.exists() and DOC.exists()
 d=dirty()
 if not d:
  for tool in ['check_boundary13_decode_state_audit.py','check_boundary13a_prefill_state.py','check_boundary13b_incremental_ngram.py','check_boundary13c_incremental_window_kv.py','check_boundary13d_block0_engram1.py']:
   assert subprocess.run([sys.executable,str(ROOT/'tools'/tool)],cwd=ROOT).returncode==0, tool
 else: assert d <= ALLOWED, d
 r=json.loads(ART.read_text()); assert r['schema']=='ds41f.native-first-incremental-block1-layer2-entry-validation.v1'; assert r['ok'] and r['not_omlx_derived']; assert r['base_head']=='8d6d957bca064f67afdf7d914e2454763e1c2f1b'
 h=r['Boundary13d_handoff']; assert h['Block0_x_out']=='ecbf76fb8c272dc20399cf4e9dc839e291adeac096d05556891fc30c3169740f'; assert h['Block0_ffn_pre']=='036a63ede630537263cbe5125569fc36680270c7bd35e841bc789705d992cf74'; assert h['layer1_hash']=='4eb8fc730c0e52176b64a388dcfcb7bb29c5ac6ee619c93efd218eef5a6df373'; assert h['post_Engram1']=='b651ee96bcf7b1f82a5ea5f18678efc85668b4766a69c2fc2b117ee75be0794b'
 b=r['Block1']; assert b['attention_input']['digest']=='18aacab2745ee7ae9ae471b4e44e480b12e9ba5ba55ff3d76fb5d2e5cff894cf'; assert b['q_rotary']['digest']=='55d7d660ed14947c532b335e58d50a5d75e7767c53aba33cd8cb7412cafeeaf8'; assert b['kv_rotary']['digest']=='dcd9dcb91809d4642c7760d123dfcaa03475c7a45c4018ebec98a5dae8dd0908'; assert b['window_slot2']['digest']=='3d228d39da6765c963f23452577f92d3183d29c8b7e972016f02dce73db6c270'; assert b['window_visible']['digest']=='9cb5e36038c87b179bfb218f2a11a9e3c67d51f10d8e0bcb32b42b7400f8487c'; assert b['sparse_attn']['digest']=='1f12ab38c201683c95d165ff32826fbd6a1f0c3c73a58ed853e18f9347e0c2aa'; assert b['attention_output']['digest']=='31355e39b300c0040c73f682f8d09243577e308a35226d410b0c26d56251e142'; assert b['x_out']['digest']=='52292729411f7435c93fc6fcca5ff37acc2a063d504d14498f587949e3e76aa9'; assert b['ffn_pre']['digest']=='27df7d1f2ec08036253e5317b759a27ccd7f038843263143e529ca0fbe3b4549'; assert b['moe_routing_indices']==[[382,172,340,228,372,343]]; assert b['selected_expert_set']==[172,228,340,343,372,382]
 l=r['Layer2_entry']; assert l['attention_input']['digest']=='671eb57c53277fd649783ab3268a7d9082df33c167bd74bf22f8fc4b3c56abd8'; assert l['q_rotary']['digest']=='47f1c92be599832c3b9b689489e219191057c277f49f452e85e0b5bf7d81a42d'; assert l['kv_rotary']['digest']=='abb75ba331a32fa275a07a22788ba9a1e870c3b3931c72b28a3bc8018b4122fb'; assert l['window_slot2']['digest']=='99c6ec24f7a8b12bda774381a4dccc3ce0938adab97825f790ad4820e6b39da2'; assert l['window_visible']['digest']=='3ff90635bf6889c0e870fca1782b34021bb575f2061cd17572dfb447355617ff'
 cp=l['compressor_partial']; assert cp['ratio']==2 and cp['start_pos']==2 and cp['written_slot']==0 and cp['valid_partial_slots']==[0]; assert cp['kv_state_slot0']['digest']=='63689a837034bf7baf329cd9cedf2f3b89d1966b2fa455e741159008d39e71b9'; assert cp['score_state_slot0']['digest']=='5aef9d58018e2c0dcb45c9342f76b4bffdbac9a3b267e34922ad140ce8fa6792'; assert cp['future_first_read_position']==3 and cp['group_complete'] is False and cp['new_latent_produced'] is False
 ck=l['compress_kv_cache']; assert ck['before_digest']==ck['after_digest']=='838e6e26d9889ef668bc6be7d345be10542466bb86a8c656f3aa5665376143ae'; assert ck['new_cache_write'] is False and ck['compress_len']==1
 ix=l['indexer']; assert ix['query_rotary_absolute_position']==2; assert ix['query_digest']=='7c22372541e7bc0b3787d89f39a6ca43698f8324caf71bfa91e914b4f3eb44c4'; assert ix['topk']['values']==[[[128]]] and ix['topk_digest']=='50c8ba3a6170f0a2fb6736ece8a603576ef6309a35e810911599bc6211b554a9'; assert ix['new_key_publication'] is False and ix['prefill_topk_reused'] is False
 assert all(r['persistent_state_nonmutation'].values()); assert all(v is False for v in r['STOP_flags'].values()); assert all(r['gates'].values()), [k for k,v in r['gates'].items() if not v]
 doc=DOC.read_text();
 for p in ['Boundary13e','Block1 completion','layer2 Compressor partial state','Indexer lifecycle','Stopped before layer2 sparse attention']: assert p in doc, p
 print(f'Boundary13e Block1->layer2 entry check PASS: {ART}'); return 0
if __name__=='__main__': raise SystemExit(main())
