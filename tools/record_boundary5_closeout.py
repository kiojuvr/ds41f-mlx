#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

VALIDATIONS = {
    '5a_compressor_compressed_kv': 'artifacts/native-compressed-kv-official-reference-validation.json',
    '5b_indexer_topk': 'artifacts/native-indexer-topk-official-reference-validation.json',
    '5c_candidate_publication': 'artifacts/native-candidate-block-official-reference-validation.json',
    '5d_candidate_consumer': 'artifacts/native-candidate-consumer-official-reference-validation.json',
    '5e_compressed_sparse_attn': 'artifacts/native-compressed-sparse-attn-validation.json',
    '5f_output_projection': 'artifacts/native-compressed-attention-integration-validation.json',
}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/boundary5-closeout.json'); args=ap.parse_args()
    loaded={k:json.loads(Path(v).read_text()) for k,v in VALIDATIONS.items()}
    b5f=loaded['5f_output_projection']
    rec={
        'schema':'ds41f.boundary5-closeout.v1',
        'classification':'official_reference_derived_closeout_no_new_model_math',
        'not_omlx_derived':True,
        'purpose':'Boundary 5 compressed Attention bounded prefill path closeout through Attention output; stop before Block/HC',
        'scope':{'layer':24,'batch':1,'sequence':2,'start_pos':0,'prefill':True,'world_size':1,'compress_ratio':1},
        'authority_chain':{k:{'path':VALIDATIONS[k],'ok':loaded[k].get('ok'),'schema':loaded[k].get('schema'),'classification':loaded[k].get('classification')} for k in VALIDATIONS},
        'validated_scope':['Compressor','Indexer','candidate publication/consumer wiring','window + compressed KV/index assembly','sparse_attn','inverse rotary','grouped wo_a','wo_b','Attention output'],
        'final_attention_output_digest':b5f['digests']['final_attention_output_bf16_uint16_sha256'],
        'candidate_pruning_status':'short fixture candidate mask retains all reachable positions; candidate machinery wiring validated, production pruning effect not validated',
        'non_claims':['decode / ring-buffer / partial compression group','production-scale candidate pruning','Block / Hyper-Connections','MoE','full layer / logits / full model correctness','performance / fusion'],
        'ok':all(loaded[k].get('ok') is True for k in VALIDATIONS),
    }
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n'); print(out); return 0 if rec['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
