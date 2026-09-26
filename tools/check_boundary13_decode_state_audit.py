#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH = Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
ART = ROOT / 'artifacts/decode-incremental-state-source-audit.json'
DOC = ROOT / 'docs/decode-incremental-state-source-audit.md'

EXPECTED = {
    'inference/model.py': '4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65',
    'inference/engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'inference/config.json': '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809',
    'inference/generate.py': '8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0',
}
SPAN_EXPECTED = {
    'generate_function': ('inference/generate.py', 23, 75, 'd1994eadba6f2269f79ba57c5a85df08c7903eb4f6052a7fbf8d3980b5f9db85'),
    'generate_loop': ('inference/generate.py', 55, 70, '109c68796bf2265de649cd8a7a10953d920dc79a0985002d1346a4870a402425'),
    'generate_model_call': ('inference/generate.py', 57, 62, 'bd03704a617f3f572e58676e4a6c16e7f7fa540cb2bea235389255a761ba5b92'),
    'generate_advance': ('inference/generate.py', 67, 68, '78db02f983aa0c4b2a4b0158bfa6ca5d46627b3c08e4e1896b860502e42bc993'),
    'ngram_state': ('inference/engram.py', 118, 184, '4948807de7a25944b0c3dfaf09ea0d8fc9c9e77cc8a9efaddc2226a3e5413bbe'),
    'attention': ('inference/model.py', 613, 788, '6922d64f9accc5af89f8e5f23c21870beaac5d2a8793ab9ca09dfb2599021b82'),
    'transformer_forward': ('inference/model.py', 1183, 1273, '3503e2aca986abba129ded558eb3db80d5045900f0c2cbe78cec768827f5c02c'),
}

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def span_sha(rel: str, a: int, b: int) -> str:
    lines = (AUTH / rel).read_bytes().splitlines(True)
    return hashlib.sha256(b''.join(lines[a-1:b])).hexdigest()

def main() -> int:
    assert ART.exists(), ART
    assert DOC.exists(), DOC
    r = json.loads(ART.read_text())
    assert r['schema'] == 'ds41f.boundary13.decode-incremental-state-source-audit.v1'
    assert r['ok'] is True and r['not_omlx_derived'] is True
    assert r['base_head'] == '5127118a4e15c920e2efe420ef259180fe12c191'
    assert r['authority_root'] == str(AUTH)
    assert r['no_numeric_decode_claims'] is True
    for rel, digest in EXPECTED.items():
        assert sha(AUTH / rel) == digest, rel
    ids = r['source_identities']
    assert ids['generate_py']['sha256'] == EXPECTED['inference/generate.py']
    assert ids['generate_py']['line_count'] == 218 and ids['generate_py']['byte_count'] == 8722
    assert ids['model_py']['sha256'] == EXPECTED['inference/model.py']
    assert ids['engram_py']['sha256'] == EXPECTED['inference/engram.py']
    assert ids['config_json']['sha256'] == EXPECTED['inference/config.json']
    assert r['canonical_span_method'] == 'raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1'
    for name, (rel, a, b, digest) in SPAN_EXPECTED.items():
        got = r['canonical_spans'][name]
        assert (got['file'], got['start_line'], got['end_line'], got['sha256']) == (rel, a, b, digest), name
        assert span_sha(rel, a, b) == digest, name
    seq = r['official_generation_call_sequence']
    assert seq['call1']['input_ids'] == [[0,3]] and seq['call1']['start_pos'] == 0 and seq['call1']['S'] == 2
    assert seq['candidate_deterministic_prefill_sample']['temperature'] == 0
    assert seq['candidate_deterministic_prefill_sample']['sampled_token'] == 15
    assert seq['call2']['input_ids'] == [[15]] and seq['call2']['start_pos'] == 2 and seq['call2']['S'] == 1
    assert 'prev_pos = cur_pos' in seq['advance_contract']
    assert r['attention_mask_s1_decode']['new_triangular_mask_constructed'] is False
    topo = r['config_topology']
    assert topo['kv_source_layers'] == [2,8,14,20]
    assert topo['index_source_layers'] == [2,8,14,20,24,28,32,36]
    assert topo['candidate_source_layer'] == 20
    assert topo['engram_layer_ids'] == [1,14]
    cls = r['state_lifetime_classification']
    assert 'persistent' in cls['ngram_cache']
    assert 'persistent' in cls['window_kv_cache']
    assert 'persistent' in cls['compress_kv_cache']
    assert 'persistent' in cls['indexer_k_cache']
    assert 'current-call' in cls['candidates'] and 'current-call' in cls['topk_idxs']
    assert cls['pre_mix_h_main_hiddens'] == 'per-forward-call locals'
    assert cls['mlx_rng_next_session_key'].startswith('target-runtime-owned')
    hist = r['ngram_candidate_history_decode_step0']
    assert hist['shift0'].startswith('current decode token 15')
    assert hist['4gram'] == ['current','prefill_pos1','prefill_pos0','pad']
    assert hist['numeric_hashes_claimed'] is False
    ps = r['persistent_state_schema_after_prefill']
    assert ps['start_pos_next'] == 2
    assert ps['target_runtime_rng_state']['next_session_key_digest'] == '175f65a6ffece3dbac3719d14f5ab4678412d4187c967463dae2fc4688bc50f4'
    dec = r['decode_step0_input_schema']
    assert dec['input_ids'] == [[15]] and dec['start_pos'] == 2
    mt = r['first_decode_mutation_table']
    for k in ['NgramHashState.cache','window_kv_cache','compress_kv_cache_ratio2_sources','Indexer.k_cache_ratio1_owner20']:
        assert k in mt
    assert 'No explicit official reset API' in r['reset_semantics']
    pc = r['pass_conditions']
    assert all(pc.values()), [k for k,v in pc.items() if not v]
    doc = DOC.read_text()
    for phrase in ['Boundary13: decode / incremental-state source audit', 'Official generation call sequence', 'Persistent-state ownership matrix', 'No numeric hash values are claimed', 'no explicit official reset API', 'No incremental numerical authority']:
        assert phrase in doc, phrase
    st = subprocess.run(['git','status','--short'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip().splitlines()
    allowed = {'docs/decode-incremental-state-source-audit.md', 'artifacts/decode-incremental-state-source-audit.json', 'tools/check_boundary13_decode_state_audit.py'}
    dirty = {line[3:] for line in st if line[:2].strip() in {'M','A','??'} or line.startswith('??')}
    assert dirty <= allowed, dirty
    print(f'Boundary13 decode/incremental-state source audit check PASS: {ART}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
