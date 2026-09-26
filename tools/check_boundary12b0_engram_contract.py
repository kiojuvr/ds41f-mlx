#!/usr/bin/env python3
"""Check Boundary12b0 Engram semantic foundation contract artifact."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "engram-semantic-foundation-contract.json"
CKPT = Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")

EXPECTED = {
    "model": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "engram": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "tokenizer": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "tokenizer_config": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "parallel_span": "23f3b61ac17a7fe38947171365138fb294d61ee73a41973aff322a45a655b23d",
    "engram_forward_span": "ff04df4f5312b4d5170b844364ff3e0e823a51f94b7c527656d4e5b08beecaf9",
    "layout_span": "6ba33720ed214987cf1f7d689f99353ee310f04ef14e648d243db3d5d3f3ea85",
    "hash_state_span": "698c03c890bc8e1b55e93cd540a678f68845d7541c9da1d3d8e9a41f1cdde64d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary12a_main_hidden_dependency.py")], cwd=ROOT).returncode == 0
    rec = json.loads(ART.read_text())

    assert rec["schema"] == "ds41f.boundary12b0-engram-semantic-foundation-contract.v1"
    assert rec["ok"] is True
    assert rec["base_head"] == "85caa26be426fdbaaa5f305723d671758752e7e6"
    assert rec["classification"] == "official-source-derived Engram semantic/data contract authority"
    assert rec["not_omlx_derived"] is True
    assert rec["source_hash_method"] == "raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1"

    ids = rec["source_identities"]
    assert ids["model_py"]["sha256"] == EXPECTED["model"] == sha(CKPT / "inference/model.py")
    assert ids["engram_py"]["sha256"] == EXPECTED["engram"] == sha(CKPT / "inference/engram.py")
    assert ids["tokenizer_json"]["sha256"] == EXPECTED["tokenizer"] == sha(CKPT / "tokenizer.json")
    assert ids["tokenizer_config_json"]["sha256"] == EXPECTED["tokenizer_config"] == sha(CKPT / "tokenizer_config.json")
    assert ids["model_safetensors_index_json"]["sha256"] == sha(CKPT / "model.safetensors.index.json")

    spans = rec["hard_gated_canonical_spans"]
    assert spans["ParallelEngramEmbedding"]["line_start"] == 288
    assert spans["ParallelEngramEmbedding"]["line_end"] == 323
    assert spans["ParallelEngramEmbedding"]["span_sha256"] == EXPECTED["parallel_span"]
    assert spans["Engram.forward"]["line_start"] == 325
    assert spans["Engram.forward"]["line_end"] == 373
    assert spans["Engram.forward"]["span_sha256"] == EXPECTED["engram_forward_span"]
    assert spans["EngramLayout"]["span_sha256"] == EXPECTED["layout_span"]
    assert spans["NgramHashState"]["span_sha256"] == EXPECTED["hash_state_span"]

    layout = rec["engram_layout_contract"]
    assert layout["derived_fields"]["layer_ids"] == [1, 14]
    assert layout["derived_fields"]["max_ngram_size"] == 4
    assert layout["derived_fields"]["n_heads"] == 8
    assert layout["derived_fields"]["head_dim"] == 256
    assert layout["n_hash_cols"] == 24
    assert layout["layer_id_to_layer_hash_index"] == {"1": 0, "14": 1}
    assert len(layout["derived_fields"]["primes"]) == 2
    assert len(layout["per_layer_offsets"][0]) == 24

    hs = rec["ngram_hash_state_contract"]
    assert hs["returned_tensor"]["shape"] == "[B,L,num_engram_layers,(max_ngram_size-1)*n_heads]"
    assert hs["returned_tensor"]["released_shape_for_B1_S2"] == "[1,2,2,24]"
    assert hs["returned_tensor"]["dtype"] == "torch int64 from int64 token map/cache/multipliers/primes/offsets"
    assert hs["returned_tensor"]["axis_order"] == ["batch", "sequence", "engram layer in layout.layer_ids order", "hash columns ordered by ngram order then head"]
    assert hs["hash_coefficients"]["values"] == [
        [76632096046245, 4839876093313, 35959672319349, 73987337458391],
        [67716810739261, 51510806800915, 30921347202721, 82619226485591],
    ]
    assert "writes compressed tokens" in hs["start_pos_semantics"]
    assert "False means image token/no ngram participation" in hs["image_mask_interaction"]

    iface = rec["transformer_to_ngram_hash_state_interface"]
    assert iface["engram_hashes"] == "shape [B,S,num_engram_layers,n_hash_cols], dtype int64"
    assert iface["layer_axis_indexed_by"] == "layer.engram.layer_hash_index"
    assert iface["selection_expression"] == "engram_hashes[:, :, layer.engram.layer_hash_index, :]"

    tok = rec["tokenizer_dependency_classification"]
    assert tok["hashing_is_over"].startswith("compressed ids derived from decoded token strings")
    assert "tokenizer.json" in tok["tokenizer_files_used"]
    assert tok["runtime_requirement"].startswith("official source builds a static token_map")
    assert tok["tokenizer_subset_digests"]["model_vocab_count"] == 128000
    assert tok["tokenizer_subset_digests"]["added_tokens_count"] == 1283
    assert tok["tokenizer_subset_digests"]["special_added_tokens_count"] == 1229

    fields = {x["field"] for x in rec["engram_config_fields"]}
    for f in ["engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size", "engram_vocab_size", "engram_n_heads", "engram_head_dim", "engram_pad_id", "engram_compressed_vocab_size", "dim", "hc_mult", "norm_eps", "dtype"]:
        assert f in fields

    inv = rec["checkpoint_inventory"]["tensors"]
    assert len(inv) == 12
    by_name = {x["tensor"]: x for x in inv}
    assert by_name["layers.1.engram.embed.weight"]["shard"] == "model-00047-of-00048.safetensors"
    assert by_name["layers.14.engram.embed.weight"]["shard"] == "model-00048-of-00048.safetensors"
    assert by_name["layers.1.engram.embed.weight"]["checkpoint_dtype"] == "F8_E4M3"
    assert by_name["layers.1.engram.embed.weight"]["shape"] == [384006168, 256]
    assert by_name["layers.14.engram.embed.weight"]["shape"] == [384016682, 256]
    assert by_name["layers.1.engram.wkv.weight"]["shape"] == [25600, 6144]
    assert by_name["layers.1.engram.q_weight"]["shape"] == [4, 5120]
    assert rec["checkpoint_inventory"]["payload_hash_policy"].startswith("no giant tensor payload SHA256")
    assert all(rec["shape_consistency_gates"].values())

    pe = rec["parallel_engram_embedding_contract"]
    assert pe["output_shape"] == "input indices shape plus final dim layout.head_dim, so [B,S,n_hash_cols,engram_head_dim]"
    assert pe["output_dtype"] == "torch.bfloat16"
    assert pe["world_size1_path"].startswith("rank=0")

    ef = rec["engram_forward_contract"]
    assert ef["inputs"]["x"] == "[B,L,hc_mult,dim]"
    assert ef["output_shape"] == "same as x [B,L,hc_mult,dim]"
    assert any("gate=sigmoid" in x for x in ef["source_order"])

    fixture = rec["future_bounded_fixture_shape_contract"]
    assert fixture["tokens"] == "[[0,3]]"
    assert fixture["engram_hashes"] == "[1,2,2,24] int64"
    assert fixture["layer1_selected_hashes"] == "[1,2,24] int64 via layer_hash_index 0"
    assert fixture["ParallelEngramEmbedding_output"] == "[1,2,24,256] bfloat16"

    world = rec["world_size_contract"]
    assert world["current_scope"] == "world_size=1"
    assert "distributed partition correctness" in world["unvalidated"]
    assert rec["ssd_offload_authority_separation"]["omlx_ssd_offload_semantic_donor"] is False

    gate12b1 = rec["promotion_gate_for_12b1"]
    assert all(gate12b1.values())
    assert rec["next_boundary"].startswith("Boundary 12b1")

    claims = rec["claims"]
    assert claims["Engram_semantic_data_contract_authority"] is True
    assert claims["NgramHashState_numeric_authority"] is False
    assert claims["Engram_numerical_authority"] is False
    assert claims["main_hidden_numeric_authority"] is False
    for phrase in ["no NgramHashState numeric authority", "no Engram numerical authority", "no distributed Engram correctness", "no SSD/offload semantic qualification"]:
        assert phrase in rec["non_claims"]

    assert all(rec["authority_source_guards"].values())
    assert all(rec["gates"].values())

    print(f"Boundary12b0 Engram contract check PASS: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
