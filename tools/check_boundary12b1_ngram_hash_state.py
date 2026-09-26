#!/usr/bin/env python3
"""Check Boundary12b1 NgramHashState bounded validation artifact."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "native-ngram-hash-state-validation.json"
CKPT = Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary12b0_engram_contract.py")], cwd=ROOT).returncode == 0
    rec = json.loads(ART.read_text())

    assert rec["schema"] == "ds41f.native-ngram-hash-state-validation.v1"
    assert rec["ok"] is True
    assert rec["base_head"] == "2daf6309cade66352ba20972102d9fa508034273"
    assert rec["classification"] == "official-reference-derived bounded NgramHashState numerical authority"
    assert rec["not_omlx_derived"] is True

    src = rec["source_identities"]
    assert src["engram_py"]["sha256"] == "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897" == sha(CKPT / "inference/engram.py")
    assert src["NgramHashState"]["span_sha256"] == "698c03c890bc8e1b55e93cd540a678f68845d7541c9da1d3d8e9a41f1cdde64d"
    assert src["tokenizer_json"]["sha256"] == "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b" == sha(CKPT / "tokenizer.json")
    assert src["tokenizer_config_json"]["sha256"] == "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547" == sha(CKPT / "tokenizer_config.json")

    scope = rec["scope"]
    assert scope == {"tokens": [[0, 3]], "B": 1, "S": 2, "start_pos": 0, "prefill": True, "world_size": 1, "engram_mask": None, "fresh_state_per_primary_run": True, "stop": "engram_hashes generated"}

    tm = rec["token_map"]
    assert tm["len_tokenizer"] == 129280
    assert tm["shape"] == [129280]
    assert tm["dtype"] == "int64"
    assert tm["digest"] == "26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496"
    assert tm["independent_digest"] == tm["digest"]
    assert tm["source_vs_independent_exact"] is True
    assert tm["compressed_vocab_size"] == 99092
    assert tm["independent_compressed_vocab_size"] == 99092
    assert tm["number_of_unique_compressed_ids"] == 99092
    assert tm["engram_pad_id"] == 2
    assert tm["token_map_0"] == 0
    assert tm["token_map_3"] == 3
    assert tm["token_map_2_compressed_pad_id"] == 2

    const = rec["static_constants"]
    assert const["layer_order"] == [1, 14]
    assert const["layer_hash_index"] == {"1": 0, "14": 1}
    assert const["max_ngram_size"] == 4
    assert const["heads"] == 8
    assert const["n_hash_cols"] == 24
    assert const["multipliers"] == [
        [76632096046245, 4839876093313, 35959672319349, 73987337458391],
        [67716810739261, 51510806800915, 30921347202721, 82619226485591],
    ]
    assert const["multipliers_digest"]
    assert const["primes_digest"]
    assert const["offsets_digest"]

    overflow = rec["int64_overflow_semantics"]
    assert overflow["ok"] is True
    assert any(r["mathematical_product_overflows_i64"] for r in overflow["rows"])
    assert all(r["product_match"] and r["xor_match"] for r in overflow["rows"])

    pf = rec["primary_fixture"]
    assert pf["input_ids"] == [[0, 3]]
    assert pf["input_shape"] == [1, 2]
    assert pf["start_pos"] == 0
    assert pf["token_mask"] is None
    assert pf["compressed_token_ids"]["values"] == [[0, 3]]
    assert pf["cache_state_evidence"]["relevant_cache_slice_after_write"] == [[0, 3]]
    assert pf["cache_state_evidence"]["no_dead_written"] is True
    assert pf["history_source_order"]["values"] == [[[0, 2, 2, 2], [3, 0, 2, 2]]]
    assert pf["history_independent"]["values"] == pf["history_source_order"]["values"]
    assert pf["source_vs_independent_history_exact"] is True

    per = rec["per_layer_intermediates"]
    assert set(per) == {"1", "14"}
    for layer in ["1", "14"]:
        assert per[layer]["products"]["shape"] == [1, 2, 4]
        assert "mathematical_i64_overflow_any" in per[layer]
        assert len(per[layer]["rolling_by_order"]) == 3
        assert [x["ngram_order"] for x in per[layer]["rolling_by_order"]] == [2, 3, 4]
        assert len(per[layer]["offsets"]) == 24
        assert len(per[layer]["hash_before_offset"][0][0]) == 24
        assert len(per[layer]["final_hash_after_offset"][0][0]) == 24
        assert per[layer]["digest"]

    buckets = rec["bucket_checks"]
    assert buckets["per_column_legal_bucket_all"] is True
    assert buckets["layer1_table_bounds_all"] is True
    assert buckets["layer14_table_bounds_all"] is True

    final = rec["final_output"]
    assert final["engram_hashes"]["shape"] == [1, 2, 2, 24]
    assert final["engram_hashes"]["dtype"] == "int64"
    assert final["engram_hashes"]["digest"] == "f5a64799492bea90bc6d87f0cb8a67bb983441e1cd9c12ec2e7b8df6cbcfee1d"
    assert final["layer1_hashes"]["shape"] == [1, 2, 24]
    assert final["layer1_hashes"]["digest"] == "8e0187ea859a7db65517a540eb5210457ec907fd0b2f44042d830366cbedbda5"
    assert final["layer14_hashes"]["shape"] == [1, 2, 24]
    assert final["layer14_hashes"]["digest"] == "33e046238287e6e8b7c3466bdb4b7b47182ad19f9115afac8a1983ede219ba80"
    assert final["independent_engram_hashes_digest"] == final["engram_hashes"]["digest"]
    assert final["source_vs_independent_final_exact"] is True

    eq = rec["none_mask_vs_all_true_equivalence"]
    assert eq["accepted_by_source_call"] is True
    assert eq["fresh_state"] is True
    assert eq["exact"] is True
    assert eq["all_true_digest"] == final["engram_hashes"]["digest"]

    assert rec["engram_checkpoint_tensor_payloads_read"] is False
    assert "layer1 and layer14 hash publication" in rec["validates"]
    for phrase in [
        "no incremental/decode NgramHashState correctness",
        "no False/image-mask DEAD crossing numeric authority",
        "no ParallelEngramEmbedding numeric authority",
        "no Engram.forward numeric authority",
        "no Engram checkpoint tensor arithmetic",
        "no main_hidden numeric authority",
    ]:
        assert phrase in rec["non_claims"]
    assert rec["next_boundary"] == "Boundary 12b2: ParallelEngramEmbedding + Engram@layer1"
    assert all(rec["authority_source_guards"].values())
    assert all(rec["gates"].values())

    print(f"Boundary12b1 NgramHashState check PASS: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
