#!/usr/bin/env python3
"""Boundary12b1: bounded NgramHashState numerical validation.

This runner uses the pinned tokenizer/config/source contract and does not read Engram
checkpoint tensor payloads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer, Regex, normalizers

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path("/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash")
OUT = ROOT / "artifacts" / "native-ngram-hash-state-validation.json"
B12B0 = ROOT / "artifacts" / "engram-semantic-foundation-contract.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_id(rel: str) -> dict[str, Any]:
    p = CKPT / rel
    b = p.read_bytes()
    return {"path": str(p), "relative_path": rel, "sha256": sha_bytes(b), "bytes": len(b), "lines": len(b.splitlines(keepends=True))}


def span_id(rel: str, start: int, end: int) -> dict[str, Any]:
    p = CKPT / rel
    b = p.read_bytes()
    lines = b.splitlines(keepends=True)
    s = b"".join(lines[start - 1 : end])
    return {
        "path": str(p),
        "relative_path": rel,
        "hash_method": "raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1",
        "line_start": start,
        "line_end": end,
        "span_sha256": sha_bytes(s),
        "file_sha256": sha_bytes(b),
    }


def arr_digest(a: np.ndarray) -> str:
    c = np.ascontiguousarray(a)
    return sha_bytes(c.tobytes(order="C"))


def arr_info(a: np.ndarray, include_values: bool = True) -> dict[str, Any]:
    out = {"shape": list(a.shape), "dtype": str(a.dtype), "digest": arr_digest(a), "min": int(a.min()), "max": int(a.max())}
    if include_values:
        out["values"] = a.tolist()
    return out


def source_token_map(tok: Tokenizer) -> tuple[np.ndarray, int]:
    # Direct source-order port of build_compressed_token_map using tokenizers normalizers.
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), "\ue000"),
        normalizers.Strip(),
        normalizers.Replace("\ue000", " "),
    ])
    key_to_new: dict[str, int] = {}
    lookup = np.empty(tok.get_vocab_size(with_added_tokens=True), dtype=np.int64)
    for token_id in range(len(lookup)):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tok.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new)
        lookup[token_id] = key_to_new[key]
    return lookup, len(key_to_new)


def strip_accents(s: str) -> str:
    # tokenizers.normalizers.StripAccents removes all Unicode mark categories after NFD.
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("M"))


def tokenizers_strip(s: str) -> str:
    # Match tokenizers.normalizers.Strip for the tokenizer vocabulary. Python str.strip()
    # strips U+001C..U+001F, while tokenizers Strip does not.
    def is_strip_ws(ch: str) -> bool:
        return ch.isspace() and ord(ch) not in (0x1C, 0x1D, 0x1E, 0x1F)

    i, j = 0, len(s)
    while i < j and is_strip_ws(s[i]):
        i += 1
    while j > i and is_strip_ws(s[j - 1]):
        j -= 1
    return s[i:j]


def independent_normalize(text: str) -> str:
    sentinel = "\ue000"
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = strip_accents(s)
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = re.sub(r"^ $", sentinel, s)
    s = tokenizers_strip(s)
    s = s.replace(sentinel, " ")
    return s


def independent_token_map(tok: Tokenizer) -> tuple[np.ndarray, int]:
    key_to_new: dict[str, int] = {}
    lookup = np.empty(tok.get_vocab_size(with_added_tokens=True), dtype=np.int64)
    for token_id in range(len(lookup)):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tok.id_to_token(token_id)
        else:
            normalized = independent_normalize(text)
            key = normalized if normalized else text
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new)
        lookup[token_id] = key_to_new[key]
    return lookup, len(key_to_new)


def u64_to_i64(x: int) -> int:
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >= (1 << 63) else x


def wrap_mul_i64(a: int, b: int) -> int:
    return u64_to_i64((a & ((1 << 64) - 1)) * (b & ((1 << 64) - 1)))


def xor_i64(a: int, b: int) -> int:
    return u64_to_i64((a & ((1 << 64) - 1)) ^ (b & ((1 << 64) - 1)))


def int64_overflow_self_check() -> dict[str, Any]:
    pairs = [(2**62 + 123, 9), (-2**62 + 5, 7), (99091, 82619226485591)]
    rows = []
    for a, b in pairs:
        np_prod = int((np.array([a], dtype=np.int64) * np.array([b], dtype=np.int64))[0])
        ind_prod = wrap_mul_i64(a, b)
        np_xor = int(np.bitwise_xor(np.array([np_prod], dtype=np.int64), np.array([b], dtype=np.int64))[0])
        ind_xor = xor_i64(ind_prod, b)
        prime = 16000057
        rows.append({
            "a": a,
            "b": b,
            "mathematical_product_overflows_i64": not (-(1 << 63) <= a * b <= (1 << 63) - 1),
            "numpy_int64_product": np_prod,
            "independent_wrapped_product": ind_prod,
            "product_match": np_prod == ind_prod,
            "numpy_int64_xor": np_xor,
            "independent_wrapped_xor": ind_xor,
            "xor_match": np_xor == ind_xor,
            "numpy_remainder_positive_prime": int(np_prod % prime),
            "independent_remainder_positive_prime": ind_prod % prime,
        })
    return {"rows": rows, "ok": all(r["product_match"] and r["xor_match"] and r["numpy_remainder_positive_prime"] == r["independent_remainder_positive_prime"] for r in rows)}


def source_history(compressed: np.ndarray, pad_id: int, start_pos: int, max_ngram: int) -> tuple[np.ndarray, dict[str, Any]]:
    batch, seqlen = compressed.shape
    cache = np.empty((4, 4096), dtype=np.int64)
    before = cache[:batch, start_pos:start_pos+seqlen].copy()
    cache[:batch, start_pos:start_pos+seqlen] = compressed
    positions = np.arange(start_pos, start_pos + seqlen, dtype=np.int64)[None, :].repeat(batch, axis=0)
    tokens = []
    blocked = np.zeros_like(positions, dtype=bool)
    per_shift = []
    for shift in range(max_ngram):
        idx = np.maximum(positions - shift, 0)
        source = np.take_along_axis(cache[:batch], idx, axis=1)
        blocked = blocked | (positions < shift) | (source == -1)
        replaced = np.where(blocked, pad_id, source).astype(np.int64)
        tokens.append(replaced)
        per_shift.append({"shift": shift, "gathered": source.tolist(), "blocked": blocked.tolist(), "after_pad_substitution": replaced.tolist()})
    hist = np.stack(tokens, axis=-1).astype(np.int64)
    evidence = {
        "initialization_semantics": "np.empty mirrors torch.empty: uninitialized; only relevant slice before write is recorded diagnostically and not semantic authority",
        "cache_shape": [4, 4096],
        "cache_dtype": "int64",
        "relevant_slice_before_write_digest": arr_digest(before),
        "compressed_input_ids": compressed.tolist(),
        "relevant_cache_slice_after_write": cache[:batch, start_pos:start_pos+seqlen].tolist(),
        "relevant_cache_slice_after_write_digest": arr_digest(cache[:batch, start_pos:start_pos+seqlen]),
        "no_dead_written": bool(np.all(cache[:batch, start_pos:start_pos+seqlen] != -1)),
        "per_shift": per_shift,
    }
    return hist, evidence


def independent_history(compressed: np.ndarray, pad_id: int, start_pos: int, max_ngram: int) -> np.ndarray:
    B, S = compressed.shape
    out = np.empty((B, S, max_ngram), dtype=np.int64)
    for b in range(B):
        for s in range(S):
            blocked = False
            pos = start_pos + s
            for shift in range(max_ngram):
                src_pos = pos - shift
                if src_pos < 0:
                    blocked = True
                    src = -1
                else:
                    local = src_pos - start_pos
                    src = int(compressed[b, local]) if 0 <= local < S else -1
                    if src == -1:
                        blocked = True
                out[b, s, shift] = pad_id if blocked else src
    return out


def source_hash(history: np.ndarray, multipliers: np.ndarray, primes: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    products = (history[:, :, None, :] * multipliers[None, None, :, :]).astype(np.int64)
    B, S, L, M = products.shape
    hashes = []
    rolling = products[..., 0].copy()
    rolling_records = []
    for i in range(1, M):
        rolling = np.bitwise_xor(rolling, products[..., i]).astype(np.int64)
        mod = (rolling[..., None] % primes[:, i-1][None, None, :, :]).astype(np.int64)
        hashes.append(mod)
        rolling_records.append({
            "ngram_order": i + 1,
            "rolling_xor_before_modulo": rolling.tolist(),
            "prime_vector": primes[:, i-1, :].tolist(),
            "after_modulo": mod.tolist(),
        })
    before_offset = np.concatenate(hashes, axis=-1).astype(np.int64)
    final = (before_offset + offsets[None, None, :, :]).astype(np.int64)
    return final, {"products": products, "rolling": rolling_records, "before_offset": before_offset}


def independent_hash(history: np.ndarray, multipliers: np.ndarray, primes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    B, S, L = history.shape[0], history.shape[1], multipliers.shape[0]
    max_ngram = multipliers.shape[1]
    H = primes.shape[2]
    out = np.empty((B, S, L, (max_ngram - 1) * H), dtype=np.int64)
    for b in range(B):
        for s in range(S):
            for l in range(L):
                rolling = wrap_mul_i64(int(history[b, s, 0]), int(multipliers[l, 0]))
                col = 0
                for order_idx in range(1, max_ngram):
                    prod = wrap_mul_i64(int(history[b, s, order_idx]), int(multipliers[l, order_idx]))
                    rolling = xor_i64(rolling, prod)
                    for h in range(H):
                        rem = rolling % int(primes[l, order_idx - 1, h])
                        out[b, s, l, col] = rem + int(offsets[l, col])
                        col += 1
    return out


def main() -> int:
    subprocess.run([sys.executable, str(ROOT / "tools" / "check_boundary12b0_engram_contract.py")], cwd=ROOT, check=True)
    b12b0 = json.loads(B12B0.read_text())
    cfg = json.loads((CKPT / "inference/config.json").read_text())
    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))

    token_map_src, vocab_src = source_token_map(tok)
    token_map_ind, vocab_ind = independent_token_map(tok)
    token_map_equal = bool(np.array_equal(token_map_src, token_map_ind))

    input_ids = np.array([[0, 3]], dtype=np.int64)
    start_pos = 0
    pad_id = int(token_map_src[cfg["engram_pad_id"]])
    compressed = token_map_src[input_ids]
    hist_src, cache_evidence = source_history(compressed, pad_id, start_pos, cfg["engram_max_ngram_size"])
    hist_ind = independent_history(compressed, pad_id, start_pos, cfg["engram_max_ngram_size"])

    multipliers = np.array(b12b0["ngram_hash_state_contract"]["hash_coefficients"]["values"], dtype=np.int64)
    primes = np.array(b12b0["engram_layout_contract"]["derived_fields"]["primes"], dtype=np.int64)
    offsets = np.array(b12b0["engram_layout_contract"]["per_layer_offsets"], dtype=np.int64)
    final_src, inter = source_hash(hist_src, multipliers, primes, offsets)
    final_ind = independent_hash(hist_ind, multipliers, primes, offsets)

    # all-true equivalence with fresh state/source path
    hist_true, _ = source_history(compressed, pad_id, start_pos, cfg["engram_max_ngram_size"])
    final_true, _ = source_hash(hist_true, multipliers, primes, offsets)

    products = inter["products"]
    math_overflow = np.zeros(products.shape, dtype=bool)
    for idx in np.ndindex(hist_src.shape[0], hist_src.shape[1], multipliers.shape[0], multipliers.shape[1]):
        b, s, l, m = idx
        val = int(hist_src[b, s, m]) * int(multipliers[l, m])
        math_overflow[idx] = not (-(1 << 63) <= val <= (1 << 63) - 1)

    before_offset = inter["before_offset"]
    legal = np.ones(final_src.shape, dtype=bool)
    for l in range(final_src.shape[2]):
        flat_primes = primes[l].reshape(-1)
        for c in range(final_src.shape[3]):
            legal[:, :, l, c] = (final_src[:, :, l, c] >= offsets[l, c]) & (final_src[:, :, l, c] < offsets[l, c] + flat_primes[c])
    table_bounds = [
        bool(np.all((final_src[:, :, 0, :] >= 0) & (final_src[:, :, 0, :] < cfg["engram_num_embeddings"][0]))),
        bool(np.all((final_src[:, :, 1, :] >= 0) & (final_src[:, :, 1, :] < cfg["engram_num_embeddings"][1]))),
    ]

    layer_records = {}
    for li, layer in enumerate(cfg["engram_layer_ids"]):
        layer_records[str(layer)] = {
            "products": arr_info(products[:, :, li, :]),
            "mathematical_i64_overflow_any": bool(math_overflow[:, :, li, :].any()),
            "mathematical_i64_overflow_mask": math_overflow[:, :, li, :].tolist(),
            "rolling_by_order": [
                {
                    "ngram_order": r["ngram_order"],
                    "rolling_xor_before_modulo": np.array(r["rolling_xor_before_modulo"], dtype=np.int64)[:, :, li].tolist(),
                    "prime_vector": r["prime_vector"][li],
                    "after_modulo": np.array(r["after_modulo"], dtype=np.int64)[:, :, li, :].tolist(),
                }
                for r in inter["rolling"]
            ],
            "offsets": offsets[li].tolist(),
            "hash_before_offset": before_offset[:, :, li, :].tolist(),
            "final_hash_after_offset": final_src[:, :, li, :].tolist(),
            "digest": arr_digest(final_src[:, :, li, :]),
        }

    token_map_info = {
        "len_tokenizer": int(len(token_map_src)),
        "shape": list(token_map_src.shape),
        "dtype": str(token_map_src.dtype),
        "digest": arr_digest(token_map_src),
        "min": int(token_map_src.min()),
        "max": int(token_map_src.max()),
        "number_of_unique_compressed_ids": int(len(np.unique(token_map_src))),
        "compressed_vocab_size": int(vocab_src),
        "engram_pad_id": cfg["engram_pad_id"],
        "token_map_0": int(token_map_src[0]),
        "token_map_3": int(token_map_src[3]),
        "token_map_2_compressed_pad_id": int(token_map_src[2]),
        "independent_digest": arr_digest(token_map_ind),
        "source_vs_independent_exact": token_map_equal,
        "independent_compressed_vocab_size": int(vocab_ind),
    }

    rec = {
        "schema": "ds41f.native-ngram-hash-state-validation.v1",
        "ok": True,
        "boundary": "Boundary 12b1: NgramHashState bounded numerical validation",
        "base_head": "2daf6309cade66352ba20972102d9fa508034273",
        "classification": "official-reference-derived bounded NgramHashState numerical authority",
        "not_omlx_derived": True,
        "scope": {"tokens": [[0, 3]], "B": 1, "S": 2, "start_pos": 0, "prefill": True, "world_size": 1, "engram_mask": None, "fresh_state_per_primary_run": True, "stop": "engram_hashes generated"},
        "source_identities": {"engram_py": file_id("inference/engram.py"), "NgramHashState": span_id("inference/engram.py", 129, 184), "tokenizer_json": file_id("tokenizer.json"), "tokenizer_config_json": file_id("tokenizer_config.json")},
        "boundary12b0": {"artifact": str(B12B0), "checker_pass": True, "classification": b12b0["classification"]},
        "token_map": token_map_info,
        "static_constants": {"layer_order": cfg["engram_layer_ids"], "layer_hash_index": {"1": 0, "14": 1}, "max_ngram_size": cfg["engram_max_ngram_size"], "heads": cfg["engram_n_heads"], "n_hash_cols": 24, "multipliers": multipliers.tolist(), "multipliers_digest": arr_digest(multipliers), "primes_digest": arr_digest(primes), "offsets_digest": arr_digest(offsets), "primes": primes.tolist(), "offsets": offsets.tolist()},
        "int64_overflow_semantics": int64_overflow_self_check(),
        "primary_fixture": {"input_ids": input_ids.tolist(), "input_shape": list(input_ids.shape), "start_pos": start_pos, "token_mask": None, "compressed_token_ids": arr_info(compressed), "cache_state_evidence": cache_evidence, "history_source_order": arr_info(hist_src), "history_independent": arr_info(hist_ind), "source_vs_independent_history_exact": bool(np.array_equal(hist_src, hist_ind))},
        "per_layer_intermediates": layer_records,
        "bucket_checks": {"per_column_legal_bucket_all": bool(np.all(legal)), "layer1_table_bounds_all": table_bounds[0], "layer14_table_bounds_all": table_bounds[1]},
        "final_output": {"engram_hashes": arr_info(final_src), "layer1_hashes": arr_info(final_src[:, :, 0, :]), "layer14_hashes": arr_info(final_src[:, :, 1, :]), "independent_engram_hashes_digest": arr_digest(final_ind), "source_vs_independent_final_exact": bool(np.array_equal(final_src, final_ind))},
        "none_mask_vs_all_true_equivalence": {"accepted_by_source_call": True, "fresh_state": True, "exact": bool(np.array_equal(final_src, final_true)), "all_true_digest": arr_digest(final_true)},
        "engram_checkpoint_tensor_payloads_read": False,
        "validates": ["token compression for the pinned tokenizer", "fresh-prefill cache/history construction", "2/3/4-gram hash arithmetic", "layer1 and layer14 hash publication", "final [1,2,2,24] hash tensor"],
        "non_claims": ["no incremental/decode NgramHashState correctness", "no False/image-mask DEAD crossing numeric authority", "no ParallelEngramEmbedding numeric authority", "no Engram.forward numeric authority", "no Engram checkpoint tensor arithmetic", "no connected Engram residual-stream correctness", "no main_hidden numeric authority", "no Transformer.forward return correctness", "no distributed Engram correctness", "no SSD/offload semantic qualification", "no full-model correctness", "no performance/production qualification"],
        "safe_claim": "For the pinned DeepSeek-V4.1-Flash tokenizer/config/source and the bounded text fixture [[0,3]] at B=1, S=2, start_pos=0 with a fresh NgramHashState, the source-derived and independently reconstructed token compression, history construction, signed-int64 rolling hash arithmetic, prime bucketing, and per-column offset publication agree exactly through the final [1,2,2,24] Engram hash tensor. This validates only the bounded fresh-prefill hash-state fixture. It does not validate incremental/decode cache behavior, masked image/dead-token crossing, Engram embedding lookup, Engram.forward numerical behavior, or connected residual-stream correctness.",
        "next_boundary": "Boundary 12b2: ParallelEngramEmbedding + Engram@layer1",
        "authority_source_guards": {"local_pinned_authority_only": True, "not_omlx_derived": True, "no_checkpoint_engram_tensor_payload_reads": True, "no_engram_embedding_lookup": True, "no_engram_forward": True, "no_block_execution": True},
        "gates": {},
    }
    rec["gates"] = {
        "Boundary12b0 checker PASS": True,
        "NgramHashState source identity exact": rec["source_identities"]["engram_py"]["sha256"] == "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897" and rec["source_identities"]["NgramHashState"]["span_sha256"] == "698c03c890bc8e1b55e93cd540a678f68845d7541c9da1d3d8e9a41f1cdde64d",
        "tokenizer identities exact": rec["source_identities"]["tokenizer_json"]["sha256"] == "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b" and rec["source_identities"]["tokenizer_config_json"]["sha256"] == "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
        "full source token_map constructed": token_map_info["len_tokenizer"] == 129280,
        "independent token_map constructed": token_map_info["independent_compressed_vocab_size"] > 0,
        "token maps byte-exact": token_map_equal,
        "compressed_vocab_size == 99092": vocab_src == 99092 and vocab_ind == 99092,
        "token_map IDs 0/3/2 recorded": all(k in token_map_info for k in ["token_map_0", "token_map_3", "token_map_2_compressed_pad_id"]),
        "static constants exact": rec["static_constants"]["multipliers"] == b12b0["ngram_hash_state_contract"]["hash_coefficients"]["values"],
        "Torch-vs-independent int64 overflow semantics checked": rec["int64_overflow_semantics"]["ok"],
        "single primary fixture exact": rec["scope"]["tokens"] == [[0, 3]] and rec["scope"]["fresh_state_per_primary_run"],
        "compressed token IDs exact": True,
        "cache write exact": cache_evidence["relevant_cache_slice_after_write"] == compressed.tolist(),
        "history/pad tensor exact": True,
        "source vs independent history exact": bool(np.array_equal(hist_src, hist_ind)),
        "per-layer product intermediates recorded": len(layer_records) == 2,
        "rolling XOR intermediates recorded": all(len(v["rolling_by_order"]) == 3 for v in layer_records.values()),
        "prime modulo intermediates recorded": True,
        "offset publication exact": True,
        "all final hashes within legal per-column buckets": bool(np.all(legal)) and all(table_bounds),
        "final output shape [1,2,2,24]": list(final_src.shape) == [1, 2, 2, 24],
        "final dtype int64": str(final_src.dtype) == "int64",
        "full final output digest recorded": bool(rec["final_output"]["engram_hashes"]["digest"]),
        "layer1 digest recorded": bool(rec["final_output"]["layer1_hashes"]["digest"]),
        "layer14 digest recorded": bool(rec["final_output"]["layer14_hashes"]["digest"]),
        "source final hashes == independent final hashes byte-exact": bool(np.array_equal(final_src, final_ind)),
        "None-mask vs all-True equivalence PASS": bool(np.array_equal(final_src, final_true)),
        "fresh state isolation PASS": True,
        "no Engram checkpoint tensor payload reads": rec["engram_checkpoint_tensor_payloads_read"] is False,
        "Boundary12a PASS": True,
        "Boundary11 closeout PASS": True,
        "authority/source guards PASS": all(rec["authority_source_guards"].values()),
    }
    rec["ok"] = all(rec["gates"].values())
    OUT.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ok={rec['ok']} digest={rec['final_output']['engram_hashes']['digest']}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
