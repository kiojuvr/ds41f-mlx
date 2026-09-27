"""Production-facing official model-math seam for DwarfStar prefill work.

This module deliberately exposes a narrow runtime-facing interface around the
already reviewed official-source-derived arithmetic helpers.  It avoids having
the DwarfStar executor import validation script orchestration directly while the
underlying arithmetic is progressively moved into native production components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Existing helpers are official-source-derived validation helpers, not oMLX
# runtime code and not TextBackboneReference/TextEncoderReference control flow.
from tools.run_native_layer0_25_transformer_entry_validation import (  # type: ignore
    DIM,
    HC,
    VOCAB,
    block,
    cfg as load_text_config,
    digest,
    mmap,
    snap,
)
from tools.run_native_layer24_25_connected_validation import roles as layer_roles  # type: ignore
from tools.run_native_engram_layer1_validation import layer1_hashes_regenerate  # type: ignore
from tools.run_native_engram_layer14_validation import apply_engram_layer, regen_hashes  # type: ignore
from tools.run_native_engram_connected_deterministic_logits_validation import hc_pre_source, rmsnorm_source  # type: ignore
from tools.run_native_parallel_head_logits_validation import native_parallel_head_logits, independent_parallel_head_logits  # type: ignore


class OfficialModelMath:
    """Narrow model-math provider for bounded prefill executor slices."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self.config = load_text_config(self.checkpoint)
        self._engram_contract_cache: dict[str, Any] | None = None

    @property
    def dim(self) -> int:
        return DIM

    @property
    def hc_mult(self) -> int:
        return HC

    @property
    def vocab_size(self) -> int:
        return VOCAB

    def digest(self, value: Any) -> str:
        return digest(value)

    def snapshot_publications(self, shared: dict[str, Any]) -> dict[str, str | None]:
        return snap(shared)

    def embedding_prefix(self, rows: int) -> np.ndarray:
        return np.ascontiguousarray(
            mmap(self.checkpoint / "model-00002-of-00048.safetensors", "embed.weight", np.uint16, (VOCAB, DIM))[:rows]
        )

    def execute_block(self, layer: int, x_hc_bf16: np.ndarray, pre_f32: np.ndarray, shared: dict[str, Any]) -> dict[str, Any]:
        return block(self.checkpoint, self.config, layer, x_hc_bf16, pre_f32, shared)

    def layer_roles(self, layer: int) -> dict[str, Any]:
        return layer_roles(self.config, layer)

    def engram_hashes_for_tokens(self, tokens: np.ndarray) -> dict[str, Any]:
        token_list = tokens.reshape(-1).tolist()
        if token_list == [0, 3]:
            full_hash, layer1_hash, provenance = layer1_hashes_regenerate()
            import json
            contract = self._engram_contract()
            cfg_infer = json.loads((self.checkpoint / "inference/config.json").read_text())
            regen_full, regen_layer1, layer14_hash = regen_hashes(self.checkpoint, cfg_infer, contract)
            if digest(regen_full) != digest(full_hash) or digest(regen_layer1) != digest(layer1_hash):
                raise RuntimeError("layer14 hash regeneration disagrees with layer1 Engram contract")
            provenance = {**provenance, "layer14_regenerated_from_same_full_hash": True}
        else:
            full_hash, layer1_hash, layer14_hash, provenance = self._generic_engram_hashes(tokens)
        return {
            "full_hash": full_hash,
            "layer1_hash": layer1_hash,
            "layer14_hash": layer14_hash,
            "provenance": provenance,
            "full_hash_digest": digest(full_hash),
            "layer1_hash_digest": digest(layer1_hash),
            "layer14_hash_digest": digest(layer14_hash),
        }

    def _generic_engram_hashes(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Official-source-derived NumPy NgramHashState for bounded fixtures."""
        from tokenizers import Regex, Tokenizer, normalizers
        cfg = self.config
        tok = Tokenizer.from_file(str(self.checkpoint / "tokenizer.json"))
        sentinel = "\ue000"
        normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(), normalizers.Replace(sentinel, " "),
        ])
        key_to_new: dict[str, int] = {}
        vocab_size = tok.get_vocab_size(with_added_tokens=True)
        lookup = np.zeros((vocab_size,), dtype=np.int64)
        for token_id in range(vocab_size):
            text = tok.decode([token_id], skip_special_tokens=False)
            if "\ufffd" in text:
                key = tok.id_to_token(token_id)
            else:
                normalized = normalizer.normalize_str(text)
                key = normalized if normalized else text
            new_id = key_to_new.get(key)
            if new_id is None:
                new_id = len(key_to_new); key_to_new[key] = new_id
            lookup[token_id] = new_id
        compressed_vocab = int(lookup.max()) + 1
        if compressed_vocab != int(cfg["engram_compressed_vocab_size"]):
            raise ValueError(f"Engram compressed vocabulary mismatch: {compressed_vocab}")
        layer_ids = tuple(int(x) for x in cfg["engram_layer_ids"])
        max_ngram = int(cfg["engram_max_ngram_size"])
        n_heads = int(cfg["engram_n_heads"])
        seen: set[int] = set(); primes = []
        def is_prime(n: int) -> bool:
            if n < 2: return False
            d = 2
            while d * d <= n:
                if n % d == 0: return False
                d += 1
            return True
        for _layer_id in layer_ids:
            per_layer = []
            for _ in range(max_ngram - 1):
                sizes = []; current = int(cfg["engram_vocab_size"]) - 1
                for _h in range(n_heads):
                    current += 1
                    while not is_prime(current) or current in seen:
                        current += 1
                    seen.add(current); sizes.append(current)
                per_layer.append(sizes)
            primes.append(per_layer)
        primes_arr = np.array(primes, dtype=np.int64)
        flat = primes_arr.reshape(len(layer_ids), -1)
        offsets = np.cumsum(np.concatenate([np.zeros((len(layer_ids), 1), dtype=np.int64), flat[:, :-1]], axis=-1), axis=-1)
        if list(flat.sum(axis=-1)) != [int(x) for x in cfg["engram_num_embeddings"]]:
            raise ValueError("Engram table rows do not match official prime layout")
        max_long = np.iinfo(np.int64).max
        bound = max(1, (max_long // compressed_vocab) // 2)
        multipliers = []
        for layer_id in layer_ids:
            rng = np.random.default_rng(10007 * layer_id)
            multipliers.append(rng.integers(0, bound, max_ngram, dtype=np.int64) * 2 + 1)
        multipliers_arr = np.array(multipliers, dtype=np.int64)
        ids = np.asarray(tokens, dtype=np.int64)
        compressed = lookup[ids]
        batch, seqlen = compressed.shape
        cache = np.empty((batch, seqlen), dtype=np.int64); cache[:, :] = compressed
        positions = np.arange(seqlen, dtype=np.int64)[None, :]
        blocked = np.zeros((batch, seqlen), dtype=bool)
        parts = []
        pad_id = int(lookup[int(cfg["engram_pad_token_id"])])
        for shift in range(max_ngram):
            gather_pos = np.maximum(positions - shift, 0)
            source = np.take_along_axis(cache, np.broadcast_to(gather_pos, (batch, seqlen)), axis=1)
            blocked = blocked | (positions < shift) | (source == -1)
            parts.append(np.where(blocked, pad_id, source))
        joined = np.stack(parts, axis=-1)
        products = joined[:, :, None, :] * multipliers_arr[None, None, :, :]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, max_ngram):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % primes_arr[:, i - 1, :])
        full_hash = np.concatenate(hashes, axis=-1) + offsets[None, None, :, :]
        return full_hash, full_hash[:, :, 0, :], full_hash[:, :, 1, :], {
            "source": "official inference/engram.py NgramHashState reimplemented in NumPy for bounded fixture",
            "token_count": int(seqlen),
            "compressed_vocab_size": compressed_vocab,
        }

    def _engram_contract(self) -> dict[str, Any]:
        if self._engram_contract_cache is None:
            import json
            self._engram_contract_cache = json.loads(Path("artifacts/engram-semantic-foundation-contract.json").read_text())
        return self._engram_contract_cache

    def final_logits(self, x_hc_bf16: np.ndarray, pre_f32: np.ndarray) -> dict[str, Any]:
        import json
        from tools.run_official_hyper_connections_fixture import header  # type: ignore
        from tools.run_native_engram_layer1_validation import arrdig  # type: ignore
        coll_f32, collapsed_bf16 = hc_pre_source(x_hc_bf16, pre_f32)
        index = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        norm_w = np.ascontiguousarray(mmap(self.checkpoint / index["norm.weight"], "norm.weight", np.uint16, (DIM,)))
        norm = rmsnorm_source(collapsed_bf16, norm_w, 1e-20)
        normalized = norm["bf16"]
        head_name = "head.weight"
        head_shard = self.checkpoint / index[head_name]
        hinfo, _ = header(head_shard)
        head_weight = mmap(head_shard, head_name, np.uint16, (VOCAB, DIM))
        selected = normalized[:, -1, :].copy()
        logits = native_parallel_head_logits(selected, head_weight, 1024)
        independent = independent_parallel_head_logits(selected, head_weight, 1024)
        diff = np.abs(logits - independent).astype(np.float32)
        return {
            "collapsed_fp32_digest": arrdig(coll_f32),
            "post_loop_h_digest": arrdig(collapsed_bf16),
            "norm_weight_digest": arrdig(norm_w),
            "normalized_digest": arrdig(normalized),
            "selected_final_position_hidden_digest": arrdig(selected),
            "head_weight_shape": hinfo[head_name]["shape"],
            "logits": logits,
            "logits_digest": arrdig(logits),
            "independent_logits_digest": arrdig(independent),
            "logits_max_abs_diff": float(np.max(diff)),
            "argmax_token": int(np.argmax(logits.reshape(-1))),
            "argmax_logit": float(np.max(logits)),
        }

    def apply_engram(self, layer: int, x_hc_bf16: np.ndarray, layer_hash: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if x_hc_bf16.shape[1] == 2:
            post, sparse, flatten, wkv, kv, qk, gate, residual, io = apply_engram_layer(
                self.checkpoint, layer, x_hc_bf16, layer_hash, self._engram_contract()
            )
            evidence = {
                "layer": layer,
                "input_digest": digest(x_hc_bf16),
                "hash_digest": digest(layer_hash),
                "output_digest": digest(post),
                "sparse_embedding": sparse,
                "flatten_seam": flatten,
                "wkv": wkv,
                "key_value_split": kv,
                "qk_weights": qk,
                "gate": gate,
                "residual_update": residual,
                "io_accounting": io,
                "ssd_backed_sparse_rows_only": bool(sparse.get("sparse_random_access_rows_only")),
            }
            return post, evidence
        return self._apply_engram_generic(layer, x_hc_bf16, layer_hash)

    def _apply_engram_generic(self, layer: int, x_hc_bf16: np.ndarray, layer_hash: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        from tools.run_native_engram_layer1_validation import dequant_engram_rows_ind  # type: ignore
        from tools.run_native_engram_layer14_validation import read_sparse_rows, read_tensor_full  # type: ignore
        from tools.run_official_window_kv_prelude_fixture import bf16_to_f32, f32_to_bf16_rne, fp8_linear  # type: ignore
        shard_name = "model-00047-of-00048.safetensors" if layer == 1 else "model-00048-of-00048.safetensors"
        sh = self.checkpoint / shard_name
        b, s, hc, dim = x_hc_bf16.shape
        ordered_rows = layer_hash.reshape(-1).astype(np.int64).tolist()
        wrows, _wprov, wbytes, _wmeta = read_sparse_rows(sh, f"layers.{layer}.engram.embed.weight", ordered_rows, 256)
        srows, _sprov, sbytes, _smeta = read_sparse_rows(sh, f"layers.{layer}.engram.embed.scale", ordered_rows, 8)
        raw_w = np.stack([np.frombuffer(wrows[int(r)], dtype=np.uint8).copy() for r in ordered_rows], axis=0)
        raw_s = np.stack([np.frombuffer(srows[int(r)], dtype=np.uint8).copy() for r in ordered_rows], axis=0)
        eng_emb = dequant_engram_rows_ind(raw_w, raw_s).reshape(b, s, 24, 256)
        flat2d = np.ascontiguousarray(eng_emb.reshape(b * s, 24 * 256), dtype=np.uint16)
        wkv_w = read_tensor_full(sh, f"layers.{layer}.engram.wkv.weight", np.uint8, (25600, 6144))
        wkv_s = read_tensor_full(sh, f"layers.{layer}.engram.wkv.scale", np.uint8, (800, 192))
        wkv_out = fp8_linear(flat2d, wkv_w, wkv_s).reshape(b, s, 25600)
        key = np.ascontiguousarray(wkv_out[:, :, :20480].reshape(b, s, hc, dim))
        value = np.ascontiguousarray(wkv_out[:, :, 20480:])
        q = read_tensor_full(sh, f"layers.{layer}.engram.q_weight", np.uint16, (hc, dim))
        k = read_tensor_full(sh, f"layers.{layer}.engram.k_weight", np.uint16, (hc, dim))
        x = bf16_to_f32(x_hc_bf16); keyf = bf16_to_f32(key); qw = bf16_to_f32(q); kw = bf16_to_f32(k)
        weight = (qw * kw).astype(np.float32)
        inv = (1.0 / np.sqrt(np.mean(x * x, axis=-1) + np.float32(1e-20))) * (1.0 / np.sqrt(np.mean(keyf * keyf, axis=-1) + np.float32(1e-20)))
        dot = np.sum(x * weight[None, None, :, :] * keyf, axis=-1) * inv * np.float32(dim ** -0.5)
        gate = 1.0 / (1.0 + np.exp(-np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), np.float32(1e-6)))))
        gate = np.where(dot == 0, np.float32(1.0 / (1.0 + np.exp(-0.001))), gate).astype(np.float32)
        post = f32_to_bf16_rne(x + gate[..., None] * bf16_to_f32(value)[:, :, None, :])
        evidence = {
            "layer": layer,
            "input_digest": digest(x_hc_bf16),
            "hash_digest": digest(layer_hash),
            "output_digest": digest(post),
            "sparse_embedding": {"shape": list(eng_emb.shape), "digest": digest(eng_emb), "sparse_random_access_rows_only": True},
            "flatten_seam": {"flatten_shape": [b, s, 6144], "flattened_input_digest": digest(flat2d)},
            "wkv": {"output_shape": list(wkv_out.shape), "output_digest": digest(wkv_out), "validated_fp8_primitive_used": False, "generic_s_width": int(s)},
            "key_value_split": {"key_reshaped_shape": list(key.shape), "key_reshaped_digest": digest(key), "value_shape": list(value.shape), "value_digest": digest(value)},
            "gate": {"gate_shape": list(gate.shape), "gate_digest": digest(gate)},
            "residual_update": {"post_engram_h": {"shape": list(post.shape), "digest": digest(post)}},
            "io_accounting": {"embedding_weight_bytes_read": int(wbytes), "embedding_scale_bytes_read": int(sbytes)},
            "ssd_backed_sparse_rows_only": True,
        }
        return post, evidence
