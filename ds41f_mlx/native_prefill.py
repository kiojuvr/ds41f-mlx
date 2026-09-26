"""ctypes bridge for the native DwarfStar-style prefill planner."""

from __future__ import annotations

import ctypes as C
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ds41f_mlx.m2_state_publication import build_publication_frontiers
except ModuleNotFoundError:  # allow static native planner tools without oMLX/numpy env
    build_publication_frontiers = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = ROOT / "ds41f_mlx" / "native"

DS4_AUTHORITY_SHA = "0aaea5a238fb41a35106a551e73c8409dfb751ac"
DS4_AUTHORITY_REMOTE = "https://github.com/antirez/ds4.git"
DS4_AUTHORITY_CAVEAT = "pin contains ds41_graph_prefill_sweep/deferred-decoder authority; native planner records V4.1 topology/lifetime contract and bounded Metal-backed buffer submission without model math"

FRONTIER_BITS = {
    "engram": 0x01,
    "kv": 0x02,
    "index_k": 0x04,
    "idx": 0x08,
    "candidates": 0x10,
}


class NativeConfig(C.Structure):
    _fields_ = [
        ("n_layers", C.c_uint32),
        ("n_chunks", C.c_uint32),
        ("dim", C.c_uint32),
        ("hc_mult", C.c_uint32),
        ("vocab_size", C.c_uint32),
        ("bytes_per_hidden_scalar", C.c_uint32),
        ("chunk_lengths", C.c_uint32 * 256),
        ("frontier_masks", C.c_uint32 * 128),
    ]


class NativeSweepConfig(C.Structure):
    _fields_ = [
        ("ctx", C.c_uint32),
        ("remaining", C.c_uint32),
        ("dim", C.c_uint32),
        ("hc_mult", C.c_uint32),
        ("vocab_size", C.c_uint32),
        ("bytes_per_hidden_scalar", C.c_uint32),
        ("encoder_resident", C.c_uint32),
        ("encoder_only", C.c_uint32),
        ("resume_encoder", C.c_uint32),
        ("frontier_masks", C.c_uint32 * 128),
        ("memory_budget_bytes", C.c_uint64),
    ]


class NativeSweepAllocation(C.Structure):
    _fields_ = [
        ("semantic_role", C.c_char * 48),
        ("size_bytes", C.c_uint64),
        ("alignment", C.c_uint32),
        ("representation", C.c_char * 24),
        ("first_use_step", C.c_uint32),
        ("last_use_step", C.c_uint32),
        ("owner", C.c_char * 32),
        ("alias_reuse_class", C.c_char * 32),
        ("persistence_class", C.c_uint32),
        ("survives_encoder_only", C.c_uint32),
        ("survives_deferred_decoder", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | str | bool]:
        return {
            "semantic_role": bytes(self.semantic_role).split(b"\0", 1)[0].decode(),
            "size_bytes": int(self.size_bytes),
            "alignment": int(self.alignment),
            "representation": bytes(self.representation).split(b"\0", 1)[0].decode(),
            "first_use_step": int(self.first_use_step),
            "last_use_step": int(self.last_use_step),
            "owner": bytes(self.owner).split(b"\0", 1)[0].decode(),
            "alias_reuse_class": bytes(self.alias_reuse_class).split(b"\0", 1)[0].decode(),
            "persistence_class": PERSISTENCE_NAMES.get(int(self.persistence_class), str(int(self.persistence_class))),
            "survives_encoder_only": bool(self.survives_encoder_only),
            "survives_deferred_decoder": bool(self.survives_deferred_decoder),
        }


class NativeOfficialLinearResult(C.Structure):
    _fields_ = [
        ("output_checksum", C.c_uint64),
        ("rows", C.c_uint32),
        ("in_dim", C.c_uint32),
        ("out_dim", C.c_uint32),
        ("metal_enabled", C.c_uint32),
        ("metal_command_buffers", C.c_uint32),
        ("metal_compute_encoders", C.c_uint32),
        ("metal_completion_waits", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | bool]:
        return {
            "output_checksum": int(self.output_checksum),
            "rows": int(self.rows),
            "in_dim": int(self.in_dim),
            "out_dim": int(self.out_dim),
            "metal_enabled": bool(self.metal_enabled),
            "metal_command_buffers": int(self.metal_command_buffers),
            "metal_compute_encoders": int(self.metal_compute_encoders),
            "metal_completion_waits": int(self.metal_completion_waits),
        }


class NativeOfficialEmbeddingResult(C.Structure):
    _fields_ = [
        ("output_checksum", C.c_uint64),
        ("n_tokens", C.c_uint32),
        ("dim", C.c_uint32),
        ("vocab_rows", C.c_uint32),
        ("metal_enabled", C.c_uint32),
        ("metal_command_buffers", C.c_uint32),
        ("metal_compute_encoders", C.c_uint32),
        ("metal_completion_waits", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | bool]:
        return {
            "output_checksum": int(self.output_checksum),
            "n_tokens": int(self.n_tokens),
            "dim": int(self.dim),
            "vocab_rows": int(self.vocab_rows),
            "metal_enabled": bool(self.metal_enabled),
            "metal_command_buffers": int(self.metal_command_buffers),
            "metal_compute_encoders": int(self.metal_compute_encoders),
            "metal_completion_waits": int(self.metal_completion_waits),
        }


class NativeOfficialRMSNormResult(C.Structure):
    _fields_ = [
        ("output_checksum", C.c_uint64),
        ("rows", C.c_uint32),
        ("dim", C.c_uint32),
        ("metal_enabled", C.c_uint32),
        ("metal_command_buffers", C.c_uint32),
        ("metal_compute_encoders", C.c_uint32),
        ("metal_completion_waits", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | bool]:
        return {
            "output_checksum": int(self.output_checksum),
            "rows": int(self.rows),
            "dim": int(self.dim),
            "metal_enabled": bool(self.metal_enabled),
            "metal_command_buffers": int(self.metal_command_buffers),
            "metal_compute_encoders": int(self.metal_compute_encoders),
            "metal_completion_waits": int(self.metal_completion_waits),
        }


class NativeOfficialRotaryResult(C.Structure):
    _fields_ = [
        ("freqs_checksum", C.c_uint64),
        ("rotated_checksum", C.c_uint64),
        ("inverse_checksum", C.c_uint64),
        ("batch", C.c_uint32),
        ("seqlen", C.c_uint32),
        ("heads", C.c_uint32),
        ("dim", C.c_uint32),
        ("original_seq_len", C.c_uint32),
        ("metal_enabled", C.c_uint32),
        ("metal_command_buffers", C.c_uint32),
        ("metal_compute_encoders", C.c_uint32),
        ("metal_completion_waits", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | bool]:
        return {
            "freqs_checksum": int(self.freqs_checksum),
            "rotated_checksum": int(self.rotated_checksum),
            "inverse_checksum": int(self.inverse_checksum),
            "batch": int(self.batch),
            "seqlen": int(self.seqlen),
            "heads": int(self.heads),
            "dim": int(self.dim),
            "original_seq_len": int(self.original_seq_len),
            "metal_enabled": bool(self.metal_enabled),
            "metal_command_buffers": int(self.metal_command_buffers),
            "metal_compute_encoders": int(self.metal_compute_encoders),
            "metal_completion_waits": int(self.metal_completion_waits),
        }


class NativeSubmissionBuffer(C.Structure):
    _fields_ = [
        ("semantic_role", C.c_char * 48),
        ("size_bytes", C.c_uint64),
        ("checksum", C.c_uint64),
        ("allocated", C.c_uint32),
        ("touched", C.c_uint32),
        ("persistence_class", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | str | bool]:
        return {
            "semantic_role": bytes(self.semantic_role).split(b"\0", 1)[0].decode(),
            "size_bytes": int(self.size_bytes),
            "checksum": int(self.checksum),
            "allocated": bool(self.allocated),
            "touched": bool(self.touched),
            "persistence_class": PERSISTENCE_NAMES.get(int(self.persistence_class), str(int(self.persistence_class))),
        }


class NativeSubmissionInfo(C.Structure):
    _fields_ = [
        ("total_owned_bytes", C.c_uint64),
        ("memory_budget_bytes", C.c_uint64),
        ("n_buffers", C.c_uint32),
        ("submitted_commands", C.c_uint32),
        ("submitted_command_buffers", C.c_uint32),
        ("encoded_buffer_ops", C.c_uint32),
        ("checkpoint_valid", C.c_uint32),
        ("current_hc_slot", C.c_uint32),
        ("failed_command_index", C.c_uint32),
        ("metal_enabled", C.c_uint32),
        ("metal_buffers_allocated", C.c_uint32),
        ("metal_command_buffers_committed", C.c_uint32),
        ("metal_blit_encoders_committed", C.c_uint32),
        ("metal_completion_waits", C.c_uint32),
        ("command_buffers_encode_rows", C.c_uint32),
        ("command_buffers_decoder_prepare", C.c_uint32),
        ("command_buffers_swap_hc", C.c_uint32),
        ("command_buffers_output", C.c_uint32),
        ("command_buffers_by_phase", C.c_uint32 * 8),
        ("command_buffers_by_layer", C.c_uint32 * 128),
        ("buffers", NativeSubmissionBuffer * 32),
    ]

    def to_json(self) -> dict[str, Any]:
        return {
            "total_owned_bytes": int(self.total_owned_bytes),
            "memory_budget_bytes": int(self.memory_budget_bytes),
            "n_buffers": int(self.n_buffers),
            "submitted_commands": int(self.submitted_commands),
            "submitted_command_buffers": int(self.submitted_command_buffers),
            "encoded_buffer_ops": int(self.encoded_buffer_ops),
            "checkpoint_valid": bool(self.checkpoint_valid),
            "current_hc_slot": int(self.current_hc_slot),
            "failed_command_index": None if int(self.failed_command_index) == 0xFFFFFFFF else int(self.failed_command_index),
            "metal_enabled": bool(self.metal_enabled),
            "metal_buffers_allocated": int(self.metal_buffers_allocated),
            "metal_command_buffers_committed": int(self.metal_command_buffers_committed),
            "metal_blit_encoders_committed": int(self.metal_blit_encoders_committed),
            "metal_completion_waits": int(self.metal_completion_waits),
            "command_buffers_by_category": {
                "encode_rows": int(self.command_buffers_encode_rows),
                "decoder_prepare_suffix": int(self.command_buffers_decoder_prepare),
                "swap_hc_layer": int(self.command_buffers_swap_hc),
                "final_output_read": int(self.command_buffers_output),
            },
            "command_buffers_by_phase": {SWEEP_PHASE_NAMES.get(i, str(i)): int(self.command_buffers_by_phase[i]) for i in range(8) if int(self.command_buffers_by_phase[i])},
            "command_buffers_by_layer": {str(i): int(self.command_buffers_by_layer[i]) for i in range(128) if int(self.command_buffers_by_layer[i])},
            "buffers": [self.buffers[i].to_json() for i in range(int(self.n_buffers))],
        }


class NativeSweepCommand(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32),
        ("layer", C.c_uint32),
        ("offset", C.c_uint32),
        ("rows", C.c_uint32),
        ("phase", C.c_uint32),
        ("checkpoint_valid", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int | str | bool | None]:
        layer = int(self.layer)
        return {
            "kind": int(self.kind),
            "kind_name": SWEEP_COMMAND_NAMES.get(int(self.kind), f"unknown_{int(self.kind)}"),
            "layer": None if layer == 0xFFFFFFFF else layer,
            "offset": int(self.offset),
            "rows": int(self.rows),
            "phase": SWEEP_PHASE_NAMES.get(int(self.phase), str(int(self.phase))),
            "checkpoint_valid": bool(self.checkpoint_valid),
        }


class NativeSweepPlan(C.Structure):
    _fields_ = [
        ("count", C.c_uint32),
        ("prefill_cap", C.c_uint32),
        ("encoder_chunk", C.c_uint32),
        ("wide", C.c_uint32),
        ("decoder_suffix", C.c_uint32),
        ("encoder_only", C.c_uint32),
        ("resume_encoder", C.c_uint32),
        ("defer_decoder_candidate", C.c_uint32),
        ("checkpoint_valid_during_sweep", C.c_uint32),
        ("checkpoint_valid_after_sweep", C.c_uint32),
        ("encoder_row_layer_work", C.c_uint64),
        ("decoder_suffix_row_layer_work", C.c_uint64),
        ("total_row_layer_work", C.c_uint64),
        ("n_allocations", C.c_uint32),
        ("n_commands", C.c_uint32),
        ("n_prefetch_events", C.c_uint32),
        ("n_checkpoint_transitions", C.c_uint32),
        ("allocations", NativeSweepAllocation * 32),
        ("commands", NativeSweepCommand * 2048),
    ]

    def to_json(self) -> dict[str, Any]:
        commands = [self.commands[i].to_json() for i in range(int(self.n_commands))]
        counts: dict[str, int] = {}
        for c in commands:
            name = str(c["kind_name"])
            counts[name] = counts.get(name, 0) + 1
        return {
            "source_authority": {"remote": DS4_AUTHORITY_REMOTE, "commit": DS4_AUTHORITY_SHA},
            "count": int(self.count),
            "prefill_cap": int(self.prefill_cap),
            "encoder_chunk": int(self.encoder_chunk),
            "wide": bool(self.wide),
            "decoder_suffix": bool(self.decoder_suffix),
            "encoder_only": bool(self.encoder_only),
            "resume_encoder": bool(self.resume_encoder),
            "defer_decoder_candidate": bool(self.defer_decoder_candidate),
            "checkpoint_valid_during_sweep": bool(self.checkpoint_valid_during_sweep),
            "checkpoint_valid_after_sweep": bool(self.checkpoint_valid_after_sweep),
            "encoder_row_layer_work": int(self.encoder_row_layer_work),
            "decoder_suffix_row_layer_work": int(self.decoder_suffix_row_layer_work),
            "total_row_layer_work": int(self.total_row_layer_work),
            "allocations": [self.allocations[i].to_json() for i in range(int(self.n_allocations))],
            "commands": commands,
            "command_counts": counts,
            "prefetch_event_count": int(self.n_prefetch_events),
            "checkpoint_transition_count": int(self.n_checkpoint_transitions),
        }


PERSISTENCE_NAMES = {1: "sweep-persistent", 2: "layer-persistent", 3: "deferred-decoder/suffix", 4: "stage-local reusable scratch"}
SWEEP_PHASE_NAMES = {0: "sweep", 1: "encoder", 2: "decoder_full", 3: "decoder_suffix", 4: "deferred_decoder", 5: "final_output"}
SWEEP_COMMAND_NAMES = {
    101: "begin_sweep_checkpoint_invalid",
    102: "prefetch_engram_table0",
    103: "prefetch_engram_table1",
    104: "ssd_read_ahead",
    105: "begin_layer",
    106: "encode_rows",
    107: "decoder_prepare_suffix",
    108: "swap_hc_after_layer",
    109: "publish_state_frontier",
    110: "encoder_only_complete_still_invalid",
    111: "decoder_pending_still_invalid",
    112: "encode_output_head",
    113: "read_logits",
    114: "checkpoint_may_commit",
    115: "end_layer",
}


class NativePlan(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_uint32),
        ("max_chunk_tokens", C.c_uint32),
        ("layer_steps", C.c_uint32),
        ("command_batches", C.c_uint32),
        ("publication_frontiers", C.c_uint32),
        ("output_rows_retained", C.c_uint32),
        ("carry_buffer_bytes", C.c_uint64),
        ("full_prompt_hidden_bytes", C.c_uint64),
        ("last_chunk_logits_bytes", C.c_uint64),
        ("full_prompt_logits_bytes", C.c_uint64),
    ]

    def to_json(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _typ in self._fields_}


class NativeArenaInfo(C.Structure):
    _fields_ = [
        ("carry_buffer_bytes", C.c_uint64),
        ("token_buffer_bytes", C.c_uint64),
        ("total_owned_bytes", C.c_uint64),
        ("current_carry_checksum", C.c_uint64),
        ("next_carry_checksum", C.c_uint64),
        ("n_tokens_uploaded", C.c_uint32),
        ("current_carry_index", C.c_uint32),
        ("executed_encode_chunks", C.c_uint32),
        ("last_encoded_layer", C.c_uint32),
        ("last_encoded_chunk", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _typ in self._fields_}


class NativeCommand(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32),
        ("layer", C.c_uint32),
        ("chunk", C.c_uint32),
        ("token_offset", C.c_uint32),
        ("token_count", C.c_uint32),
        ("frontier_mask", C.c_uint32),
    ]

    def to_json(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _typ in self._fields_}


COMMAND_NAMES = {
    1: "upload_tokens",
    2: "upload_embeddings_hc",
    3: "begin_layer",
    4: "encode_layer_chunk",
    5: "publish_frontier",
    6: "swap_carry",
    7: "end_layer",
    8: "encode_output_head",
    9: "read_logits",
}


@dataclass
class NativePrefillContext:
    native: "NativePrefillLibrary"
    ptr: C.c_void_p

    def close(self) -> None:
        if self.ptr:
            self.native.lib.ds41f_prefill_native_context_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __enter__(self) -> "NativePrefillContext":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def info(self) -> dict[str, int]:
        out = NativeArenaInfo()
        fn = self.native.lib.ds41f_prefill_native_context_info
        fn.argtypes = [C.c_void_p, C.POINTER(NativeArenaInfo)]
        fn.restype = C.c_int
        rc = int(fn(self.ptr, C.byref(out)))
        if rc != 0:
            raise RuntimeError(f"native context info failed rc={rc}")
        return out.to_json()

    def upload_tokens(self, offset: int, tokens: list[int]) -> None:
        arr_t = C.c_int32 * len(tokens)
        arr = arr_t(*tokens)
        fn = self.native.lib.ds41f_prefill_native_upload_tokens
        fn.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_int32), C.c_uint32]
        fn.restype = C.c_int
        rc = int(fn(self.ptr, int(offset), arr, len(tokens)))
        if rc != 0:
            raise RuntimeError(f"native token upload failed rc={rc}")

    def swap_carry(self) -> None:
        fn = self.native.lib.ds41f_prefill_native_swap_carry
        fn.argtypes = [C.c_void_p]
        fn.restype = C.c_int
        rc = int(fn(self.ptr))
        if rc != 0:
            raise RuntimeError(f"native carry swap failed rc={rc}")

    def carry_addresses(self) -> dict[str, int | None]:
        cur = self.native.lib.ds41f_prefill_native_current_carry
        nxt = self.native.lib.ds41f_prefill_native_next_carry
        cur.argtypes = nxt.argtypes = [C.c_void_p]
        cur.restype = nxt.restype = C.c_void_p
        return {"current": cur(self.ptr), "next": nxt(self.ptr)}

    def build_commands(self) -> None:
        fn = self.native.lib.ds41f_prefill_native_build_commands
        fn.argtypes = [C.c_void_p]
        fn.restype = C.c_int
        rc = int(fn(self.ptr))
        if rc != 0:
            raise RuntimeError(f"native command build failed rc={rc}")

    def command_count(self) -> int:
        fn = self.native.lib.ds41f_prefill_native_command_count
        fn.argtypes = [C.c_void_p]
        fn.restype = C.c_uint32
        return int(fn(self.ptr))

    def command_at(self, index: int) -> dict[str, int | str]:
        out = NativeCommand()
        fn = self.native.lib.ds41f_prefill_native_command_at
        fn.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(NativeCommand)]
        fn.restype = C.c_int
        rc = int(fn(self.ptr, int(index), C.byref(out)))
        if rc != 0:
            raise RuntimeError(f"native command_at failed rc={rc}")
        d = out.to_json()
        d["kind_name"] = COMMAND_NAMES.get(d["kind"], f"unknown_{d['kind']}")
        return d

    def command_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in range(self.command_count()):
            name = str(self.command_at(i)["kind_name"])
            counts[name] = counts.get(name, 0) + 1
        return counts

    def execute_noop_graph(self) -> int:
        submitted = C.c_uint32()
        fn = self.native.lib.ds41f_prefill_native_execute_noop_graph
        fn.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
        fn.restype = C.c_int
        rc = int(fn(self.ptr, C.byref(submitted)))
        if rc != 0:
            raise RuntimeError(f"native noop graph failed rc={rc}")
        return int(submitted.value)


@dataclass
class NativeSubmissionContext:
    native: "NativePrefillLibrary"
    ptr: C.c_void_p

    def close(self) -> None:
        if self.ptr:
            self.native.lib.ds41f_prefill_submission_context_destroy(self.ptr)
            self.ptr = C.c_void_p()

    def __enter__(self) -> "NativeSubmissionContext":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def info(self) -> dict[str, Any]:
        out = NativeSubmissionInfo()
        fn = self.native.lib.ds41f_prefill_submission_context_info
        fn.argtypes = [C.c_void_p, C.POINTER(NativeSubmissionInfo)]
        fn.restype = C.c_int
        rc = int(fn(self.ptr, C.byref(out)))
        if rc != 0:
            raise RuntimeError(f"native submission info failed rc={rc}")
        return out.to_json()

    def submit(self) -> int:
        submitted = C.c_uint32()
        fn = self.native.lib.ds41f_prefill_submission_submit
        fn.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
        fn.restype = C.c_int
        rc = int(fn(self.ptr, C.byref(submitted)))
        if rc != 0:
            raise RuntimeError(f"native submission failed rc={rc}")
        return int(submitted.value)


@dataclass(frozen=True)
class NativePrefillLibrary:
    path: Path
    lib: Any

    def version(self) -> str:
        self.lib.ds41f_prefill_native_version.restype = C.c_char_p
        return self.lib.ds41f_prefill_native_version().decode("utf-8")

    def build_sweep_plan(self, cfg: NativeSweepConfig) -> NativeSweepPlan:
        out = NativeSweepPlan()
        fn = self.lib.ds41f_prefill_native_build_sweep_plan
        fn.argtypes = [C.POINTER(NativeSweepConfig), C.POINTER(NativeSweepPlan)]
        fn.restype = C.c_int
        rc = int(fn(C.byref(cfg), C.byref(out)))
        if rc != 0:
            raise RuntimeError(f"native sweep planner failed rc={rc}")
        return out

    def official_embedding_gather_bf16(self, embedding_bf16: Any, tokens: Any) -> tuple[Any, dict[str, int | bool]]:
        import numpy as np
        weights = np.ascontiguousarray(embedding_bf16, dtype=np.uint16)
        toks = np.ascontiguousarray(tokens, dtype=np.int32)
        if weights.ndim != 2:
            raise ValueError("embedding_bf16 must be [vocab_rows, dim]")
        if toks.ndim != 1:
            raise ValueError("tokens must be 1-D")
        out = np.empty((int(toks.shape[0]), int(weights.shape[1])), dtype=np.uint16)
        result = NativeOfficialEmbeddingResult()
        fn = self.lib.ds41f_official_embedding_gather_bf16
        fn.argtypes = [
            C.POINTER(C.c_uint16), C.c_uint32, C.c_uint32,
            C.POINTER(C.c_int32), C.c_uint32,
            C.POINTER(C.c_uint16), C.POINTER(NativeOfficialEmbeddingResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            weights.ctypes.data_as(C.POINTER(C.c_uint16)), int(weights.shape[0]), int(weights.shape[1]),
            toks.ctypes.data_as(C.POINTER(C.c_int32)), int(toks.shape[0]),
            out.ctypes.data_as(C.POINTER(C.c_uint16)), C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official embedding gather failed rc={rc}")
        return out, result.to_json()

    def official_bf16_linear_f32(self, input_bf16: Any, weight_bf16: Any) -> tuple[Any, dict[str, int | bool]]:
        import numpy as np
        x = np.ascontiguousarray(input_bf16, dtype=np.uint16)
        w = np.ascontiguousarray(weight_bf16, dtype=np.uint16)
        if x.ndim != 2 or w.ndim != 2:
            raise ValueError("input and weight must be 2-D")
        if x.shape[1] != w.shape[1]:
            raise ValueError("input/weight inner dimensions differ")
        out = np.empty((int(x.shape[0]), int(w.shape[0])), dtype=np.float32)
        result = NativeOfficialLinearResult()
        fn = self.lib.ds41f_official_bf16_linear_f32
        fn.argtypes = [
            C.POINTER(C.c_uint16), C.POINTER(C.c_uint16),
            C.c_uint32, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_float), C.POINTER(NativeOfficialLinearResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            x.ctypes.data_as(C.POINTER(C.c_uint16)),
            w.ctypes.data_as(C.POINTER(C.c_uint16)),
            int(x.shape[0]), int(x.shape[1]), int(w.shape[0]),
            out.ctypes.data_as(C.POINTER(C.c_float)), C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official bf16 linear failed rc={rc}")
        return out, result.to_json()

    def official_f32_linear_f32(self, input_f32: Any, weight_f32: Any) -> tuple[Any, dict[str, int | bool]]:
        import numpy as np
        x = np.ascontiguousarray(input_f32, dtype=np.float32)
        w = np.ascontiguousarray(weight_f32, dtype=np.float32)
        if x.ndim != 2 or w.ndim != 2:
            raise ValueError("input and weight must be 2-D")
        if x.shape[1] != w.shape[1]:
            raise ValueError("input/weight inner dimensions differ")
        out = np.empty((int(x.shape[0]), int(w.shape[0])), dtype=np.float32)
        result = NativeOfficialLinearResult()
        fn = self.lib.ds41f_official_f32_linear_f32
        fn.argtypes = [
            C.POINTER(C.c_float), C.POINTER(C.c_float),
            C.c_uint32, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_float), C.POINTER(NativeOfficialLinearResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            x.ctypes.data_as(C.POINTER(C.c_float)),
            w.ctypes.data_as(C.POINTER(C.c_float)),
            int(x.shape[0]), int(x.shape[1]), int(w.shape[0]),
            out.ctypes.data_as(C.POINTER(C.c_float)), C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official f32 linear failed rc={rc}")
        return out, result.to_json()

    def official_fp8_linear_bf16(self, input_bf16: Any, weight_fp8: Any, weight_scale_e8m0: Any, block_size: int = 32) -> tuple[Any, dict[str, int | bool]]:
        import numpy as np
        x = np.ascontiguousarray(input_bf16, dtype=np.uint16)
        w = np.ascontiguousarray(weight_fp8, dtype=np.uint8)
        s = np.ascontiguousarray(weight_scale_e8m0, dtype=np.uint8)
        if x.ndim != 2 or w.ndim != 2 or s.ndim != 2:
            raise ValueError("input, weight, and scale must be 2-D")
        if x.shape[1] != w.shape[1]:
            raise ValueError("input/weight inner dimensions differ")
        if x.shape[1] % int(block_size) != 0:
            raise ValueError("inner dimension must be divisible by block_size")
        expected_s = ((int(w.shape[0]) + int(block_size) - 1) // int(block_size), int(x.shape[1]) // int(block_size))
        if tuple(s.shape) != expected_s:
            raise ValueError(f"scale shape {s.shape} != {expected_s}")
        out = np.empty((int(x.shape[0]), int(w.shape[0])), dtype=np.uint16)
        result = NativeOfficialLinearResult()
        fn = self.lib.ds41f_official_fp8_linear_bf16
        fn.argtypes = [
            C.POINTER(C.c_uint16), C.POINTER(C.c_uint8), C.POINTER(C.c_uint8),
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_uint16), C.POINTER(NativeOfficialLinearResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            x.ctypes.data_as(C.POINTER(C.c_uint16)),
            w.ctypes.data_as(C.POINTER(C.c_uint8)),
            s.ctypes.data_as(C.POINTER(C.c_uint8)),
            int(x.shape[0]), int(x.shape[1]), int(w.shape[0]), int(block_size),
            out.ctypes.data_as(C.POINTER(C.c_uint16)), C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official fp8 linear failed rc={rc}")
        return out, result.to_json()

    def official_act_quant_bf16(self, input_bf16: Any, block_size: int = 32) -> tuple[Any, Any, Any, dict[str, int | bool]]:
        import numpy as np
        x = np.ascontiguousarray(input_bf16, dtype=np.uint16)
        if x.ndim != 2:
            raise ValueError("input must be 2-D")
        if x.shape[1] % int(block_size) != 0:
            raise ValueError("inner dimension must be divisible by block_size")
        q = np.empty(x.shape, dtype=np.uint8)
        s = np.empty((int(x.shape[0]), int(x.shape[1]) // int(block_size)), dtype=np.uint8)
        out = np.empty(x.shape, dtype=np.uint16)
        result = NativeOfficialLinearResult()
        fn = self.lib.ds41f_official_act_quant_bf16
        fn.argtypes = [
            C.POINTER(C.c_uint16), C.c_uint32, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint8), C.POINTER(C.c_uint16),
            C.POINTER(NativeOfficialLinearResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            x.ctypes.data_as(C.POINTER(C.c_uint16)), int(x.shape[0]), int(x.shape[1]), int(block_size),
            q.ctypes.data_as(C.POINTER(C.c_uint8)), s.ctypes.data_as(C.POINTER(C.c_uint8)), out.ctypes.data_as(C.POINTER(C.c_uint16)),
            C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official act_quant failed rc={rc}")
        return q, s, out, result.to_json()

    def official_sparse_attn_bf16(self, q_bf16: Any, kv_bf16: Any, attn_sink_f32: Any, topk_idxs: Any, softmax_scale: float) -> tuple[Any, dict[str, int | bool]]:
        import numpy as np
        q = np.ascontiguousarray(q_bf16, dtype=np.uint16)
        kv = np.ascontiguousarray(kv_bf16, dtype=np.uint16)
        sink = np.ascontiguousarray(attn_sink_f32, dtype=np.float32)
        idx = np.ascontiguousarray(topk_idxs, dtype=np.int32)
        if q.ndim != 4 or kv.ndim != 3 or idx.ndim != 3:
            raise ValueError("q must be [B,S,H,D], kv [B,N,D], topk [B,S,T]")
        if q.shape[0] != kv.shape[0] or q.shape[0] != idx.shape[0] or q.shape[1] != idx.shape[1] or q.shape[3] != kv.shape[2]:
            raise ValueError("sparse attention shape mismatch")
        if sink.shape != (q.shape[2],):
            raise ValueError("attn_sink must be [heads]")
        out = np.empty(q.shape, dtype=np.uint16)
        result = NativeOfficialLinearResult()
        fn = self.lib.ds41f_official_sparse_attn_bf16
        fn.argtypes = [
            C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), C.POINTER(C.c_float), C.POINTER(C.c_int32),
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_float,
            C.POINTER(C.c_uint16), C.POINTER(NativeOfficialLinearResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            q.ctypes.data_as(C.POINTER(C.c_uint16)), kv.ctypes.data_as(C.POINTER(C.c_uint16)), sink.ctypes.data_as(C.POINTER(C.c_float)), idx.ctypes.data_as(C.POINTER(C.c_int32)),
            int(q.shape[0]), int(q.shape[1]), int(kv.shape[1]), int(q.shape[2]), int(q.shape[3]), int(idx.shape[2]), float(softmax_scale),
            out.ctypes.data_as(C.POINTER(C.c_uint16)), C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official sparse_attn failed rc={rc}")
        return out, result.to_json()

    def official_rmsnorm_bf16(self, input_bf16: Any, weight_bf16: Any, eps: float = 1.0e-6) -> tuple[Any, dict[str, int | bool]]:
        import numpy as np
        x = np.ascontiguousarray(input_bf16, dtype=np.uint16)
        w = np.ascontiguousarray(weight_bf16, dtype=np.uint16)
        if x.ndim != 2 or w.ndim != 1:
            raise ValueError("input must be 2-D and weight must be 1-D")
        if x.shape[1] != w.shape[0]:
            raise ValueError("input/weight dimensions differ")
        out = np.empty(x.shape, dtype=np.uint16)
        result = NativeOfficialRMSNormResult()
        fn = self.lib.ds41f_official_rmsnorm_bf16
        fn.argtypes = [
            C.POINTER(C.c_uint16), C.POINTER(C.c_uint16),
            C.c_uint32, C.c_uint32, C.c_float,
            C.POINTER(C.c_uint16), C.POINTER(NativeOfficialRMSNormResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            x.ctypes.data_as(C.POINTER(C.c_uint16)),
            w.ctypes.data_as(C.POINTER(C.c_uint16)),
            int(x.shape[0]), int(x.shape[1]), float(eps),
            out.ctypes.data_as(C.POINTER(C.c_uint16)), C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official rmsnorm failed rc={rc}")
        return out, result.to_json()

    def official_rotary_f32(self, input_f32: Any, params: dict[str, Any]) -> tuple[Any, Any, Any, Any, dict[str, int | bool]]:
        import numpy as np
        x = np.ascontiguousarray(input_f32, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("input_f32 must be [batch, seqlen, heads, dim]")
        if x.shape[3] % 2:
            raise ValueError("rotary dim must be even")
        batch, seqlen, heads, dim = [int(v) for v in x.shape]
        freqs_real = np.empty((seqlen, dim // 2), dtype=np.float32)
        freqs_imag = np.empty((seqlen, dim // 2), dtype=np.float32)
        rotated = np.empty_like(x)
        inverse = np.empty_like(x)
        result = NativeOfficialRotaryResult()
        fn = self.lib.ds41f_official_rotary_f32
        fn.argtypes = [
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32,
            C.c_uint32, C.c_float, C.c_float, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float),
            C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(NativeOfficialRotaryResult),
        ]
        fn.restype = C.c_int
        rc = int(fn(
            batch, seqlen, heads, dim,
            int(params["original_seq_len"]), float(params["base"]), float(params["factor"]), int(params["beta_fast"]), int(params["beta_slow"]),
            x.ctypes.data_as(C.POINTER(C.c_float)),
            freqs_real.ctypes.data_as(C.POINTER(C.c_float)),
            freqs_imag.ctypes.data_as(C.POINTER(C.c_float)),
            rotated.ctypes.data_as(C.POINTER(C.c_float)),
            inverse.ctypes.data_as(C.POINTER(C.c_float)),
            C.byref(result),
        ))
        if rc != 0:
            raise RuntimeError(f"native official rotary failed rc={rc}")
        return freqs_real, freqs_imag, rotated, inverse, result.to_json()

    def create_submission_context(self, cfg: NativeSweepConfig) -> NativeSubmissionContext:
        ptr = C.c_void_p()
        fn = self.lib.ds41f_prefill_submission_context_create
        fn.argtypes = [C.POINTER(NativeSweepConfig), C.POINTER(C.c_void_p)]
        fn.restype = C.c_int
        rc = int(fn(C.byref(cfg), C.byref(ptr)))
        if rc != 0:
            raise RuntimeError(f"native submission context create failed rc={rc}")
        self.lib.ds41f_prefill_submission_context_destroy.argtypes = [C.c_void_p]
        self.lib.ds41f_prefill_submission_context_destroy.restype = None
        return NativeSubmissionContext(native=self, ptr=ptr)

    def build_plan(self, cfg: NativeConfig) -> NativePlan:
        out = NativePlan()
        fn = self.lib.ds41f_prefill_native_build_plan
        fn.argtypes = [C.POINTER(NativeConfig), C.POINTER(NativePlan)]
        fn.restype = C.c_int
        rc = int(fn(C.byref(cfg), C.byref(out)))
        if rc != 0:
            raise RuntimeError(f"native prefill planner failed rc={rc}")
        return out

    def create_context(self, cfg: NativeConfig) -> NativePrefillContext:
        ptr = C.c_void_p()
        fn = self.lib.ds41f_prefill_native_context_create
        fn.argtypes = [C.POINTER(NativeConfig), C.POINTER(C.c_void_p)]
        fn.restype = C.c_int
        rc = int(fn(C.byref(cfg), C.byref(ptr)))
        if rc != 0:
            raise RuntimeError(f"native prefill context create failed rc={rc}")
        self.lib.ds41f_prefill_native_context_destroy.argtypes = [C.c_void_p]
        self.lib.ds41f_prefill_native_context_destroy.restype = None
        return NativePrefillContext(native=self, ptr=ptr)


def compile_native_prefill_library(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if __import__("sys").platform == "darwin" else ".so"
    out = out_dir / f"libds41f_prefill_native{suffix}"
    import sys
    cmd = ["cc", "-O2", "-shared", "-fPIC", "-I", str(NATIVE_DIR)]
    if sys.platform == "darwin":
        cmd.extend(["-x", "objective-c", "-DDS41F_ENABLE_METAL=1"])
    else:
        cmd.extend(["-std=c11"])
    cmd.extend([str(NATIVE_DIR / "ds41f_prefill_native.c"), "-o", str(out)])
    if sys.platform == "darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    subprocess.run(cmd, check=True)
    return out


def load_native_prefill_library(path: Path) -> NativePrefillLibrary:
    return NativePrefillLibrary(path=path, lib=C.CDLL(str(path)))


def native_sweep_config_from_model(
    language_model: Any,
    *,
    ctx: int,
    remaining: int,
    encoder_resident: bool = False,
    encoder_only: bool = False,
    resume_encoder: bool = False,
    memory_budget_bytes: int = 0,
) -> NativeSweepConfig:
    cfg_obj = language_model._config
    n_layers = int(getattr(cfg_obj, "n_layers", 40) or 40)
    if n_layers > 128:
        raise ValueError("native sweep planner supports at most 128 layers")
    cfg = NativeSweepConfig()
    cfg.ctx = int(ctx)
    cfg.remaining = int(remaining)
    cfg.dim = int(getattr(cfg_obj, "dim"))
    cfg.hc_mult = int(getattr(cfg_obj, "hc_mult"))
    cfg.vocab_size = int(getattr(cfg_obj, "vocab_size"))
    cfg.bytes_per_hidden_scalar = 2
    cfg.encoder_resident = 1 if encoder_resident else 0
    cfg.encoder_only = 1 if encoder_only else 0
    cfg.resume_encoder = 1 if resume_encoder else 0
    cfg.memory_budget_bytes = int(memory_budget_bytes)
    if build_publication_frontiers is None:
        raise RuntimeError("build_publication_frontiers unavailable; install runtime deps for model-derived config")
    for layer, frontier in build_publication_frontiers(cfg_obj).items():
        mask = 0
        for name in frontier.publishes:
            mask |= FRONTIER_BITS[name]
        cfg.frontier_masks[layer] = mask
    return cfg


def native_sweep_config_static(
    *,
    ctx: int,
    remaining: int,
    dim: int = 5120,
    hc_mult: int = 4,
    vocab_size: int = 129280,
    encoder_resident: bool = False,
    encoder_only: bool = False,
    resume_encoder: bool = False,
    memory_budget_bytes: int = 0,
) -> NativeSweepConfig:
    cfg = NativeSweepConfig()
    cfg.ctx = int(ctx)
    cfg.remaining = int(remaining)
    cfg.dim = int(dim)
    cfg.hc_mult = int(hc_mult)
    cfg.vocab_size = int(vocab_size)
    cfg.bytes_per_hidden_scalar = 2
    cfg.encoder_resident = 1 if encoder_resident else 0
    cfg.encoder_only = 1 if encoder_only else 0
    cfg.resume_encoder = 1 if resume_encoder else 0
    cfg.memory_budget_bytes = int(memory_budget_bytes)
    # DeepSeek-V4.1 publication frontiers observed in M2; enough for static planner artifacts.
    for layer, names in {
        1: ("engram",), 2: ("kv", "index_k", "idx"), 8: ("kv", "index_k", "idx"),
        14: ("engram", "kv", "index_k", "idx"), 20: ("kv", "index_k", "idx", "candidates"),
        24: ("index_k", "idx"), 28: ("index_k", "idx"), 32: ("index_k", "idx"), 36: ("index_k", "idx"),
    }.items():
        mask = 0
        for name in names:
            mask |= FRONTIER_BITS[name]
        cfg.frontier_masks[layer] = mask
    return cfg


def native_config_from_model(language_model: Any, chunk_lengths: list[int]) -> NativeConfig:
    cfg_obj = language_model._config
    n_layers = int(getattr(cfg_obj, "n_layers", 40) or 40)
    if n_layers > 128:
        raise ValueError("native planner supports at most 128 layers")
    if len(chunk_lengths) > 256:
        raise ValueError("native planner supports at most 256 chunks")
    cfg = NativeConfig()
    cfg.n_layers = n_layers
    cfg.n_chunks = len(chunk_lengths)
    cfg.dim = int(getattr(cfg_obj, "dim"))
    cfg.hc_mult = int(getattr(cfg_obj, "hc_mult"))
    cfg.vocab_size = int(getattr(cfg_obj, "vocab_size"))
    cfg.bytes_per_hidden_scalar = 2  # BF16 hidden/carry target from historical oMLX adapter config; architecture sizing only.
    for i, n in enumerate(chunk_lengths):
        cfg.chunk_lengths[i] = int(n)
    if build_publication_frontiers is None:
        raise RuntimeError("build_publication_frontiers unavailable; install runtime deps for model-derived config")
    for layer, frontier in build_publication_frontiers(cfg_obj).items():
        mask = 0
        for name in frontier.publishes:
            mask |= FRONTIER_BITS[name]
        cfg.frontier_masks[layer] = mask
    return cfg
