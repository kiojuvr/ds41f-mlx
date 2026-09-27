"""Architecture-neutral prefill session handoff contract.

Milestone 2 produces this state after a successful DwarfStar-derived prefill
transaction.  It deliberately does not choose the Milestone 3 decode runtime;
it records the committed model/session frontier a decoder candidate must accept
without recomputing prefill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefillSessionHandoff:
    schema: str
    token_frontier: int
    tokens_digest: str
    last_logits_digest: str
    committed_shared_state: dict[str, str | None]
    current_hc_digest: str
    pre_mix_digest: str
    engram_history: dict[str, str | None]
    decode_architecture_selected: bool = False
    requires_no_prefill_recompute: bool = True
    extra_state: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "classification": "architecture_neutral_prefill_to_decode_contract",
            "token_frontier": self.token_frontier,
            "tokens_digest": self.tokens_digest,
            "last_logits_digest": self.last_logits_digest,
            "committed_shared_state": self.committed_shared_state,
            "current_hc_digest": self.current_hc_digest,
            "pre_mix_digest": self.pre_mix_digest,
            "engram_history": self.engram_history,
            "decode_architecture_selected": self.decode_architecture_selected,
            "requires_no_prefill_recompute": self.requires_no_prefill_recompute,
            "extra_state": self.extra_state or {},
        }
