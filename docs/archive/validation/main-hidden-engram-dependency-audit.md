# Boundary 12a: Transformer `main_hidden` producer-chain / Engram dependency audit

Status: PASS (structural/dependency audit only).  This is not a new model-math implementation and does not promote numeric `main_hidden` authority.

## Authority

Local pinned source/config only:

`/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/`

Reviewed minimum files: `inference/model.py`, `inference/engram.py`, `inference/config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors.index.json`.  Source hashes use `raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1`; machine-readable identities are in `artifacts/main-hidden-engram-dependency-audit.json`.

## Local config facts

From local `inference/config.json`:

- `dspark_target_layer_ids`: `[37, 38, 39]`
- `engram_layer_ids`: `[1, 14]`
- `hc_mult`: `4`
- `dim`: `5120`
- `n_layers`: `40`
- `n_mtp_layers`: `3`

These match the expected public-checkpoint values for the audited keys.

## Exact producer ordering

`Transformer.forward` source establishes this order:

1. `engram_hashes = self.engram_hash(...)`
2. `h = self.embed(input_ids)`
3. `h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)`
4. `main_hiddens = []`
5. loop over `self.layers`
6. if `layer.engram is not None`, update `h = layer.engram(...)`
7. if `i in self.target_layer_ids`, capture `main_hiddens.append(h.mean(dim=2))`
8. run `h, pre_mix = layer(...)`
9. after logits/sampling, assemble `main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None`
10. return `output_ids, logits, main_hidden`

Therefore the target-layer capture occurs after the optional Engram update for that layer and before `Block.forward`.  The captured value is the pre-Block attention input of target layers, not those layers' output.

## Engram dependency conclusion

Although the capture targets are layers 37/38/39, the hidden stream `h` has already passed through configured upstream Engram insertions at layers 1 and 14.  Dependency must follow the upstream residual stream; it is insufficient to ask whether the target layer itself is an Engram layer.

Dependency graph:

```text
tokens
-> engram hash state
-> embedding
-> Block0
-> Engram@1
-> ...
-> Engram@14
-> ...
-> pre-Block37 h
   -> mean(HC)
-> Block37

-> pre-Block38 h
   -> mean(HC)
-> Block38

-> pre-Block39 h
   -> mean(HC)
-> Block39
```

## Structural `main_hidden` contract

For the released config, `target_layer_ids` is non-empty, so the `torch.cat(...)` branch is selected.  If `target_layer_ids` were empty, source contract is `main_hidden = None`.

With `target_layer_ids = [37, 38, 39]`, `hc_mult = 4`, and `dim = 5120`:

- pre-target `h`: `[B,S,HC,D]`
- capture: `h.mean(dim=2)` -> `[B,S,D]`
- concat of three captures along `dim=-1`: `[B,S,3*D]`
- fixture `B=1`, `S=2`, `D=5120`: `[1,2,15360]`

## Mean dtype/arithmetic review

The source operation is `h.mean(dim=2)` with no explicit `dtype=` argument.  Input dtype inherits the current `h` dtype after embedding, blocks, and Engram updates.  Return dtype follows the PyTorch `Tensor.mean` contract for this source expression.  Boundary12a does not promote backend accumulation details to numeric authority; in particular it does not claim a `BF16 -> FP32 mean -> BF16` path.

## Current repo Engram authority inventory

The current `ds41f-mlx` repo has no official-semantic native implementation and no numerical validation authority for:

- `EngramLayout`
- `NgramHashState`
- `ParallelEngramEmbedding`
- `Engram.forward`

Existing Engram references are historical oMLX/perf/trace artifacts or non-Engram boundaries with explicit `no Engram` non-claims. They are not current native official numeric Engram authority.

Thus `engram_numeric_authority_present = false`.

## Promotion gate

All hard-gate inputs are true:

- official config has non-empty `engram_layer_ids`
- `Transformer.forward` applies Engram upstream of target hidden captures
- current runtime lacks validated Engram numerical authority

Therefore:

`main_hidden_numeric_promotion_blocked_by_engram = true`

Boundary12a may close structural source semantics, but must not claim numeric `main_hidden` authority, full `Transformer.forward` authority, or return tuple correctness.

## Critical distinction

Current no-Engram connected path can validate:

- structural placement of `main_hidden` capture
- target-layer selection
- concat shape contract

Current no-Engram connected path cannot validate:

- official numeric `main_hidden` values
- official full `Transformer.forward` return

## Return packaging source contract

Source return order is:

```python
return output_ids, logits, main_hidden
```

This audit records the source contract only; it does not execute or validate the return.

## Recommended next boundary

Boundary 12b: Engram semantic foundation

Suggested decomposition:

- 12b0: Engram source/checkpoint/tokenizer contract
- 12b1: `NgramHashState` for bounded text fixture `[[0,3]]`
- 12b2: `ParallelEngramEmbedding` + `Engram@layer1`
- 12b3: `Engram@layer14` with upstream connected state
- 12b4: replay connected model path with both Engram insertions
- 12c: `main_hidden` captures at 37/38/39 + concat

## Non-claims

- no numeric `main_hidden` authority
- no `Transformer.forward` return correctness
- no DSpark/`forward_spec` correctness
- no MTP correctness
- no decode semantics
- no cache persistence qualification
- no full-model correctness
- no performance/production qualification
