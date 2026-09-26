# Boundary12c: Engram-connected `main_hidden` validation

Status: **PASS / authority promoted**.

Boundary12c validates the bounded `[[0,3]]` prefill fixture for the source contract in `Transformer.forward`:

```python
main_hiddens = []
...
if i in self.target_layer_ids:
    main_hiddens.append(h.mean(dim=2))
...
main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
```

## Initial failed attempt

The first Boundary12c attempt failed because the project `.venv` could not import PyTorch:

```text
ModuleNotFoundError("No module named 'torch'")
```

No authority was promoted from that failure.

## Successful reference setup

PyTorch remains a **reference-only bounded framework semantic oracle**, not a production/native runtime dependency:

```text
pytorch_role = reference-only bounded framework semantic oracle
pytorch_is_production_dependency = false
```

No torch package was installed into the repo `.venv`, and no dependency files were changed. The validation uses an isolated repo-external reference environment selected via:

```bash
DS41F_TORCH_REFERENCE_PYTHON=/Users/kioju/.cache/ds41f-pytorch-reference/bin/python
```

Official local model metadata contains a PyTorch version requirement in:

```text
/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/requirements.txt
```

with:

```text
torch>=2.10.0
```

The reference environment used PyTorch `2.14.0`, satisfying that range, on native arm64 macOS Python 3.12.

## Source/config identity

Recorded source gates:

- `inference/model.py` SHA256: `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`
- `Transformer.forward` lines 1242-1272 SHA256: `6f309bf24d79649472588cb0fda651646f9d8ee46c5b5f0dc5de482cd6f94a0c`
- narrower main-hidden contract span lines 1259-1272 SHA256: `56ba7227f4202c0791c33e08300943a0675ef07b62ee824af92889d1e7dd57a3`
- `inference/config.json` SHA256: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`
- config regressions: `dspark_target_layer_ids == [37,38,39]`, `hc_mult == 4`, `dim == 5120`

## Authority result

For the pinned DeepSeek-V4.1-Flash source/checkpoint/tokenizer and the bounded Engram-connected `[[0,3]]` prefill fixture, the target-layer hidden states captured immediately before Blocks37, 38, and 39 are reproduced from the same connected trajectory as the current Boundary12b4 logits.

For each target layer, PyTorch CPU BF16 `h.mean(dim=2)` on exact captured BF16 bytes agrees byte-exactly with the independently predeclared reduction:

```text
acc = float32(0)
acc = float32(acc + h0)
acc = float32(acc + h1)
acc = float32(acc + h2)
acc = float32(acc + h3)
mean = float32(acc * 0.25)
output = BF16 round-to-nearest-even(mean)
```

The captures concatenate in source encounter order `[37,38,39]` to validated `main_hidden`:

```text
shape: [1,2,15360]
dtype: torch.bfloat16
digest: 4956b1b8bce5101fd567b3f03a4db9dc7739776f7d18d54d26eadffc597e7997
```

Segment digests:

```text
37: dd1c77d0351824cf30e058453ea0b7de3fcb07d014d48add2ac45544ffcc70bb
38: 8b683f5f29e0caae1a2e1666e99f145b22722a5a4babded9cea0cdb7adf2a34c
39: b8998f15b11afc7a32234cb15cdd2095b345e1aa3ac5269893a10acfff48ef80
```

The historical Boundary12b4 diagnostic mean digests match the newly validated authority, but were not used as expected values to choose the arithmetic.

Final logits non-mutation regression remained exact:

```text
7be45bd57cf7b2763a9f91e6a070c285e5b7d02a58953f0f1b5634779584b6fd
```

## Scope limit

This validates `main_hidden` capture and concatenation only. It does not yet validate the complete `Transformer.forward` return tuple or execute sampling + `main_hidden` assembly + return as one literal integrated source-order call.

Non-claims remain: no complete `Transformer.forward` return authority, no default PyTorch stochastic RNG bitstream authority, no PyTorch/MLX RNG parity, no decode/incremental `NgramHashState` correctness, no False/image-mask DEAD crossing numerical authority, no distributed/world_size>1 correctness, no long-context qualification, no full-model correctness, and no performance/production qualification.

Next boundary: **Boundary12d: integrated bounded `Transformer.forward` return packaging**.
