# Milestone 9 KV persistence / restore / resume architecture

Status: **M9_KV_RESTORE_RESUME_QUALIFIED** for same-backend text-only M8 idle-state persistence and restore.

Primary evidence: `artifacts/m9/kv-persistence-qualification.json`.

## Persistence boundary

M9 persists only the qualified M8 idle boundary:

```text
DeepseekV41Cache[40] + exact all_tokens + provenance + bounded diagnostics
```

It is invalid to persist an active `GenerationBatch`, an unsealed/failed P6 append, or any state while ownership is ambiguous. The artifact is dormant storage and is never a second executable authority. Restore validates storage and constructs one request-local live `DeepseekV41Cache[40]`, which becomes the sole authority before M8 continuation resumes.

## State inventory

The inventory is derived from the actual oMLX `DeepseekV41Cache` class. Each of 40 layers contains seven slots:

0. scalar offset/frontier;
1. packed window KV;
2. packed compressed KV;
3. packed index K;
4. compressor pending KV;
5. compressor pending gates/scores;
6. Engram/ngram history lookback.

For each layer M9 records `class_name`, `compress_ratio`, offset, per-slot shape/dtype, and tensor name. The exact `all_tokens` list is stored in the manifest and must match the restored frontier. Publication/candidate state is not separately serialized because at the M8 idle boundary the executable continuation state is represented in the cache slots consumed by P6/P5/GenerationBatch; P6 transaction objects and diagnostic publication managers are not live authorities.

## Artifact format

Artifact directory:

```text
m9-<timestamp>-<uuid>/
  manifest.json
  cache.safetensors
  COMMITTED
```

Writes go to a sibling `.tmp` directory, write `cache.safetensors`, write and rename `manifest.json.tmp`, write `COMMITTED`, then atomically rename the directory. Restore requires all files, schema match, commit marker, tensor SHA-256 match, checkpoint fingerprint match, all expected tensors, shape/dtype match, and cache-structure validation against the loaded model config.

Default artifact root for this environment:

```text
/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv
```

## Qualification closeout

The closeout run used two separate Python processes. The save phase reached an M8 idle frontier of 45 on the official checkpoint, persisted 280 cache tensors (40 layers x 7 slots) to `/Volumes/USB-SSD-RAID-0/ds41f-mlx/kv/...`, and tore down the runtime. The resume phase loaded a fresh runtime/model, validated the artifact, restored frontier 45, verified the next real recipe turn was an exact prefix extension, appended only the new 10-token pre-terminal suffix, consumed terminal `<think>` once, decoded four tokens, and returned to idle frontier 60.

Results:

- prompt replay: 0;
- full-cache repack/reconstruction: 0;
- restored cache offsets: all 40 layers matched frontier;
- restored continuation exact-prefix-extension: true;
- tensor payload integrity: SHA-256 validated;
- tensor inventory: 280/280 tensors restored with shape/dtype validation;
- checkpoint compatibility: safetensors index fingerprint validated;
- process boundary: save and resume ran in separate Python processes;
- resume decode: ~20.36 tok/s in the canonical recorded short-context run.

## Qualification contract

The M9 runner executes two separate Python processes:

1. save phase: official checkpoint -> M8 idle boundary -> persist artifact -> teardown;
2. resume phase: fresh model/runtime process -> validate/restore artifact -> construct M8 live continuation session -> encode next recipe turn -> exact-prefix extension -> append/decode -> idle.

Production path remains:

```text
DENSE_P0_P7 -> P5 -> GenerationBatch MTP-OFF -> M8 continuation lifecycle
```

No replay/fresh-prefill fallback is permitted during restore/resume.

## Non-goals

M9 does not claim cross-runtime portability, batching, multimodal, tool execution, MTP/DSpark, speculative decode, or persistence of active scheduler/append internals.
