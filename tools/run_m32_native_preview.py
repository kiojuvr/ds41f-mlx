"""Reproduce M31 source preview rows using the actual installed Python binding."""
import hashlib
import json
from pathlib import Path
import sys
import time
import deepseek_recipe as d
import deepseek_recipe._native as native

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'artifacts/m31/source-preview-mapping.json').read_text())
fixtures = {c['name']: c for c in json.loads((ROOT / 'artifacts/m31/parity-fixtures.json').read_text())}
t = d.Tokenizer.from_file(sys.argv[1])
rows, timings = [], []
for case in source['cases']:
    fixture = fixtures[case['name']]
    options = d.ParsingOptions()
    options.stop_sequences = fixture['stops']
    options.parse_json_output = fixture.get('json', False)
    if fixture.get('reasoning'):
        options.reasoning_initial_stage = d.ReasoningStage.Reasoning
    p = d.StreamProcessor(d.ChatCompletionChunkGenerator('preview', 'v41', True, fixture.get('reasoning', False)), options, t)
    for row in case['rows']:
        before = p.semantic_snapshot()
        assert before[1] == row['pending_canonical_ids_before']
        assert before[2] == len(before[1])
        assert before[3] == row['source_bytes_before']
        def fields(r):
            return dict(safe_token_count=r.safe_token_count, completing_token_index=r.completing_token_index,
                        mapping_exact=r.mapping_exact, candidate_ids_processed=r.candidate_ids_processed,
                        max_pending_ids=r.max_pending_ids,
                        terminal=None if r.terminal_kind is None else dict(kind=r.terminal_kind, source_start=r.source_start, source_end=r.source_end))
        r = p.preview_tokens(row['candidate_ids'])
        actual = fields(r)
        assert actual == {key: row[key] for key in actual}, (case['name'], actual, row)
        for _ in range(200):
            start = time.perf_counter_ns()
            repeat = p.preview_tokens(row['candidate_ids'])
            timings.append(time.perf_counter_ns() - start)
            assert fields(repeat) == actual
            assert p.semantic_snapshot() == before
            assert p.semantic_terminal is None
        # Advance only the canonically emitted prefix, never rejected tails.
        used = r.completing_token_index + 1 if r.terminal_kind else len(row['candidate_ids'])
        for token in row['candidate_ids'][:used]:
            p.push(d.InferenceChunk.token(token))
        after = p.semantic_snapshot()
        assert after[1] == row['pending_canonical_ids_after']
        assert after[2] == len(after[1])
        assert after[3] == row['source_bytes_after']
        obs = p.semantic_terminal
        canonical = None if obs is None else dict(kind=obs.kind, source_start=obs.source_start, source_end=obs.source_end)
        assert canonical == row['canonical_terminal']
        if obs is not None and not p.finished:
            closed = p.preview_tokens([token])
            assert not closed.mapping_exact and closed.safe_token_count == 0
        rows.append(dict(case=case['name'], candidate_ids=row['candidate_ids'], before=before, after=after, preview=actual, canonical_terminal=canonical))
    p.close()
sorted_times = sorted(timings)
result = dict(scope='Real native Python binding, synthetic M31 corpus, NOT live model cycles',
              native_path=native.__file__, native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              source_agreement=True, nonmutation=True, deterministic=True, row_count=len(rows),
              calls_per_diagnostic_row=201, rows=rows,
              preview_latency=dict(samples=len(timings), median_ns=sorted_times[len(timings)//2], p95_ns=sorted_times[int(len(timings)*.95)], max_ns=max(timings)),
              clone_latency=None, clone_latency_status='not separately instrumented in Python binding', live_calls_per_cycle=None)
(ROOT / 'artifacts/m32/native-preview.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
