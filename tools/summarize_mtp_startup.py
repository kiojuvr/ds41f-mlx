"""Summarize retained startup controls and uninstrumented R1 workflow receipts."""
import argparse
import json
import gzip
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    controls = json.loads((a.directory/'controls-production.json').read_text())
    summary = dict(schema='ds41f.mtp.startup-summary.v1', controls=[], workflows=[], phase_paths={})
    for filename in ('phases-detail', 'phases-joint-split', 'phases-after', 'phases-first-proposal'):
        path = a.directory / (filename+'.json')
        text = path.read_text() if path.exists() else gzip.open(path.with_suffix('.json.gz'), 'rt').read()
        rows = json.loads(text)['rows']
        paths = []
        for start in [r for r in rows if r['name'].endswith('._start') and r.get('prefix', 0) > 100]:
            inside = [r for r in rows if start['start'] <= r['start'] < start['start']+start['wall_s']]
            def duration(name):
                return next((r['wall_s'] for r in inside if r['name'].endswith(name)), None)
            item = dict(prefix_positions=start['prefix']-1, committed=start['committed'],
                        startup_s=start['wall_s'], prefix_enqueue_s=duration('DeferredPrefillAppend.execute_all'),
                        hidden_eval_s=duration('diagnostic.target_hidden_eval'),
                        constructor_s=duration('OMLXMTPGenerationSession.__post_init__'),
                        terminal_s=duration('OMLXMTPGenerationSession.start'),
                        root_eval_s=sum(r['wall_s'] for r in inside if r['name']=='mlx.core.eval' and
                                        r['parent']==['InternalMTPQualificationBackend._start']),
                        wired_calls=[r for r in inside if r['name']=='mlx.core.set_wired_limit'])
            following = [r for r in rows if r['name'].endswith('._next') and
                         r['start'] >= start['start']+start['wall_s']]
            if following:
                first = min(following, key=lambda r:r['start'])
                drafts = [r for r in rows if r['name'].endswith('._dspark_next_drafts') and
                          first['start'] <= r['start'] < first['start']+first['wall_s']]
                item['first_next_s'] = first['wall_s']
                item['first_proposal_entry_s'] = None if not drafts else drafts[0]['start']-first['start']
            paths.append(item)
        summary['phase_paths'][filename] = paths
    for length in sorted({r['length'] for r in controls['rows']}):
        rows = [r for r in controls['rows'] if r['length'] == length]
        entry = dict(length=length, token_equivalence=len({tuple(r['tokens']) for r in rows}) == 1,
                     target_equivalence=len({r['target_digest'] for r in rows}) == 1,
                     ring_equivalence=len({r['rings_digest'] for r in rows}) == 1,
                     frontier_equivalence=len({r['frontier'] for r in rows}) == 1, modes={})
        for mode in ('baseline', 'joint', 'early-wire'):
            selected = [r for r in rows if r['mode'] == mode and not
                        (length == 64 and mode == 'baseline' and r['repeat'] == 0)]
            entry['modes'][mode] = {k:mean(r[k] for r in selected) for k in
                ('startup_s', 'prefix_enqueue_s', 'context_ready_s', 'constructor_s', 'terminal_s', 'decode_s')}
        summary['controls'].append(entry)
    before = json.loads(Path('artifacts/mtp-verification/r1-mtp-gates/mtp-model.json').read_text())
    after = json.loads((a.directory/'r1-mtp-final-gates/mtp-model.json').read_text())
    assert controls['status'] == before['status'] == after['status'] == 'PASS'
    def choices_without_issued_ids(outcome):
        choices = json.loads(json.dumps(outcome['response']['choices']))
        for choice in choices:
            for call in choice.get('message', {}).get('tool_calls', []):
                call.pop('id', None)
        return choices
    for b, c in zip(before['cases'], after['cases']):
        assert b['name'] == c['name']
        old, new = b['outcome'], c['outcome']
        if old['outcome_state'] != 'recoverable' or new['outcome_state'] != 'recoverable':
            continue
        bm, cm = old['metrics'], new['metrics']
        summary['workflows'].append(dict(name=b['name'], before=bm, after=cm,
            message_equivalence_except_request_issued_tool_ids=choices_without_issued_ids(old) == choices_without_issued_ids(new),
            acceptance_equivalence=all(bm[k] == cm[k] for k in ('generated','considered_drafts','accepted_drafts')),
            lifecycle_equivalence=all(bm[k] == cm[k] for k in ('prompt_replay','full_cache_repack','aligned_idle','settlement'))))
    (a.directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

if __name__ == '__main__':
    main()
