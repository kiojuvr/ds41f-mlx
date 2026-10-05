#!/usr/bin/env python3
"""Close the finite very-long OFF campaign from raw, fail-closed process receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_very_long_context import qualify


def close(root):
    groups = [('512k-wired', '512k-restored'), ('768k-initial', '768k-restored'),
              ('1m-initial', '1m-restored', '1m-ceiling-restored')]
    frontiers = []
    identities = set()
    for names in groups:
        paths = [root/(n+'.json') for n in names]
        phases = [json.loads(p.read_text()) for p in paths]
        result = qualify(phases)
        if result['decision'] != 'QUALIFIED_BOUNDED_VERY_LONG_FIRST_PARTY_OFF':
            raise ValueError('unqualified practical decode frontier')
        for phase in phases:
            for name, expected in phase['runtime_sources'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                    raise ValueError('current runtime no longer matches receipt: '+name)
            identities.add(phase['admission']['resource_set_sha256'])
        result['receipts'] = [str(p) for p in paths]
        frontiers.append(result)
    if len(identities) != 1:
        raise ValueError('admitted resource identity changed between frontiers')
    maximum = max(f['max_qualified_frontier'] for f in frontiers)
    configured = {f['configured_max_seq_len'] for f in frontiers}
    if configured != {1048576} or maximum != 1048576:
        raise ValueError('full checkpoint frontier not actually qualified')
    if not all(json.loads((root/(n+'.json')).read_text()).get('pre_policy_comparison')
               for group in groups for n in group if n != '1m-ceiling-restored'):
        raise ValueError('same-fixture retirement correctness comparison missing')
    retirement_resources=[]
    for names in groups:
        for name in names:
            if name == '1m-ceiling-restored': continue
            old=json.loads((root/'before-certificate-retirement'/(name+'.json')).read_text())
            new=json.loads((root/(name+'.json')).read_text())
            for key in ('persisted_history_sha256', 'persisted_slots', 'probe_tokens', 'probe_slots', 'final_frontier'):
                if old[key] != new[key]:
                    raise ValueError('retirement changed physical state/history: '+name+'/'+key)
            if len(old['turns']) != len(new['turns']) or any(
                [r['token'] for r in a['reports']] != [r['token'] for r in b['reports']]
                for a,b in zip(old['turns'],new['turns'])):
                raise ValueError('retirement changed decoded tokens: '+name)
            if new['phase'] == 'initial':
                if old['fixture'] != new['fixture'] or (
                    [r['token'] for r in old['initial_decode']['reports']] !=
                    [r['token'] for r in new['initial_decode']['reports']]):
                    raise ValueError('retirement changed initial fixture/decode')
                def idle_span(p):
                    values=[r['memory']['mlx_active_bytes'] for r in p['turns']]
                    return dict(first=values[0], last=values[-1], min=min(values), max=max(values))
                retirement_resources.append(dict(context=new['fixture']['count'],
                    before_idle_active_bytes=idle_span(old), after_idle_active_bytes=idle_span(new)))
    failed=json.loads((root/'512k-diagnostic.json').read_text())
    corrected=json.loads((root/'512k-wired.json').read_text())
    if (failed['fixture'] != corrected['fixture'] or
            (root/'512k-diagnostic.exit').read_text().strip() != '137'):
        raise ValueError('failed 512K fixture not actually requalified')
    return dict(schema='ds41f.standard-off.very-long-completion.v1',
        decision='QUALIFIED_BOUNDED_VERY_LONG_FIRST_PARTY_OFF_THROUGH_1M',
        baseline_commit='31a7e1d', first_party_standard_off_512k_qualified=True,
        maximum_qualified_frontier=maximum, maximum_initial_context=1040090,
        practical_supported_context_ceiling=maximum, configured_max_seq_len=1048576,
        ceiling_basis='actual prefill/decode/continuation/restart within admitted checkpoint context range; not physical exhaustion',
        physical_memory_exhaustion_reached=False, resource_set_sha256=identities.pop(),
        process_count=sum(f['process_count'] for f in frontiers),
        fresh_process_restores=sum(f['fresh_process_restores'] for f in frontiers),
        continued_turns=sum(f['continued_turns'] for f in frontiers),
        decoded_tokens=sum(f['decoded_tokens'] for f in frontiers),
        cancelled_turns=sum(f['cancelled_turns'] for f in frontiers),
        corrected_wall_s=sum(f['wall_s'] for f in frontiers),
        zero_replay_repack=True, exact_restart_tokens_and_all_280_slots=True,
        production_defects=['zero wired budget outside active generation exposed prefill compression/jetsam at 512K',
                            'retired P6 certificate/setup backlink cycle retained old publications until cyclic GC'],
        production_fixes=['existing admitted resource owner holds recommended wired residency through model lifetime; synchronizes/restores on retirement',
                          'P5 transfer/burn retires only the passive admission backlink, preserving live cache and all revocation guards'],
        exact_before_after_retirement_comparison=True,
        retired_publication_resource_comparison=retirement_resources,
        failed_fixture_requalified=True, failed_fixture_sha256=failed['fixture']['sha256'],
        evidence_tools_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('summarize_very_long_context.py'),
             Path(__file__).with_name('qualify_standard_off_long_session.py'),
             Path(__file__).with_name('summarize_standard_off_long_session.py'),
             Path(__file__).with_name('requalify_very_long_context.sh')]},
        frontiers=frontiers,
        remaining_constraints=['finite single-flight maintenance/document text core workload, not arbitrary prompt quality or unlimited lifetime',
            'above-checkpoint extrapolation and absolute physical-memory context ceiling not established',
            'configured maximum alone is not a tested hard overlength request guard; reserve prompt/output/suffix capacity',
            '13-32 minute initial ingestion plus fresh admission cost; 2K suffix still seconds',
            'warm selected-row SSD behavior, not cold-all-pages/high-diversity worst-case latency',
            'no new long HTTP/client/tool recovery, concurrency, crash durability, active or cross-backend persistence',
            'no Vision, MTP/DSpark/speculation, full R1, release/clean-room or runtime promotion'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/very-long-context'))
    a=p.parse_args()
    result=close(a.root)
    (a.root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
