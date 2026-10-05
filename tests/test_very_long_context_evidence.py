"""Very-long qualification requires actual production lifecycle, not admission."""
import pytest
from tests.test_standard_off_long_session_evidence import receipts
from tools.summarize_very_long_context import qualify


def long_receipts():
    phases = receipts()
    phases[0]['fixture']['count'] = 524288
    for p in phases:
        p['runtime_sources'] = {'runtime.py': 'identity'}
        p['configured_max_seq_len'] = 1048576
        p['admission'].update(wired_limit_bytes=498216206336, previous_wired_limit_bytes=0)
        p['retired_wired_limit_bytes'] = 0
        p['resources'][0].update(system_available_bytes=100, mlx_cache_bytes=1)
        for key in ('start_memory', 'loaded_memory', 'closed_memory'):
            p[key] = dict(swap_used_bytes=0, disk_io={})
    phases[0]['prefill'].update(production_prefill_selector='DENSE_P0_P7',
        token_count=524287, frontier=524287, portable_state_exported=False,
        full_cache_repack_count=0, p7_scheduling_evidence=dict(enabled=True,
        policy='FULL_RESIDENT_BACKBONE_SSD_ENGRAM', foreground_fallback=0,
        foreground_engram_fallback=0, consume_events=512, logical_match_consumptions=512))
    phases[0]['initial_decode']['bootstrap_s'] = 9
    row = phases[0]['turns'][0]
    row['suffix_tokens'] = 2049
    row['frontier_before'] = 524288+32
    row['frontier_after'] = row['frontier_before']+2049+len(row['reports'])
    frontier = row['frontier_after']
    phases[0]['persisted_frontier'] = frontier
    phases[0]['turns'][1].update(frontier_before=frontier, frontier_after=frontier+33)
    phases[0]['final_frontier'] = frontier+33
    phases[1]['turns'][0].update(frontier_before=frontier, frontier_after=frontier+33)
    phases[1]['turns'][1].update(frontier_before=frontier+33, frontier_after=frontier+66)
    phases[1].update(persisted_frontier=frontier+33, final_frontier=frontier+66)
    return phases


def test_actual_lifecycle():
    result = qualify(long_receipts())
    assert result['decision'] == 'QUALIFIED_BOUNDED_VERY_LONG_FIRST_PARTY_OFF'
    assert result['context_tokens'] == 524288
    assert result['configured_max_seq_len'] == 1048576


@pytest.mark.parametrize('case', ['admitted_only', 'short_decode', 'runtime_changed',
                                'alternate_prefill', 'export', 'fallback', 'missing_2k', 'wrong_frontier', 'wired_retirement', 'beyond_checkpoint', 'discontinuity'])
def test_reject_incomplete_envelope(case):
    p = long_receipts()
    if case == 'admitted_only': p = p[:1]
    elif case == 'short_decode': p[0]['initial_decode']['reports'] = [{'token': 7}]
    elif case == 'runtime_changed': p[1]['runtime_sources'] = {'runtime.py': 'changed'}
    elif case == 'alternate_prefill': p[0]['prefill']['production_prefill_selector'] = 'other'
    elif case == 'export': p[0]['prefill']['portable_state_exported'] = True
    elif case == 'fallback': p[0]['prefill']['p7_scheduling_evidence']['foreground_fallback'] = 1
    elif case == 'missing_2k': p[0]['turns'][0]['suffix_tokens'] = 1; p[0]['turns'][0]['frontier_after'] = 200033
    elif case == 'wrong_frontier': p[0]['prefill']['frontier'] += 1
    elif case == 'wired_retirement': p[1]['retired_wired_limit_bytes'] = 1
    elif case == 'beyond_checkpoint': p[0]['configured_max_seq_len'] = 524288; p[1]['configured_max_seq_len'] = 524288
    elif case == 'discontinuity':
        p[1]['turns'][1]['frontier_before'] += 1; p[1]['turns'][1]['frontier_after'] += 1
    with pytest.raises(ValueError): qualify(p)


def test_decode_floor_not_inferred_from_maximum():
    p = long_receipts(); p[1]['turns'][1]['decode_tok_s'] = 14
    assert qualify(p)['decision'] == 'BLOCKED_PRACTICAL_DECODE'
