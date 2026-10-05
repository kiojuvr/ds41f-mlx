"""Receipt closure must not mistake prefill-only or incomplete restore for success."""
from copy import deepcopy
import pytest
from tools.summarize_standard_off_long_session import summarize


def receipts():
    memory=dict(mlx_peak_bytes=10, rss_bytes=8, swap_used_bytes=0)
    def turn(before, cancelled=False):
        return dict(frontier_before=before, frontier_after=before+33, suffix_tokens=1,
                    reports=[dict(token=7)]*32, decode_tok_s=18, bootstrap_s=.1,
                    cancelled=cancelled, replay=0, repack=0, same_list=True, memory=memory)
    base=dict(status='PASS', diagnostics=dict(total_prompt_replay_count=0,
              total_full_cache_repack_count=0, cache_layer_count=40, all_cache_offsets_equal_frontier=True),
              backend_active_sessions=0, wall_s=100, resources=[memory], persisted_frontier=200100,
              admission=dict(allocator_cache_limit_bytes=32*1024**3,previous_allocator_cache_limit_bytes=100*1024**3),
              retired_allocator_cache_limit_bytes=100*1024**3,
              final_frontier=200133, artifact_bytes=100, save_s=.2, probe_tokens=[7]*32)
    first=deepcopy(base)
    first.update(phase='initial', fixture=dict(count=200000),
                 p5=dict(handoff_count=1,same_list=True,exported=False,replay=0,repack=0),
                 prefill=dict(seconds=1000, tok_s=200),
                 initial_decode=dict(decode_tok_s=18, reports=[dict(token=7)]*32),
                 turns=[turn(200000,True),turn(200100)])
    second=deepcopy(base)
    second.update(phase='restore', fresh_process_restore_exact=True, restore_s=.1,
                  turns=[turn(200100),turn(200133)],persisted_frontier=200166,final_frontier=200199)
    return [first,second]


def test_complete_receipts():
    assert summarize(receipts())['decision']=='QUALIFIED_BOUNDED_200K_FIRST_PARTY_OFF'


@pytest.mark.parametrize('case', ['prefill_only','incomplete','restore','frontier','replay','repack','list','offset','cleanup','cancel','allocator','retirement','short_context','p5'])
def test_fail_closed(case):
    rows=receipts()
    if case=='prefill_only': rows=rows[:1]
    elif case=='incomplete': rows[1]['status']='RUNNING'
    elif case=='restore': rows[1]['fresh_process_restore_exact']=False
    elif case=='frontier': rows[1]['turns'][0]['frontier_before']+=1
    elif case in ('replay','repack'): rows[1]['turns'][1][case]=1
    elif case=='list': rows[1]['turns'][1]['same_list']=False
    elif case=='offset': rows[1]['diagnostics']['all_cache_offsets_equal_frontier']=False
    elif case=='cleanup': rows[1]['backend_active_sessions']=1
    elif case=='cancel': rows[0]['turns'][0]['cancelled']=False
    elif case=='allocator': rows[1]['admission']['allocator_cache_limit_bytes']=None
    elif case=='retirement': rows[1]['retired_allocator_cache_limit_bytes']=0
    elif case=='short_context': rows[0]['fixture']['count']=4096
    elif case=='p5': rows[0]['p5']['handoff_count']=2
    with pytest.raises(ValueError): summarize(rows)


def test_practical_decode_is_separate_gate():
    rows=receipts(); rows[1]['turns'][1]['decode_tok_s']=8
    assert summarize(rows)['decision']=='BLOCKED_PRACTICAL_DECODE'
