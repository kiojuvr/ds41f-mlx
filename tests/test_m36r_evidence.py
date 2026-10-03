"""M36R evidence and controlled negative admission gates."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from ds41f_mlx.serving.recovery_certificate import LocalToolLedger
from test_m35_transport_lease import FixtureBackend, request
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/m36r'

def load(name):return json.loads((OUT/name).read_text())


def test_recipe_matrix_semantics_not_json_or_parser_state_labels():
    rows={r['name']:r for r in load('recipe-matrix.json')['rows']}
    positive=['text','unicode_full','length','eos','marker_prefix','reasoning_transition_exact','complete_tool','complete_two_tools','complete_auto_tool']
    assert len(rows)==28
    for name,row in rows.items():
        cert=row['certificate']
        assert cert['representable']==(name in positive)
        if cert['representable']:assert cert['exact_prefix'] and cert['semantic_complete']
        if 'witness' in cert:
            from ds41f_mlx.serving.server import load_v41_tokenizer, prepare_request
            ids=prepare_request('chat_completions',json.dumps(cert['witness']).encode(),tokenizer=load_v41_tokenizer(Path('/tmp/ds41f-m32-recipe'))).token_ids
            assert cert['exact_prefix']==(len(ids)>len(row['canonical']) and ids[:len(row['canonical'])]==row['canonical'])
        if name.startswith('valid_json_canonical_prefix_'):
            assert cert['exact_prefix'] and not cert['semantic_complete'] and not cert['executable_tools']
            json.loads(row['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    assert rows['complete_tool_alternate_spelling']['response']['choices'][0]['finish_reason']=='tool_calls'
    assert not rows['complete_tool_alternate_spelling']['certificate']['exact_prefix']


@pytest.mark.parametrize('row',load('recipe-matrix.json')['rows'],ids=lambda r:r['name'])
def test_negative_official_reconstruction_admission_is_atomic(row):
    if row['certificate']['representable']:return
    async def run():
        b=FixtureBackend();rec=await b.create_stateful_session(session_id='negative')
        rec.canonical=row['canonical'];rec.certificate=row['certificate'];rec.poisoned=rec.unrecoverable=True
        before=json.dumps(rec.to_json(),sort_keys=True)
        req=request();req.token_ids=rec.canonical+[1]
        with pytest.raises(RuntimeError,match='DELETE required'):
            await b.qualification_response('negative',req,tokenizer=None)
        assert json.dumps(rec.to_json(),sort_keys=True)==before and not b._lock.locked() and not b.session_traces
        ledger=LocalToolLedger()
        with pytest.raises(ValueError):ledger.execute('negative',1,dict(sequence=1,outcome_state='unrecoverable',certificate=row['certificate'],response=row['response']),lambda _:pytest.fail('tool executed'))
        b.close()
    asyncio.run(run())


def test_real_http_recovery_fence_tools_and_zero_replay():
    p=load('http.json');assert p['status']=='PASS'
    assert p['counts']['replay']==p['counts']['repack']==0
    by={c['name']:c for c in p['cases']}
    for name in ['unicode','partial-tool','completed-tool']:
        c=by[name];assert c['duplicate_nonmutating'] and c['body_mismatch_atomic']
        assert c['recovered']['request_fence']['sequence']==1
    assert by['partial-tool']['recovered']['outcome_state']=='unrecoverable'
    assert by['partial-tool']['tool_executions']==0 and by['partial-tool']['continuation_rejected_atomic']
    for name in ['unicode','completed-tool']:
        c=by[name];assert c['continuation_exact_prefix'] and c['expired_retry_atomic']
        for rec in [c['recovered'],c['after']]:
            t=rec['last_turn'];assert t['prompt_replay']==t['full_cache_repack']==0
            assert set(t['target_offsets']+t['dspark_offsets'])=={rec['canonical_frontier']}
    assert by['completed-tool']['tool_executions']==1 and by['completed-tool']['execution_reobservations']==4
    assert by['active']['active_retry_no_generation']


def test_final_qualification_current_sources_and_evidence():
    p=load('qualification.json');assert p['decision']=='QUALIFIED_BOUNDED_CERTIFIED_RECOVERY'
    for key in ['source_hashes','evidence_hashes']:
        for path,digest in p[key].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    assert p['gates']['m35_http_turns']==18 and p['gates']['native_interruption_cases']==16
