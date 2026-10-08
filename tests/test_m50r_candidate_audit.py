"""Audit tooling tests only; no model/topology/state qualification claim."""
import hashlib
import json
import subprocess
import sys

import pytest

from tools.freeze_m50r_candidate import dispersion
from tools.report_m50r_candidate import summarize
from tools.probe_m50r_prefix_oracle import prefix_indices, ring_append_reference
from tools.report_m50r_movement import interval_union_ns, table
from tools.qualify_m50r_closure import qualified_prefix


def test_absolute_prefix_and_wrap_reference():
    import numpy as np
    assert prefix_indices(124, 8, 129, 126) == [2, 3, 4]
    capacity = 8
    # Independent fixture: slots contain absolute integer positions. Before 10,
    # physical positions 0,1 hold 8,9, and new 10..13 overwrite slots 2..5.
    old = np.array([8,9,2,3,4,5,6,7]).reshape(1,1,8,1)
    new = np.arange(10,14).reshape(1,1,4,1)
    result = ring_append_reference(old,10,new,capacity)
    assert result.reshape(-1).tolist() == [8,9,10,11,12,13,6,7]
    assert result.tobytes() != old.tobytes()  # equal shape/offset metadata isn't contents


def test_device_union_is_not_sum_of_overlapping_intervals():
    assert interval_union_ns([(3,8),(1,5),(8,9),(12,13)]) == 9


def test_xctrace_references_and_unknown_process(tmp_path):
    path = tmp_path / 'table.xml'
    path.write_text('''<root><schema><col><mnemonic>start</mnemonic></col>
        <col><mnemonic>process</mnemonic></col></schema>
        <row><start-time id="1">3</start-time><process id="2" fmt="Python">
        <pid id="3">44</pid></process></row>
        <row><start-time ref="1"/><process ref="2"/></row>
        <row><start-time ref="1"/><sentinel/></row></root>''')
    rows = table(path)
    assert rows[0] == rows[1] == dict(start=3,process=dict(pid=44,name='Python'))
    assert rows[2]['process'] is None


def test_real_prefix_receipt_and_negative_controls():
    from pathlib import Path
    from copy import deepcopy
    path = Path('artifacts/m50r/prefix-oracle.json')
    if not path.exists(): pytest.skip('local/committed real evidence unavailable')
    receipt = json.loads(path.read_text())
    assert qualified_prefix(receipt)
    for field in ('state_content_exact','tap_content_exact','logit_content_exact'):
        bad = deepcopy(receipt); bad['cases'][0][field] = False
        assert not qualified_prefix(bad)
    bad = deepcopy(receipt); bad['cases'][0]['index_publications_exact']['2.indices'] = False
    assert not qualified_prefix(bad)
    bad = deepcopy(receipt); bad['cases'][0]['ring_content_exact'][2] = False
    assert not qualified_prefix(bad)
    bad = deepcopy(receipt); bad['cases'][0]['oracle_ids'] = bad['cases'][0]['input_ids']
    assert not qualified_prefix(bad)
    bad = deepcopy(receipt); bad['live_cases'][0]['owner_retired_before_oracle'] = False
    assert not qualified_prefix(bad)


def test_dispersion_uses_sample_standard_deviation():
    stats = dispersion([9., 10., 11.])
    assert stats == dict(n=3, mean=10., median=10., stdev=1., min=9., max=11., cv=.1)


def test_too_few_trials_reject_before_environment_or_model_import():
    result = subprocess.run([sys.executable, '-m', 'tools.freeze_m50r_candidate',
                             '--mode', 'performance', '--output', '/unused', '--repeats', '2'],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'at least three fresh measured sessions required' in result.stderr


def test_full_rate_receipts_cannot_authorize_without_state_oracle(tmp_path):
    # Synthetic serializer/gate fixture, NOT model evidence. Even perfect
    # scalar observations cannot qualify independent physical state/dispatch.
    source = b'fixture only\n'
    row = dict(sample=0, warmup=False, retired=True, request_sha256='request',
               decode_tok_s=10., settled_decode_tok_s=9., http_s=5.,
               request_tok_s=8., ttft_content_s=1.,
               topology={}, causal_events=[], cancel_requested=False,
               trace=dict(canonical_generated=[1], canonical_frontier=3,
                          target_offsets=[3]*40, dspark_offsets=[3]*3,
                          queue_empty=True, prediction_retired=True, prompt_replay=0,
                          full_cache_repack=0, cancelled=False, terminal_matches=[],
                          prefill_handoff_s=1., cleanup_s=.1, decode_s=.1, elapsed_s=2.,
                          mtp_stats=dict(backbone_ms=80., mtp_head_ms=5., sample_ms=1., cache_ops_ms=1.),
                          quiescence=dict(counters=dict(new_verify_cycles=0, new_proposals=0))))
    weights = [dict(name=str(i), bytes=1, sha256='digest') for i in range(48)]
    common = dict(head='head', identity=dict(identity_sha256='identity', checkpoint=dict(shards=[
        dict(name=w['name'], bytes=1, lfs_sha256='digest') for w in weights])),
        tool_sha256=hashlib.sha256(source).hexdigest())
    identity = dict(common, mode='identity', status='IDENTITY_CAPTURED', checkpoint_bytes=weights)
    performance = dict(common, mode='performance', status='CAPTURED_NOT_GATE_PASS', token_identity=True,
                       model_config={}, server=dict(protocol='LocalH11Protocol'),
                       rows=[dict(row, warmup=True), row, row, row])
    trace = dict(common, mode='trace', status='CAPTURED_NOT_GATE_PASS',
                 server=performance['server'], rows=[row])
    for name, receipt in [('identity', identity), ('performance', performance), ('trace', trace)]:
        (tmp_path / f'{name}.json').write_text(json.dumps(receipt))
        (tmp_path / f'{name}.tool.py').write_bytes(source)
    summary = summarize(tmp_path)
    assert summary['decision'] == 'BLOCK'
    assert summary['m51r_authorized'] is False
    assert any('prefix-state' in reason for reason in summary['blockers'])
    # Identity drift must fail before any rates can be used.
    trace['identity'] = dict(trace['identity'], identity_sha256='foreign')
    (tmp_path / 'trace.json').write_text(json.dumps(trace))
    with pytest.raises(AssertionError):
        summarize(tmp_path)
