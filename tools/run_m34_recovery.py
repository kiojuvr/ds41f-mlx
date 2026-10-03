"""Real ordinary recipe re-entry after midstream cancellation, no raw-ID splice.
Uses the soak's unchanged P6/P5 ownership and canonical-prefix admission checks.
"""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'tools/run_m34_operational_soak.py'
code=source.read_text()
changes={
    "OUT = ROOT / 'artifacts/m34/soak.json'":"OUT = ROOT / 'artifacts/m34/recovery.json'",
    'for fresh in range(3):':'for fresh in range(1):',
    "memory('model_loaded')":"""memory('model_loaded')
    import tempfile
    from ds41f_mlx.runtime.kv_persistence import save_m8_idle_state, restore_m8_idle_state, M9PersistenceError
    denied=[]
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)/'must-not-exist'
        for name in ('save','restore'):
            try:
                if name=='save':save_m8_idle_state(artifact_root=root,model=model,live_cache=[],all_tokens=[1],checkpoint=root)
                else:restore_m8_idle_state(artifact_path=root,model=model,checkpoint=root)
                raise AssertionError('MTP persistence unexpectedly admitted')
            except M9PersistenceError as exc:
                assert 'MTP-OFF' in str(exc) and not root.exists()
                denied.append(dict(operation=name,fail_closed_before_io=True,error=str(exc)))
    result['unsupported_live']=denied
    save()""",
    'for turn in range(30):':'for turn in range(12):',
    'cancel_at=193 if turn==29 else None':'cancel_at=193 if turn==8 else None',
    'if turn==29:break':'if turn==11:break',
    "assert choice['finish_reason']=='stop', 'long response must close before ordinary user re-entry'":"assert choice['finish_reason']=='stop' or rec['cancelled'], 'only explicit cancellation permits partial assistant re-entry'",
    "harness_sha256=sha(__file__), turns=[]":"harness_sha256=sha(__file__), recovery_driver_sha256=sha(ROOT/'tools/run_m34_recovery.py'), turns=[]",
}
for old,new in changes.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'),dict(__file__=str(source),__name__='__main__'))
