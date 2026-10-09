"""One M51R risk: publication failure after an actual native verify mutation.
Reuses the frozen public request/server harness; this is NOT a rate run.
Installed admission is never modified. No fault while a native region is live.
"""
import hashlib
import json
from pathlib import Path
import sys

from ds41f_mlx import mtp_identity as identity
identity.RECORD = Path('artifacts/m51r/qualification-identity.json')
from ds41f_mlx.runtime.mtp_lifecycle import OMLXMTPGenerationSession
from ds41f_mlx.serving.internal_mtp import InternalMTPQualificationBackend
from tools.freeze_m50r_candidate import main

out = {'scope': 'post-verify connection publication failure; not rates or M52R',
       'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
original_bind = OMLXMTPGenerationSession._bind_connection
original_close = OMLXMTPGenerationSession.close
original_retire = InternalMTPQualificationBackend._retire

def bind(self):
    original_bind(self)
    state = getattr(self._bg._generation_batch, '_omlx_mtp_state', None)
    if state is not None and state.stats.cycles >= 1 and 'injected' not in out:
        out['injected'] = dict(cycles=state.stats.cycles,
            consumed_frontier=len(self.connection.consumed_tokens),
            emitted_frontier=self.history.canonical_frontier,
            queue_ahead=list(self.connection.queue_ahead),
            pending_prediction=self.connection.pending_prediction)
        raise RuntimeError('M51R injected publication failure after native verify')

def close(self):
    original_close(self)
    if 'injected' in out:
        out['burn'] = dict(disposition=self.connection.disposition,
            failed=self._operation_failed, closed=self._closed,
            bg_retired=self._bg is None, rings_retired=self.dspark_context is None,
            model_alias_retired=self.model is None and self.language_model is None,
            observation_retired=self.bound_observation is None, uid_retired=self.uid is None)

def retire(self, rec):
    original_retire(self, rec)
    if 'injected' in out:
        out['retirement'] = dict(poisoned=rec.poisoned,
            owner_retired=rec.owner is None, cache_retired=rec.cache is None,
            rings_retired=rec.rings is None, processor_retired=rec.processor is None)

OMLXMTPGenerationSession._bind_connection = bind
OMLXMTPGenerationSession.close = close
InternalMTPQualificationBackend._retire = retire
sys.argv = ['freeze_m50r_candidate', '--mode', 'performance', '--repeats', '3',
            '--output', 'artifacts/m51r/fault-harness.json']
try:
    main()
except BaseException as exc:
    out['harness_exit'] = repr(exc)
harness = Path('artifacts/m51r/fault-harness.json')
receipt = json.loads(harness.read_text())
receipt['instrumentation'] = 'M51R post-verify fault injection and retirement observer; NOT rate evidence'
receipt['expected_fault'] = True
harness.write_text(json.dumps(receipt, indent=2) + '\n')
assert out.get('injected', {}).get('cycles', 0) >= 1, out
assert out['burn']['disposition'] == 'fault' and all(v for k,v in out['burn'].items() if k != 'disposition'), out
assert all(out['retirement'].values()), out
out['status'] = 'PASS'
Path('artifacts/m51r/fault-boundary.json').write_text(json.dumps(out, indent=2) + '\n')
