"""Supported one-thread/living-client facade for mtp-singleton-v1.

SSE is display-only. Explicit reconcile, effects, retirement and fresh creation;
no automatic replacements, effect retries, persistence or crash-safe ledger.
"""
import hashlib
import json
import re
from urllib.parse import urlsplit
from .internal_local_client import InternalLocalClient, OwnedStream, ClientStateError
from .web_client import RuntimeClient, RuntimeHTTPError
from .mtp_profile import PROFILE, LIMITS, validate_chat, validate_completed_calls


class LocalMTPClient(InternalLocalClient):
    def __init__(self, base_url='http://127.0.0.1:8000', *, ledger_limit=128):
        url = urlsplit(base_url)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path or url.query or url.fragment or
                url.username is not None or url.password is not None):
            raise ValueError('literal loopback HTTP authority required')
        runtime = RuntimeClient(base_url)
        health = runtime.health()
        identity = health.get('dependency_identity')
        if (health.get('profile') != PROFILE or json.dumps(health.get('limits'),sort_keys=True) != json.dumps(LIMITS,sort_keys=True) or
                not isinstance(identity,str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity)):
            raise ClientStateError('runtime does not implement the supported local MTP profile identity')
        super().__init__(runtime, ledger_limit=ledger_limit)
        self.dependency_identity = identity

    def create(self, session_id=None):
        if session_id is not None:
            raise ClientStateError('server-issued IDs only')
        return super().create()

    def _valid_created_record(self, rec):
        sid = rec.get('id')
        return (super()._valid_created_record(rec) and rec.get('profile') == PROFILE and
                re.fullmatch(r'mtp_[0-9a-f]{32}_[0-9a-f]{32}', sid) is not None and int(sid[-32:],16) > 0 and
                type(rec.get('next_sequence')) is int and rec.get('state') == 'empty' and
                type(rec.get('canonical_frontier')) is int and rec['canonical_frontier'] == 0 and
                type(rec.get('request_count')) is int and rec['request_count'] == 0 and
                rec.get('request_fence') is None and rec.get('certificate') is None)

    def _send(self, *, outcome_projection=False):
        if type(self._pending.sequence) is not int or not 1 <= self._pending.sequence <= (1 << 64)-1:
            self._stop('request sequence exhausted; explicit retire/fresh decision required')
        validate_chat(self._pending.body)
        try:
            return super()._send(outcome_projection=outcome_projection)
        except RuntimeHTTPError as exc:
            try:
                code = json.loads(exc.body)['error']['code']
            except (ValueError, KeyError, TypeError):
                code = None
            if code in ('request_body_mismatch','request_expired','request_sequence_gap',
                        'retained_envelope_changed','certified_history_changed','canonical_prefix_mismatch'):
                self._stop('profile conflict: '+code+'; explicit retire/fresh decision required')
            raise

    def _accept(self, out):
        if out.get('profile') != PROFILE or out.get('body_sha256') != hashlib.sha256(self._pending.body).hexdigest():
            self._stop('public outcome profile/body identity mismatch')
        return super()._accept(out)

    def _validate_message_binding(self, cert, request, msg):
        validate_completed_calls(msg)
        if type(cert.get('frontier')) is not int or not 0 <= cert['frontier'] <= LIMITS['context_tokens']:
            raise ValueError('certificate exceeds bounded canonical frontier')
        binding = dict(messages=request['messages'], assistant=msg)
        digest = hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if (cert.get('projection') != 'ds41f.mtp.certificate.v1' or cert.get('ordinary_binding_sha256') != digest or
                cert.get('profile_supported') is not True):
            raise ValueError('public certificate ordinary binding mismatch')


__all__ = ['LocalMTPClient', 'OwnedStream', 'ClientStateError']
