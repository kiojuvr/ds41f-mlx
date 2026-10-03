"""Bounded in-process singleton request identity; no generation authority."""
import hashlib
from copy import deepcopy


def digest(body):
    return hashlib.sha256(body).hexdigest()


def observe_retry(rec, sequence, body):
    if sequence is None:
        if rec.fence is not None:
            raise ValueError('fenced session requires request sequence')
        return None
    if sequence < 1 or body is None:
        raise ValueError('positive request sequence and body required')
    slot = rec.fence
    if slot is not None and sequence == slot['sequence']:
        if digest(body) != slot['body_sha256']:
            raise ValueError('request sequence body mismatch')
        if slot['state'] == 'active':
            raise RuntimeError('active request; observe session outcome')
        if slot['state'] != 'not_admitted':
            return dict(sequence=sequence, outcome_state=slot['state'],
                        certificate=deepcopy(rec.certificate),
                        response=deepcopy(rec.last_turn.get('response')) if rec.last_turn else None)
        return None
    if sequence != rec.consumed_sequence + 1:
        raise ValueError('expired or out-of-order request sequence')
    if slot is None and rec.request_count:
        raise ValueError('cannot switch legacy session to fenced mode')
    return None


def reserve(rec, sequence, body):
    if sequence is not None:
        rec.fence = dict(sequence=sequence, body_sha256=digest(body), state='active', started=False)


def finish(rec, unstarted=False):
    if rec.fence is None:
        return
    if unstarted:
        assert not rec.fence['started']
        rec.fence['state'] = 'not_admitted'
        return
    rec.consumed_sequence = rec.fence['sequence']
    rec.fence['state'] = ('unrecoverable' if rec.unrecoverable else 'poisoned' if rec.poisoned
                          else 'recoverable' if rec.certificate and rec.certificate['representable']
                          else 'poisoned')
