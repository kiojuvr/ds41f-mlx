"""One bounded exact-byte response reservation on the standard session record.

Not a second transcript/effect ledger. Replay transfers already serialized
bytes; no recipe/target/tool operation runs. Delivery is not acknowledgement.
"""
from dataclasses import dataclass, field
from hashlib import sha256

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_CHUNKS = 8192


@dataclass
class ResponseReservation:
    sequence: int
    request_bytes: bytes
    media_type: str
    state: str = 'active'
    request_id: str | None = None
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0

    def append(self, data):
        if self.state != 'active':
            raise RuntimeError('response reservation is frozen')
        data = data.encode() if isinstance(data, str) else bytes(data)
        if self.size + len(data) > MAX_RESPONSE_BYTES or len(self.chunks) >= MAX_RESPONSE_CHUNKS:
            self.state = 'uncertain'
            raise RuntimeError('response reservation byte ceiling; no automatic retry')
        self.chunks.append(data)
        self.size += len(data)
        return data

    def complete(self):
        if self.state != 'active':
            raise RuntimeError('response reservation cannot complete')
        self.state = 'completed'

    def burn(self):
        self.state = 'uncertain'

    def retire(self):
        self.state = 'retired'
        self.request_bytes = b''
        self.chunks.clear()
        self.size = 0

    def certificate(self):
        # Hashing releases the GIL. Snapshot before hashing so a concurrent
        # worker append cannot pair a prefix hash with a later completed size.
        state, request_bytes, chunks = self.state, self.request_bytes, tuple(self.chunks)
        return dict(sequence=self.sequence, outcome_state=state,
                    request_sha256=sha256(request_bytes).hexdigest(),
                    response_sha256=sha256(b''.join(chunks)).hexdigest(),
                    response_bytes=sum(map(len, chunks)), request_id=self.request_id)


def observe_retry(rec, sequence, body):
    if sequence is None:
        if rec.response_reservation is not None:
            raise ValueError('fenced session requires request sequence')
        return None
    if not isinstance(sequence, int) or sequence < 1 or len(body) > MAX_REQUEST_BYTES:
        raise ValueError('positive request sequence and bounded exact body required')
    slot = rec.response_reservation
    if slot is not None and sequence == slot.sequence:
        if bytes(body) != slot.request_bytes:
            raise ValueError('request sequence exact-byte mismatch')
        if slot.state != 'completed':
            raise RuntimeError('active or uncertain reservation; observe session, never regenerate')
        return slot
    expected = 1 if slot is None else slot.sequence + 1
    if sequence != expected:
        raise ValueError('expired or out-of-order request sequence')
    if slot is not None and slot.state != 'completed':
        raise RuntimeError('prior response reservation is not coherent')
    if slot is None and rec.request_count:
        raise ValueError('cannot switch legacy session to fenced mode')
    return None


def reserve(rec, sequence, body, media_type):
    if sequence is None:
        return None
    if observe_retry(rec, sequence, body) is not None:
        raise RuntimeError('completed reservation must replay, not reserve again')
    if rec.busy or rec.recovery_state == 'unrecoverable':
        raise RuntimeError('session does not admit response reservation')
    slot = ResponseReservation(sequence, bytes(body), media_type)
    rec.response_reservation = slot
    return slot
