"""Bounded application-owned acquisition receipts, never runtime/history authority.

A random capability identifies an immutable transfer (including original URLs).
SQLite transactions publish bytes and metadata together. No URL cache/retry,
runtime model state lives here. Page-scoped parser material and explicitly
non-authoritative model observations share this bounded retention. Browser history owns
observed image bytes independently of this seven-day staging retention.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path

from ds41f_mlx.web_tools import AcquiredResource, ToolError

ARTIFACT_BYTES = 256 * 1024 * 1024
ARTIFACT_COUNT = 512
ARTIFACT_LIFETIME_SECONDS = 7 * 24 * 3600
# JSON-escaped source/final URLs (8192 chars each), bounded MIME metadata,
# digest and flags. Reserve before transfer, then charge exact stored bytes.
ARTIFACT_METADATA_RESERVATION = 256 * 1024


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: str
    resource: AcquiredResource
    expires_at: float

    def metadata(self):
        r = self.resource
        return dict(artifact_id=self.id, sha256=r.sha256, url=r.url,
                    source_url=r.source_url, content_type=r.content_type,
                    acquired_bytes=len(r.data), network_truncated=r.truncated,
                    redirects=r.redirects, expires_at=self.expires_at)


class ArtifactStore:
    def __init__(self, path, *, max_bytes=ARTIFACT_BYTES,
                 max_count=ARTIFACT_COUNT, lifetime=ARTIFACT_LIFETIME_SECONDS):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_bytes, self.max_count, self.lifetime = max_bytes, max_count, lifetime
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS artifacts '
                       '(id TEXT PRIMARY KEY, kind TEXT NOT NULL, metadata TEXT NOT NULL, '
                       'data BLOB NOT NULL, expires REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS interpretations '
                       '(artifact_id TEXT NOT NULL, key TEXT NOT NULL, result TEXT NOT NULL, '
                       'PRIMARY KEY(artifact_id,key), FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE)')
        self.path.chmod(0o600)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute('PRAGMA secure_delete=ON')
            db.execute('PRAGMA foreign_keys=ON')
            with db:
                yield db
        finally:
            db.close()

    def _check_capacity(self, db, size):
        db.execute('DELETE FROM artifacts WHERE expires <= ?', (time.time(),))
        count, used = self._usage(db)
        if count >= self.max_count or used + size > self.max_bytes:
            raise ToolError('artifact storage ceiling reached; clear staged artifacts explicitly, no automatic download retry', code='artifact_resource_ceiling')

    @staticmethod
    def _metadata(resource):
        return json.dumps(dict(source_url=resource.source_url, url=resource.url,
                               content_type=resource.content_type, truncated=resource.truncated,
                               redirects=resource.redirects, sha256=resource.sha256))

    def _publish(self, db, resource, kind):
        if kind not in {'text', 'image', 'pdf'}:
            raise ValueError('invalid artifact kind')
        identity = secrets.token_hex(32)
        expires = time.time() + self.lifetime
        db.execute('INSERT INTO artifacts VALUES (?,?,?,?,?)',
                   (identity, kind, self._metadata(resource), resource.data, expires))
        return Artifact(identity, kind, resource, expires)

    def put(self, resource, kind):
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._check_capacity(db, len(resource.data) + len(self._metadata(resource).encode('utf-8')))
            artifact = self._publish(db, resource, kind)
        return artifact

    def put_document(self, identity, resource):
        """Client-known upload receipt permits GET reconciliation, not POST retry."""
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ToolError('upload identity must be 64 lowercase hex characters')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT metadata FROM artifacts WHERE id=?', (identity,)).fetchone()
            if row:
                raise ToolError('upload identity already exists; observe via GET, do not repeat upload')
            self._check_capacity(db, len(resource.data) + len(self._metadata(resource).encode('utf-8')))
            expires = time.time() + self.lifetime
            db.execute('INSERT INTO artifacts VALUES (?,?,?,?,?)',
                       (identity, 'pdf', self._metadata(resource), resource.data, expires))
        return Artifact(identity, 'pdf', resource, expires)

    def acquire(self, kind, max_bytes, fetch):
        # Reserve worst-case transfer storage BEFORE an external GET. Holding
        # the write transaction serializes staging acquisitions across processes;
        # reads remain possible. On crash the effect is uncertain, never retried.
        completed = False
        try:
            with self._connect() as db:
                db.execute('BEGIN IMMEDIATE')
                self._check_capacity(db, max_bytes + ARTIFACT_METADATA_RESERVATION)
                resource = fetch()
                completed = True
                if len(resource.data) > max_bytes:
                    raise ValueError('acquisition exceeded its reserved resource boundary')
                artifact = self._publish(db, resource, kind)
            return artifact
        except (sqlite3.Error, OSError) as exc:
            code = 'uncertain_external_effect_outcome' if completed else 'artifact_resource_ceiling'
            raise ToolError('artifact publication failed; ' + ('download completed but persistence outcome is uncertain; automatic retry forbidden' if completed else 'no download started'), code=code) from exc

    def get(self, identity, kind):
        if not isinstance(identity, str) or len(identity) != 64:
            raise ToolError('invalid retained artifact identity', code='artifact_unavailable')
        with self._connect() as db:
            row = db.execute('SELECT kind, metadata, data, expires FROM artifacts WHERE id=?', (identity,)).fetchone()
        if not row or row[0] != kind or row[3] <= time.time():
            raise ToolError('retained artifact unavailable/expired; reacquisition requires an explicit new URL call', code='artifact_unavailable')
        meta, data = json.loads(row[1]), row[2]
        if hashlib.sha256(data).hexdigest() != meta.pop('sha256'):
            raise ToolError('retained artifact integrity failure', code='artifact_integrity_failure')
        return Artifact(identity, kind, AcquiredResource(data=data, **meta), row[3])

    def _usage(self, db):
        count, size = db.execute('SELECT count(*), coalesce(sum(length(data) + length(CAST(metadata AS BLOB))),0) FROM artifacts').fetchone()
        size += db.execute('SELECT coalesce(sum(length(CAST(result AS BLOB))),0) FROM interpretations').fetchone()[0]
        return count, size

    def interpretation(self, identity, key):
        # Caller first resolves a nonexpired original-byte artifact.
        with self._connect() as db:
            row = db.execute('SELECT result FROM interpretations WHERE artifact_id=? AND key=?', (identity, key)).fetchone()
        return json.loads(row[0]) if row else None

    def retain_interpretation(self, identity, key, result):
        encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM interpretations WHERE artifact_id=? AND key=?', (identity, key)).fetchone():
                return  # Immutable first interpretation wins; never replace.
            _, size = self._usage(db)
            if size + len(encoded.encode('utf-8')) > self.max_bytes:
                raise ToolError('artifact interpretation storage ceiling reached; original acquisition retained', code='artifact_resource_ceiling')
            db.execute('INSERT INTO interpretations VALUES (?,?,?)', (identity, key, encoded))

    def unsettled_observation(self, identity, page):
        """A policy/version change must not implicitly retry an uncertain page."""
        with self._connect() as db:
            row = db.execute("SELECT result FROM interpretations WHERE artifact_id=? "
                             "AND key LIKE 'pdf-page-observation-%' AND json_extract(result,'$.page')=? "
                             "AND json_extract(result,'$.state') IN ('reserved','uncertain') LIMIT 1",
                             (identity, page)).fetchone()
        return json.loads(row[0]) if row else None

    def reserve_interpretation(self, identity, key, result):
        """Reserve observation storage BEFORE inference, across processes."""
        encoded = json.dumps(result, ensure_ascii=False)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM interpretations WHERE artifact_id=? AND key=?', (identity, key)).fetchone():
                return False
            if key.startswith('pdf-page-observation-') and db.execute(
                    "SELECT 1 FROM interpretations WHERE artifact_id=? AND key LIKE 'pdf-page-observation-%' "
                    "AND json_extract(result,'$.page')=? AND json_extract(result,'$.state') IN ('reserved','uncertain')",
                    (identity, result['page'])).fetchone():
                return False  # atomic cross-policy fence, not just a key cache
            _, size = self._usage(db)
            # Padding is charged by _usage, including after process death.
            if size + 65536 > self.max_bytes:
                raise ToolError('observation storage ceiling; inference not started', code='artifact_resource_ceiling')
            if len(encoded.encode('utf-8')) > 65536:
                raise ToolError('observation reservation exceeds ceiling')
            encoded += ' ' * (65536 - len(encoded.encode('utf-8')))
            db.execute('INSERT INTO interpretations VALUES (?,?,?)', (identity, key, encoded))
        return True

    def settle_interpretation(self, identity, key, result):
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded.encode('utf-8')) > 65536:
            raise ToolError('observation output exceeds reservation; outcome remains reserved', code='uncertain_document_observation')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT result FROM interpretations WHERE artifact_id=? AND key=?', (identity, key)).fetchone()
            if not row or json.loads(row[0]).get('state') != 'reserved':
                raise ToolError('observation reservation not settleable')
            db.execute('UPDATE interpretations SET result=? WHERE artifact_id=? AND key=?', (encoded, identity, key))

    def document_page_states(self, identity):
        # Project metadata in SQLite; never load all retained PNGs into Python
        # just to refresh the document list.
        with self._connect() as db:
            rows = db.execute("SELECT key, json_extract(result,'$.page'), "
                              "json_extract(result,'$.state'), json_extract(result,'$.pages[0].page'), "
                              "coalesce(length(json_extract(result,'$.pages[0].text')),0)>0, "
                              "coalesce(length(json_extract(result,'$.pages[0].image')),0)>0 "
                              "FROM interpretations WHERE artifact_id=?", (identity,)).fetchall()
        pages = {}
        for key, observation_page, status, parser_page, has_text, has_raster in rows:
            if key.startswith('pdf-page-observation-'):
                pages.setdefault(observation_page, {'page': observation_page})['observation_state'] = status
            elif key.startswith('pdf-v1:') and parser_page is not None:
                state = pages.setdefault(parser_page, {'page': parser_page})
                if has_text:
                    state['text_extracted'] = True
                if has_raster:
                    state['raster_retained'] = True
        return [pages[p] for p in sorted(pages)]

    def cleanup(self):
        with self._connect() as db:
            db.execute('DELETE FROM artifacts WHERE expires <= ?', (time.time(),))

    def delete(self, identity):
        with self._connect() as db:
            db.execute('DELETE FROM artifacts WHERE id=?', (identity,))
