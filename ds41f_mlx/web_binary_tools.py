"""Public binary acquisition -> immutable receipt -> local interpretation.

Tool images are ordinary Chat Completions image parts (not base64 tool text).
The existing recipe, multimodal preparation and runtime budget own consumption.
"""
import base64
import hashlib
import http.client
from importlib.metadata import distribution
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
from urllib.parse import urlparse, unquote

from ds41f_mlx.web_tools import ToolError, ToolResult, acquire_public_resource, FETCH_TEXT_BYTES, _json_dumps
from ds41f_mlx.runtime.multimodal import validate_encoded_image, ImageConstraintError


def validate_acquisition(arguments, allowed):
    if set(arguments) - allowed:
        raise ToolError('unknown acquisition arguments')
    identity, url = arguments.get('artifact_id'), arguments.get('url')
    if (identity is None) == (url is None):
        raise ToolError('supply exactly one of url or retained artifact_id')
    if identity is not None and (not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity)):
        raise ToolError('invalid retained artifact identity', code='artifact_unavailable')
    if url is not None:
        if not isinstance(url, str) or not url.strip() or len(url) > 8192:
            raise ToolError('url must be a nonempty URL of at most 8192 characters')
        try:
            parsed = urlparse(url.strip())
            port = parsed.port
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
                raise ValueError('only public HTTP(S) URLs without credentials')
        except ValueError as exc:
            raise ToolError(str(exc), code='security_rejection') from exc


def artifact_for(arguments, store, kind):
    identity, url = arguments.get('artifact_id'), arguments.get('url')
    if (identity is None) == (url is None):
        raise ToolError('supply exactly one of url or retained artifact_id')
    if identity is not None:
        return store.get(identity, kind)
    if not isinstance(url, str) or not url.strip() or len(url) > 8192:
        raise ToolError('url must be a nonempty URL of at most 8192 characters')
    try:
        return store.acquire(kind, FETCH_TEXT_BYTES, lambda: acquire_public_resource(url.strip(), kind=kind, max_bytes=FETCH_TEXT_BYTES))
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise ToolError('network acquisition failed: ' + str(exc), code='acquisition_failure') from exc


def image_part(data, media):
    return dict(type='image_url', image_url=dict(url=f'data:{media};base64,' + base64.b64encode(data).decode('ascii')))


def binary_failure(artifact, code, error, tool):
    # Neither model-facing text nor compact UI exposes parser implementation
    # diagnostics. Keep typed, actionable outcomes and immutable receipt identity.
    summaries = {'pdf_parser_failure': 'PDF could not be safely parsed; choose a valid static PDF.',
                 'malformed_image': 'Image failed complete PNG/JPEG/WebP validation; choose a valid original image.'}
    actions = {'network_resource_ceiling': 'Choose a different source within the acquisition byte ceiling; retained prefixes cannot be parsed as complete binaries.',
               'malformed_image': 'Choose a valid original PNG/JPEG/WebP; do not automatically retry the same URL.',
               'vision_constraint_rejection': 'Choose a different original image within the Vision envelope; do not resize/transcode to fit.',
               'pdf_parser_failure': 'Choose a valid static PDF; the failed acquisition remains retained, with no automatic URL retry.',
               'unsafe_pdf_capability': 'Use a static PDF without active forms/actions or embedded resources.',
               'pdf_resource_ceiling': 'Use retained artifact_id with text mode, fewer pages or a different page; do not redownload automatically.'}
    payload = {**artifact.metadata(), 'error_code': code, 'error': summaries.get(code, str(error)),
               'next_action': actions.get(code, 'Use retained artifact_id for local interpretation; do not automatically reacquire the URL.')}
    return ToolResult(_json_dumps(payload), dict(tool=tool, **payload))


class FetchImageTool:
    name = 'fetch_image'
    schema = {'type': 'function', 'function': {
        'name': name, 'description': 'Acquire a public PNG/JPEG/WebP without transformation and view it with Vision. Retains original bytes; artifact_id reuses a completed acquisition. Shares the full-history Vision envelope.',
        'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}, 'artifact_id': {'type': 'string'}}, 'additionalProperties': False}}}

    def __init__(self, store): self.store = store

    def validate(self, arguments):
        validate_acquisition(arguments, {'url', 'artifact_id'})

    def run(self, arguments, context=None):
        self.validate(arguments)
        artifact = artifact_for(arguments, self.store, 'image')
        r = artifact.resource
        if r.truncated:
            return binary_failure(artifact, 'network_resource_ceiling', 'Image exceeds 8 MiB acquisition ceiling; partial file not interpreted', self.name)
        try:
            metadata = validate_encoded_image(r.data, r.content_type)
        except ImageConstraintError as exc:
            return binary_failure(artifact, 'vision_constraint_rejection', exc, self.name)
        except Exception as exc:
            return binary_failure(artifact, 'malformed_image', exc, self.name)
        payload = {**artifact.metadata(), **metadata, 'filename': unquote(Path(urlparse(r.url).path).name)[:200], 'interpretation': 'vision'}
        return ToolResult([dict(type='text', text=_json_dumps(payload)), image_part(r.data, metadata['content_type'])], dict(tool=self.name, **payload))


@lru_cache(maxsize=1)
def pdf_dependency_identity():
    identity = {}
    for name, required in [('pypdf', '6.10.0'), ('pypdfium2', '5.3.0')]:
        dist = distribution(name)
        if dist.version != required:
            raise ToolError(f'{name} {required} required; found {dist.version}', code='pdf_parser_failure')
        digest = hashlib.sha256()
        for entry in sorted(dist.files or [], key=str):
            if str(entry).endswith(('.py', '.so', '.dylib', '.dll')):
                digest.update(str(entry).encode('utf-8'))
                digest.update(Path(dist.locate_file(entry)).read_bytes())
        identity[name] = dict(version=dist.version, executable_source_sha256=digest.hexdigest())
    return identity


def interpret_pdf(data, pages, mode):
    # Temporary original-byte copy belongs to this one local interpretation.
    # communicate() doesn't retain unbounded parser stdout/stderr: discarded.
    with tempfile.TemporaryDirectory(prefix='ds41f-pdf-') as directory:
        source, target = Path(directory) / 'input.pdf', Path(directory) / 'result.json'
        source.write_bytes(data)
        process = subprocess.Popen([sys.executable, '-m', 'ds41f_mlx.web_pdf_worker', str(source), str(target),
                                    _json_dumps(dict(pages=pages, mode=mode))],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 20
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise ToolError('PDF interpretation wall deadline reached', code='pdf_resource_ceiling')
                if sys.platform == 'darwin':
                    sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, timeout=1)
                    rss = sample.stdout.strip()
                    if rss and int(rss) > 512 * 1024:
                        raise ToolError('PDF parser 512 MiB RSS circuit breaker reached', code='pdf_resource_ceiling')
                time.sleep(.05)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        if process.returncode != 0 or not target.exists():
            limited_output = target.exists() and target.stat().st_size >= 32 * 1024 * 1024
            code = 'pdf_resource_ceiling' if limited_output or process.returncode in {-signal.SIGXCPU, -signal.SIGXFSZ} else 'pdf_parser_failure'
            raise ToolError('PDF parser worker terminated (malformed document or parser resource ceiling)', code=code)
        if target.stat().st_size > 32 * 1024 * 1024:
            raise ToolError('PDF interpretation output ceiling reached', code='pdf_resource_ceiling')
        return json.loads(target.read_text())


class FetchPDFTool:
    name = 'fetch_pdf'
    schema = {'type': 'function', 'function': {
        'name': name, 'description': 'Read a public PDF or additional pages from retained artifact_id without another download. Defaults to page 1. auto uses selected text layers and Vision for scans/figures/tables; text forces text; visual renders selected pages. No whole-document context injection or active PDF execution.',
        'parameters': {'type': 'object', 'properties': {'url': {'type': 'string'}, 'artifact_id': {'type': 'string'},
            'pages': {'type': 'array', 'items': {'type': 'integer', 'minimum': 1}, 'minItems': 1, 'maxItems': 4},
            'mode': {'type': 'string', 'enum': ['auto', 'text', 'visual']},
            'offset': {'type': 'integer', 'minimum': 0, 'description': 'Character offset in each selected text page, for retained text continuation.'}}, 'additionalProperties': False}}}

    def __init__(self, store): self.store = store

    def validate(self, arguments):
        validate_acquisition(arguments, {'url', 'artifact_id', 'pages', 'mode', 'offset'})
        pages, mode = arguments.get('pages', [1]), arguments.get('mode', 'auto')
        offset = arguments.get('offset', 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError('offset must be a nonnegative integer')
        if (not isinstance(pages, list) or not 1 <= len(pages) <= 4 or
                any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in pages) or len(set(pages)) != len(pages)):
            raise ToolError('pages must select 1–4 distinct positive page numbers')
        if not isinstance(mode, str) or mode not in {'auto', 'text', 'visual'}:
            raise ToolError('mode must be auto, text or visual')

    def run(self, arguments, context=None):
        self.validate(arguments)
        pages, mode = arguments.get('pages', [1]), arguments.get('mode', 'auto')
        offset = arguments.get('offset', 0)
        artifact = artifact_for(arguments, self.store, 'pdf')
        if artifact.resource.truncated:
            return binary_failure(artifact, 'network_resource_ceiling', 'PDF exceeds 8 MiB acquisition ceiling; partial file not parsed', self.name)
        if not artifact.resource.data.startswith(b'%PDF-'):
            return binary_failure(artifact, 'pdf_parser_failure', 'PDF magic header missing', self.name)
        try:
            # Versioned immutable page interpretations retain the exact generated
            # PNG bytes across reloads. No historical page raster is replaced.
            dependencies = pdf_dependency_identity()
            policy = hashlib.sha256(Path(__file__).with_name('web_pdf_worker.py').read_bytes() + _json_dumps(dependencies).encode('utf-8')).hexdigest()
            keys = [f'pdf-v1:{policy}:pypdf-6.10.0:pdfium-5.3.0:scale-1.5:{mode}:{p}' for p in pages]
            cached = [self.store.interpretation(artifact.id, key) for key in keys]
            if all(value is not None for value in cached):
                result = {**cached[0], 'pages': [value['pages'][0] for value in cached]}
            else:
                result = interpret_pdf(artifact.resource.data, pages, mode)
                if not result.get('error'):
                    for key, page in zip(keys, result['pages']):
                        self.store.retain_interpretation(artifact.id, key, {**result, 'pages': [page]})
                    cached = [self.store.interpretation(artifact.id, key) for key in keys]
                    result = {**cached[0], 'pages': [value['pages'][0] for value in cached]}
        except ToolError as exc:
            return binary_failure(artifact, exc.code, exc, self.name)
        except Exception as exc:
            return binary_failure(artifact, 'pdf_parser_failure', str(exc)[:500], self.name)
        if result.get('error'):
            return binary_failure(artifact, result['error_code'], result['error'], self.name)
        images = []
        for page in result['pages']:
            text = page['text']
            page.update(text=text[offset:], offset=offset, total_chars=len(text),
                        returned_chars=len(text[offset:]), next_offset=offset + len(text[offset:]), context_truncated=False)
            encoded = page.pop('image', None)
            if encoded:
                data = base64.b64decode(encoded, validate=True)
                try:
                    metadata = validate_encoded_image(data, 'image/png')
                except Exception as exc:
                    return binary_failure(artifact, 'vision_constraint_rejection', exc, self.name)
                page['image_identity'] = metadata
                images.append(image_part(data, 'image/png'))
        payload = {**artifact.metadata(), **result, 'filename': unquote(Path(urlparse(artifact.resource.url).path).name)[:200], 'mode': mode, 'interpretation_identity': policy, 'dependency_identities': dependencies}
        text = _json_dumps(payload)
        return ToolResult([dict(type='text', text=text), *images] if images else text, dict(tool=self.name, **payload))
