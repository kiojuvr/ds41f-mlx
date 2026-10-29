"""Request-owned image inputs; no generation or cache authority.

Only inline encoded images are accepted: no implicit filesystem/network access.
Committed images are identified by encoded-byte digest + expanded span, never
re-encoded on continuation/restore. Their executable representation is the KV.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import base64
import re
import warnings
from time import perf_counter
from typing import Any

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 2048 * 2048
MAX_IMAGE_TOKENS = 1014  # Retired nonordinary/OFF qualification; not an MTP limit.
MAX_MULTIMODAL_CONTEXT = 8192  # Legacy standard-OFF only.
# Harness rc.2 occurrence envelope; not a model execution guarantee.
MAX_REQUEST_IMAGES = 600
# Encoded bytes bound queued source residency. Pixels/patches bound fresh work.
# Ordinary wire history can defer early work charges, never prove a cache hit:
# the validated physical paired frontier decides actual fresh work before start.
# These are not Harness request byte budgets.
MAX_REQUEST_IMAGE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_PIXELS = 16 * 1024 * 1024
MAX_REQUEST_PATCHES = 36_864


@dataclass(frozen=True)
class ImageSpan:
    start: int
    length: int
    digest: str
    grid: tuple[int, int]
    kinds: tuple[int, ...]
    source_pixels: int

    def identity(self) -> dict:
        return dict(start=self.start, length=self.length, sha256=self.digest,
                    grid=list(self.grid))


@dataclass(frozen=True)
class MultimodalInput:
    token_ids: tuple[int, ...]
    spans: tuple[ImageSpan, ...]
    # Only encoded immutable bytes wait in the request queue. This request-owned
    # transport storage is discarded after encoding/admission failure; identities
    # and geometry remain available to publication and reconstruction.
    _sources: dict[int, bytes] = field(repr=False, compare=False)
    _config: Any = field(repr=False, compare=False)

    def release_sources(self) -> None:
        self._sources.clear()

    def identities(self) -> list[dict]:
        return [s.identity() for s in self.spans]

    def validate_prefix(self, frontier: int, committed: list[dict]) -> None:
        old = [s.identity() for s in self.spans if s.start < frontier]
        if any(s.start < frontier < s.start + s.length for s in self.spans):
            raise ValueError('committed frontier intersects an image span')
        if old != committed:
            raise ValueError('continuation changed committed image identity')

    def validate_workload(self, frontier: int = 0) -> None:
        """Physical fresh workload, only after the caller validates paired restore."""
        if frontier < 0 or any(s.start < frontier < s.start + s.length for s in self.spans):
            raise ValueError('committed frontier intersects an image span')
        fresh = [s for s in self.spans if s.start >= frontier]
        if sum(s.source_pixels for s in fresh) > MAX_REQUEST_PIXELS:
            raise ValueError('aggregate decoded pixel budget exceeded (fresh Vision workload)')
        if sum(s.grid[0] * s.grid[1] for s in fresh) > MAX_REQUEST_PATCHES:
            raise ValueError('aggregate image patch budget exceeded (fresh Vision workload)')

    def validate_sources(self, frontier: int = 0) -> None:
        """Recheck all fresh immutable sources before any native work starts."""
        for span in self.spans:
            if span.start >= frontier:
                data = self._sources.get(span.start)
                if data is None or sha256(data).hexdigest() != span.digest:
                    raise ValueError('new image source unavailable or changed')

    def encode_new(self, model: Any, frontier: int = 0, *, span_encoder=None) -> 'ImageEmbeddings':
        """Consume request sources once, preparing only spans beyond the frontier."""
        # The caller owns the admitted paired/idle frontier. Never turn a
        # partially covered image into a cache hit or an implicit replay.
        start = perf_counter()
        rows = []
        try:
            self.validate_workload(frontier)
            # OFF owns its resource capability; ordinary MTP supplies its admitted
            # pinned Model embedding adapter. Neither authority admits the other.
            if span_encoder is None:
                from ds41f_mlx.runtime.resource_admission import resources_for
                resources_for(model.language_model).validate_binding(model.language_model)
            for span in self.spans:
                if span.start < frontier:
                    self._sources.pop(span.start, None)
                    continue
                data = self._sources.pop(span.start, None)
                try:
                    if data is None or sha256(data).hexdigest() != span.digest:
                        raise ValueError('new image source unavailable or changed')
                    encoder = model.encode_image_span if span_encoder is None else span_encoder
                    value = _encode_span(data, span, self._config, encoder)
                    rows.append((span.start, value))
                finally:
                    data = value = None
            return ImageEmbeddings(tuple(rows), model.config.image_token_id, perf_counter() - start)
        finally:
            self.release_sources()
            rows.clear()


def _encode_span(data: bytes, span: ImageSpan, config: Any, encoder) -> Any:
    """Worker-local patch lifetime, ending only after MLX consumes the graph."""
    import mlx.core as mx
    from PIL import Image
    from ds41f_mlx.model_execution.processing import image_patch_array
    patches = pixels = value = None
    try:
        # Admission already verified/decoded this exact immutable byte sequence.
        # Use the original checkpoint processor, not a metadata-only substitute.
        with Image.open(BytesIO(data)) as image:
            patches, h, w, kinds = image_patch_array(image, config)
        if (h, w) != span.grid or tuple(kinds) != span.kinds:
            raise ValueError('image preparation differs from admitted geometry')
        pixels = mx.array(patches).astype(mx.bfloat16)
        value = encoder(pixels, h, w, span.kinds)
        mx.eval(value)
        return value
    finally:
        # Also drop large inputs from exception tracebacks; no request, span or
        # passive prefill certificate owns normalized CPU patches.
        data = patches = pixels = value = None


@dataclass(frozen=True)
class ImageEmbeddings:
    """Short-lived sparse image rows, indexed in absolute token coordinates."""
    rows: tuple[tuple[int, Any], ...]
    image_token_id: int
    seconds: float = 0.0

    def merge(self, h: Any, start: int) -> Any:
        end = start + h.shape[1]
        for origin, values in self.rows:
            left, right = max(start, origin), min(end, origin + values.shape[0])
            if left < right:
                h[:, left-start:right-start] = values[left-origin:right-origin].astype(h.dtype)
        return h


def image_bytes(source: Any) -> bytes:
    if getattr(source, 'detail', 'high') not in ('high', 'auto'):
        raise ValueError('low-detail image resizing is not qualified; use high/auto')
    if source.kind == 'bytes':
        data = source.data
    elif source.kind == 'data_url':
        url = source.data_url
        if not isinstance(url, str) or len(url) > MAX_IMAGE_BYTES * 4 // 3 + 1024:
            raise ValueError('invalid or oversized image data URL')
        header, sep, payload = url.partition(',')
        if not sep or not re.fullmatch(r'data:image/(png|jpeg|webp);base64', header):
            raise ValueError('only base64 PNG/JPEG/WebP inline images are supported')
        try:
            data = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError('invalid image base64') from exc
    else:
        raise ValueError('external image URLs/files are unsupported; supply inline image data')
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError('empty or oversized image bytes')
    return data


class ImageConstraintError(ValueError):
    """Valid image outside the qualified Vision resource envelope."""


def validate_encoded_image(data: bytes, media_type: str | None = None, *, remaining_pixels: int | None = None) -> dict:
    """CPU validation shared by manual and acquired images; never transforms bytes.

    Expanded positions and full-conversation capacity remain prepare/admission
    authority. Verify the complete file, not merely its header dimensions.
    """
    from PIL import Image
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ImageConstraintError('empty or oversized image bytes')
    magic = ('image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else
             'image/jpeg' if data.startswith(b'\xff\xd8\xff') else
             'image/webp' if data.startswith(b'RIFF') and data[8:12] == b'WEBP' else None)
    if magic is None or media_type is not None and media_type.split(';', 1)[0].strip().lower() != magic:
        raise ValueError('image MIME/magic mismatch or unsupported format')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            if image.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('unsupported image format')
            if getattr(image, 'n_frames', 1) != 1:
                raise ImageConstraintError('animated image outside the qualified single-frame Vision envelope')
            if not 0 < image.width * image.height <= MAX_PIXELS:
                raise ImageConstraintError(f'image pixel limit {MAX_PIXELS} exceeded ({image.width}x{image.height})')
            if remaining_pixels is not None and image.width * image.height > remaining_pixels:
                raise ImageConstraintError('aggregate decoded pixel budget exceeded')
            metadata = dict(content_type=magic, width=image.width, height=image.height,
                            sha256=sha256(data).hexdigest())
            image.verify()
        # verify() alone does not force JPEG/WebP pixel decoding.
        with Image.open(BytesIO(data)) as image:
            image.load()
    return metadata


def prepare_multimodal(token_ids: list[int], sources: list[Any], config: Any, *, ordinary: bool = False,
                       historical: tuple[bool, ...] = ()) -> MultimodalInput:
    """Fully validate sources and plan geometry, without normalized patch arrays."""
    from ds41f_mlx.model_execution.processing import plan_image_grid, image_token_kinds
    if not config.vision_enabled:
        raise ValueError('checkpoint has no vision encoder')
    if not sources or (not ordinary and len(sources) > MAX_IMAGES):
        raise ValueError(f'inline image count must be 1..{MAX_IMAGES}' if not ordinary else 'inline images required')
    if ordinary and len(sources) > MAX_REQUEST_IMAGES:
        raise ValueError('image occurrence budget exceeded (600)')
    if token_ids.count(config.image_token_id) != len(sources):
        raise ValueError('image placeholder/source count mismatch')
    expanded, spans, encoded = [], [], {}
    total_bytes = fresh_pixels = total_patches = expansion = 0
    if historical and (not ordinary or len(historical) != len(sources)):
        raise ValueError('invalid historical image origins')
    origins = iter(historical or (False,) * len(sources))
    # Bound expansion before allocation; output reservation is checked by serving.
    context = config.max_seq_len if ordinary else MAX_MULTIMODAL_CONTEXT
    image_token_limit = config.vision_max_n_token if ordinary else MAX_IMAGE_TOKENS
    if len(token_ids) > context:
        raise ValueError('expanded multimodal context exceeds context envelope')
    sources = iter(sources)
    for token in token_ids:
        if token != config.image_token_id:
            expanded.append(token)
            continue
        data = image_bytes(next(sources))
        is_historical = next(origins)
        total_bytes += len(data)
        if total_bytes > MAX_REQUEST_IMAGE_BYTES:
            raise ValueError('aggregate encoded image byte budget exceeded')
        try:
            # Fully validate/decode one source at a time, even on a later hit;
            # only encoded bytes/metadata survive. Wire age defers an early
            # charge, not actual admission: validate_workload uses the verified
            # paired frontier (or genuine miss) before any native start.
            metadata = validate_encoded_image(data, remaining_pixels=(
                None if is_historical else MAX_REQUEST_PIXELS-fresh_pixels))
            if not is_historical:
                fresh_pixels += metadata['width'] * metadata['height']
            nh, nw, height, width = plan_image_grid(metadata['width'], metadata['height'], config)
            h, w = height // config.vision_patch_size, width // config.vision_patch_size
            kinds = image_token_kinds(nh, nw)
            if len(kinds) > image_token_limit:
                raise ValueError('expanded image exceeds processor token envelope' if ordinary
                                 else 'expanded image exceeds qualified token envelope')
            # Deferral is not a cache hit. Scheduler still validates complete
            # physical paired coverage and the actual fresh workload.
            total_patches += 0 if is_historical else h * w
            if total_patches > MAX_REQUEST_PATCHES:
                raise ValueError('aggregate image patch budget exceeded')
            # Include all as-yet unexpanded tokens in the lower bound.
            expansion += len(kinds) - 1
            if len(token_ids) + expansion > context:
                raise ValueError('expanded multimodal context exceeds context envelope')
        except Exception as exc:
            raise ValueError(f'invalid image: {exc}') from exc
        encoded[len(expanded)] = data
        spans.append(ImageSpan(len(expanded), len(kinds), metadata['sha256'],
                               (h, w), tuple(kinds), metadata['width'] * metadata['height']))
        expanded.extend([token] * len(kinds))
    if len(expanded) > context:
        raise ValueError('expanded multimodal context exceeds qualified envelope')
    if expanded[-1] == config.image_token_id:
        raise ValueError('held-out terminal must be a text token')
    return MultimodalInput(tuple(expanded), tuple(spans), encoded, config)
