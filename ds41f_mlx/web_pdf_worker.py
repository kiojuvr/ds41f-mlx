"""One-shot isolated untrusted PDF interpretation. No JS/forms/network APIs.

Invoked only by web_binary_tools with application-created input/output paths.
Native PDFium rasterization and pypdf inspection run outside the Web process.
"""
import base64
from contextlib import closing
import io
import json
import math
import resource
import sys

MAX_PAGES = 10000
MAX_OBJECTS = 100000
MAX_PAGE_TEXT = 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
MAX_PAGE_POINTS = 14400
RASTER_SCALE = 1.5


class Refusal(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def inspect_pdf(data, pages, mode):
    from pypdf import PdfReader
    from pypdf.generic import DictionaryObject, ArrayObject, IndirectObject
    import pypdfium2 as pdfium
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise Refusal('unsafe_pdf_capability', 'encrypted PDFs are not supported')
    count = len(reader.pages)
    if not 0 < count <= MAX_PAGES:
        raise Refusal('pdf_resource_ceiling', 'PDF page-count inspection ceiling reached')
    # Inspect reachable document graph without executing actions or extracting
    # attachments. Refuse active features, even when PDFium would ignore them.
    pending, visited, objects = [reader.trailer], set(), 0
    dangerous = {'/JS', '/JavaScript', '/AA', '/AcroForm', '/XFA',
                 '/EmbeddedFiles', '/RichMedia', '/Launch', '/SubmitForm', '/ImportData'}
    actions = {'/JavaScript', '/Launch', '/SubmitForm', '/ImportData', '/GoToR', '/GoToE', '/Rendition'}
    while pending:
        value = pending.pop()
        if isinstance(value, IndirectObject):
            key = (value.idnum, value.generation)
            if key in visited:
                continue
            visited.add(key)
            value = value.get_object()
        objects += 1
        if objects > MAX_OBJECTS:
            raise Refusal('pdf_resource_ceiling', 'PDF object inspection ceiling reached')
        if isinstance(value, DictionaryObject):
            opening = value.get('/OpenAction')
            if opening is not None:
                opening = opening.get_object()
                # A static initial page destination is inert reading metadata,
                # not an action/form execution. It is ignored, never followed.
                if not isinstance(opening, ArrayObject) and not (isinstance(opening, DictionaryObject) and opening.get('/S') == '/GoTo'):
                    raise Refusal('unsafe_pdf_capability', 'active PDF open action refused')
            if dangerous.intersection(value.keys()) or value.get('/S') in actions or value.get('/Type') in {'/Filespec', '/EmbeddedFile'}:
                raise Refusal('unsafe_pdf_capability', 'active/embedded PDF capability refused; provide a static PDF')
            pending.extend(value.values())
        elif isinstance(value, ArrayObject):
            pending.extend(value)
    if any(page > count for page in pages):
        raise Refusal('pdf_page_selection', f'page selection exceeds page count {count}')
    info = reader.metadata or {}
    # Discovery metadata is not a document-wide text injection channel.
    standard = {'/Title', '/Author', '/Subject', '/Keywords', '/Creator', '/Producer', '/CreationDate', '/ModDate', '/Trapped'}
    metadata = {str(k).lstrip('/'): str(v)[:1000] for k, v in info.items() if k in standard}
    metadata_truncated = any(k not in standard or len(str(v)) > 1000 for k, v in info.items())
    outputs = []
    with pdfium.PdfDocument(data) as document:
        for number in pages:
            with closing(document[number - 1]) as page:
                width, height = page.get_size()
                if not all(math.isfinite(v) and 0 < v <= MAX_PAGE_POINTS for v in (width, height)):
                    raise Refusal('pdf_resource_ceiling', 'PDF page dimensions exceed inspection ceiling')
                with closing(page.get_textpage()) as textpage:
                    if textpage.count_chars() > MAX_PAGE_TEXT:
                        raise Refusal('pdf_resource_ceiling', 'PDF page text exceeds extraction ceiling')
                    text = textpage.get_text_range()
                if len(text.encode('utf-8')) > MAX_PAGE_TEXT:
                    raise Refusal('pdf_resource_ceiling', 'PDF page text exceeds extraction byte ceiling')
                # Images and vector paths indicate layout-sensitive figures,
                # charts/tables. Auto retains text too; it never invokes OCR.
                layout = any(obj.type in (pdfium.raw.FPDF_PAGEOBJ_IMAGE, pdfium.raw.FPDF_PAGEOBJ_PATH)
                             for obj in page.get_objects())
                visual = mode == 'visual' or mode == 'auto' and (len(text.strip()) < 40 or layout)
                output = dict(page=number, width_points=width, height_points=height,
                              text=text if mode != 'visual' else '', interpretation='visual' if visual else 'text', links=[])
                for annot in reader.pages[number - 1].get('/Annots', []):
                    action = annot.get_object().get('/A')
                    if action:
                        action = action.get_object()
                        uri = action.get('/URI')
                        if uri and len(output['links']) < 100:
                            from urllib.parse import urlparse
                            parsed = urlparse(str(uri))
                            if parsed.scheme in {'http', 'https'} and parsed.hostname and not parsed.username and not parsed.password and len(str(uri)) <= 8192:
                                output['links'].append(str(uri))
                if visual:
                    w, h = math.ceil(width * RASTER_SCALE), math.ceil(height * RASTER_SCALE)
                    if not 0 < w * h <= 4194304 or not .5 <= w / h <= 2:
                        raise Refusal('pdf_resource_ceiling', 'PDF page raster exceeds Vision pixel/aspect envelope; request text or a different page')
                    # No form environment, JS runtime, action processing, or
                    # external-resource callbacks are ever initialized.
                    bitmap = page.render(scale=RASTER_SCALE, may_draw_forms=False)
                    try:
                        image = bitmap.to_pil()
                        buffer = io.BytesIO()
                        image.save(buffer, format='PNG')
                        output['image'] = base64.b64encode(buffer.getvalue()).decode('ascii')
                    finally:
                        bitmap.close()
                outputs.append(output)
    return dict(metadata=metadata, metadata_truncated=metadata_truncated, page_count=count, pages=outputs,
                parser='pypdf 6.10.0 / pypdfium2 5.3.0', raster_scale=RASTER_SCALE)


def main():
    # CPU, virtual address space, output file, and wall timeout (parent) are
    # independent of page/model capacity. Fail closed on worker death.
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    # Darwin rejects useful RLIMIT_AS values; the parent observes RSS and
    # kills the worker instead. Linux additionally enforces virtual memory.
    if sys.platform != 'darwin':
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_RESULT_BYTES,) * 2)
    source, target, selection = sys.argv[1:]
    try:
        with open(source, 'rb') as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise Refusal('pdf_resource_ceiling', 'PDF acquisition byte ceiling reached')
        options = json.loads(selection)
        result = inspect_pdf(data, options['pages'], options['mode'])
    except MemoryError:
        result = dict(error='PDF parser allocation ceiling reached', error_code='pdf_resource_ceiling')
    except Refusal as exc:
        result = dict(error=str(exc), error_code=exc.code)
    except Exception as exc:
        result = dict(error='PDF parser failed: ' + str(exc)[:500], error_code='pdf_parser_failure')
    with open(target, 'w') as stream:
        json.dump(result, stream, ensure_ascii=False)


if __name__ == '__main__':
    main()
