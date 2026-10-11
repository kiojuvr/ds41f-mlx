"""Application dogfood against a RUNNING Chat Server and official real model.

No mock model. Generated known-answer PDFs supplement a real public paper.
Outputs full transcripts/receipts locally; does not retry uncertain POSTs.
"""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import secrets
import urllib.request

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
from ds41f_mlx.chat_application import ChatApplication


def text_pdf(count):
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')}))
    for number in range(1, count + 1):
        page = writer.add_blank_page(width=600, height=780)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        content = DecodedStreamObject()
        content.set_data(f'BT /F1 16 Tf 40 700 Td (Operations manual page {number}) Tj 0 -30 Td (Department: Sector{number}. Inventory: {number * 17} units.) Tj 0 -30 Td (Document control: Aurora retained workflow.) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(content)
    stream = io.BytesIO(); writer.write(stream); return stream.getvalue()


def scan_pdf(count):
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 30)
    images = []
    for number in range(1, count + 1):
        image = Image.new('RGB', (800, 1040), 'white'); draw = ImageDraw.Draw(image)
        draw.text((50, 55), f'SCANNED INVENTORY - PAGE {number}', font=font, fill='black')
        draw.text((50, 115), f'Warehouse: Depot{number}', font=font, fill='black')
        draw.text((50, 170), f'Audit code: SCAN-{number:03d}', font=font, fill='black')
        for y in [260, 320, 380, 440]: draw.line((50, y, 750, y), fill='black', width=2)
        for x in [50, 410, 750]: draw.line((x, 260, x, 440), fill='black', width=2)
        for y, left, right in [(275, 'Product', 'Quantity'), (335, 'Bolts', str(number * 11)), (395, 'Nuts', str(number * 7))]:
            draw.text((70, y), left, font=font, fill='black'); draw.text((440, y), right, font=font, fill='black')
        draw.text((50, 520), 'Figure: relative stock levels', font=font, fill='black')
        draw.rectangle((80, 620, 80 + number * 15, 670), fill='#174b96')
        draw.text((80, 710), 'Blue bar: bolts inventory', font=font, fill='black')
        images.append(image)
    stream = io.BytesIO(); images[0].save(stream, format='PDF', save_all=True, append_images=images[1:], resolution=96)
    return stream.getvalue()


async def qualify(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    app = ChatApplication(args.app)
    records = []
    async def upload(name, data):
        identity = secrets.token_hex(32)
        entry = dict(kind='upload', name=name, artifact_id=identity, state='unconfirmed')
        records.append(entry)
        (output / (args.phase + '.json')).write_text(json.dumps(records, ensure_ascii=False, indent=2))
        receipt = await app.api('/api/documents', 'POST', dict(id=identity, name=name, data=base64.b64encode(data).decode()))
        info = await app.api('/api/documents/' + identity)
        entry.update(receipt=receipt, info=info, state=info.get('state'))
        print('UPLOAD', name, identity, info.get('page_count'), flush=True)
        assert info.get('state') == 'retained'
        return identity
    async def turn(name, prompt):
        print('TURN', name, flush=True)
        start_index = len(app.messages)
        await app.send(prompt)
        record = dict(kind='turn', name=name, start_index=start_index, messages=list(app.messages), session=app.record,
                      problem=app.problem, notice=app.notice)
        records.append(record)
        (output / (args.phase + '.json')).write_text(json.dumps(records, ensure_ascii=False, indent=2))
        assert not app.problem and not app.pending and not app.calls() and not app.awaiting_response, record
        images = [p for m in app.messages if isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url']
        assert not images, 'Document workflow leaked images into retained chat'
        print('ANSWER', app.messages[-1].get('content'), flush=True)
        return app.messages[-1].get('content', '')
    def check_values(answer, values):
        import re
        for value in values:
            assert re.search(r'(?<!\d)' + str(value) + r'(?!\d)', answer.replace(',', '')), (value, answer)
    def tool_pages():
        latest = records[-1]
        result = set()
        for message in latest['messages'][latest['start_index']:]:
            if message['role'] == 'tool' and isinstance(message['content'], str):
                value = json.loads(message['content'])
                result.update(p['page'] for p in value.get('pages', []) if 'text' in p or p.get('observation', {}).get('state') == 'completed')
        return result
    try:
        await app.new()
        app.settings.values.update(max_tokens=1500, temperature=0)
        if args.phase == 'text':
            ordinary = await upload('ordinary.pdf', text_pdf(3))
            answer = await turn('ordinary', f'Read PDF artifact_id={ordinary}, pages 1 and 3 using document_pdf and report the inventories with page citations. Do not use fetch_pdf or fetch_url.')
            check_values(answer, [17, 51])
            assert tool_pages() == {1, 3}
            long = await upload('operations-100-pages.pdf', text_pdf(100))
            answer = await turn('long-search', f'In retained PDF artifact_id={long} (100 pages), find Sector99 via document_pdf search. Continue search with next_page until found; report inventory with citation. Do not reacquire.')
            check_values(answer, [1683])
            answer = await turn('noncontiguous', f'Using the same PDF {long}, read pages 2, 50 and 99 with document_pdf and compare inventories, citing each page.')
            check_values(answer, [34, 850, 1683])
            assert tool_pages() == {2, 50, 99}
            answer = await turn('over-four-pages', f'Inspect pages 1 through 12 of retained PDF {long}, using document_pdf read in batches of at most four pages. Actually call tools for all twelve pages. List department and inventory for each page with citations.')
            check_values(answer, [p * 17 for p in range(1, 13)])
            assert tool_pages() == set(range(1, 13))
            if args.public_paper:
                with urllib.request.urlopen('https://arxiv.org/pdf/1706.03762', timeout=60) as response:
                    data = response.read(8 * 1024 * 1024 + 1)
                paper = await upload('attention-is-all-you-need.pdf', data)
                await turn('public-paper-text', f'For retained PDF {paper}, search for BLEU then read the matching page text. Report a table value and its exact source page. Cite source, distinguish parser text from any observation.')
        else:
            scan = await upload('scanned-inventory-12-pages.pdf', scan_pdf(12))
            answer = await turn('scanned-12-pages', f'Retained scanned PDF artifact_id={scan} has 12 pages, no text layer. Use document_pdf observe for EACH page 1 through 12 (exactly one page per call). Interpret all 12 pages, not only four. Report warehouse and bolts quantity for each with page citations. Observations are model-derived, not original text. Do not use fetch_pdf or bring images into chat.')
            check_values(answer, [p * 11 for p in range(1, 13)])
            assert tool_pages() == set(range(1, 13))
            before = await app.api('/api/documents/' + scan)
            answer = await turn('scanned-reuse-noncontiguous', f'Reuse observations for retained PDF {scan}, pages 2, 7 and 12 using action observation. Compare bolts quantities and explain the blue bar on page 7. No upload, download or fresh observe needed.')
            check_values(answer, [22, 77, 132])
            assert tool_pages() == {2, 7, 12}
            after = await app.api('/api/documents/' + scan)
            assert before['sha256'] == after['sha256']
            assert len([p for p in after['pages'] if p.get('observation_state') == 'completed']) == 12
            records.append(dict(kind='reuse', before=before, after=after))
            if args.paper_artifact:
                answer = await turn('public-figures-tables', f'In the previously retained real paper PDF {args.paper_artifact}, use document_pdf observe on pages 3 and 8. Explain Figure 1 encoder/decoder layout and give Transformer (big) BLEU values from Table 2. Cite pages. Explicitly label these Vision results model-derived, not original text. No reacquisition.')
                assert tool_pages() == {3, 8}
                assert '28.4' in answer and '41.8' in answer
                # Stop the OUTER document workflow while isolated page inference
                # owns a reserved effect. It may finish once; never resend it.
                paper_state = await app.api('/api/documents/' + args.paper_artifact)
                observed_pages = {p['page'] for p in paper_state['pages'] if p.get('observation_state')}
                choices = [4, *range(1, paper_state['page_count'] + 1)]
                fresh_page = next((p for p in choices if p <= paper_state['page_count'] and p not in observed_pages), None)
                assert fresh_page is not None, 'No unobserved source page for tool-stop probe; no reacquisition or retry'
                pending = asyncio.create_task(app.send(f'Use document_pdf observe on retained PDF {args.paper_artifact}, page {fresh_page}, then briefly describe its content.'))
                reserved = False
                for _ in range(36000):
                    if app.effects and app.effects['state'] == 'reserved':
                        current = await app.api('/api/documents/' + args.paper_artifact)
                        reserved = any(p['page'] == fresh_page and p.get('observation_state') == 'reserved' for p in current['pages'])
                        if reserved: break
                    if pending.done(): break
                    await asyncio.sleep(.05)
                assert reserved, 'No reserved page inference to stop; never retry'
                await app.stop(); await pending; await app.reconcile(wait=True)
                assert not app.problem and app.awaiting_response and not app.calls()
                completed = await app.api('/api/documents/' + args.paper_artifact + '/pages/' + str(fresh_page))
                assert completed['observation']['state'] == 'completed'
                before_count = app.count
                await app.resume()
                after_observation = await app.api('/api/documents/' + args.paper_artifact + '/pages/' + str(fresh_page))
                assert completed['observation']['response_id'] == after_observation['observation']['response_id']
                assert app.count == before_count + 1 and not app.problem and not app.awaiting_response
                tool_ids = [m['tool_call_id'] for m in app.messages if m['role'] == 'tool']
                assert len(tool_ids) == len(set(tool_ids))
                records.append(dict(kind='cancel-during-observation-continuation', record=app.record, messages=list(app.messages), observation=completed['observation']))
        # Existing canonical cancellation/reconciliation, through Chat Server.
        task = asyncio.create_task(app.send('Write a very long detailed audit checklist for this PDF, 200 numbered items.'))
        for _ in range(36000):
            if app.pending and app.pending.get('request_id') and app.live and len(app.live.get('content', '')) > 20: break
            if task.done(): break
            await asyncio.sleep(.05)
        assert not task.done() and app.live and len(app.live.get('content', '')) > 20, 'No active generation to cancel; never retry'
        await app.stop(); await task; await app.reconcile(wait=True)
        records.append(dict(kind='cancel-reconcile', record=app.record, messages=list(app.messages), problem=app.problem))
        assert not app.problem and not app.pending
        assert app.record['last_turn'].get('cancelled'), 'Stop did not exercise cancellation'
        if app.calls() or app.awaiting_response:
            await app.resume()
        await turn('after-cancel', 'Briefly confirm you can continue this audit, without rereading the PDF.')
    finally:
        (output / (args.phase + '.json')).write_text(json.dumps(records, ensure_ascii=False, indent=2))
        await app.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', default='http://127.0.0.1:8090')
    parser.add_argument('--phase', choices=['text', 'scan'], default='text')
    parser.add_argument('--output', default='artifacts/pdf-documents')
    parser.add_argument('--public-paper', action='store_true')
    parser.add_argument('--paper-artifact')
    asyncio.run(qualify(parser.parse_args()))
