"""PDF fixtures reused by browser/application probes; no second chat frontend."""
import io

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject


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
