"""Immutable fixture authoring/verification only, no model input answers."""
import hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
old=ROOT/'artifacts/multimodal-runtime/phase3-step3a-evidence'
rows=[]
for x in json.loads((old/'images.json').read_text()):
 rows.append(dict(label=x['label'],source=x['source'],sha256=x['source_sha256'],bytes=x['bytes'],width=384,height=384,media_type='image/png',visible_facts=x['visual_fact']))
for label,shape,color,marker in [('E','diamond','orange','E51'),('F','five-point star','black','F62'),('G','horizontal ellipse','teal','G73'),('H','crescent','magenta','H84'),('I','two vertical bars','brown','I95')]:
 p=OUT/('image-'+label+'.png')
 if not p.exists():
  im=Image.new('RGB',(384,384),'white');d=ImageDraw.Draw(im)
  if label=='E':d.polygon([(192,40),(315,155),(192,270),(69,155)],fill=color)
  elif label=='F':d.polygon([(192+(115 if i%2==0 else 48)*math.cos(-math.pi/2+i*math.pi/5),155+(115 if i%2==0 else 48)*math.sin(-math.pi/2+i*math.pi/5)) for i in range(10)],fill=color)
  elif label=='G':d.ellipse((45,95,339,225),fill=color)
  elif label=='H':d.ellipse((75,35,305,265),fill=color);d.ellipse((145,15,335,245),fill='white')
  else:d.rectangle((95,55,155,265),fill=color);d.rectangle((225,55,285,265),fill=color)
  font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',40)
  d.text((192,320),marker,fill='black',font=font,anchor='mm');im.save(p)
 b=p.read_bytes()
 rows.append(dict(label=label,source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),width=384,height=384,media_type='image/png',visible_facts=dict(shape=shape,color=color,marker=marker)))
for r in rows:
 b=(ROOT/r['source']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256'];assert len(b)==r['bytes']
 with Image.open(ROOT/r['source']) as im:assert im.size==(384,384) and im.format=='PNG';im.load()
 r.update(pixels=147456,expected_patch_grid=[39,39],expected_patches=1521,expected_expanded_positions=184,independent_source_admission='Expected PASS: PNG 384x384, 1:1, <16 MiB, <4194304 pixels; verified by admission.py',retention='Keep authenticated immutable source available through final continuation; A required again')
assert len({r['sha256'] for r in rows})==9
(OUT/'images.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS: nine distinct authenticated images; four reused, five new')
