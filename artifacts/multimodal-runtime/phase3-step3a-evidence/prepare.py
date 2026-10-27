"""Finite Step 3A configuration; reuses F0 client scaffolding, no model calls."""
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=Path(__file__).resolve().parent
F0=ROOT/'artifacts/harness-files/gate-f0-evidence'
TEMP=Path(tempfile.mkdtemp(prefix='ds41f-step3a-session-'))
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,default=str)+'\n')
for d in ['workspace','agents','runtime','home/profiles/headless']: (TEMP/d).mkdir(parents=True)
save('location.json',{'temp':str(TEMP)})
# Independently identifiable original fixtures; answers never enter the user task.
rows=[]
for label,shape,color,marker in [('A','circle','red','A17'),('B','square','blue','B28'),('C','triangle','green','C39'),('D','cross','purple','D40')]:
 p=OUT/f'image-{label}.png'; im=Image.new('RGB',(384,384),'white'); d=ImageDraw.Draw(im)
 if shape=='circle': d.ellipse((80,80,304,304),fill=color)
 elif shape=='square': d.rectangle((80,80,304,304),fill=color)
 elif shape=='triangle': d.polygon([(192,60),(60,300),(324,300)],fill=color)
 else: d.rectangle((150,60,234,324),fill=color); d.rectangle((60,150,324,234),fill=color)
 d.text((135,335),marker,fill='black',font_size=32); im.save(p); p.chmod(0o444)
 dest=TEMP/'workspace'/p.name; shutil.copyfile(p,dest); dest.chmod(0o444)
 rows.append(dict(label=label,source=str(p.relative_to(ROOT)),copy=str(dest),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,width=384,height=384,pixels=384**2,visual_fact={'shape':shape,'color':color,'marker':marker}))
save('images.json',rows)
profile=TEMP/'home/profiles/headless'
(profile/'package.json').write_text(json.dumps(dict(name='ds41f-step3a',private=True,dependencies={},dsh=dict(profile=dict(bundles=['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless'])))))
(profile/'cordis.yml').write_text('[]\n')
patch=(F0/'profile.patch.yml').read_text().replace('/var/folders/jk/0kx3l0t93zj6gwlgth99c6lc0000gn/T/ds41f-gate-f0-cfccpse4/workspace',str(TEMP/'workspace')).replace(str(F0/'budget-guard.mjs'),str(OUT/'budget-guard.mjs')).replace('    thinking: disabled','    maxImagesPerRequest: 3\n    imageOffloadCountQuantum: 1\n    thinking: disabled')
(profile/'cordis.patch.yml').write_text(patch); (OUT/'profile.patch.yml').write_text(patch)
prompt='Inspect four images sequentially using only read_image, one tool call per response: image-A.png, then image-B.png, then image-C.png, then image-D.png. After each actual result describe its distinctive visible shape, color and marker, and compare it with the previously observed images before requesting the next image. After inspecting D, explicitly read image-A.png once more as a new read_image call, then compare that newly read image with the others and finish. Use exactly five read_image calls. Do not batch calls, use other tools, or infer visual contents from filenames. Do not edit anything.'
(TEMP/'prompt.txt').write_text(prompt+'\n'); save('task.json',{'prompt':prompt})
save('budgets.json',dict(conversations=1,requests=9,image_effects=5,max_tokens_per_request=1200,total_output_tokens=10800,total_s=900,generation_wall_s=180,startup_s=1200,body_bytes=16777216,prepared_positions=32768,no_progress_s=600,retry=False,model_loads=1))
old=json.loads((F0/'identities.json').read_text())
install=Path.home()/'.local/share/deepseek-harness/node_modules/@deepseek-ai'
actual={}
for n in [*old['packages'],'dsh-compaction-image-offload','dsh-llm','dsh-session']:
 p=install/n; actual[n]=dict(version=json.loads((p/'package.json').read_text())['version'],implementation_sha256=hashlib.sha256((p/('lib/bin.js' if n=='dsh' else 'lib/index.js')).read_bytes()).hexdigest())
 if n in old['packages']: assert actual[n]==old['packages'][n],n
save('identities.json',dict(node=subprocess.check_output(['node','--version'],text=True).strip(),packages=actual,all_17_match_f0=True,source_audit='687f282e8625c6b8d11ff58211a90524be2e4317',build_git_sha='not established by package manifests'))
observer=(F0/'observer.mjs').read_text().replace('++generations>5','++generations>9')
# Retain the exact native wire tree; sanitize only binary payloads (Files are opaque IDs).
observer=observer.replace("Object.assign(r,{body_bytes:","r.messages_projection=j;\n    Object.assign(r,{body_bytes:")
observer=observer.replace("    for(const [i,m] of (j.messages??[]).entries()) await walk(m.content,'messages.'+i+'.content');","    for(const [i,m] of (j.messages??[]).entries()) await walk(m.content,'messages.'+i+'.content');\n    const sanitize=x=>{if(!x||typeof x!=='object')return; if(x.type==='base64'&&typeof x.data==='string'){x.sha256=hash(Buffer.from(x.data,'base64'));delete x.data;} for(const v of Object.values(x))if(typeof v==='object')sanitize(v);}; sanitize(r.messages_projection);")
(OUT/'observer.mjs').write_text(observer)
env=dict(os.environ,DSH_HOME=str(TEMP/'home'),DSH_AGENTS_HOME=str(TEMP/'agents'),PHASE3_WORKSPACE=str(TEMP/'workspace'),PHASE3_EVIDENCE=str(OUT),DS41F_LOCAL_API_KEY='local-only-placeholder')
c=subprocess.run(['node',str(install/'dsh/lib/bin.js'),'--profile','headless','--dump-config'],cwd=TEMP/'workspace',env=env,capture_output=True,text=True,timeout=60)
(OUT/'composed-config.yml').write_text(c.stdout); (OUT/'config-stderr.log').write_text(c.stderr)
assert c.returncode==0,c.stderr
assert 'maxImagesPerRequest: 3' in c.stdout and 'imageOffloadCountQuantum: 1' in c.stdout
save('configuration-preflight.json',dict(success=True,weights_loaded=False,count=3,countQuantum=1,byte_thresholds='unchanged installed defaults; see composed configuration',session_log=False,only_tool='read_image',files_transport='normal DeepSeek Messages adapter',official_packages_modified=False))
print(TEMP)
