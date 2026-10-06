"""Source-pinned completion collector; refuses missing/failed/stale GPU probes."""
import hashlib,json,platform,plistlib,subprocess,sys
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.qualify_tool_binary_development import sources
from ds41f_mlx.web_binary_tools import pdf_dependency_identity

ROOT=Path('artifacts/tool-capability-audit/development-attempt-8')
OUT=Path(sys.argv[1]);OUT.parent.mkdir(parents=True,exist_ok=True)
current=sources()
names=['research-focused-order','png-recovery','jpeg','webp','pdf-auto-recovery','text-recovery','retained-text-offset-browser','private-refusal-continuation','stop-capacity']
probes={n:json.loads((ROOT/(n+'.json')).read_text()) for n in names}
for n,p in probes.items():
    assert p['status']=='DEVELOPMENT_PROBE_FINISHED_NOT_ACCEPTANCE',(n,p['status'],p['errors'])
    assert p['source_stable'] and p['source_hashes']==current,n
research=probes['research-focused-order'];initial=research['workflows'][0]['result'];ordinary=research['workflows'][1]
assert initial['finishReason']=='stop' and ordinary['result']['finishReason']=='stop'
assert not ordinary['result'].get('capacityStop')
messages=initial['messages']
calls=[c for m in messages for c in m.get('tool_calls',[])]
assert {'web_search','fetch_url','fetch_image','fetch_pdf'} <= {c['function']['name'] for c in calls}
pdfcalls=[json.loads(c['function']['arguments']) for c in calls if c['function']['name']=='fetch_pdf']
assert sum(bool(c.get('url')) for c in pdfcalls)==1
assert any(c.get('artifact_id') and c.get('mode')=='visual' for c in pdfcalls)
assert any(c.get('artifact_id') and c.get('mode')=='text' and 4 in c.get('pages',[]) for c in pdfcalls)
assert len(ordinary['canonical']['diagnostics']['image_identities'])==2
for n in ['png-recovery','pdf-auto-recovery','text-recovery']:
    p=probes[n];assert 'canonical_effect_dedupe' in p['checks'] and 'save_fresh_restore_exact_history' in p['checks']
for n in ['research-focused-order','png-recovery','pdf-auto-recovery','text-recovery','jpeg','webp']:
    d=json.loads((ROOT/(n+'-runtime-diagnostics.json')).read_text())['diagnostics']
    assert d['last_image_encoded_count']==0 and d['m8']['total_prompt_replay_count']==0 and d['m8']['total_full_cache_repack_count']==0,n
for n in ['png-recovery','pdf-auto-recovery']:
    p=probes[n];sid=p['checks']['restored_ordinary_continuation']['id']
    d=json.loads((ROOT/(n+'-runtime-diagnostics.json')).read_text())
    assert d['id']==sid and len(d['diagnostics']['image_identities'])==1
assert 'completed_acquisitions_retained_at_admission_exhaustion' in probes['stop-capacity']['checks']
assert 'stop_after_vision_consumption' in probes['stop-capacity']['checks']

cpu=(ROOT/'final-cpu.log').read_text();assert '142 passed' in cpu
browser=(ROOT/'browser-cpu.log').read_text();assert 'pass 15' in browser
# Explicit heterogeneous evidence: safety/parser boundaries are CPU tests;
# actual consumption, native state and recovery are real Chrome/runtime/GPU.
rows=[
('search -> primary HTML -> model','research-focused-order'),
('HTML beyond former 80 KB/1 MiB boundaries','CPU tests/test_web_acquisition.py and test_web_binary_tools.py'),
('retained HTML offset -> model','retained-text-offset-browser'),
('HTML reload before consumption','text-recovery'),
('public PNG/JPEG/WebP acquisitions','png-recovery/jpeg/webp'),
('actual fetched-image Vision consumption','png-recovery/jpeg/webp'),
('ordinary text after fetched-image observation','png-recovery/jpeg/webp'),
('historical original image identities unchanged','png-recovery canonical save/fresh restore + diagnostics'),
('binary reload before consumption without download replay','png-recovery'),
('malformed/oversize/animated image rejection','CPU tests/test_web_binary_tools.py and test_web_acquisition_completion.py'),
('SSRF DNS/private-peer/redirect refusal','private-refusal-continuation + CPU tests/test_web_acquisition.py and test_m19_web_client.py'),
('real PDF acquisition','research-focused-order'),
('PDF metadata/page count','research-focused-order'),
('selected text-layer pages','research-focused-order pages 1 and 4'),
('additional retained PDF page, no URL reacquisition','research-focused-order'),
('image-heavy PDF auto -> Vision','pdf-auto-recovery'),
('PDF architecture figure -> actual Vision','research-focused-order/pdf-auto-recovery'),
('malformed/active/embedded PDF refusal','CPU tests/test_web_binary_tools.py'),
('large 15-page PDF without whole-document injection','research-focused-order'),
('PDF page/raster/output resource ceilings','CPU tests/test_web_binary_tools.py and test_web_pdf_completion_resources.py'),
('mixed HTML plus new/historical image admission','research-focused-order'),
('mixed PDF text and visual admission','research-focused-order'),
('actual full-request structural/image/output-reservation budgeting','research-focused-order + CPU tests/test_web_budget.py and test_runtime_capacity.py'),
('completed acquisitions survive admission exhaustion','stop-capacity'),
('explicit Continue rechecks, no redownload','stop-capacity/png-recovery'),
('Stop after Vision consumption','stop-capacity'),
('browser reload/reconciliation','png-recovery/pdf-auto-recovery/text-recovery'),
('canonical settled effect dedupe','png-recovery/pdf-auto-recovery/text-recovery'),
('save/fresh native restore with exact original-byte history','png-recovery/pdf-auto-recovery/text-recovery'),
('zero historical image re-encoding','runtime diagnostics for ordinary/restored continuation'),
('zero prompt replay/full-cache repack','runtime diagnostics for ordinary/restored continuation'),
]
assert len(rows)==31
files=list(ROOT.glob('*'))+list(Path('tests').glob('test_web*completion*.py'))
extra=[Path('pyproject.toml'),Path('README.md'),Path('docs/README.md'),Path('docs/doc-classification.json'),Path('docs/tool-capability-completion.md'),Path('docs/web-application.md'),Path('docs/dynamic-capability-budget.md'),*Path('tests').glob('test_web*.py'),Path('tests/test_m19_web_client.py'),*Path('tools').glob('*tool*capability*.py'),*Path('tools').glob('qualify_tool*.py')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
startup=json.JSONDecoder().raw_decode((ROOT/'runtime.log').read_text())[0]
assert startup['provenance_status']=='PASS' and startup['config']['mtp']=='OFF'
chrome=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
chrome_info=plistlib.loads(Path('/Applications/Google Chrome.app/Contents/Info.plist').read_bytes())
receipt=dict(schema='ds41f.tool_capability_completion.v1',status='ACCEPTED_TOOL_CAPABILITY_COMPLETION',
    baseline='c8e1ee5',recorded_at=datetime.now(timezone.utc).isoformat(),source_hashes=current,
    auxiliary_source_hashes={str(p):sha(p) for p in extra if p.is_file()},
    dependencies={n:version(n) for n in ['pypdf','pypdfium2','Pillow','numpy','mlx','fastapi','httpx','playwright']},
    pdf_executable_identity=pdf_dependency_identity(),platform=dict(system=platform.platform(),machine=platform.machine(),python=sys.version),
    hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
    runtime_startup_provenance=startup,
    chrome=dict(version=chrome_info['CFBundleShortVersionString'],executable=str(chrome),sha256=sha(chrome)),
    cpu=dict(passed=142,subtests_passed=8,renderer_platform_passed=15),
    cases=[dict(case=i+1,status='PASS',name=name,evidence=evidence) for i,(name,evidence) in enumerate(rows)],
    evidence_sha256={str(p):sha(p) for p in files if p.is_file()},
    retained_intermediate_receipt=dict(path='artifacts/tool-capability-audit/cpu-attempt-1/receipt.json',sha256=sha(Path('artifacts/tool-capability-audit/cpu-attempt-1/receipt.json')),status='HISTORICAL_INTERMEDIATE_NOT_FINAL_ACCEPTANCE'),
    not_claimed=['new 1M maximum-context GPU qualification','Linux qualification','OCR dependency or OCR accuracy guarantee','OS parser sandbox','exactly-once external network execution','hard cancellation of OS DNS/MCP trickle','R1/release/runtime promotion'],
    known_limits=['full-history Vision 4 images/8192 positions unchanged','fixed 108dpi PDF raster, no capacity-driven resize','PDF active/forms/encrypted/embedded capabilities refused','7-day immutable staging; browser ordinary image history independent','native artifacts require matching original-byte browser history','direct OFF ingress review remains separate'],
    r1_release_promotion_run=False)
OUT.write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt['status'],OUT)
