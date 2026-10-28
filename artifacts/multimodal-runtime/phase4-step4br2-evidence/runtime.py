"""Task integration of retained R3 worker; unchanged lifespan/native trace ownership.
Source substitutions affect only task bounds, paths, and passive preparation.
"""
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx');OUT=Path(__file__).resolve().parent
prior=OUT.parent/'phase3-step3br3-evidence'
text=(prior/'runtime.py').read_text()
text=text.replace("proof=prove_installed_admission(checkpoint=True); save('admission.json',proof)", "proof=prove_installed_admission(checkpoint=True); assert proof['result_sha256']=='185d7692ae717f721a3e33ed4661b857c3582ca9ad4eb01691c8415c05bb7d99'; save('admission.json',proof)")
text=text.replace('from references import snapshot as reference_snapshot',"sys.path.append("+repr(str(prior))+ "); from references import snapshot as reference_snapshot")
text=text.replace("'d7b1d37ff0e00c0011b324eef309534d01947e44'","'147367541f750a611a242f727032e5593138f7d6'")
start=text.index(' tokenizer=serving.load_v41_tokenizer(')
end=text.index(' def observed(protocol,body,**kwargs):',start)
admission=''' safety_path=ROOT/'artifacts/multimodal-runtime/phase4-step4ar1-evidence'
 sys.path.append(str(safety_path))
 from safety import admit_prepared
 tokenizer=serving.load_v41_tokenizer(Path(sys.prefix)/'share/ds41f-mtp/recipe')
 admission_rows=[]
 for labels in [*'ABCDEFGHI','ABC','DEF','GHI']:
  blocks=[dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode((TEMP/'workspace'/f'image-{label}.png').read_bytes()).decode())) for label in labels]
  blocks.append(dict(type='text',text='Model-free source admission only. Describe the supplied images.'))
  body=json.dumps(dict(model='deepseek-flash',max_tokens=1200,messages=[dict(role='user',content=blocks)])).encode()
  p=original('messages',body,tokenizer=tokenizer,checkpoint=Path(os.environ['DS41F_CHECKPOINT']),ordinary=True)
  admit_prepared(p,1200);p.multimodal.validate_sources(0);p.multimodal.validate_workload(0)
  admission_rows.append(dict(labels=labels,N=len(p.token_ids),output=1200,images=p.multimodal.identities()))
  p.multimodal.release_sources()
 save('source-admission.json',dict(model_free=True,not_native_evidence=True,rows=admission_rows))
 p=None;blocks=None;body=None;tokenizer=None
'''
text=text[:start]+admission+text[end:]
text=text.replace("  if len(p.token_ids)>16384:raise ValueError('frozen acceptance prepared-position budget')", "  admit_prepared(p,json.loads(body)['max_tokens'])")
# Define backend before the no-weight preflight branch; collector runs only later.
text=text.replace(" serving.prepare_request=observed", " backend=None\n serving.prepare_request=observed")
text=text.replace("sources=hashes", "source_files_matched=len(hashes),aggregate_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()")
text=text.replace(" print('progress: installed admission, actual pre_woa and serving imports passed; waiting for supervisor',flush=True)"," print('progress: installed admission, actual pre_woa and serving imports passed; waiting for supervisor',flush=True)")
exec(compile(text,str(OUT/'runtime.py'),'exec'),{'__file__':str(OUT/'runtime.py'),'__name__':'__main__'})
