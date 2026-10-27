"""Model-free Recipe/source admission. No Backend/Model construction or weights.
Uses the existing tests' separately authenticated dependency selection, NOT
installed serving admission. Step 4B must repeat real installed pre_woa before load.
"""
import base64,hashlib,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(OUT))
from safety import admit_prepared
from conftest import installed_test_variant,admitted_test_numerics
from test_cold_messages_reconstruction import CHECKPOINT,prepare_fixture
from ds41f_mlx.serving.server import prepare_request

def test_envelopes():
 rows=json.loads((OUT.parent/'phase4-step4a-evidence/images.json').read_text());results=[]
 p,tokenizer,_,_=prepare_fixture();p.multimodal.release_sources()
 def image(label):
  row=next(x for x in rows if x['label']==label);b=(ROOT/row['source']).read_bytes()
  assert hashlib.sha256(b).hexdigest()==row['sha256']
  return dict(type='image',source=dict(type='base64',media_type='image/png',data=base64.b64encode(b).decode()))
 # Source admission and exact Recipe geometry, including every new fixture.
 cases=[('source-'+r['label'],r['label'],1200,0) for r in rows]
 cases += [('initial-ABC','ABC',1200,1800),('first-offload-candidate-ABCD','ABCD',1200,3000),('changed-BCD','BCD',1200,6000),('all-current-miss-DEF','DEF',1200,11500),('summary-selected-envelope','DEF',2048,11500),('post-compaction-cold','',1200,5000),('post-compaction-image-tail','EF',1200,5000),('final-HIA','HIA',1200,11500)]
 for name,labels,cap,text_words in cases:
  blocks=[image(l) for l in labels]+[dict(type='text',text='Model-free request envelope only. '+('bounded review '*text_words))]
  # The fourth represented image is a conservative Runtime control, NEVER
  # submitted by the official budget-3 adapter to HTTP in the intended run.
  body=json.dumps(dict(model='deepseek-flash',max_tokens=cap,messages=[dict(role='user',content=blocks)])).encode()
  q=prepare_request('messages',body,tokenizer=tokenizer,checkpoint=CHECKPOINT,ordinary=True)
  mm=q.multimodal
  if len(labels)<=3:admit_prepared(q,cap) # exact task wrapper on real Recipe preparation
  assert len(q.token_ids)+cap<=32768
  result=dict(name=name,current_labels=labels,input_positions=len(q.token_ids),output=cap,prepared_plus_output=len(q.token_ids)+cap,body_bytes=len(body))
  if mm:
   mm.validate_sources(0);mm.validate_workload(0)
   assert all(s.grid==(39,39) and s.length==184 for s in mm.spans)
   result.update(patches=sum(s.grid[0]*s.grid[1] for s in mm.spans),pixels=sum(s.source_pixels for s in mm.spans),identities=mm.identities(),cold_sources_valid=True)
   # Source/workload-only frontier check, not a forged paired hit. Genuine
   # authority selection is separately rechecked by retained source tests.
   C=mm.spans[0].start+mm.spans[0].length
   mm.validate_prefix(C,[mm.identities()[0]]);mm.validate_sources(C);mm.validate_workload(C)
   result['whole_image_frontier']=C;result['covered_patches']=1521;result['fresh_suffix_patches']=1521*(len(labels)-1)
   mm.release_sources()
  results.append(result)
 (OUT/'gate-c-admission.json').write_text(json.dumps(dict(kind='MODEL_FREE_SOURCE_TEST_ADMISSION_NOT_INSTALLED_SERVING_GATE',weights_loaded=0,context=1048576,prepared_plus_output_cap=32768,body_limit=16777216,fresh_patch_limit=36864,conditional_pair_limit=12582912,conditional_root_limit=201326592,cases=results),indent=2)+'\n')
