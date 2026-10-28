"""Fresh installed full-checkpoint gate; never loads model weights."""
import hashlib,json,os,subprocess,sys,sysconfig,traceback
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx')
OUT=Path(__file__).resolve().parent
sys.path.append(str(ROOT))
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='993fd338107430ecde5e80c06c8159ac08a06c85'
try:
 from tools.qualify_installed_admission import prove_installed_admission
 proof=prove_installed_admission(checkpoint=True)
 save('admission.json',proof)
 import ds41f_mlx
 installed=Path(ds41f_mlx.__file__).parent
 count=0
 for source in (ROOT/'ds41f_mlx').rglob('*.py'):
  relative=source.relative_to(ROOT/'ds41f_mlx')
  assert source.read_bytes()==(installed/relative).read_bytes(),str(relative)
  count+=1
 from ds41f_mlx.model_execution import activation
 function=activation.pre_woa
 modules={name:sys.modules[name] for name in proof['origins']}
 from ds41f_mlx.serving.server import create_app
 assert activation.pre_woa is function
 assert all(sys.modules[name] is module for name,module in modules.items())
 save('preflight.json',dict(status='PASS',weights_loaded=0,source_files_matched=count,installed_root=str(installed),full_checkpoint=True,zero_weight=True,post_serving_function_identity=True))
except BaseException as error:
 save('preflight.json',dict(status='BLOCKED',weights_loaded=0,full_checkpoint_admission_completed=False,error=repr(error),traceback=traceback.format_exc(),python=sys.executable,site=sysconfig.get_paths()['purelib'],checkpoint=os.environ.get('DS41F_CHECKPOINT'),no_overlay=True))
 traceback.print_exc()
 sys.exit(1)
