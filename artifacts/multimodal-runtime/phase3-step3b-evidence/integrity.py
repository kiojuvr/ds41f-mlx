"""Read-only exact baseline inventory check; never imports operational Runtime."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path('/Volumes/SDXC-512/ds41f-mlx');OUT=Path(__file__).resolve().parent
baseline=json.loads((ROOT/'artifacts/multimodal-runtime/phase2-step2br2-original-before.json').read_text())['files']
install=Path('/Volumes/SDXC-512/ds41f-1017');files={}
for name in baseline:
 p=install/name;s=p.stat();files[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
result=dict(paths=len(files),baseline='phase2-step2br2-original-before.json',unchanged=files==baseline,files=files)
(OUT/f'integrity-{sys.argv[1]}.json').write_text(json.dumps(result,indent=2)+'\n')
assert len(files)==817 and files==baseline
print('817 operational paths match retained baseline')
