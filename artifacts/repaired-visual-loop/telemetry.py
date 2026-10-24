"""Passive physical/allocator/host observer. Flags stop; no admission changes."""
import ctypes,json,os,re,subprocess,threading,time
from pathlib import Path
OUT=Path(os.environ['PHASE3_EVIDENCE']); GiB=1024**3
lib=ctypes.CDLL('/usr/lib/libproc.dylib')
def command(*args): return subprocess.check_output(args,text=True)
def host():
 return dict(at=time.time(),vm=command('vm_stat'),swap=command('sysctl','vm.swapusage'),pressure=command('memory_pressure','-Q'))
def start(backend,mx):
 stop=threading.Event(); pressure_count=0
 def sample(event):
  b=ctypes.create_string_buffer(1024);rc=lib.proc_pid_rusage(os.getpid(),4,ctypes.byref(b))
  assert rc==0
  row=dict(at=time.time(),event=event,mlx_active=mx.get_active_memory(),mlx_cache=mx.get_cache_memory(),mlx_peak=mx.get_peak_memory(),**{n:ctypes.c_uint64.from_buffer(b,o).value for n,o in [('resident',64),('footprint',72),('lifetime_max_footprint',240)]})
  with (OUT/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if max(row['footprint'],row['mlx_active']+row['mlx_cache'])>360*GiB: flag('360 GiB process/allocator ceiling',row)
  return row
 def flag(reason,row):
  (OUT/'SAFETY_STOP.json').write_text(json.dumps(dict(reason=reason,sample=row)))
 def watch():
  nonlocal pressure_count
  while not stop.wait(2):
   try:
    sample('periodic'); h=host()
    with (OUT/'host.jsonl').open('a') as f:f.write(json.dumps(h)+'\n')
    page=int(re.search(r'page size of (\d+)',h['vm'])[1]);pages=lambda n:int(re.search(n+r':\s+(\d+)',h['vm'])[1])*page
    reserve=sum(pages(n) for n in ['Pages free','Pages inactive','Pages speculative'])
    if reserve<100*GiB:flag('host reserve proxy below 100 GiB',h)
    if re.search(r'used = (?!0\.00M)\S+',h['swap']):flag('swap usage growth',h)
    if pages('Swapins') or pages('Swapouts'):flag('nonzero swap counters',h)
    pct=int(re.search(r'free percentage: (\d+)%',h['pressure'])[1]);pressure_count=pressure_count+1 if pct<25 else 0
    if pressure_count>=2:flag('sustained memory pressure',h)
   except Exception as e:flag('telemetry failure',dict(error=repr(e)))
 t=threading.Thread(target=watch,daemon=True);t.start()
 # Observe existing worker lifecycle without altering its numerical/cache result.
 from ds41f_mlx.serving.paired_checkpoint import PairedCheckpoint,PairedCheckpointAuthority
 for cls,name in [(PairedCheckpoint,'restore'),(PairedCheckpoint,'capture'),(PairedCheckpointAuthority,'publish')]:
  original=getattr(cls,name)
  def wrap(self,*a,_original=original,_name=name,**kw):
   sample(_name+':before')
   try:return _original(self,*a,**kw)
   finally:sample(_name+':after')
  setattr(cls,name,wrap)
 return stop,t,sample
