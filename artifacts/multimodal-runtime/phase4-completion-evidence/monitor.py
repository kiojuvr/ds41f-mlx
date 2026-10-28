"""Task-local host sampling. Python is a candidate, never proof of model ownership.
Only verified descendants of this Supervisor are task-owned; unknown candidates
remain blocking. PID + creation time prevents PID-reuse whitelist accidents.
"""
import ctypes,json,os,re,signal,subprocess,time,urllib.request
import psutil
G=1024**3
runtime=harness=None
start=time.monotonic()
ROOT_PID=os.getpid()
ROOT_BIRTH=psutil.Process(ROOT_PID).create_time()
lib=ctypes.CDLL('/usr/lib/libproc.dylib')
lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

def provenance(pid,root_pid=ROOT_PID,root_birth=ROOT_BIRTH):
 chain=[]
 try:
  p=psutil.Process(pid)
  # Each fresh Process object binds create_time; parent() checks PID reuse.
  while p is not None:
   with p.oneshot():
    row=dict(pid=p.pid,ppid=p.ppid(),created=p.create_time(),name=p.name(),cmd=p.cmdline(),rss=p.memory_info().rss,pgid=os.getpgid(p.pid))
   chain.append(row)
   if p.pid==root_pid:
    return dict(ownership='task' if row['created']==root_birth else 'unknown',chain=chain)
   p=p.parent()
  return dict(ownership='unknown',chain=chain)
 except (psutil.NoSuchProcess,ProcessLookupError):
  # Not a competing live owner. Persist the partial chain, do not guess ancestry.
  return dict(ownership='exited',chain=chain)
 except (psutil.AccessDenied,PermissionError) as error:
  return dict(ownership='unknown',chain=chain,error=repr(error))

def process_screen():
 candidates=[]
 # No process_iter object cache: use fresh identities for every sample.
 for pid in psutil.pids():
  try:
   p=psutil.Process(pid);name=p.name().lower()
   if 'python' not in name and name not in ('omlx','ds41f'):continue
   row=provenance(pid);row['pid']=pid;candidates.append(row)
  except psutil.NoSuchProcess:continue
  except psutil.AccessDenied:candidates.append(dict(pid=pid,ownership='unknown',error='access denied'))
 return candidates

def sample():
 b=(ctypes.c_uint64*64)();rc=-1 if runtime is None else lib.proc_pid_rusage(runtime.pid,4,ctypes.byref(b))
 vm=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
 candidates=process_screen()
 return dict(elapsed=time.monotonic()-start,available=psutil.virtual_memory().available,pressure=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True)),swap_used=psutil.swap_memory().used,swapouts=int(re.search(r'Swapouts:\s+(\d+)',vm)[1]),footprint=0 if rc else b[9],lifetime_footprint=0 if rc else b[30],competitors=[r for r in candidates if r['ownership']=='unknown'],process_candidates=candidates)

def retire(p):
 if p is None or p.poll() is not None:return
 os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=90)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def health():
 with urllib.request.urlopen('http://127.0.0.1:18083/health',timeout=2) as response:return json.load(response)
