#!/usr/bin/env python3
"""Bounded installed HTTP acceptance. Starts/stops only its own idle Runtime."""
import argparse
import base64
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18057)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base, control = f'http://127.0.0.1:{a.port}', 'http://127.0.0.1:18441'
    def call(url, body=None, method=None):
        data = None if body is None else json.dumps(body).encode()
        assert data is None or len(data) < 8*1024**2
        try:
            with urlopen(Request(url, data=data, method=method,
                    headers={'Content-Type':'application/json'}), timeout=180) as r:
                return r.status, json.load(r)
        except HTTPError as exc:
            return exc.code, json.load(exc)
    for url in (base+'/health', control+'/ds41f/status'):
        try:
            call(url)
        except URLError:
            pass
        else:
            raise RuntimeError('existing listener; refusing start')
    env = dict(os.environ, DS41F_CHECKPOINT='/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
    with open(os.devnull, 'wb') as null:
        proc = subprocess.Popen([str(a.env/'bin/ds41f'), 'start', '--host', '127.0.0.1', '--port', str(a.port)],
            env=env, cwd='/tmp', stdout=null, stderr=null)
    lib = ctypes.CDLL('/usr/lib/libproc.dylib')
    stop, unsafe = threading.Event(), []
    receipt = dict(requests=[])
    def observe():
        try:
            while not stop.wait(2):
                cmd = lambda *args: subprocess.check_output(args, text=True)
                vm, swap, pressure = cmd('vm_stat'), cmd('sysctl', 'vm.swapusage'), cmd('memory_pressure', '-Q')
                pages = lambda name: int(re.search(name+r':\s+(\d+)', vm)[1])*16384
                reserve = sum(pages(n) for n in ('Pages free', 'Pages inactive', 'Pages speculative'))
                buf = ctypes.create_string_buffer(1024)
                assert lib.proc_pid_rusage(proc.pid, 4, ctypes.byref(buf)) == 0
                footprint = ctypes.c_uint64.from_buffer(buf, 72).value
                row = dict(t=time.monotonic(), footprint=footprint,
                    lifetime_max_footprint=ctypes.c_uint64.from_buffer(buf, 240).value,
                    reserve_proxy=reserve, vm=vm, swap=swap, pressure=pressure)
                with (a.output/'host.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                if footprint > 360*1024**3 or reserve < 100*1024**3:
                    unsafe.append('footprint/reserve stop')
                if re.search(r'used = (?!0\.00M)\S+', swap):
                    unsafe.append('nonzero swap')
                if int(re.search(r'free percentage: (\d+)%', pressure)[1]) < 25:
                    unsafe.append('memory pressure stop')
        except BaseException:
            if proc.poll() is None:
                unsafe.append('observer failure')
    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        deadline = time.monotonic()+420
        while True:
            assert proc.poll() is None and not unsafe
            try:
                code, health = call(base+'/health')
                if code == 200 and health['model_ready']:
                    break
            except URLError:
                pass
            assert time.monotonic() < deadline, 'readiness deadline'
            time.sleep(1)
        receipt['initial_health'] = health
        # Preserve recoverable failure/READY semantics without recording body.
        body = dict(model='deepseek-v4.1-flash', max_tokens=16, temperature=-1,
            messages=[dict(role='user', content='Synthetic admission check.')])
        code, _ = call(base+'/v1/messages', body)
        assert code in (400, 422)
        status = call(control+'/ds41f/status')[1]
        assert status['state'] == 'READY' and status['failures'][-1]['stage'] == 'preparation'
        receipt['recoverable_failure'] = status['failures'][-1]
        history = []
        image = base64.b64encode(a.image.read_bytes()).decode()
        for n in range(13):
            assert not unsafe and time.monotonic() < deadline+480, 'finite request deadline'
            content = [dict(type='text', text='State the dashboard title only.' if n < 8 else
                            ' x'*1800+' State the dashboard title only.')]
            if n < 8:
                content.append(dict(type='image', source=dict(type='base64', media_type='image/png', data=image)))
            history.append(dict(role='user', content=content))
            body.update(temperature=0, thinking={'type':'disabled'}, messages=history)
            code, response = call(base+'/v1/messages', body)
            assert code == 200 and response['content'], 'Messages acceptance failed'
            history.append(dict(role='assistant', content=response['content']))
            status = call(control+'/ds41f/status')[1]
            last = status['last_request']
            assert status['state'] == 'READY' and last['finish_reason'] != 'error'
            m = last['metrics']
            assert m['aligned'] and m['prompt_replay'] == m['full_cache_repack'] == 0
            receipt['requests'].append(dict(request_id=last['request_id'], metrics=m,
                native_started=last['native_started'], canonical_settled=last['canonical_settled'],
                cache_published=last['cache_published']))
            if m.get('legacy_pair_bytes_exceeded'):
                assert m['historical_planned_patches'] > 36864 and m['fresh_encoder_patches'] == 0
                assert last['cache_published'] and last['canonical_settled']
                break
        else:
            raise AssertionError('old byte boundary not crossed')
        health = call(base+'/health')[1]
        assert health['status'] == 'ready' and health['fatal_error'] is None
        assert health['active_requests'] == health['queued_requests'] == 0
        assert not unsafe
        receipt.update(final_health=health, status='COMPLETED')
    finally:
        # Never interrupt settlement via instrumentation; stop through owner.
        receipt['forced_stop'] = False
        try:
            if proc.poll() is None:
                try:
                    code, _ = call(control+'/ds41f/control/shutdown', method='POST')
                    assert code == 202
                except (OSError, AssertionError):
                    proc.terminate()  # owned process only; control may not have started
            try:
                receipt['exit_code'] = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                receipt['forced_stop'] = True
                proc.kill()
                receipt['exit_code'] = proc.wait(timeout=30)
        finally:
            stop.set()
            watcher.join(timeout=30)
            receipt['unsafe'] = unsafe
            (a.output/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert receipt['exit_code'] == 0


if __name__ == '__main__':
    main()
