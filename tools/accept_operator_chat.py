import os, pty, fcntl, termios, struct, subprocess, time, select, json, pathlib, re, signal, hashlib
from urllib.request import urlopen
import pyte
# Run with the final-wheel Python (pyte is collector-only, not a runtime dependency).
# Starts services through actual TUI keys; leaves them READY after detachable quit.
import sys
OUT=pathlib.Path(os.environ['TUI_EVIDENCE']).resolve(); OUT.mkdir(parents=True, exist_ok=True)
PY=sys.executable
WHEEL=pathlib.Path(os.environ['TUI_WHEEL']).resolve()
os.environ['TERM']='xterm-256color'; os.environ['LANG']='en_US.UTF-8'
def api(port,path='/ds41f/status'):
    with urlopen(f'http://127.0.0.1:{port}'+path,timeout=2) as response:return json.load(response)
class UI:
    def __init__(self):
        self.fd,slave=pty.openpty(); self.raw=bytearray()
        fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack('HHHH',40,132,0,0))
        self.screen=pyte.Screen(132,40); self.stream=pyte.ByteStream(self.screen)
        self.child=subprocess.Popen([PY,'-I','-m','ds41f_mlx.ops'],stdin=slave,stdout=slave,stderr=slave,cwd='/tmp',env=os.environ.copy());os.close(slave)
    def pump(self,seconds=.4):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if select.select([self.fd],[],[],.03)[0]:
                try:data=os.read(self.fd,65536)
                except OSError:break
                self.raw.extend(data);self.stream.feed(data)
    def text(self):
        # pyte 0.8 can leave orphan wide-cell stubs after resize/repaint.
        # Preserve raw ANSI evidence; repair only its screenshot projection.
        try:return '\n'.join(self.screen.display)
        except IndexError:
            for row in self.screen.buffer.values():
                for col, char in list(row.items()):
                    if not char.data:row[col]=char._replace(data=' ')
            return '\n'.join(self.screen.display)
    def key(self,key):os.write(self.fd,key);self.pump()
    def save(self,name):
        self.pump();(OUT/(name+'.txt')).write_text(self.text());(OUT/(name+'.ansi')).write_bytes(self.raw)
    def wait(self,predicate,seconds=120):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.pump(.1)
            try:
                result=predicate()
                if result:return result
            except (OSError,KeyError,ValueError):pass
        self.save('failure');raise RuntimeError('acceptance wait timeout')
ui=UI();ui.pump(1);ui.key(b's')
ui.wait(lambda:api(18441)['state']=='READY',360);ui.save('dashboard-ready');print('runtime READY',flush=True)
ui.key(b'c');ui.save('chat-stopped');assert 'server STOPPED' in ui.text()
ui.key(b'\x1bOQ') # F2 xterm
ui.wait(lambda:api(18443)['state']=='READY',30);ui.save('chat-server-ready')
ui.key(b'\x0e');ui.wait(lambda:re.search(r'tui-[0-9a-f-]{36}',ui.text()))
sid=re.search(r'tui-[0-9a-f-]{36}',ui.text()).group();print('session',sid,flush=True)
assert 'production-sessions-v1' in ui.text() and 'T=0 p=1 reasoning=none output=auto' in ui.text()
negotiated=api(8080,'/api/status')['runtime'];ui.save('negotiated-defaults')
settlements={}
def session():
    rec=api(8080,'/api/session/'+sid)
    if rec['state']=='idle' and rec['request_count']:
        diag=rec['diagnostics']['m8']
        assert diag['all_cache_offsets_equal_frontier']
        assert diag['total_prompt_replay_count']==diag['total_full_cache_repack_count']==0
        settlements[rec['request_count']]=rec
    return rec
def send(text):
    ui.key(b'\x1b[200~'+text.encode()+b'\x1b[201~');ui.save('paste')
    assert session()['request_count']==0 if text.startswith('こんにちは') else True
    assert text.split('\n')[0] in ui.text()
    ui.key(b'\x07')
stream_seen=[]
# 500-line draft, no admission; preserve it through mode switch and resize.
long_paste='日本語 paste line\n'*500
ui.key(b'\x1b[200~'+long_paste.encode()+b'\x1b[201~');ui.save('500-line-paste')
assert session()['request_count']==0
ui.key(b'\x14');ui.key(b'c');ui.save('500-line-return')
# Remove the unsubmitted draft with editor keys, not model/server calls.
ui.key(b'\x15'+b'\x7f\x15'*500)
send('こんにちは。日本語で短く挨拶してください。\nこれはペーストの2行目です。')
ui.wait(lambda:'UNSETTLED' in ui.text());ui.save('streaming');stream_seen.append(ui.text())
ui.key(b'\x14');ui.save('dashboard-live-stream')
assert 'request received' in ui.text() or 'RUNTIME LOG' in ui.text()
ui.key(b'c')
ui.wait(lambda:session()['request_count']==1 and session()['state']=='idle');ui.wait(lambda:'turns 1' in ui.text());ui.save('turn1');assert not 'UNSETTLED' in ui.text()
print('turn1',flush=True)
ui.key(b'\x14');ui.save('dashboard-during-chat');assert 'RUNTIME LOG' in ui.text()
ui.key(b'c');ui.save('chat-return');assert sid in ui.text() and 'こんにちは' in ui.text()
ui.key('さっきの挨拶を英語で一文にしてください。'.encode());ui.key(b'\x07')
ui.wait(lambda:session()['request_count']==2 and session()['state']=='idle');ui.wait(lambda:'turns 2' in ui.text());ui.save('turn2');print('turn2',flush=True)
ui.key('必ず fetch_url ツールを使って https://example.com/ を取得し、ページのタイトルを短く教えてください。'.encode());ui.key(b'\x07')
ui.wait(lambda:session()['request_count']>=4 and session()['state']=='idle',180);ui.wait(lambda:'turns 4' in ui.text());ui.save('tool-continuation')
assert 'tool fetch_url' in ui.text() and 'excerpt' in ui.text(),ui.text()
tool_rec=session();m=api(18441)['last_request']['metrics'];assert m['aligned'] and m['prompt_replay']==m['full_cache_repack']==0
print('tool complete',json.dumps(m),flush=True)
ui.key(b'\x1b[5~');ui.save('scroll-paused');assert 'scroll paused' in ui.text()
ui.key(b'\x0c');ui.save('latest');assert 'latest' in ui.text()
ui.key('下書き日本語\nresize draft'.encode())
for width,height in ((80,28),(40,10),(132,40)):
    fcntl.ioctl(ui.fd,termios.TIOCSWINSZ,struct.pack('HHHH',height,width,0,0));ui.screen.resize(height,width);os.kill(ui.child.pid,signal.SIGWINCH);ui.pump(1);ui.save(f'resize-{width}x{height}')
assert '下書き日本語' in ui.text() and 'resize draft' in ui.text()
# Erase draft via Ctrl-U twice (first second line, then join/delete first).
ui.key(b'\x15\x7f\x15')
ui.key('1から200までの整数を、一行に一つずつ数字だけ書いてください。説明やコードブロックは不要です。'.encode());ui.key(b'\x07')
ui.wait(lambda:'UNSETTLED' in ui.text() and re.search(r'^ \d+\s*$',ui.text(),re.M),30)
ui.pump(.3);ui.save('live-output-before-stop');ui.key(b'\x03')
ui.wait(lambda:session()['request_count']==5 and session()['state']=='idle',120);ui.wait(lambda:'turns 5' in ui.text());ui.save('stopped-canonical')
cancel_rec=session();assert cancel_rec['last_turn']['cancelled'],cancel_rec['last_turn']
m2=api(18441)['last_request']['metrics'];assert m2['aligned'] and m2['prompt_replay']==m2['full_cache_repack']==0
print('cancel canonical',json.dumps(m2),flush=True)
ui.key('Stop後も安全に続けられますか。「はい」とだけ答えてください。'.encode());ui.key(b'\x07')
ui.wait(lambda:session()['request_count']==6 and session()['state']=='idle',120)
ui.wait(lambda:'turns 6' in ui.text());ui.save('safe-after-stop')
after_stop=session()
# Simulate response transport loss by stopping Chat Server externally, not TUI F3.
ui.key('1から200までの整数を、一行に一つずつ数字だけ書いてください。説明は不要です。'.encode());ui.key(b'\x07')
ui.wait(lambda:'UNSETTLED' in ui.text())
from urllib.request import Request
with urlopen(Request('http://127.0.0.1:18443/ds41f/control/shutdown', method='POST', data=b''),timeout=5):pass
ui.wait(lambda:'server STOPPED' in ui.text(),120);ui.save('external-disconnect')
ui.key(b'\x1bOQ');ui.wait(lambda:api(18443)['state']=='READY',30)
ui.wait(lambda:'working' not in ui.text().splitlines()[0],90)
ui.key(b'\x12');ui.wait(lambda:'turns 7' in ui.text(),120);ui.save('disconnect-reconciled')
disconnected=session();assert disconnected['state']=='idle' and disconnected['request_count']==7
ui.key(b'\x14');ui.save('dashboard-final');assert 'MTP settled' in ui.text()
ui.key(b'c');assert sid in ui.text();ui.key(b'\x1bOR') # F3 stop backend retires own session
ui.wait(lambda:'server STOPPED' in ui.text(),120);ui.save('server-stopped-after-chat')
ui.key(b'\x1bOQ');ui.wait(lambda:api(18443)['state']=='READY',30);ui.save('server-restarted')
ui.key(b'\x0e');ui.wait(lambda:re.search(r'tui-[0-9a-f-]{36}',ui.text()));sid2=re.search(r'tui-[0-9a-f-]{36}',ui.text()).group();assert sid2!=sid
ui.key('再起動後の確認です。「はい」とだけ答えてください。'.encode());ui.key(b'\x07')
ui.wait(lambda:api(8080,'/api/session/'+sid2)['request_count']==1 and api(8080,'/api/session/'+sid2)['state']=='idle');ui.wait(lambda:'turns 1' in ui.text());ui.save('restarted-turn')
ui.key(b'\x11');ui.child.wait(timeout=90);ui.pump();assert ui.child.returncode==0
assert api(18441)['state']=='READY' and api(18443)['state']=='READY'
try:api(8080,'/api/session/'+sid2)
except Exception as exc:assert getattr(exc,'code',None)==404,exc
else:raise AssertionError('quit did not retire session')
wheel=WHEEL
final=api(18441)['last_request']['metrics']
assert final['aligned'] and final['prompt_replay']==final['full_cache_repack']==0
receipt=dict(status='PTY_MODEL_PASS',settlements=settlements,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),negotiated=negotiated,after_stop=after_stop,disconnected=disconnected,final_metrics=final,installed_python=PY,session=sid,tool_settlement=m,cancel_settlement=m2,tool_turn=tool_rec,cancel_turn=cancel_rec,session_quit_retired=True,services_detachable=True,unicode_multiline_paste=True,physical_ime='not exercised; PTY injection is not a physical IME check')
(OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('PASS',flush=True)
