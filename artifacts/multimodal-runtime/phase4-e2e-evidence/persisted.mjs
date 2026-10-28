// Read-only official persistence handle and official Session projection after CLI exit.
import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const install='file://'+process.env.HOME+'/.local/share/deepseek-harness/node_modules/@deepseek-ai/';
const {Context}=await import(install+'cordis/lib/index.js');
const Persistence=await import(install+'dsh-session-persistence-jsonl/lib/index.js');
const {Session,SessionId}=await import(install+'dsh-session/lib/index.js');
const load=name=>import(install+name+'/lib/index.js');
const out=process.env.PHASE4_INSPECT_OUTPUT?new URL('file://'+process.env.PHASE4_INSPECT_OUTPUT+'/'):new URL('.',import.meta.url);
const temp=process.env.PHASE4_INSPECT_ROOT??JSON.parse(readFileSync(new URL('location.json',out))).temp;
const ctx=new Context();let handle;
try{
 // Official plugins own projection definitions. Mount services only, no Agent,
 // loop, provider, tool registration or compaction engine is instantiated.
 for(const name of ['dsh-llm','dsh-session','dsh-session-projection','dsh-system-prompt','dsh-tools','dsh-agent'])await ctx.plugin((await load(name)).default);
 await ctx.plugin(await load('dsh-compaction-image-offload'));
 await ctx.plugin(Persistence,{root:temp+'/home/sessions'});
 handle=await ctx.sessionPersistence.open(SessionId(process.argv[2]),'read');
 const {events}=await handle.read();
 const session=Session.create(handle.id,events,handle.header,handle.inheritedEventCount,ctx.sessions.messageProjections);
 const chain=events.filter(e=>e.type.startsWith('compaction/'));
 const summary=chain.find(e=>e.type==='compaction/summary');
 const replacement=events.find(e=>e.surfaceOp?.op==='replace'&&e.data.source?.kind==='compaction');
 // Actual checkpoint source kind is retained verbatim below, never guessed for projection.
 if(summary){
  const checkpoint=events.find(e=>e.surfaceOp?.op==='replace'&&e.sourceEventSeqs?.includes(summary.seq));
  assert(checkpoint);assert(checkpoint.data.content.every(b=>b.type==='text'));
  assert(summary.data.shadowedSeqs.every(s=>events.some(e=>e.seq===s)&&!session.surface.nodes.includes(s)));
  assert(session.surface.nodes.includes(checkpoint.seq));
 }
 writeFileSync(new URL('persisted-'+process.argv[3]+'.json',out),JSON.stringify({session_id:handle.id,header:handle.header,events,compaction:chain,nodes:[...session.surface.nodes],messages:session.deriveMessages(),requestHeader:session.requestHeader(),officialReadOnlyHandle:true},null,2)+'\n');
 console.log('PASS: official read-only handle; events='+events.length+'; compaction summaries='+Number(Boolean(summary)));
}finally{if(handle)await handle.close();await ctx.fiber.dispose();}
