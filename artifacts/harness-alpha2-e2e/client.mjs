// Genuine official browser client automation; no provider/session substitution.
import {chromium} from '/tmp/ds41f-alpha2-e2e/client/node_modules/playwright/index.mjs';
import {readFileSync,writeFileSync,existsSync,appendFileSync} from 'node:fs';
const out='/Volumes/SDXC-512/ds41f-mlx/artifacts/harness-alpha2-e2e',root='/tmp/ds41f-alpha2-e2e';
const browser=await chromium.launch({headless:true});
const ctx=await browser.newContext({viewport:{width:1440,height:1100},storageState:root+'/browser-state.json'});
const page=await ctx.newPage();
page.on('pageerror',e=>appendFileSync(out+'/browser-errors.jsonl',JSON.stringify({at:new Date().toISOString(),error:String(e)})+'\n'));
await page.goto(readFileSync(out+'/harness.log','utf8').match(/dsh web: (http[^\s]+)/)[1]);
await page.waitForTimeout(3000);
if(await page.getByRole('button',{name:'Continue',exact:true}).count())await page.getByRole('button',{name:'Continue',exact:true}).click();
writeFileSync(root+'/client-ready','ready');
let seq=Number(process.env.START_SEQ??0);const start=Date.now();
try{
while(!existsSync(root+'/client-stop')&&Date.now()-start<2400000){
 const path=root+'/action-'+seq+'.json';
 if(!existsSync(path)){await page.waitForTimeout(200);continue;}
 const a=JSON.parse(readFileSync(path,'utf8'));let error=null;
 appendFileSync(out+'/client-actions.jsonl',JSON.stringify({seq,at:new Date().toISOString(),action:a})+'\n');
 try{
 if(a.kind==='send'||a.kind==='command'){
  const box=page.locator('[contenteditable="true"]').first();await box.fill(a.text);
  if(a.kind==='command')await box.press('Enter');
  else await page.getByRole('button',{name:'Send message',exact:true}).click();
 }else if(a.kind==='text')await page.getByText(a.text,{exact:true}).first().click();
 else if(a.kind==='click')await page.getByRole(a.role??'button',{name:a.name,exact:true}).nth(a.index??0).click();
 else if(a.kind==='fill')await page.getByRole(a.role??'textbox',{name:a.name}).fill(a.text);
 else if(a.kind==='key')await page.keyboard.press(a.key);
 await page.waitForTimeout(a.wait??3000);
 }catch(e){error=String(e);}
 const text=await page.locator('body').innerText();
 const controls=await page.locator('input,textarea,[contenteditable],button').evaluateAll(es=>es.map(e=>({tag:e.tagName,role:e.getAttribute('role'),placeholder:e.getAttribute('placeholder'),aria:e.getAttribute('aria-label'),text:e.innerText?.slice(0,120)})));
 writeFileSync(out+'/ui-'+seq+'.json',JSON.stringify({seq,url:page.url(),error,text,controls},null,2));
 await page.screenshot({path:out+'/ui-'+seq+'.png'});
 writeFileSync(root+'/done-'+seq,JSON.stringify({error}));seq++;
}
}finally{await browser.close();}
