/** Real-browser regression contract. No fake search results: delayed responses
 * come from native fetch() against the running API. Fault cases only inject
 * transport errors. Evidence goes to an explicit directory outside source.
 * PLAYWRIGHT_MODULE=/path/to/playwright node behavior.cjs BASE DATA OUT
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const BASE = process.argv[2] || 'http://127.0.0.1:18070';
const DATA = process.argv[3];
const OUT = process.argv[4];
if (!DATA || !OUT) throw Error('Usage: node behavior.cjs BASE DATA_DIRECTORY OUTPUT_DIRECTORY');
fs.mkdirSync(OUT, {recursive:true});
const qid='a4f2a13bd2b54360a921c8ef7366e535';
const frame=path.join(DATA,'images',qid+'.jpg');
const other=path.join(DATA,'images','5cfbbd42352245fb9ab4e93f0e17452a.jpg');
const result={base:BASE,started:new Date().toISOString(),cases:[],requests:[],browserErrors:[]};
let browser;
async function fillBox(page, x=849) {
  // Exact input through the same number-field event an operator uses.
  for(const [id,v] of Object.entries({bx:x,by:300,bw:736,bh:471})) {
    await page.locator('#'+id).fill(String(v));
    await page.locator('#'+id).blur();
  }
}
async function ready(page) {
  await page.goto(BASE);
  await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));
  await page.locator('#file').setInputFiles(frame);
  await page.waitForFunction(()=>document.querySelector('#file-name').textContent.includes('a4f2a13b'));
  await fillBox(page);
}
async function search(page) {
  await page.locator('#run').click();
  await page.waitForFunction(()=>!document.querySelector('#export').hidden);
  assert(await page.locator('#cards .card').count()>0,'positive real API search has candidates');
}
async function hold(page, endpoint) {
  // Delay delivery AFTER the actual browser fetch: replaying Chromium multipart
  // uploads through route.fetch loses their file part in this Playwright version.
  await page.evaluate(endpoint => {
    const original=window.fetch;
    window.__held={ready:false,data:null,release:null};
    window.fetch=async (...args)=>{
      if(String(args[0])!==endpoint) return original(...args);
      window.fetch=original;
      const response=await original(...args);
      window.__held.data={status:response.status,body:await response.clone().json()};
      window.__held.ready=true;
      await new Promise(resolve=>{window.__held.release=resolve;});
      return response;
    };
  }, endpoint);
  return {
    arrived: page.waitForFunction(()=>window.__held.ready).then(()=>page.evaluate(()=>({
      transportError: window.__held.data.status!==200, detail:window.__held.data
    }))),
    release: ()=>page.evaluate(()=>window.__held.release()),
  };
}
async function noExport(page) {
  assert(await page.locator('#export').isHidden(),'changed query must not retain/export the previous response');
}
async function test(name, fn) {
  const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
  const page=await context.newPage();
  page.on('request',r=>result.requests.push(r.url()));
  page.on('pageerror',e=>result.browserErrors.push({test:name,error:e.message}));
  try {await ready(page);await fn(page);assert.equal(result.browserErrors.filter(e=>e.test===name).length,0,'no uncaught browser error');result.cases.push({name,status:'PASS'});console.log('PASS',name);}
  catch(e){result.cases.push({name,status:'FAIL',error:e.message});console.log('FAIL',name,e.message);await page.screenshot({path:path.join(OUT,name+'.png'),fullPage:true});}
  finally {await context.close();}
}
(async()=>{
 browser=await chromium.launch({executablePath:process.env.CHROME || '/usr/bin/google-chrome',headless:true,args:['--disable-gpu']});
 try {
  await test('positive-search-export',async page=>{
    await search(page);
    const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-json').click()]);
    const file=await download.path();const data=JSON.parse(fs.readFileSync(file,'utf8'));
    assert.equal(data.query.x,849); assert.equal(data.query.image,qid+'.jpg');
    assert(data.candidates.length>0);assert.equal(data.score_scale,'cosine');
    await page.screenshot({path:path.join(OUT,'positive-search.png'),fullPage:true});
  });
  await test('bbox-invalidates-export',async page=>{await search(page);await page.locator('#bx').fill('850');await page.locator('#bx').blur();await noExport(page);});
  await test('late-search-after-bbox',async page=>{
    const gate=await hold(page,'/api/search');await page.locator('#run').click();assert(!(await gate.arrived).transportError,'delayed real API must succeed');
    await page.locator('#bx').fill('850');await page.locator('#bx').blur();await gate.release();
    await page.waitForTimeout(300);await noExport(page);
  });
  await test('late-search-after-file',async page=>{
    const gate=await hold(page,'/api/search');await page.locator('#run').click();assert(!(await gate.arrived).transportError,'delayed real API must succeed');
    await page.locator('#file').setInputFiles(other);await page.waitForFunction(()=>document.querySelector('#file-name').textContent.includes('5cfbbd42'));
    await gate.release();await page.waitForTimeout(300);await noExport(page);
  });
  await test('network-error-removes-prior-export',async page=>{
    await search(page);await page.route('**/api/search',route=>route.abort('connectionrefused'),{times:1});
    await page.locator('#run').click();await page.waitForTimeout(300);await noExport(page);
  });
  await test('late-below-after-bbox',async page=>{
    await page.locator('#thr').fill('1.1');await page.locator('#thr').blur();
    await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#refusal').hidden);
    const gate=await hold(page,'/api/search');await page.locator('#show-below').click();assert(!(await gate.arrived).transportError,'delayed real API must succeed');
    await page.locator('#bx').fill('850');await page.locator('#bx').blur();await gate.release();
    await page.waitForTimeout(300);await noExport(page);assert.equal(await page.locator('#cards .card').count(),0);
  });
  await test('late-explain-after-file',async page=>{
    await search(page);const gate=await hold(page,'/api/explain');await page.locator('#cards .act-explain').first().click();assert(!(await gate.arrived).transportError,'delayed real API must succeed');
    await page.locator('#file').setInputFiles(other);await page.waitForFunction(()=>document.querySelector('#file-name').textContent.includes('5cfbbd42'));
    await gate.release();await page.waitForTimeout(300);await noExport(page);
    assert(await page.locator('#explain').isHidden(),'stale explanation is hidden');
    assert.equal(await page.locator('#ex-facts').textContent(),'','stale explanation does not write facts after frame change');
  });
 } finally {await browser.close();}
 result.finished=new Date().toISOString();result.passed=result.cases.filter(c=>c.status==='PASS').length;result.failed=result.cases.length-result.passed;
 result.externalRequests=[...new Set(result.requests.filter(u=>!u.startsWith(BASE+'/')&&!u.startsWith('blob:')&&!u.startsWith('data:')))];
 fs.writeFileSync(path.join(OUT,'behavior.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:result.passed,failed:result.failed,external:result.externalRequests.length}));
 process.exitCode=result.failed||result.externalRequests.length?1:0;
})().catch(e=>{console.error(e);process.exitCode=2;});
