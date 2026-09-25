/** Focused acceptance: real UI/API, explicit simulated transport/health failures.
 * No production writes. Browser profiles and recordings are owned by this run.
 */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const [BASE,DATA,OUT]=process.argv.slice(2);if(!OUT)throw Error('BASE DATA OUT required');
fs.mkdirSync(OUT,{recursive:true});
const report={started:new Date().toISOString(),cases:[],screens:[],external:[],errors:[]};
report.evidence=require("./evidence.cjs")(DATA);
let browser;
async function check(name,fn,options={}){
 if(process.env.TEST_FILTER&&!new RegExp(process.env.TEST_FILTER).test(name))return;
 const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true,...options});
 const page=await context.newPage(); const errors=[];
 page.on('request',r=>{const u=r.url();if(!u.startsWith(BASE+'/')&&!u.startsWith('blob:')&&!u.startsWith('data:'))report.external.push(u);});
 page.on('pageerror',e=>errors.push(e.message));
 try{await fn(page,context);assert.equal(errors.length,0,errors.join('\n'));report.cases.push({name,status:'PASS'});console.log('PASS',name);}
 catch(e){report.cases.push({name,status:'FAIL',error:e.message});console.log('FAIL',name,e.message);await page.screenshot({path:path.join(OUT,name+'-failure.png'),fullPage:true});}
 finally{report.errors.push(...errors);await context.close();}
}
async function open(page){await page.goto(BASE);await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));await page.evaluate(()=>document.fonts.ready);}
async function demo(page){await open(page);await page.locator('#try-demo').click();await page.waitForFunction(()=>!document.querySelector('#run').disabled);}
async function run(page){await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#export').hidden);}
async function shot(page,name){await page.evaluate(async()=>{await document.fonts.ready;for(const image of document.images)image.loading='eager';await Promise.all([...document.images].map(image=>image.decode().catch(()=>{})));});await page.screenshot({path:path.join(OUT,name+'.png'),fullPage:true});report.screens.push(name+'.png');}
async function noOverflow(page){assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1), 'no horizontal document overflow');}
async function hold(page,endpoint){
 await page.evaluate(endpoint=>{const original=window.fetch;window.__gate={ready:false};window.fetch=async(...args)=>{if(String(args[0])!==endpoint)return original(...args);window.fetch=original;const response=await original(...args);await new Promise(resolve=>{window.__gate.release=resolve;window.__gate.ready=true;});return response;};},endpoint);
}
(async()=>{browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--disable-gpu']});
report.evidence.browser=browser.version();
try{
 for(const width of [390,768,1440])for(const theme of ['light','dark'])await check(`layout-${width}-${theme}`,async page=>{
  await demo(page);await noOverflow(page);await shot(page,`${width}-${theme}-query`);
  await run(page);assert(await page.locator('#cards .card').count()>0);await noOverflow(page);
  await shot(page,`${width}-${theme}-search`);
  await page.locator('#cards .act-compare').first().click();await page.locator('#compare-g').waitFor({state:'visible'});
  await page.waitForFunction(()=>document.querySelector('#compare-g').complete);await noOverflow(page);await shot(page,`${width}-${theme}-compare`);
  await page.locator('#compare-view').click();assert((await page.locator('#compare-g').getAttribute('src')).includes('view=frame'));
  await page.locator('#compare-explain').click();await page.waitForFunction(()=>document.querySelector('#ex-facts').children.length>0);await noOverflow(page);await shot(page,`${width}-${theme}-explain`);
  await page.locator('.demo-example[data-id="edge"]').click();await page.waitForFunction(()=>!document.querySelector('#run').disabled);
  await hold(page,'/api/search');await page.locator('#run').click();await page.waitForFunction(()=>window.__gate.ready);await noOverflow(page);await shot(page,`${width}-${theme}-loading`);await page.evaluate(()=>window.__gate.release());
  await page.waitForFunction(()=>!document.querySelector('#refusal').hidden);await noOverflow(page);await shot(page,`${width}-${theme}-refusal`);
  await page.route('**/api/search',r=>r.abort('connectionrefused'),{times:1});await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#error').hidden);await noOverflow(page);await shot(page,`${width}-${theme}-error`);

 },{viewport:{width,height:1000},colorScheme:theme});
 await check('bbox-mouse-numeric-zoom',async page=>{
  await demo(page);await page.locator('#clear-box').click();
  const canvas=page.locator('#canvas');await canvas.scrollIntoViewIfNeeded();let box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width*.12,box.y+box.height*.2);await page.mouse.down();await page.mouse.move(box.x+box.width*.65,box.y+box.height*.7,{steps:8});await page.mouse.up();
  const before=await page.locator('#bx').inputValue();assert(Number(await page.locator('#bw').inputValue())>100);
  await page.mouse.move(box.x+box.width*.4,box.y+box.height*.4);await page.mouse.down();await page.mouse.move(box.x+box.width*.46,box.y+box.height*.45,{steps:6});await page.mouse.up();assert.notEqual(await page.locator('#bx').inputValue(),before,'pointer moves bbox');
  await page.locator('#bx').fill('999999');await page.locator('#bx').blur();assert.equal(await page.locator('#bx').inputValue(),'1919');assert.equal(await page.locator('#bw').inputValue(),'1');
  await page.locator('#whole-frame').click();assert.equal(await page.locator('#bx').inputValue(),'0');assert.equal(await page.locator('#bw').inputValue(),'1920');
  await page.evaluate(()=>{document.documentElement.style.zoom='2';window.dispatchEvent(new Event('resize'));});await noOverflow(page);await shot(page,'200percent-query');
  await page.locator('#clear-box').click();await canvas.scrollIntoViewIfNeeded();box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width*.2,box.y+box.height*.2);await page.mouse.down();await page.mouse.move(box.x+box.width*.6,box.y+box.height*.6,{steps:8});await page.mouse.up();
  assert(Math.abs(Number(await page.locator('#bx').inputValue())-384)<=2,'bbox X respects the rendered scale at 200%');
  assert(Math.abs(Number(await page.locator('#bw').inputValue())-768)<=2,'bbox width respects the rendered scale at 200%');

  await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#export').hidden);await noOverflow(page);await shot(page,'200percent-result');
 });
 await check('bbox-touch',async(page,context)=>{
  await demo(page);await page.locator('#clear-box').click();await page.locator('#canvas').scrollIntoViewIfNeeded();const box=await page.locator('#canvas').boundingBox();
  const cdp=await context.newCDPSession(page);
  const send=(type,x,y)=>cdp.send('Input.dispatchTouchEvent',{type,touchPoints:type==='touchEnd'?[]:[{x,y,id:1,radiusX:4,radiusY:4}]});
  await send('touchStart',box.x+box.width*.12,box.y+box.height*.2);
  for(let i=1;i<=8;i++)await send('touchMove',box.x+box.width*(.12+.55*i/8),box.y+box.height*(.2+.5*i/8));await send('touchEnd');
  assert(Number(await page.locator('#bw').inputValue())>500);assert(Number(await page.locator('#bh').inputValue())>100);await noOverflow(page);await shot(page,'390-touch-bbox');
 },{viewport:{width:390,height:844},isMobile:true,hasTouch:true});
 await check('loading-refusal-and-export',async page=>{
  await demo(page);await page.locator('#advanced > summary').click();await page.locator('#thr').fill('1.1');await hold(page,'/api/search');await page.locator('#run').click();
  await page.waitForFunction(()=>window.__gate.ready);assert(await page.locator('#search-loading').isVisible());assert(await page.locator('#export').isHidden());await shot(page,'loading');await page.evaluate(()=>window.__gate.release());
  await page.waitForFunction(()=>!document.querySelector('#refusal').hidden);assert.equal(await page.locator('#cards .card').count(),0);await shot(page,'refusal');
  const [json]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-json').click()]);const data=JSON.parse(fs.readFileSync(await json.path(),'utf8'));assert.equal(data.refusal,true);assert.equal(data.candidates.length,0);assert.equal(data.threshold,1.1);
  const [csv]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-csv').click()]);const text=fs.readFileSync(await csv.path(),'utf8');assert.equal(text.trim().split('\n').length,2,'CSV refusal keeps one row');
  await page.locator('#show-below').click();await page.waitForFunction(()=>document.querySelectorAll('#cards .below').length>0);assert(await page.locator('#refusal').isVisible());await shot(page,'below-threshold');
 });
 await check('invalid-file-and-search-error',async page=>{
  await demo(page);await run(page);
  await page.locator('#file').setInputFiles({name:'bad.txt',mimeType:'text/plain',buffer:Buffer.from('not an image')});assert(await page.locator('#error').isVisible());assert(await page.locator('#export').isHidden());assert(await page.locator('#run').isHidden());await shot(page,'invalid-file');
  await page.locator('#try-demo').click();await page.waitForFunction(()=>!document.querySelector('#run').disabled);
  await page.route('**/api/search',r=>r.abort('connectionrefused'),{times:1});await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#error').hidden);assert(await page.locator('#export').isHidden());await shot(page,'network-error');
 });
 await check('explain-error-preserves-search',async page=>{
  await demo(page);await run(page);const count=await page.locator('#cards .card').count();
  await page.route('**/api/explain',r=>r.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'controlled unavailable explanation'})}),{times:1});
  await page.locator('#cards .act-explain').first().click();await page.waitForFunction(()=>document.querySelector('#explain-lead').textContent.includes('503'));
  assert(await page.locator('#export').isVisible());assert.equal(await page.locator('#cards .card').count(),count);assert(await page.locator('#explain-body').isHidden());await shot(page,'explain-error');
 });
 await check('empty-gallery-recovers',async page=>{
  await page.route('**/api/ui/state',async r=>{const response=await r.fetch();const data=await response.json();data.gallery_points=0;await r.fulfill({json:data});},{times:1});
  await page.goto(BASE);await page.waitForFunction(()=>!document.querySelector('#banner').hidden);await shot(page,'gallery-preparing');
  await page.locator('#retry-state').click();await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));assert(await page.locator('#banner').isHidden());
 });
 await check('explain-waits-for-gallery-pixels',async page=>{
  await demo(page);await run(page);let release,arrive;const arrived=new Promise(r=>arrive=r),gate=new Promise(r=>release=r);
  await page.route('**/api/gallery/*/crop?size=416',async route=>{const response=await route.fetch();arrive();await gate;await route.fulfill({response});},{times:1});
  await page.locator('#cards .act-explain').first().click();await arrived;
  const factsBeforePixels=await page.locator('#ex-facts').textContent();
  release();assert.equal(factsBeforePixels,'','explanation must not claim completion before candidate pixels load');await page.waitForFunction(()=>document.querySelector('#ex-facts').children.length>0);
  const colors=await page.locator('#ex-g').evaluate(canvas=>{const data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;const colors=new Set();for(let i=0;i<data.length;i+=128)colors.add(`${data[i]},${data[i+1]},${data[i+2]}`);return colors.size;});assert(colors>100,'candidate explanation contains real image data');
 });
 await check('below-json-ranks-match-screen',async page=>{
  await demo(page);await run(page);assert(await page.locator('#cards .card:not(.below)').count()>0);
  await page.locator('#show-below-2').click();await page.waitForFunction(()=>document.querySelectorAll('#cards .below').length>0);
  const rows=await page.locator('#cards .below').evaluateAll(items=>items.map(item=>({gallery_id:item.dataset.gid,rank:Number(item.querySelector('.rank').textContent)})));
  const [json]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-json').click()]);const data=JSON.parse(fs.readFileSync(await json.path(),'utf8'));
  assert.deepEqual(data.below_threshold.map(({gallery_id,rank})=>({gallery_id,rank})),rows,'JSON ranks correspond to the displayed full top-k ranks');
 });
 await check('demo-load-disables-search',async page=>{
  await demo(page);await run(page);
  await page.evaluate(()=>{
    const original=window.fetch;window.__demoGate={ready:false};
    window.fetch=async(...args)=>{
      if(!String(args[0]).includes('/api/demo/edge/'))return original(...args);
      window.fetch=original;const response=await original(...args);
      await new Promise(resolve=>{window.__demoGate.release=resolve;window.__demoGate.ready=true;});return response;
    };
  });
  await page.locator('.demo-example[data-id="edge"]').click();await page.waitForFunction(()=>window.__demoGate.ready);
  assert(await page.locator('#run').isDisabled(),'search stays disabled while a new example is loading');assert(await page.locator('#export').isHidden());
  await page.evaluate(()=>window.__demoGate.release());await page.waitForFunction(()=>document.querySelector('#file-name').textContent.includes('edge')&&!document.querySelector('#run').disabled);
  await run(page);assert(await page.locator('#refusal').isVisible());
 });
 await check('long-cyrillic-filename-export',async page=>{
  await open(page);const filename='Кадр автомобиля, "проверка" — '+ 'длинное_название_'.repeat(8)+'.jpg';
  await page.locator('#file').setInputFiles({name:filename,mimeType:'image/jpeg',buffer:fs.readFileSync(path.join(DATA,'images','a4f2a13bd2b54360a921c8ef7366e535.jpg'))});
  await page.waitForFunction(()=>!document.querySelector('#stage').hidden);await page.locator('#whole-frame').click();await run(page);await noOverflow(page);
  const [json]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-json').click()]);const data=JSON.parse(fs.readFileSync(await json.path(),'utf8'));assert.equal(data.query.image,filename);
  const [csv]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-csv').click()]);assert(fs.readFileSync(await csv.path(),'utf8').includes('""проверка""'));await shot(page,'390-long-filename');
 },{viewport:{width:390,height:844}});
 await check('missing-demo-keyboard-theme',async page=>{
  await page.route('**/api/demo',r=>r.fulfill({json:{examples:[]}}));await open(page);assert(await page.locator('#try-demo').isDisabled());assert((await page.locator('#demo-status').textContent()).includes('недоступны'));
  await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.activeElement.className),'skip-link');await page.keyboard.press('Enter');
  await page.locator('#theme-btn').focus();await page.keyboard.press('Enter');assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');await page.reload();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).scrollBehavior),'auto');await shot(page,'no-demo-keyboard');
 },{colorScheme:'light'});
}finally{await browser.close();}
report.finished=new Date().toISOString();report.passed=report.cases.filter(c=>c.status==='PASS').length;report.failed=report.cases.length-report.passed;report.external=[...new Set(report.external)];fs.writeFileSync(path.join(OUT,'acceptance.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,failed:report.failed,external:report.external.length}));process.exitCode=report.failed||report.external.length?1:0;
})().catch(e=>{console.error(e);process.exitCode=2;});
