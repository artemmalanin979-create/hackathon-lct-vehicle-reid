/** Regression P2: pointer gestures must preserve all source-image coordinates.
 * Own browser contexts only, real demo/search API, no service mutations.
 */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [BASE, DATA, OUT] = process.argv.slice(2);
if (!OUT) throw Error('BASE DATA OUT required');
fs.mkdirSync(OUT, {recursive: true});
const report = {started:new Date().toISOString(), cases:[], errors:[], external:[], evidence:require('./evidence.cjs')(DATA)};
const keys = ['x','y','w','h'];
async function fields(page) { return page.evaluate(() => ['bx','by','bw','bh'].map(id=>document.getElementById(id).value)); }
function near(actual, expected, message) {
  assert(actual.every((n,i)=>Math.abs(Number(n)-expected[i])<=3), `${message}: actual ${actual}, expected ${expected}`);
}
(async () => {
 const browser = await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true});
 report.evidence.browser = browser.version();
 try {
  for (const theme of ['light','dark']) for (const zoom of [1,2]) for (const input of ['mouse','touch']) {
   const name = `${theme}-${zoom*100}-${input}`;
   if (process.env.TEST_FILTER && !new RegExp(process.env.TEST_FILTER).test(name)) continue;
   const item = {name};
   const context = await browser.newContext({viewport:{width:1440,height:1000},colorScheme:theme,hasTouch:input==='touch',acceptDownloads:true});
   const page = await context.newPage();
   await page.addInitScript(() => {
     const nativeFetch=window.fetch;
     window.fetch=function(url,options) {
       if (String(url)==='/api/search') window.__sentBox=['x','y','w','h'].map(key=>Number(options.body.get(key)));
       return nativeFetch.apply(this,arguments);
     };
   });
   page.on('pageerror',e=>report.errors.push(e.message));
   page.on('request',r=>{if (!r.url().startsWith(BASE+'/') && !/^(data|blob):/.test(r.url())) report.external.push(r.url());});
   try {
    await page.goto(BASE);
    await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));
    await page.evaluate(()=>document.fonts.ready);
    await page.locator('#try-demo').click();
    await page.waitForFunction(()=>!document.querySelector('#run').disabled);
    await page.evaluate(z=>{document.documentElement.style.zoom=String(z);window.dispatchEvent(new Event('resize'));},zoom);
    await page.locator('#clear-box').click();
    const canvas=page.locator('#canvas');
    const cdp=input==='touch'?await context.newCDPSession(page):null;
    const touch=(type,p)=>cdp.send('Input.dispatchTouchEvent',{type,touchPoints:type==='touchEnd'?[]:[{...p,id:1,radiusX:4,radiusY:4}]});
    async function gesture(from,to) {
      await canvas.scrollIntoViewIfNeeded(); const rect=await canvas.boundingBox();
      const point=([x,y])=>({x:rect.x+rect.width*x/1920,y:rect.y+rect.height*y/1080});
      const a=point(from),b=point(to);
      if(cdp) await touch('touchStart',a); else {await page.mouse.move(a.x,a.y);await page.mouse.down();}
      const down=await canvas.boundingBox();
      for(let step=1;step<=8;step++) {
        const p={x:a.x+(b.x-a.x)*step/8,y:a.y+(b.y-a.y)*step/8};
        if(cdp)await touch('touchMove',p);else await page.mouse.move(p.x,p.y);
      }
      const moving=await canvas.boundingBox();
      if(cdp)await touch('touchEnd');else await page.mouse.up();
      return {rect,down,moving,box:await fields(page)};
    }
    item.draw=await gesture([384,216],[1152,648]);
    near(item.draw.box,[384,216,768,432],'draw geometry');
    for(const stage of ['down','moving']) assert(Math.abs(item.draw[stage].y-item.draw.rect.y)<1,'canvas must not move vertically during draw');
    item.move=await gesture([768,432],[960,540]);
    near(item.move.box,[576,324,768,432],'move geometry');
    item.resize=await gesture([1344,756],[1536,864]);
    near(item.resize.box,[576,324,960,540],'resize geometry');
    item.click=await gesture([96,54],[96,54]);
    near(item.click.box,[576,324,960,540],'click without drag restores the previous bbox');
    const expected=item.click.box.map(Number);
    const responseWait=page.waitForResponse(r=>r.url()===BASE+'/api/search' && r.request().method()==='POST');
    await page.locator('#run').click();const response=await responseWait;
    assert(response.ok(),'real API accepted the request');
    item.sent=await page.evaluate(()=>window.__sentBox);
    assert.deepEqual(item.sent,expected,'real POST source bbox equals the fields');
    await page.waitForFunction(()=>!document.querySelector('#export').hidden);
    const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#exp-json').click()]);
    const exported=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
    item.exported=keys.map(key=>exported.query[key]);
    assert.deepEqual(item.exported,expected,'export snapshot equals the sent bbox');
    await page.locator('#clear-box').click();
    item.emptyClick=await gesture([96,54],[96,54]);
    assert.deepEqual(item.emptyClick.box,['','','',''],'empty click leaves no numeric bbox');
    assert(await page.locator('#run').isDisabled());
    assert.equal(await page.locator('#query-step').textContent(),'Выделите автомобиль');
    await page.screenshot({path:path.join(OUT,name+'.png'),fullPage:true});
    item.status='PASS'; console.log('PASS',name);
   } catch(e) { item.status='FAIL';item.error=e.message;console.log('FAIL',name,e.message);await page.screenshot({path:path.join(OUT,name+'-failure.png'),fullPage:true}); }
   finally {report.cases.push(item);await context.close();}
  }
 } finally {await browser.close();}
 report.finished=new Date().toISOString();report.passed=report.cases.filter(x=>x.status==='PASS').length;report.failed=report.cases.length-report.passed;
 fs.writeFileSync(path.join(OUT,'bbox-geometry.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({passed:report.passed,failed:report.failed,errors:report.errors.length,external:report.external.length}));
 process.exitCode=report.failed||report.errors.length||report.external.length?1:0;
})().catch(e=>{console.error(e);process.exitCode=2;});
