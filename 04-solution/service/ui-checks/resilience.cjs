/** Browser controls for client resilience. The API replies here are deliberately
 * simulated; real inference and export are covered by behavior/acceptance.cjs.
 * No server is started and no request leaves this browser context.
 * PLAYWRIGHT_MODULE=/path/to/playwright node resilience.cjs OUT_DIRECTORY
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const OUT = process.argv[2];
if (!OUT) throw Error('Usage: node resilience.cjs OUTPUT_DIRECTORY');
fs.mkdirSync(OUT, {recursive:true});
const staticDir = path.resolve(__dirname, '../app/static');
const base = 'http://127.0.0.1:18726';
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/8mEAAAAASUVORK5CYII=', 'base64');
const source = Object.fromEntries(['index.html','app.css','app.js'].map(name =>
  [name, crypto.createHash('sha256').update(fs.readFileSync(path.join(staticDir,name))).digest('hex')]));
const results = {head:require('node:child_process').execFileSync('git',['-C',path.resolve(__dirname,'../../..'),'rev-parse','HEAD'],{encoding:'utf8'}).trim(),
  source_sha256:source, node:process.version, playwright:require((process.env.PLAYWRIGHT_MODULE || 'playwright')+'/package.json').version,
  command:[process.execPath,...process.argv.slice(1)], cases:[], externalRequests:[], browserErrors:[]};

function state(overrides={}) {
  return {service:'0.2.0',model:'d1_j48',score_scale:'cosine',default_threshold:0.5,
    storage_reachable:true,gallery_points:750,images_available:true,explain_available:true,...overrides};
}
async function setup(browser, settings={}) {
  const context = await browser.newContext({viewport:{width:1000,height:800},acceptDownloads:true});
  const page = await context.newPage();
  const env = {state:state(settings.state), searchStatus:200, stateFailure:false};
  page.setDefaultTimeout(6000);
  page.on('pageerror', error => results.browserErrors.push(error.message));
  page.on('request', request => {if (!request.url().startsWith(base+'/') && !request.url().startsWith('data:') && !request.url().startsWith('blob:')) results.externalRequests.push(request.url());});
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin !== base) return route.abort();
    const endpoint=url.pathname;
    if (endpoint==='/' || endpoint==='/index.html') return route.fulfill({status:200,contentType:'text/html; charset=utf-8',body:fs.readFileSync(path.join(staticDir,'index.html'))});
    if (endpoint==='/static/app.css') return route.fulfill({status:200,contentType:'text/css',body:fs.readFileSync(path.join(staticDir,'app.css'))});
    if (endpoint==='/static/app.js') return route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(staticDir,'app.js'))});
    if (endpoint.startsWith('/static/fonts/')) return route.fulfill({status:200,contentType:'font/woff2',body:fs.readFileSync(path.join(staticDir,'fonts',path.basename(endpoint)))});
    if (endpoint==='/api/ui/state') return env.stateFailure
      ? route.abort('connectionrefused') : route.fulfill({json:env.state});
    if (endpoint==='/api/demo') return route.fulfill({json:{examples:[]}});
    if (endpoint==='/api/materials') return route.fulfill({json:{items:[],deployment_status:'DEPLOY PENDING'}});
    if (endpoint==='/api/search') {
      if (env.searchStatus==='network') return route.abort('connectionrefused');
      return route.fulfill(env.searchStatus===409
        ? {status:409,json:{detail:'галерея не загружена'}}
        : env.searchStatus===503
          ? {status:503,contentType:'text/html',body:'<h1>gateway unavailable</h1>'}
          : {json:{refusal:false,threshold:0.5,best_confidence:0.9,candidates:[
            {gallery_id:'missing-frame',confidence:0.9},{gallery_id:'working-frame',confidence:0.8}]}});
    }
    if (endpoint.startsWith('/api/gallery/missing-frame/')) return route.fulfill({status:404,body:'not found'});
    if (endpoint.startsWith('/api/gallery/working-frame/')) return route.fulfill({status:200,contentType:'image/png',body:pixel});
    return route.fulfill({status:404,body:'not found'});
  });
  await page.goto(base+'/');
  await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));
  await page.locator('#file').setInputFiles({name:'query.png',mimeType:'image/png',buffer:pixel});
  await page.waitForFunction(()=>document.querySelector('#file-name').textContent==='query.png');
  await page.locator('#whole-frame').click();
  return {context,page,env};
}
async function search(page) {
  await page.locator('#run').click();
  await page.locator('#export').waitFor({state:'visible'});
}
async function check(name,browser,fn) {
  let context,page,env;
  try {
    ({context,page,env}=await setup(browser, name==='model-change-clears-export' ? {state:{images_available:false}} : {}));
    await fn(page,env);
    results.cases.push({name,status:'PASS'});
    console.log('PASS',name);
  } catch (error) {
    results.cases.push({name,status:'FAIL',error:error.message});
    console.log('FAIL',name,error.message);
    if (page) await page.screenshot({path:path.join(OUT,name+'.png'),fullPage:true});
  } finally {if (context) await context.close();}
}
(async()=>{
  const browser=await chromium.launch({executablePath:process.env.CHROME||'/usr/bin/google-chrome',headless:true,args:['--disable-gpu']});
  results.browser=browser.version();
  try {
    await check('missing-comparison-frame',browser,async page=>{
      await search(page);
      await page.locator('#cards .card').first().locator('.noimg').waitFor({state:'visible'});
      await page.locator('#cards .card').first().locator('.act-compare').click();
      await page.locator('#compare-image-status').waitFor({state:'visible'});
      await page.waitForFunction(()=>document.querySelector('#compare-image-status').textContent.includes('недоступен'));
      assert.match(await page.locator('#compare-image-status').textContent(),/недоступен/);
      assert(await page.locator('#compare-g').isHidden());
      await page.locator('#cards .card').nth(1).locator('.act-compare').click();
      await page.waitForFunction(()=>document.querySelector('#compare-g').complete && document.querySelector('#compare-g').naturalWidth>0);
      assert(await page.locator('#compare-image-status').isHidden());
      assert(await page.locator('#compare-g').isVisible());
    });
    await check('search-409-refreshes-gallery-state',browser,async (page,env)=>{
      await search(page);
      env.searchStatus=409;
      env.state=state({gallery_points:0});
      await page.locator('#run').click();
      await page.locator('#error').waitFor({state:'visible'});
      await page.waitForFunction(()=>document.querySelector('#state-chip').textContent.includes('готовится'));
      assert(await page.locator('#run').isDisabled());
      assert(await page.locator('#export').isHidden());
      env.state=state(); env.searchStatus=200;
      await page.locator('#retry-state').click();
      await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));
      assert(await page.locator('#run').isEnabled());
      await search(page);
      assert(await page.locator('#export').isVisible());
    });
    await check('model-change-clears-export',browser,async (page,env)=>{
      await search(page);
      env.state=state({model:'next-model',images_available:false});
      await page.locator('#retry-state').click();
      await page.waitForFunction(()=>document.querySelector('#foot-model').textContent==='next-model');
      assert(await page.locator('#export').isHidden(),'previous model response must not be exported with new model label');
      assert(await page.locator('#run').isEnabled());
      assert.match(await page.locator('#banner-body').textContent(),/повторите поиск/i);
    });
    await check('html-503-refreshes-gallery-state',browser,async (page,env)=>{
      await search(page);
      env.searchStatus=503;
      env.state=state({storage_reachable:false,gallery_points:null});
      await page.locator('#run').click();
      await page.waitForFunction(()=>document.querySelector('#state-chip').textContent.includes('недоступна'));
      assert.match(await page.locator('#error').textContent(),/503/);
      assert(await page.locator('#run').isDisabled());
      assert(await page.locator('#export').isHidden());
    });
    await check('network-failure-refreshes-service-state',browser,async (page,env)=>{
      await search(page);
      env.searchStatus='network'; env.stateFailure=true;
      await page.locator('#run').click();
      await page.waitForFunction(()=>document.querySelector('#state-chip').textContent.includes('Сервис недоступен'));
      assert(await page.locator('#run').isDisabled());
      assert(await page.locator('#export').isHidden());
    });
    await check('export-rechecks-ready-model',browser,async (page,env)=>{
      await search(page);
      let downloads=0;
      page.on('download',()=>downloads++);
      const positive=page.waitForEvent('download');
      await page.locator('#exp-json').click();
      await positive;
      assert.equal(downloads,1,'unchanged model must export');
      env.state=state({model:'next-model'});
      await page.locator('#exp-json').click();
      await page.locator('#export').waitFor({state:'hidden'});
      assert.equal(downloads,1,'old response must not be exported after model change');
      assert.match(await page.locator('#banner-body').textContent(),/повторите поиск/i);
      await search(page);
      const updated=page.waitForEvent('download');
      await page.locator('#exp-json').click();
      const file=await updated;
      const body=JSON.parse(fs.readFileSync(await file.path(),'utf8'));
      assert.equal(body.model,'next-model');
    });
    await check('ready-result-polls-model-change',browser,async (page,env)=>{
      await search(page);
      env.state=state({model:'next-model'});
      await page.locator('#export').waitFor({state:'hidden',timeout:22000});
      assert.match(await page.locator('#banner-body').textContent(),/повторите поиск/i);
    });
    await check('export-blocked-when-state-unavailable',browser,async (page,env)=>{
      await search(page);
      let downloads=0;
      page.on('download',()=>downloads++);
      env.stateFailure=true;
      await page.locator('#exp-json').click();
      await page.waitForFunction(()=>document.querySelector('#state-chip').textContent.includes('Сервис недоступен'));
      assert.equal(downloads,0,'export must not use an unverified old result');
      assert(await page.locator('#export').isHidden());
      assert.match(await page.locator('#banner-body').textContent(),/Проверьте соединение/);
    });
  } finally {await browser.close();}
  results.passed=results.cases.filter(x=>x.status==='PASS').length;
  results.failed=results.cases.length-results.passed;
  fs.writeFileSync(path.join(OUT,'resilience.json'),JSON.stringify(results,null,2)+'\n');
  console.log(JSON.stringify({passed:results.passed,failed:results.failed,external:results.externalRequests.length,browserErrors:results.browserErrors.length}));
  process.exitCode=results.failed||results.externalRequests.length||results.browserErrors.length?1:0;
})().catch(error=>{console.error(error);process.exitCode=2;});
