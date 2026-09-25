/** Screenshot evidence, always from the real running service. */
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const [BASE,DATA,OUT]=process.argv.slice(2);if(!OUT)throw Error('BASE DATA OUT required');
fs.mkdirSync(OUT,{recursive:true});
(async()=>{
const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--disable-gpu']});
const findings=[];
try{for(const width of [390,768,1440])for(const theme of ['light','dark']){
const ctx=await browser.newContext({viewport:{width,height:1000},colorScheme:theme});const page=await ctx.newPage();
await page.goto(BASE);await page.waitForFunction(()=>document.querySelector('#state-chip').classList.contains('chip-ok'));
await page.evaluate(()=>document.fonts.ready);await page.screenshot({path:path.join(OUT,`${width}-${theme}-start.png`),fullPage:true});
await page.locator('#file').setInputFiles(path.join(DATA,'images','a4f2a13bd2b54360a921c8ef7366e535.jpg'));
await page.waitForFunction(()=>!document.querySelector('#stage').hidden);
for(const [id,val] of Object.entries({bx:849,by:300,bw:736,bh:471})){await page.locator('#'+id).fill(String(val));await page.locator('#'+id).blur();}
await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#export').hidden);
await page.waitForFunction(()=>[...document.querySelectorAll('#cards img')].every(i=>i.complete));
await page.screenshot({path:path.join(OUT,`${width}-${theme}-search.png`),fullPage:true});
const act=page.locator('#cards .act-explain').first();await act.click();
await page.waitForFunction(()=>document.querySelector('#ex-facts').children.length>0);
await page.screenshot({path:path.join(OUT,`${width}-${theme}-explain.png`),fullPage:true});
findings.push({width,theme,overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),cards:await page.locator('#cards .card').count()});
await ctx.close();
}}finally{await browser.close();}
fs.writeFileSync(path.join(OUT,'capture.json'),JSON.stringify(findings,null,2));console.log(JSON.stringify(findings));
})().catch(e=>{console.error(e);process.exitCode=1;});
