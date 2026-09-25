/** Clean 1440px product shots; no hidden/mocked result data. */
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const [BASE,OUT]=process.argv.slice(2);if(!OUT)throw Error('BASE OUT required');fs.mkdirSync(OUT,{recursive:true});
(async()=>{const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--disable-gpu']});
try{for(const theme of ['light','dark']){
const context=await browser.newContext({viewport:{width:1440,height:1000},colorScheme:theme});const page=await context.newPage();await page.goto(BASE);await page.waitForFunction(()=>!document.querySelector('#try-demo').disabled);await page.evaluate(()=>document.fonts.ready);
async function settle(){await page.evaluate(async()=>{for(const i of document.images)i.loading='eager';await Promise.all([...document.images].map(i=>i.decode().catch(()=>{})));await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));});}
async function shoot(name,section){await settle();await page.locator(section).evaluate(el=>el.scrollIntoView({behavior:'instant',block:'start'}));await page.screenshot({path:path.join(OUT,`1440-${theme}-${name}.png`)});}
await shoot('cover','#home');await page.locator('#try-demo').click();await page.waitForFunction(()=>!document.querySelector('#run').disabled);await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#export').hidden);await shoot('search','#workspace');
await page.locator('#cards .act-compare').first().click();await shoot('comparison','#compare');await page.locator('#compare-explain').click();await page.waitForFunction(()=>document.querySelector('#ex-facts').children.length>0);await shoot('explanation','#explain');
await page.locator('.demo-example[data-id="edge"]').click();await page.waitForFunction(()=>!document.querySelector('#run').disabled);await page.locator('#run').click();await page.waitForFunction(()=>!document.querySelector('#refusal').hidden);await shoot('refusal','#workspace');await context.close();
}}finally{await browser.close();}console.log('10 real product screenshots saved');})().catch(e=>{console.error(e);process.exitCode=1;});
