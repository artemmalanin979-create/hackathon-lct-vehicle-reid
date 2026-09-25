/** One semantic mutation, only in a disposable local JS copy served to our own
 * browser context. Real local API is read-only. Never touches production code.
 * Order: baseline GREEN → non-empty diff → semantic FAIL → source hash → GREEN.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),assert=require('node:assert/strict');
const [BASE,DATA,OUT]=process.argv.slice(2);if(!OUT)throw Error('BASE DATA OUT required');
assert(['127.0.0.1','localhost'].includes(new URL(BASE).hostname),'mutations require a loopback rehearsal API');
fs.mkdirSync(OUT,{recursive:true});
const source=path.resolve(__dirname,'../app/static/app.js'),original=fs.readFileSync(source),hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const target='    if (revision !== app.revision || controller.signal.aborted) return;';
assert.equal(original.toString().split(target).length-1,1,'exactly one intended guard');
const command=[path.join(__dirname,'behavior.cjs'),BASE,DATA];
function run(label,override){
 const folder=path.join(OUT,label);const env={...process.env,TEST_FILTER:'positive-search-export|late-search-after-bbox'};
 if(override)env.UI_JS_OVERRIDE=override;else delete env.UI_JS_OVERRIDE;
 const r=cp.spawnSync(process.execPath,[...command,folder],{env,encoding:'utf8',timeout:180000});
 fs.writeFileSync(path.join(OUT,label+'.log'),(r.stdout||'')+(r.stderr||''));
 if(r.error)throw r.error;
 return {exit:r.status,...JSON.parse(fs.readFileSync(path.join(folder,'behavior.json'),'utf8'))};
}
const baseline=run('baseline');assert.equal(baseline.exit,0);assert.equal(baseline.passed,2);
const disposable=fs.mkdtempSync(path.join(OUT,'copy-')),mutant=path.join(disposable,'app.js');
const modified=original.toString().replace(target,'    // MUTATION: accept an obsolete parsed response.');
fs.writeFileSync(mutant,modified);assert.notEqual(hash(original),hash(Buffer.from(modified)));
const diff=cp.spawnSync('diff',['-u',source,mutant],{encoding:'utf8'});assert.equal(diff.status,1);assert(diff.stdout.includes('-'+target));fs.writeFileSync(path.join(OUT,'mutation.diff'),diff.stdout);
const negative=run('negative',mutant);assert.equal(negative.exit,1);assert.equal(negative.passed,1);assert.equal(negative.failed,1);
assert(negative.cases.some(c=>c.name==='late-search-after-bbox'&&c.status==='FAIL'&&c.error.includes('changed query')),'expected semantic assertion killed the mutation');
assert.equal(hash(fs.readFileSync(source)),hash(original),'production source unchanged');
const restored=run('restored');assert.equal(restored.exit,0);assert.equal(restored.passed,2);
fs.writeFileSync(path.join(OUT,'mutation.json'),JSON.stringify({source,source_sha256:hash(original),mutant_sha256:hash(Buffer.from(modified)),baseline:{exit:baseline.exit,passed:baseline.passed},negative:{exit:negative.exit,passed:negative.passed,failed:negative.failed},restored:{exit:restored.exit,passed:restored.passed},status:'KILLED: semantic stale export assertion'},null,2));
console.log('PASS: baseline 2/2 → nonempty JS diff → stale export FAIL → source hash unchanged → baseline 2/2');
