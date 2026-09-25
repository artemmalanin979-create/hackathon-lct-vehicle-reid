/** Metadata only. Does not start services, mutate the checkout, or read secrets. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process');
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
module.exports=function evidence(data){
 const root=path.resolve(__dirname,'../../..'),staticDir=path.resolve(__dirname,'../app/static');
 const git=args=>cp.execFileSync('git',['-C',root,...args],{encoding:'utf8'}).trim();
 const sources={};for(const name of ['index.html','app.css','app.js'])sources[name]=sha(path.join(staticDir,name));
 const inputs={};for(const name of ['test_query.csv','test_gallery.csv','images/a4f2a13bd2b54360a921c8ef7366e535.jpg']){const file=path.join(data,name);if(fs.existsSync(file))inputs[name]=sha(file);}
 return {head:git(['rev-parse','HEAD']),branch:git(['branch','--show-current']),source_sha256:sources,
  input_sha256:inputs,node:process.version,playwright:require((process.env.PLAYWRIGHT_MODULE||'playwright')+'/package.json').version,
  command:[process.execPath,...process.argv.slice(1)],filter:process.env.TEST_FILTER||null,
  source_override:process.env.UI_JS_OVERRIDE||null};
};
