// Local reproducible 450 dpi export of the vehicle request-and-cost table.
const {chromium}=require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),crypto=require('crypto');const {execFileSync}=require('child_process');
(async()=>{
 const root=path.resolve(__dirname,'../..'),dir=path.join(root,'data/exp/figure-table-generation/three_ward_transport_v1');
 const source=path.join(dir,'vehicle_request_and_common_cohort_cost_review.html'),output=path.join(root,'data/results/tables/Table_vehicle_request_and_common_cohort_cost.png');
 const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),before=digest(source);
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:2000,height:1400},deviceScaleFactor:450/96});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('file://'+source,{waitUntil:'load'});await page.evaluate(()=>document.fonts.ready);
 const state=await page.evaluate(()=>({rows:document.querySelectorAll('#main-table tbody tr').length,columns:document.querySelectorAll('#main-table th').length,notes:document.querySelectorAll('#main-table .notes p').length,overflow:document.documentElement.scrollWidth>innerWidth,links:Array.from(document.querySelectorAll('a')).map(a=>a.href),badCells:Array.from(document.querySelectorAll('#main-table td,#main-table th')).filter(e=>e.scrollWidth>e.clientWidth+1).length}));
 if(state.rows!==8||state.columns!==7||state.notes!==5||state.overflow||state.badCells||errors.length)throw Error(JSON.stringify({state,errors}));
 await page.locator('#main-table').screenshot({path:output});
 await page.screenshot({path:path.join(dir,'vehicle_request_and_common_cohort_cost_browser.png')});await browser.close();
 const py=`from PIL import Image\nimport sys,json,hashlib\np=sys.argv[1]\nwith Image.open(p) as im:\n    pixels=hashlib.sha256(im.tobytes()).hexdigest()\n    im.save(p,dpi=(450,450))\nwith Image.open(p) as im:\n    assert hashlib.sha256(im.tobytes()).hexdigest()==pixels\n    assert all(449<=d<=451 for d in im.info['dpi'])\n    print(json.dumps({'width':im.width,'height':im.height,'dpi':im.info['dpi']}))\n`;
 const info=JSON.parse(execFileSync(path.join(root,'.venv/bin/python'),['-c',py,output],{encoding:'utf8'}));
 for(const link of state.links)if(link.startsWith('file://')&&!fs.existsSync(decodeURIComponent(link.slice(7))))throw Error('Missing link '+link);
 if(before!==digest(source))throw Error('Source HTML changed');
 const receipt={passed:true,source:path.relative(root,source),output:path.relative(root,output),...state,errors,source_unchanged:true,pixel_resampling:false,source_sha256:before,output_sha256:digest(output),...info};
 fs.writeFileSync(path.join(dir,'vehicle_request_and_common_cohort_cost_png_export.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
})();
