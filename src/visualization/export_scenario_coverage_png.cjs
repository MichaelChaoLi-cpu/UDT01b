// Render a six-column publication preview at native 450 dpi from the detailed source.
// Detailed eight-column HTML/workbook, research plan and review status are preserved.
const {chromium}=require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');const path=require('path');const crypto=require('crypto');const {execFileSync}=require('child_process');
(async()=>{
 const root=path.resolve(__dirname,'../..');const slug='scenario_coverage_and_evidence_limits';
 const source=path.join(root,'data/exp/figure-table-generation/three_ward_transport_v1',slug+'_review.html');
 const output=path.join(root,'data/results/tables','Table_'+slug+'.png');
 const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');const before=digest(source);
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1900,height:1300},deviceScaleFactor:450/96});
 await page.goto('file://'+source,{waitUntil:'load'});await page.evaluate(()=>document.fonts.ready);
 const presentation=await page.evaluate(()=>{
  const container=document.querySelector('#main-table');
  const table=container.querySelector('table');
  const rows=Array.from(table.rows);
  const original=rows.map(r=>Array.from(r.cells).map(c=>c.textContent.trim()));
  for(const row of rows){row.deleteCell(7);row.deleteCell(6);}
  const shown=rows.map(r=>Array.from(r.cells).map(c=>c.textContent.trim()));
  if(JSON.stringify(shown)!==JSON.stringify(original.map(r=>r.slice(0,6))))throw Error('Displayed values changed');
  document.title='Scenario Coverage';document.querySelector('h1').textContent='Scenario Coverage';
  document.querySelector('aside').textContent='Simplified preview: nine evidence blocks and six columns. Detailed qualification and limitations remain in the linked workbook and supporting notes. See the hash-bound review status above.';
  const note=document.createElement('p');note.className='note';note.style.cssText='font-size:12px;line-height:1.5;margin:0;padding:10px;background:white;border:1px solid #cfdbe3';
  note.textContent='Notes: Counts overlap across blocks and must not be added as independent storms. Water stages and interventions are scenario assumptions. Field flood and congestion accuracy remain unverified; resident-hours do not count unique stranded people.';
  container.appendChild(note);
  return {columns:shown[0].length,rows:shown.length-1,retained_values_verified:true};
 });
 const compact=source.replace('_review.html','_compact_review.html');
 fs.writeFileSync(compact,await page.content());
 const table=page.locator('#main-table');const count=await table.evaluate(e=>({rows:e.querySelectorAll('tbody tr').length,columns:e.querySelectorAll('thead th').length}));
 if(count.rows!==9||count.columns!==6)throw Error('Unexpected main-table shape');
 await table.screenshot({path:output});await browser.close();
 const py=`from PIL import Image\nimport sys,json,hashlib\np=sys.argv[1]\nwith Image.open(p) as im:\n    pixels=hashlib.sha256(im.tobytes()).hexdigest()\n    im.save(p,dpi=(450,450))\nwith Image.open(p) as im:\n    assert hashlib.sha256(im.tobytes()).hexdigest()==pixels\n    assert all(449<=d<=451 for d in im.info['dpi'])\n    print(json.dumps({'width':im.width,'height':im.height,'dpi':im.info['dpi']}))\n`;
 const info=JSON.parse(execFileSync(path.join(root,'.venv/bin/python'),['-c',py,output],{encoding:'utf8'}));
 if(before!==digest(source))throw Error('Source HTML changed');
 const receipt={passed:true,source:path.relative(root,source),output:path.relative(root,output),rows:9,columns:6,detailed_columns:8,omitted_display_columns:["Qualification status","Evidence limits"],detailed_source_unchanged:true,retained_values_verified:presentation.retained_values_verified,compact_review:path.relative(root,compact),pixel_resampling:false,source_sha256:before,output_sha256:digest(output),...info};
 fs.writeFileSync(path.join(root,'data/exp/figure-table-generation/three_ward_transport_v1',slug+'_png_export.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
})();
