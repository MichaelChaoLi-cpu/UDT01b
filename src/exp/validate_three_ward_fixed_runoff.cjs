const fs=require('fs'),path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
const root=path.resolve(__dirname,'../..'),out=path.join(root,'data/exp/data-briefing/three_ward_fixed_runoff');
const f=JSON.parse(fs.readFileSync(path.join(out,'findings.json'),'utf8'));
const b=JSON.parse(fs.readFileSync(path.join(out,'water_budgets.json'),'utf8'));
if(b.length!==6||!b.every(x=>x.file_hashes_unchanged&&x.passed&&x.rain_conservation_relative_error<1e-6))throw Error('Mass or input integrity audit failed');
for(const dx of [20,10]){
const base=path.join(root,'data/processed/three_ward_sfincs_framework/fixed_runoff_comparison');
const p=JSON.parse(fs.readFileSync(path.join(base,`pilot_${dx}m/forcing_validation.json`),'utf8'));
const t=JSON.parse(fs.readFileSync(path.join(base,`urban_storage253_${dx}m_template/storage_refinement_complete.json`),'utf8'));
if(!p.passed||t.max_face_old_knot_difference!==0||t.max_storage_old_knot_elevation_difference_m!==0)throw Error('Forcing/isolated-storage preflight failed');
}
if(f.formal_grid_convergence_established||f.event_validated)throw Error('Unsupported convergence/event assertion');
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true}),checks=[];
for(const width of [1280,390]){
const context=await browser.newContext({viewport:{width,height:960}}),page=await context.newPage();
await page.goto(pathToFileURL(path.join(out,'review.html')).href);await page.waitForLoadState('load');
const s=await page.evaluate(()=>({width:innerWidth,bodyWidth:document.documentElement.scrollWidth,images:[...document.images].map(x=>({src:x.getAttribute('src'),loaded:x.complete&&x.naturalWidth>0})),rows:document.querySelectorAll('tr').length,links:[...document.querySelectorAll('a[href]')].map(x=>x.href)}));
s.localLinks=s.links.filter(x=>x.startsWith('file:')).map(x=>({url:x,exists:fs.existsSync(decodeURIComponent(new URL(x).pathname))}));
s.passed=s.bodyWidth<=width+1&&s.images.length===1&&s.images.every(x=>x.loaded)&&s.rows===16&&s.localLinks.every(x=>x.exists);
await page.screenshot({path:path.join(out,width===1280?'preview_desktop.png':'preview_mobile.png'),fullPage:false});checks.push(s);await context.close();
}
await browser.close();
const result={passed:checks.every(x=>x.passed),checks,all_6_official_runs_complete:true,all_mass_budgets_passed:true,formal_convergence_established:false,event_validated:false};
fs.writeFileSync(path.join(out,'delivery_validation.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));if(!result.passed)process.exitCode=1;
})();
