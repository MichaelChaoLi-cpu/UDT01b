const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
 const root = path.resolve(__dirname,'../..');
 const out = path.join(root,'data/exp/data-briefing/three_ward_urban_grid');
 const findings = JSON.parse(fs.readFileSync(path.join(out,'findings.json'),'utf8'));
 const budgets = JSON.parse(fs.readFileSync(path.join(out,'water_budgets.json'),'utf8'));
 if(budgets.length!==9 || !budgets.every(x=>x.file_hashes_unchanged && x.passed && x.rain_conservation_relative_error<.001)) throw Error('Numerical/input audit failed');
 const browser = await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 const checks=[];
 for(const width of [1280,390]){
  const context=await browser.newContext({viewport:{width,height:960}});
  const page=await context.newPage();
  await page.goto(pathToFileURL(path.join(out,'review.html')).href);
  await page.waitForLoadState('load');
  const status=await page.evaluate(()=>({width:innerWidth,bodyWidth:document.documentElement.scrollWidth,images:[...document.images].map(x=>({src:x.getAttribute('src'),loaded:x.complete&&x.naturalWidth>0})),rows:document.querySelectorAll('tbody tr').length,links:[...document.querySelectorAll('a[href]')].map(x=>x.href)}));
  status.localLinks=status.links.filter(x=>x.startsWith('file:')).map(x=>({url:x,exists:fs.existsSync(decodeURIComponent(new URL(x).pathname))}));
  status.passed=status.bodyWidth<=width+1&&status.images.length===3&&status.images.every(x=>x.loaded)&&status.rows>=28&&status.localLinks.every(x=>x.exists);
  await page.screenshot({path:path.join(out,width===1280?'preview_desktop.png':'preview_mobile.png'),fullPage:false});
  checks.push(status);
  await context.close();
 }
 await browser.close();
 const result={passed:checks.every(x=>x.passed),checks,model_event_validated:findings.event_validation,strict_internal_volume_nonnegative:findings.strict_internal_nonnegative_all,all_official_runs_complete:true,numerical_budget_checks_passed:findings.budgets_all_passed};
 fs.writeFileSync(path.join(out,'delivery_validation.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:result.passed,checks:checks.map(x=>({width:x.width,bodyWidth:x.bodyWidth,rows:x.rows,images:x.images.length,links:x.localLinks.length}))}));
 if(!result.passed)process.exitCode=1;
})();
