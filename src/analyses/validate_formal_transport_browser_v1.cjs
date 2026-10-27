const fs=require('fs');
const {chromium}=require('playwright');
(async()=>{
 const root=require('path').resolve(__dirname,'../../data/exp/estimation/three_ward_formal_transport_access_v1');
 const opts={headless:true};
 if(!fs.existsSync(chromium.executablePath()))opts.executablePath='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
 const browser=await chromium.launch(opts);const page=await browser.newPage({viewport:{width:1280,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file://'+root+'/review.html',{waitUntil:'load'});await page.waitForSelector('#points circle');
 const checks=[];checks.push({check:'default_map_has209_source_probes',passed:await page.locator('#points circle').count()===209});
 for(const [id,value]of [['case','R125__H__U'],['frame','peak_envelope'],['projection','strict_classified_road'],['shutdown','no_assumed_rail_shutdown'],['direction','two_direction'],['frame','0'],['case','R0__L__T']]){
  await page.selectOption('#'+id,value);checks.push({check:'switch_'+id+'_'+value,passed:await page.locator('#points circle').count()===209});
 }
 await page.locator('#points circle').first().click();checks.push({check:'probe_click_shows_source_context',passed:(await page.locator('#detail').textContent()).includes('分析点：')});
 checks.push({check:'all2496_directional_map_states_have209_codes',passed:await page.evaluate(()=>Object.values(DATA.states).every(v=>Object.values(v).every(x=>x.length===209))&&Object.keys(DATA.states).length===1248)});
 checks.push({check:'no_javascript_errors',passed:errors.length===0});
 await page.selectOption('#case','R1__H__U');await page.selectOption('#frame','14400');await page.selectOption('#projection','ground_path_supplement');await page.selectOption('#shutdown','section');await page.selectOption('#direction','outward');
 await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:root+'/report_preview.png'});await page.locator('#map').screenshot({path:root+'/interactive_map_preview.png'});
 fs.writeFileSync(root+'/browser_validation.json',JSON.stringify({all_passed:checks.every(c=>c.passed),checks,page_errors:errors},null,2)+'\n');await browser.close();if(!checks.every(c=>c.passed))throw Error('Browser QA failed');process.stdout.write('BROWSER QA PASSED '+checks.length+'\n');
})().catch(e=>{console.error(e);process.exit(1)});
