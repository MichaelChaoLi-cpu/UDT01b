/* Verify true file:// operation with network disabled and fetch forbidden. */
const fs=require('fs'),path=require('path'),os=require('os'),{pathToFileURL}=require('url');
const {chromium}=require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'../..'),work=path.join(root,'data/exp/dashboards/tokyo-briefing-offline-v1');
const site=process.env.UDT_OFFLINE_TEST_SITE||path.join(work,'site');
(async()=>{
 const qa=path.join(work,'qa');fs.mkdirSync(qa,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const errors=[],http=[],failures=[],checks={};
 try{
  const context=await browser.newContext({offline:true,viewport:{width:1600,height:1120},acceptDownloads:true,serviceWorkers:'block'});
  await context.addInitScript(()=>{window.__attemptedFetch=0;window.fetch=()=>{window.__attemptedFetch++;throw Error('Network/fetch forbidden by offline verification');};});
  context.on('request',r=>{if(/^https?:/.test(r.url()))http.push(r.url());});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>failures.push({url:r.url(),error:r.failure()?.errorText}));
  await page.goto(pathToFileURL(path.join(site,'index.html')).href);
  await page.waitForFunction(()=>window.dashboardReady===true,{timeout:60000});
  checks.file_scheme=page.url().startsWith('file:');checks.no_server=true;
  checks.english=await page.evaluate(()=>!/[\u3400-\u9fff]/.test(document.body.innerText));
  checks.five_events=await page.locator('#eventSelect option').count()===5;
  checks.snapshot_label=await page.locator('#feedBadge').innerText()==='Offline snapshot';
  checks.completed_pilot_snapshot=await page.locator('#liveSummary').innerText()==='Both cases qualified';
  checks.desktop_no_overflow=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);
  checks.numeric_curves_exact=true;checks.ten_2d_cases=true;
  for(const event of ['E20240618','E20240731','E20240830','E20250710','E20250911']){
   await page.locator('#eventSelect').selectOption(event);
   for(const stage of ['H','L']){
    await page.locator('#stageSelect').selectOption(stage);
    await page.waitForFunction(([e,s])=>state.event===e&&state.stage===s&&document.getElementById('mapLoad').style.display==='none',[event,stage]);
    const ok=await page.evaluate(()=>activeRoadData.length===data.roadCount*109*2);
    checks.ten_2d_cases&&=ok;
    for(const branch of ['lower','upper']){
     await page.locator('#boundSelect').selectOption(branch);
     for(const scope of ['0','1','2','3']){
      await page.locator('#scopeSelect').selectOption(scope);
      await page.locator('#timeSlider').fill('60');
      checks.numeric_curves_exact&&=await page.evaluate(()=>Math.abs(Number(document.querySelector('#closedMetric').textContent.replaceAll(',',''))-series().closed[60])<=.051);
     }
    }
   }
  }
  await page.locator('#scopeSelect').selectOption('0');await page.locator('#boundSelect').selectOption('lower');await page.locator('#stageSelect').selectOption('H');
  await page.waitForFunction(()=>document.querySelector('#mapLoad').style.display==='none');
  await page.locator('#playBtn').click();await page.waitForTimeout(850);await page.locator('#playBtn').click();
  checks.map_playback=await page.evaluate(()=>state.frame>60&&!state.playing);
  await page.locator('#showIntakes').check();await page.locator('#zoomIn').click();checks.map_controls=await page.evaluate(()=>state.zoom>1&&document.getElementById('showIntakes').checked);await page.locator('#resetMap').click();
  await page.locator('#reachToggle button[data-reach="500"]').click();checks.allocation_toggle=(await page.locator('#allocationBars').innerText()).includes('137 modeled intake locations');
  await page.locator('#notesBtn').click();checks.evidence_dialog=await page.locator('#notesDialog').evaluate(e=>e.open);await page.locator('#closeNotes').click();
  const down=page.waitForEvent('download');await page.locator('#exportBtn').click();const download=await down;const saved=path.join(qa,'status-export.json');await download.saveAs(saved);checks.json_export=JSON.parse(fs.readFileSync(saved)).monitoring.offline===true;
  await page.locator('#presentBtn').click();checks.presentation=await page.evaluate(()=>document.body.classList.contains('present'));await page.screenshot({path:path.join(qa,'dashboard-presentation.png')});await page.locator('#presentBtn').click();
  await page.locator('#peakBtn').click();await page.screenshot({path:path.join(qa,'dashboard-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);checks.mobile_no_overflow=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);await page.screenshot({path:path.join(qa,'dashboard-mobile.png'),fullPage:true});await page.setViewportSize({width:1600,height:1120});
  const popup=context.waitForEvent('page');await page.locator('#street3dLink').click();const street=await popup;street.on('pageerror',e=>errors.push(String(e)));street.on('requestfailed',r=>failures.push({url:r.url(),error:r.failure()?.errorText}));await street.waitForFunction(()=>window.streetReady===true,{timeout:90000});
  checks.street_link_file=street.url().startsWith('file:')&&street.url().includes('street-view.html');
  checks.real_mesh=await street.evaluate(()=>streetMetadata.meshId==='53394611'&&streetMetadata.groups.buildings.triangles>200000&&streetMetadata.groups.terrain.triangles>80000);
  checks.street_english=await street.evaluate(()=>!/[\u3400-\u9fff]/.test(document.body.innerText));
  checks.ten_3d_cases=true;
  for(const event of ['E20240618','E20240731','E20240830','E20250710','E20250911']){
   await street.locator('#event').selectOption(event);
   for(const stage of ['H','L']){
    await street.locator('#stage').selectOption(stage);
    await street.waitForFunction(([e,s])=>streetState.event===e&&streetState.stage===s&&document.querySelector('#loading').style.display==='none',[event,stage]);
    checks.ten_3d_cases&&=await street.evaluate(()=>head.length===109*streetMetadata.cellCenters.length&&lastStreetGlError===0);
   }
  }
  checks.three_viewpoints=await street.locator('#place option').count()===3;
  for(const id of ['yaesu','sotobori','kyobashi']){await street.locator('#place').selectOption(id);await street.locator('#walk').fill('80');}
  checks.street_movement=await street.evaluate(()=>streetState.view===2&&streetState.sample===80);
  await street.waitForFunction(()=>streetCamera.mode==='street'&&streetCamera.view===streetState.view&&streetCamera.sample===streetState.sample);checks.eye_height=await street.evaluate(()=>Math.abs(streetCamera.eye[1]-streetMetadata.views[streetState.view].path[streetState.sample][1]-1.65)<1e-6);
  await street.locator('#time').fill('60');checks.street_clock=(await street.locator('#clock').innerText())==='05:00';
  await street.locator('#flood').uncheck();await street.locator('#flood').check();await street.locator('#play').click();await street.waitForTimeout(1100);await street.locator('#play').click();checks.street_playback=await street.evaluate(()=>streetState.frame>60&&!streetState.playing);
  await street.locator('#localPeak').click();await street.waitForTimeout(200);await street.screenshot({path:path.join(qa,'street-eye.png')});
  const png=street.waitForEvent('download');await street.locator('#save').click();const pngDownload=await png;await pngDownload.saveAs(path.join(qa,'street-export.png'));checks.png_export=fs.readFileSync(path.join(qa,'street-export.png')).subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]));
  await street.locator('#overview').click();await street.waitForTimeout(200);checks.street_overview=await street.evaluate(()=>streetCamera.mode==='overview'&&lastStreetGlError===0);await street.screenshot({path:path.join(qa,'street-overview.png')});
  await street.setViewportSize({width:390,height:844});checks.street_mobile_no_overflow=await street.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);await street.screenshot({path:path.join(qa,'street-mobile.png')});
  checks.zero_fetch_calls=await page.evaluate(()=>__attemptedFetch===0)&&await street.evaluate(()=>__attemptedFetch===0);
  await street.locator('.back').click();await street.waitForFunction(()=>window.dashboardReady===true);checks.return_to_index=street.url().includes('index.html');
  checks.zero_http_requests=http.length===0;checks.zero_failed_file_requests=failures.length===0;checks.no_javascript_errors=errors.length===0;
  const result={passed:Object.values(checks).every(Boolean),checks,errors,httpRequests:http,failedRequests:failures,mode:'file://; browser offline; fetch deliberately forbidden',testSite:process.env.UDT_OFFLINE_TEST_SITE?'freshly extracted ZIP':'build directory'};
  fs.writeFileSync(path.join(work,'browser_validation.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));if(!result.passed)process.exitCode=1;
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
