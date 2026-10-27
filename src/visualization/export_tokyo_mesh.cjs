const fs=require('fs'),path=require('path');const {pathToFileURL}=require('url');const {chromium}=require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'data/results/tokyo_mesh_53394611');
 const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const errors=[];const ctx=await browser.newContext({viewport:{width:1800,height:1200},deviceScaleFactor:2});const page=await ctx.newPage();page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(120000);
 await page.goto(pathToFileURL(path.join(out,'tokyo_mesh_53394611.html')).href+'?export=1',{timeout:120000});await page.waitForFunction(()=>window.sceneReady===true);await page.waitForTimeout(500);
 await page.screenshot({path:path.join(out,'tokyo_mesh_53394611_clean.png')});
 const state=await page.evaluate(()=>({stats:window.sceneStats,glError:window.lastGlError,canvasWidth:document.querySelector('canvas').width,canvasHeight:document.querySelector('canvas').height,language:document.documentElement.lang,annotations:document.body.innerText,hasChineseAnnotations:/[\u3400-\u9fff]/.test(document.body.innerText),controlsHidden:getComputedStyle(document.querySelector('.panel')).display==='none'}));
 await page.evaluate(()=>window.setLayers({buildings:false},'Roads · Terrain · Water (buildings hidden)'));await page.screenshot({path:path.join(out,'tokyo_mesh_53394611_three_layers_clean.png')});
 await page.evaluate(()=>{document.body.classList.remove('export');window.setLayers({buildings:true});});
 const defaultText=await page.evaluate(()=>document.body.innerText.trim());await page.keyboard.press('h');const controlsShown=await page.locator('#buildings').isVisible();
 const switches=[];
 for(const id of ['buildings','roads','terrain','water','wire']){
  await page.locator('#'+id).click();const a=await page.evaluate(()=>window.layerState);await page.locator('#'+id).click();const b=await page.evaluate(()=>window.layerState);switches.push({id,works:a[id]!==b[id],glError:await page.evaluate(()=>window.lastGlError)});
 }
 await page.keyboard.press('h');const controlsHiddenAgain=!(await page.locator('#buildings').isVisible());await page.screenshot({path:path.join(out,'viewer_desktop.png')});
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);const mobile=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,error:window.lastGlError}));await page.screenshot({path:path.join(out,'viewer_mobile.png')});
 const result={passed:errors.length===0&&state.glError===0&&state.stats.meshId==='53394611'&&state.stats.counts.roads>500&&state.stats.counts.water_bodies>0&&state.stats.counts.terrain_triangles>80000&&state.stats.materials==='none'&&state.language==='en'&&!state.hasChineseAnnotations&&state.annotations.trim()===''&&state.controlsHidden&&defaultText===''&&controlsShown&&controlsHiddenAgain&&state.stats.palette.buildings==='#FFFFFF'&&state.stats.palette.roads==='#7D858C'&&state.stats.palette.terrain==='#D9CFB7'&&state.stats.palette.water==='#ADD8EF'&&switches.every(x=>x.works&&x.glError===0)&&mobile.scroll<=390&&mobile.error===0,errors,state,defaultText,controlsShown,controlsHiddenAgain,switches,mobile};
 fs.writeFileSync(path.join(out,'render_validation.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));await browser.close();if(!result.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
