const { chromium } = require('playwright');
(async()=>{
 const path='/Volumes/Extension/Research/UDT01b/data/exp/estimation/three_ward_demand_protection_feedback_v1/';
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file://'+path+'review.html');await page.waitForLoadState('load');
 const info=await page.evaluate(()=>({title:document.title,images:[...document.images].map(i=>({source:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0})),bodyWidth:document.body.scrollWidth,viewport:innerWidth,localLinks:[...document.querySelectorAll('a')].map(a=>a.getAttribute('href')).filter(x=>x&&!x.startsWith('http'))}));
 if(errors.length||!info.images.every(x=>x.loaded)||info.bodyWidth>info.viewport+1)throw new Error(JSON.stringify({errors,info}));
 await page.screenshot({path:path+'review_desktop.png'});
 await page.setViewportSize({width:390,height:844});
 const mobile=await page.evaluate(()=>({width:document.body.scrollWidth,viewport:innerWidth}));
 if(mobile.width>mobile.viewport+1)throw new Error('Mobile overflow '+JSON.stringify(mobile));
 await page.screenshot({path:path+'review_mobile.png'});
 require('fs').writeFileSync(path+'browser_validation.json',JSON.stringify({passed:true,errors,desktop:info,mobile},null,2));
 console.log(JSON.stringify({passed:true,title:info.title,images:info.images,mobile}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
