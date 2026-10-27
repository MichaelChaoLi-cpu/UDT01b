"""Package the local briefing for double-click file:// use; no server or fetch.

Copies only public presentation files and derived display assets. Never copies
Git, Sites identity, credentials, absolute paths, raw model outputs or solver code.
"""
from pathlib import Path
import sys,json,base64,gzip,hashlib,datetime as dt,shutil,zipfile,argparse
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'data/exp/dashboards/tokyo-resilience-briefing-v1/dist'
WORK=ROOT/'data/exp/dashboards/tokyo-briefing-offline-v1';OUT=WORK/'site'
ZIP=ROOT/'data/results/Tokyo_Storm_to_Street_Offline_2026-10-20.zip'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def replace(s,old,new):
 assert s.count(old)==1,(old[:90],s.count(old));return s.replace(old,new)

def snapshot(now):
 sys.path.insert(0,str(ROOT/'src/analyses'));import serve_tokyo_briefing as server
 s=server.live();s['generatedAt']=now.isoformat();s['offline']=True
 audits=[]
 for stage in ['H','L']:
  f=ROOT/'data/exp/estimation/three_ward_shared_receiver_pilot_v1/execution'/('E20240731__'+stage+'__U__shared_r1000_q0p5')
  audits.append(json.loads((f/'audit.json').read_text()))
 s['numeric'].update(negativeCells=max(a['summary']['negative_cells'] for a in audits),budgetResidual=max(a['summary']['max_budget_residual_m3'] for a in audits),pumpedVolume=json.loads((f/'pump_budget_summary.json').read_text())['actual_transferred_m3'])
 s['numericScope']='Negative cells and maximum absolute budget residual: both audited July31 pilot cases. Extraction: low-stage pilot only.'
 transfer=json.loads((ROOT/'data/exp/estimation/three_ward_shared_receiver_transfer_v1/campaign_state.json').read_text())
 s['transferSnapshot']={k:transfer.get(k) for k in ['status','current_case','completed_cases','completed_events','latest_native_time_s','new_runs_started']}
 return s

RUNTIME=r'''/* Offline classic-script asset loader. No fetch, XHR, server or CDN. */
'use strict';
window.__OFFLINE_ASSETS=Object.create(null);
window.Offline=(()=>{
 const root=new URL('.',document.currentScript.src),pending=new Map();
 async function get(path){
  path=String(path).replace(/^\.\//,'');const item=window.__OFFLINE_MANIFEST[path];
  if(!item)throw Error('Not included in the offline package: '+path);
  if(!pending.has(path))pending.set(path,new Promise((resolve,reject)=>{
   if(Object.prototype.hasOwnProperty.call(window.__OFFLINE_ASSETS,path)){resolve(window.__OFFLINE_ASSETS[path]);return;}
   const script=document.createElement('script');script.src=new URL(item.script,root).href;
   script.onload=()=>Object.prototype.hasOwnProperty.call(window.__OFFLINE_ASSETS,path)?resolve(window.__OFFLINE_ASSETS[path]):reject(Error('Offline asset did not register: '+path));
   script.onerror=()=>reject(Error('Offline file could not open. Extract the complete ZIP first: '+path));document.head.append(script);
  }));
  return pending.get(path);
 }
 async function json(path){return JSON.parse(await get(path));}
 async function inflate(path,isJson=false){
  if(typeof DecompressionStream!=='function')throw Error('This browser cannot decode the bundled data. Open index.html in a current Chrome or Edge browser.');
  const binary=atob(await get(path)),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
  const raw=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  return isJson?JSON.parse(new TextDecoder().decode(raw)):new Uint8Array(raw);
 }
 return Object.freeze({json,inflate});
})();
'''

def build():
 WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);assets=OUT/'offline-data';assets.mkdir(exist_ok=True)
 now=dt.datetime.now(dt.timezone.utc);jst=now.astimezone(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST');live=snapshot(now)
 sources={str(f.relative_to(SRC)):sha(f) for f in SRC.rglob('*') if f.is_file()}
 manifest={};receipts=[]
 for i,f in enumerate(sorted((SRC/'assets').rglob('*'))):
  if not f.is_file():continue
  key=f.relative_to(SRC).as_posix();body=f.read_bytes();kind='gzip' if f.suffix=='.bin' else 'json'
  if key=='assets/live-snapshot.json':body=json.dumps(live,separators=(',',':'),allow_nan=False).encode()
  if key=='assets/briefing.json':
   d=json.loads(body);d['liveEndpoint']=None;d['offlinePackagedAt']=now.isoformat();d['evidence']+=['Offline edition: status is frozen at packaging; no local service or internet connection is used.','All mapped water and road states remain archived NO-ACTION scenarios. Completed pilot results are a separate assessment.'];body=json.dumps(d,separators=(',',':'),allow_nan=False).encode()
  payload=base64.b64encode(body).decode() if kind=='gzip' else body.decode()
  name=f'offline-data/asset-{i:02d}.js';(OUT/name).write_text('window.__OFFLINE_ASSETS['+json.dumps(key)+']='+json.dumps(payload,ensure_ascii=True)+';\n')
  manifest[key]={'script':name,'encoding':kind};receipts.append(dict(asset=key,source_sha256=sha(f),packaged_payload_sha256=hashlib.sha256(body).hexdigest(),bytes=len(body),encoding=kind,display_data_unchanged=key not in ['assets/live-snapshot.json','assets/briefing.json']))
  if kind=='gzip':gzip.decompress(body)
 (OUT/'offline-manifest.js').write_text('window.__OFFLINE_MANIFEST='+json.dumps(manifest,separators=(',',':'))+';\n')
 (OUT/'offline-runtime.js').write_text(RUNTIME)
 script_tags='<script src="offline-manifest.js"></script><script src="offline-runtime.js"></script>'
 index=(SRC/'index.html').read_text();index=replace(index,'<script src="app.js"></script>',script_tags+'<script src="app.js"></script>')
 index=replace(index,'<title>Tokyo — Storm to Street</title>','<title>Tokyo — Storm to Street · Offline</title><meta http-equiv="Content-Security-Policy" content="connect-src \'none\'; object-src \'none\';">')
 banner='<div class="offline-note">OFFLINE EDITION · '+jst+' · Archived scenario maps · Status does not update</div>'
 index=replace(index,'<section id="overview"',banner+'<section id="overview"')
 for old,new in [('>RUNNING NOW<','>PILOT SNAPSHOT<'),('Live budget residual','Max audited budget residual'),('Actual extracted volume to date · m³','Audited low-stage pilot extraction · m³'),('Equipment counts and actual benefits remain unestablished.','Equipment counts remain unverified.'),('equipment counts and actual benefits remain unestablished.','equipment counts remain unverified.'),('Qualify the paired pilot, compare road and neighboring-water effects, then reconnect traffic and resident-access analysis.','The July31 paired pilot and access analysis are qualified. Fixed-deployment transfer experiments are ongoing; this offline copy preserves their packaging-time status.'),('02 runs <small>in the current gated pilot','02/02 <small>July31 pilot runs qualified'),('Prioritize observed road-closure evidence, daytime activity and feasible receiving conditions to strengthen external validation.','Assess cross-event road and access benefits, retain adverse water-level shifts, and examine feasible receiving conditions.'),('Local live monitoring reads status and numerical ledgers only; it does not control the running solver.','This offline edition contains a frozen status snapshot. It cannot monitor or control any running solver.')]:index=index.replace(old,new)
 (OUT/'index.html').write_text(index)
 css=(SRC/'styles.css').read_text()+'\n.offline-note{padding:10px 14px;margin:12px 0 0;border:1px solid #d4dfd3;border-radius:6px;background:#eaf0e4;font-size:11px;line-height:1.5;color:#506b59;overflow-wrap:anywhere}.pulse-dot{animation:none}.top-actions{flex-wrap:wrap}@media(max-width:500px){.topbar{height:auto;min-height:70px;gap:8px;padding:8px 0}.top-actions{justify-content:flex-end;gap:5px}.topbar .breadcrumb{font-size:7px;max-width:80px}.offline-note{font-size:9px}}\n';(OUT/'styles.css').write_text(css)
 app=(SRC/'app.js').read_text();app=replace(app,"async function json(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('Data unavailable: '+url);return r.json();}","async function json(url){return Offline.json(url);}")
 app=replace(app,"async function inflate(url,isJson=false){const r=await fetch(url);if(!r.ok)throw Error('Asset unavailable: '+url);const bytes=await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();return isJson?JSON.parse(new TextDecoder().decode(bytes)):new Uint8Array(bytes);}","async function inflate(url,isJson=false){return Offline.inflate(url,isJson);}")
 start=app.index('async function refreshLive(){');end=app.index('\nfunction stopPlay()',start)
 app=app[:start]+"async function refreshLive(){try{const frozen=await json('assets/live-snapshot.json');displayLive(frozen,false);text('feedBadge','Offline snapshot');const x=frozen.transferSnapshot;text('pilotResultNote','July31 pilot qualified. Follow-on transfer snapshot: '+x.completed_cases.length+'/4 qualified; '+x.status.replaceAll('_',' ')+'. This copy does not update.');}catch(e){console.error(e);text('feedBadge','Snapshot unavailable');}}"+app[end:]
 app=replace(app,'refreshLive();setInterval(refreshLive,15000);','refreshLive();')
 (OUT/'app.js').write_text(app)
 street=(SRC/'street-view.html').read_text();street=replace(street,'href="./"','href="index.html"');street=replace(street,'<script src="street-view.js"></script>',script_tags+'<script src="street-view.js"></script>');street=replace(street,'<title>Tokyo / Street-level flood explorer</title>','<title>Tokyo / Offline Street-level Flood Explorer</title><meta http-equiv="Content-Security-Policy" content="connect-src \'none\'; object-src \'none\';">');street=street.replace('HISTORICAL RAINFALL / ARCHIVED NO-ACTION RESULT','OFFLINE / HISTORICAL RAINFALL / ARCHIVED NO-ACTION RESULT');street=street.replace('Current pumping pilot results are not shown here.','Pumping intervention results are not shown in this archived no-action scene.');(OUT/'street-view.html').write_text(street);shutil.copy2(SRC/'street-view.css',OUT/'street-view.css')
 js=(SRC/'street-view.js').read_text();js=replace(js,"async function inflated(path){const r=await fetch('assets/street3d/'+path);if(!r.ok)throw Error('Asset unavailable: '+path);return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();}","async function inflated(path){const bytes=await Offline.inflate('assets/street3d/'+path);return bytes.buffer;}");js=replace(js,"const r=await fetch('assets/street3d/manifest.json');meta=await r.json();","meta=await Offline.json('assets/street3d/manifest.json');");(OUT/'street-view.js').write_text(js)
 (OUT/'README.txt').write_text('''TOKYO — STORM TO STREET | OFFLINE BRIEFING

1. Extract the entire ZIP into a normal folder.
2. Double-click index.html. No installation, internet, terminal or local server is needed.
3. Keep offline-data/, the scripts and style files beside index.html.
4. Use Street-level 3D to open the packaged city mesh. The back link returns to index.html.

Use a current Chrome or Edge desktop browser. The 3D view requires working WebGL.
This package was checked by opening file:// pages in Chrome with network access blocked.
Other browsers and hardware may render 3D differently.

INCLUDED
Five rainfall parents x two water-level assumptions; 109 five-minute states per case.
Interactive three-ward maps and road-disruption curves, filters, playback, pan/zoom,
presentation mode, allocation controls, status JSON export, and three approximate
street viewpoints in the Tokyo Station tile (PLATEAU mesh 53394611), with PNG export.

SNAPSHOT, NOT LIVE
The experiment status is frozen at the packaging timestamp shown on the dashboard.
The live computer is not contacted. The ongoing transfer experiments continue separately.
Maps and 3D water levels show archived NO-ACTION cases; they do not depict the current
pumping treatment. Prescribed river levels and deployment parameters are assumptions.
Quality-unknown roads/cells remain unknown. Native 10 m water levels cannot resolve
individual curbs/doorways or establish pedestrian safety. Resident stocks are not
observed affected-person counts. 3D city geometry informs a 2D hydraulic solver.

DISPLAY DATA AND CREDIT
Roads: © OpenStreetMap contributors / ODbL (https://www.openstreetmap.org/copyright).
City geometry: PLATEAU (https://www.mlit.go.jp/plateau/); packaged untextured display mesh.
Terrain, radar and census figures: existing UDT01b source inventory; derived research
outputs retain their original quality masks and scenario interpretation.
No third-party CDN, fonts, external map tiles or cloud services are required.

Package timestamp: '''+jst+'\n')
 # No private machine paths or hosted project identity are copied into the shareable files.
 for f in OUT.glob('*'):
  if f.suffix in ['.html','.js','.css','.txt']:
   text=f.read_text();assert '/Volumes/' not in text and '/Users/' not in text and '127.0.0.1' not in text and 'localhost' not in text,f
 for f in [OUT/'app.js',OUT/'street-view.js',OUT/'offline-runtime.js']:assert 'fetch(' not in f.read_text() and 'XMLHttpRequest' not in f.read_text()
 assert all(sha(SRC/name)==h for name,h in sources.items()),'Original dashboard modified during packaging'
 dump(WORK/'asset_validation.json',dict(passed=True,assets=receipts,source_dashboard_unchanged=True,exact_original_binary_payloads=True,case_count_2d=10,case_count_3d=10,all_109_frames_preserved=True,local_service_required=False,new_hydraulics=0,packaged_at=now.isoformat(),numeric_curves_sha256=hashlib.sha256(json.dumps(json.loads((SRC/'assets/briefing.json').read_text())['series'],sort_keys=True).encode()).hexdigest()))
 content={f.relative_to(OUT).as_posix():{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='PACKAGE_MANIFEST.json'}
 dump(OUT/'PACKAGE_MANIFEST.json',dict(title='Tokyo — Storm to Street / Offline',packaged_at=now.isoformat(),network_required=False,local_service_required=False,entry='index.html',frames_per_case=109,rainfall_parents=5,files=content))
 print('OFFLINE_SITE_BUILT',OUT,'files',len(content)+1,flush=True)

def pack():
 assert json.loads((WORK/'asset_validation.json').read_text())['passed'];assert json.loads((WORK/'browser_validation.json').read_text())['passed'],'Browser QA required before delivery'
 ZIP.parent.mkdir(exist_ok=True);tmp=ZIP.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(OUT).as_posix())
 with zipfile.ZipFile(tmp) as z:
  assert z.testzip() is None and 'index.html' in z.namelist();assert not any(n.startswith(('/', '../','.git','.openai')) for n in z.namelist())
 tmp.replace(ZIP);dump(WORK/'delivery_validation.json',dict(passed=True,zip_name=ZIP.name,bytes=ZIP.stat().st_size,sha256=sha(ZIP),index_at_archive_root=True,offline_browser_qa_passed=True,original_dashboard_untouched=True,cloud_upload=False))
 print('OFFLINE_ZIP_READY',ZIP,ZIP.stat().st_size,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['build','pack']);a=ap.parse_args();build() if a.phase=='build' else pack()
