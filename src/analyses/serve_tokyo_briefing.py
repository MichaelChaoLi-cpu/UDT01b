#!/usr/bin/env python3
"""Read-only loopback dashboard service;never modifies the active hydraulic campaign."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlsplit,unquote
import sys,json,datetime,argparse,os
ROOT=Path(__file__).resolve().parents[2];SITE=ROOT/'data/exp/dashboards/tokyo-resilience-briefing-v1';DIST=SITE/'dist';P=ROOT/'data/processed/three_ward_shared_receiver_pilot_v1';R=ROOT/'data/exp/estimation/three_ward_shared_receiver_pilot_v1'

def tail(path):
 if not path.exists():return None
 with path.open('rb') as f:
  names=f.readline().decode().strip().split(',');f.seek(0,2);n=f.tell();f.seek(max(0,n-65536));s=f.read().decode(errors='replace')
 if not s.endswith('\n'):s=s.rsplit('\n',1)[0]
 for line in reversed(s.splitlines()):
  vals=line.split(',')
  if len(vals)==len(names):
   try:return dict(zip(names,map(float,vals)))
   except ValueError:pass
 return None

def live():
 s=json.loads((R/'campaign_state.json').read_text());manifest=json.loads((P/'run_manifest.json').read_text());cases=[];latest=None
 for run in manifest['runs']:
  f=ROOT/run['folder'];row=tail(f/'udt_step_budget.csv');a=R/'execution'/run['case_id']/'audit.json';qualified=run['case_id'] in s.get('completed_cases',[]);audit_passed=json.loads(a.read_text()).get('all_checks_passed',False) if a.exists() else False
  seconds=float(row['actual_state_time']) if row else 0.
  cases.append({'stage':run['stage'],'qualified':qualified,'auditPassed':audit_passed,'progress':min(1.,seconds/32400),'modelHours':seconds/3600,'started':(f/'launch.json').exists(),'active':s.get('current_case')==run['case_id']})
  if s.get('current_case')==run['case_id']:latest=row
 post=R/'postprocessing_validation.json';complete=(R/'delivery_validation.json').exists() and json.loads((R/'delivery_validation.json').read_text()).get('passed',False)
 return {'generatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':s['status'],'cases':cases,'qualified':len(s.get('completed_cases',[])),'total':2,'started':s.get('new_runs_started',0),'activeSeconds':s.get('case_active_wall_seconds',0),'preflightPassed':json.loads((R/'preflight_validation.json').read_text())['passed'],'numeric':{'negativeCells':int(latest['negative_native_count']) if latest else None,'budgetResidual':float(latest['budget_residual']) if latest else None,'pumpedVolume':float(latest['other_sink_cumulative']) if latest else None,'rainVolume':float(latest['rain_applied_cumulative']) if latest else None},'complete':complete,'postprocessingComplete':post.exists(),'scope':'Paired July31pilot;62modeled intakes,2.5m3/s total static reservation','measureResultsAvailable':complete,'error':s.get('error')}

class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(DIST),**kwargs)
 def do_GET(self):
  path=urlsplit(self.path).path
  if path=='/api/live':
   try:body=json.dumps(live(),allow_nan=False).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
   except (OSError,ValueError,KeyError):self.send_error(503,'Live snapshot temporarily unavailable')
   return
  resolved=(DIST/unquote(path).lstrip('/')).resolve()
  if not resolved.is_relative_to(DIST.resolve()) or any(x.startswith('.') for x in Path(unquote(path)).parts if x not in ['/', '.']):self.send_error(404);return
  if resolved.is_dir() and path!='/':self.send_error(404);return
  return super().do_GET()
 def end_headers(self):self.send_header('X-Content-Type-Options','nosniff');super().end_headers()
 def log_message(self,fmt,*args):pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');p.add_argument('--port',type=int,default=8877);args=p.parse_args()
 if args.export:
  (DIST/'assets/live-snapshot.json').write_text(json.dumps(live(),indent=2,allow_nan=False)+'\n');print('SNAPSHOT_EXPORTED');return
 try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
 except OSError:server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
 url='http://127.0.0.1:'+str(server.server_address[1]);(SITE/'local_server.json').write_text(json.dumps({'url':url,'pid':os.getpid(),'bind':'loopback only','read_only_experiment_access':True})+'\n');print('DASHBOARD_READY '+url,flush=True)
 try:server.serve_forever()
 finally:server.server_close()
if __name__=='__main__':main()
