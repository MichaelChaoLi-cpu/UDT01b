"""Bounded serial execution: low stage audit must pass before raised stage starts."""
from pathlib import Path
import hashlib,json,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
R=ROOT/'data/exp/estimation/three_ward_historical_radar_experiment_v2'
PY='/private/tmp/udt01b-sfincs-builder/bin/python'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
 assert json.loads((R/'solver_qualification.json').read_text())['all_passed']
 m=json.loads((P/'run_manifest.json').read_text());assert len(m['runs'])==m['maximum_runs']==2
 assert [r['case_id'] for r in m['runs']]==['radar_low','radar_raised']
 code=[ROOT/'src/analyses/run_historical_radar_experiment_v2.py',ROOT/'src/analyses/audit_historical_radar_run_v2.py',ROOT/'src/preprocessing/reference_label_policy.py']
 binding={str(p.relative_to(ROOT)):sha(p) for p in code}
 dest=P/'execution_code_binding.json'
 if dest.exists():assert json.loads(dest.read_text())==binding
 else:dump(dest,binding)
 for run in m['runs']:
  case=run['case_id'];folder=ROOT/run['folder']
  assert all(sha(ROOT/rel)==h for rel,h in binding.items())
  assert all(sha(ROOT/rel)==h for rel,h in json.loads((P/'protected_sources.json').read_text()).items())
  if case=='radar_raised':assert json.loads((R/'execution/radar_low/audit.json').read_text())['all_checks_passed']
  if not (folder/'execution.json').exists():
   assert not any((folder/n).exists() for n in ['launch.json','sfincs.log','sfincs_map.nc']),'Incomplete attempt retained; no automatic restart'
   assert all(sha(folder/n)==h for n,h in run['input_sha256'].items())
   image=run['image_id'];inspect=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],check=True,capture_output=True,text=True).stdout.strip()
   meta=json.loads((P/'build-metadata.json').read_text());assert inspect==image==meta['containerimage.digest']
   cmd=['docker','run','--rm','--name','udt01b-historical-v2-'+case.replace('_','-'),'--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={folder},target=/data','--mount',f"type=bind,source={run['large_readonly_source']},target=/data/sfincs_subgrid.nc,readonly",'--entrypoint','/usr/local/bin/sfincs',image]
   dump(folder/'launch.json',dict(command=cmd,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),horizon_s=32400,maximum_cases=2,inspected_image_id=inspect,config_image_digest=meta['containerimage.config.digest']));print('START',case,flush=True)
   started=time.monotonic()
   with (folder/'solver_stdout.txt').open('w') as so,(folder/'solver_stderr.txt').open('w') as se:ex=subprocess.run(cmd,stdout=so,stderr=se)
   log=(folder/'sfincs.log').read_text(errors='replace') if (folder/'sfincs.log').exists() else '';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS'])
   dump(folder/'execution.json',dict(returncode=ex.returncode,completion_marker=done,runtime_seconds=time.monotonic()-started,solver_image_id=image,container_inspected_image_id=inspect,config_image_digest=meta['containerimage.config.digest'],input_sha256=run['input_sha256'],command=cmd,training=False,sealed_test=False,observed_rain_scenario_river_stage=True))
   assert ex.returncode==0 and done,'Preserve failed attempt; no second run'
  ex=json.loads((folder/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker']
  dest=R/'execution'/case;dest.mkdir(parents=True,exist_ok=True)
  with (dest/'audit.log').open('w') as f:subprocess.run([PY,str(ROOT/'src/analyses/audit_historical_radar_run_v2.py'),'execution',case],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  audit=json.loads((dest/'audit.json').read_text());assert audit['all_checks_passed']
  print('QUALIFIED',case,flush=True);dump(R/'progress.json',dict(phase=case+'_qualified',training=False,sealed_test=False))
 dump(R/'progress.json',dict(phase='both_historical_scenarios_qualified',training=False,sealed_test=False));print('BOTH CASES QUALIFIED',flush=True)

if __name__=='__main__':main()
