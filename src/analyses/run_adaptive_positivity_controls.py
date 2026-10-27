"""Only the two existing open-control adaptive companions; serial memory-bounded execution."""
from pathlib import Path
import sys,json,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_adaptive_positivity import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;m=json.loads((out/'run_manifest.json').read_text())
 assert [r['case_id']for r in m['runs']]==CASES and not m['pending_58_scenarios_authorized']
 assert json.loads((report/'build_validation.json').read_text())['all_checks_passed']
 for run in m['runs']:
  path=ROOT/run['folder']
  if(path/'execution.json').exists():
   old=json.loads((path/'execution.json').read_text());assert old['returncode']==0 and old['completion_marker'];print('RETAIN',run['case_id'],flush=True);continue
  assert not(path/'sfincs.log').exists(),'Preserve incomplete attempt'
  name='udt01b-adaptive-positivity-'+run['case_id'].replace('_','-')
  cmd=['docker','run','--rm','--name',name,'--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={path},target=/data','--mount','type=bind,source='+run['large_readonly_source']+',target=/data/sfincs_subgrid.nc,readonly','--entrypoint','/usr/local/bin/sfincs',run['image_id']]
  dump(path/'launch.json',{'command':cmd,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'authorization':'two matched adaptive controls only; no training or58batch'});print('START',run['run_id'],flush=True);t=time.monotonic()
  with(path/'solver_stdout.txt').open('w')as so,(path/'solver_stderr.txt').open('w')as se:res=subprocess.run(cmd,stdout=so,stderr=se)
  log=(path/'sfincs.log').read_text(errors='replace')if(path/'sfincs.log').exists()else'';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS'])
  receipt={'returncode':res.returncode,'completion_marker':done,'runtime_seconds':time.monotonic()-t,'command':cmd,'solver_image_id':run['image_id'],'input_sha256':run['input_sha256'],'new_parent_event':False,'training':False,'pending_58_batch_run':False};dump(path/'execution.json',receipt);print('FINISH',run['case_id'],res.returncode,done,receipt['runtime_seconds'],flush=True)
  assert res.returncode==0 and done,'Inspect saved failure logs; no automatic repeat'
 dump(report/'execution_scope.json',{'completed_case_ids':CASES,'hydrodynamic_new_runs':2,'regular_frozen_runs_reused':2,'new_parent_events':0,'pending_58_executed':0,'training':0})
if __name__=='__main__':main()
