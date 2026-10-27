"""Execute only the two explicitly authorized staged open-control validations."""
from pathlib import Path
import sys,json,subprocess,time,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT
 manifest=json.loads((out/'run_manifest.json').read_text());runs=manifest['runs']
 assert [r['case_id']for r in runs]==CASES and len(runs)==2 and not manifest['pending_58_scenarios_authorized']
 for run in runs:
  path=root/run['folder']
  if(path/'execution.json').exists():
   old=json.loads((path/'execution.json').read_text())
   if old['returncode']==0 and old['completion_marker']:print('RETAIN COMPLETED',run['run_id'],flush=True);continue
   raise FileExistsError('Failed or incomplete prior attempt retained; inspect before another attempt')
  if(path/'sfincs.log').exists():raise FileExistsError('Incomplete attempt exists')
  name='udt01b-e2e-'+run['case_id'].replace('_','-')
  command=['docker','run','--rm','--name',name,'--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount','type=bind,source='+str(path)+',target=/data','--mount','type=bind,source='+run['large_readonly_source']+',target=/data/sfincs_subgrid.nc,readonly','--entrypoint','/usr/local/bin/sfincs',run['image_id']]
  dump(path/'launch.json',{'command':command,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'authorization':'two existing open controls only','case_id':run['case_id']})
  print('START AUTHORIZED CONTROL',run['run_id'],flush=True);start=time.monotonic()
  with(path/'solver_stdout.txt').open('w')as stdout,(path/'solver_stderr.txt').open('w')as stderr:
   result=subprocess.run(command,stdout=stdout,stderr=stderr)
  log=(path/'sfincs.log').read_text(errors='replace')if(path/'sfincs.log').exists()else''
  done='Simulation finished'in log or 'Simulation completed'in log or 'Closing off SFINCS'in log
  receipt={'returncode':result.returncode,'completion_marker':done,'runtime_seconds':time.monotonic()-start,'command':command,'input_sha256':run['input_sha256'],'solver_image_id':run['image_id'],'solver_image_digest':run['image_digest'],'new_parent_event':False,'training':False,'pending_58_batch_run':False}
  dump(path/'execution.json',receipt);print('FINISH',run['run_id'],json.dumps(receipt|{'command':'see execution.json','input_sha256':'see execution.json'}),flush=True)
  if result.returncode or not done:raise RuntimeError('Control failed; inspect saved logs before proceeding')
 dump(report/'execution_scope.json',{'completed_case_ids':CASES,'hydrodynamic_control_runs':2,'new_scenario_parent_count':0,'pending_58_executed':0,'training':0})
if __name__=='__main__':main()
