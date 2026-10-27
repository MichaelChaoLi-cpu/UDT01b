"""Start authorized physical queue only after live qualification and input preflight."""
import sys,os,time,subprocess,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import dump
P=ROOT/'data/processed/three_ward_barrier_drainage_v1';R=ROOT/'data/exp/estimation'/P.name;PYTHON='/private/tmp/udt01b-sfincs-builder/bin/python'
def main():
 pid=int(sys.argv[1]);start=time.monotonic()
 try:
  dump(R/'pipeline_status.json',dict(status='waiting_for_qualification',qualifier_pid=pid))
  while not(R/'input_validation.json').exists():
   if(P/'STOP').exists():raise RuntimeError('User STOP before production')
   while(P/'PAUSE').exists():time.sleep(5)
   if time.monotonic()-start>3600:raise RuntimeError('Qualification supervision limit3600s')
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('Qualification/staging exited before passing;see qualification_execution_v2.log')
   time.sleep(5)
  assert json.loads((R/'solver_qualification.json').read_text())['passed'];assert json.loads((R/'input_validation.json').read_text())['passed']
  with(R/'preflight_execution.log').open('w')as out:subprocess.run([PYTHON,str(ROOT/'src/analyses/validate_barrier_drainage_preflight_v1.py')],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
  dump(R/'pipeline_status.json',dict(status='running_physical_campaign'))
  with(R/'campaign_execution.log').open('w')as out:subprocess.run([PYTHON,'-u',str(ROOT/'src/analyses/run_barrier_drainage_v1.py')],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
  assert json.loads((R/'delivery_validation.json').read_text())['passed'];dump(R/'pipeline_status.json',dict(status='qualified_complete'))
 except BaseException as e:
  dump(R/'pipeline_status.json',dict(status='held',reason=str(e),automatic_retry=False));(R/'pipeline_error.txt').write_text(traceback.format_exc());subprocess.run([PYTHON,str(ROOT/'src/analyses/report_barrier_drainage_v1.py')],cwd=ROOT);raise
if __name__=='__main__':main()
