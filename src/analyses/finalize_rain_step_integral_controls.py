"""Wait for the authorized executions and analyze them; never start/retry a solver."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_rain_step_integral import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;m=json.loads((out/'run_manifest.json').read_text())
 for run in m['runs']:
  p=ROOT/run['folder'];c=run['case_id'];print('WAIT COMPLETION',c,flush=True)
  while not(p/'execution.json').exists():time.sleep(20)
  ex=json.loads((p/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'],'Preserve failed execution'
  for phase in ['audit','reconstruct','exchange']:
   receipt=report/(c+'_'+{'audit':'audit','reconstruct':'label_reconstruction','exchange':'boundary_comparison'}[phase]+'.json')
   if receipt.exists():
    obj=json.loads(receipt.read_text())
    if obj.get('all_checks_passed',obj.get('quarter_ground_area_reaggregation_exact',False)):
     print('RETAIN',c,phase,flush=True);continue
   print('PROCESS',c,phase,flush=True)
   with(report/(c+'_'+phase+'.log')).open('w')as log:
    res=subprocess.run([sys.executable,'src/analyses/process_rain_step_integral_controls.py',phase,'--only',c],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if res.returncode:
    if phase=='audit'and receipt.exists():print('RETAIN NUMERICAL HOLD',c,flush=True)
    else:raise RuntimeError('Inspect saved '+c+' '+phase+' failure')
  print('PROCESSED',c,flush=True)
 audits=[json.loads((report/(c+'_audit.json')).read_text())for c in CASES];checks=[c for d in audits for c in d['checks']];dump(report/'audit.json',{'cases':{k:v for d in audits for k,v in d['cases'].items()},'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'training':0,'batch58':0})
 for cmd in [['src/analyses/process_rain_step_integral_controls.py','exchange'],['src/analyses/review_rain_step_integral.py']]:
  print('FINAL PROCESS',*cmd,flush=True)
  with(report/(Path(cmd[0]).stem+'.log')).open('w')as log:subprocess.run([sys.executable,*cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('REVIEW COMPLETE',flush=True)
if __name__=='__main__':main()
