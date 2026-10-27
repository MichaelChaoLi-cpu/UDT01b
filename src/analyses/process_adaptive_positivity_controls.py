"""Process only completed authorized controls; never starts or retries a solver."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_adaptive_positivity import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;m=json.loads((out/'run_manifest.json').read_text())
 for run in m['runs']:
  path=ROOT/run['folder'];case=run['case_id'];print('WAIT COMPLETION',case,flush=True)
  while not(path/'execution.json').exists():time.sleep(20)
  ex=json.loads((path/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'],'Preserve failed attempt'
  receipts=[report/(case+'_audit.json'),report/(case+'_label_reconstruction.json'),report/(case+'_spatial_comparison.json'),report/(case+'_boundary_comparison.json')]
  if all(p.exists()for p in receipts):
   proofs=[json.loads(p.read_text())for p in receipts]
   if proofs[0]['all_checks_passed']and proofs[1]['quarter_ground_area_reaggregation_exact']and proofs[2]['all_checks_passed']and proofs[3]['all_checks_passed']:
    print('RETAIN VERIFIED ANALYSES',case,flush=True);continue
  commands=[['src/analyses/audit_adaptive_positivity_controls.py','--only',case],['src/preprocessing/reconstruct_adaptive_positivity_labels.py','--only',case],['src/analyses/compare_adaptive_positivity.py','--only',case],['src/analyses/compare_adaptive_boundary_exchange.py','--only',case]]
  for command in commands:
   print('PROCESS',*command,flush=True)
   with(report/(case+'_'+Path(command[0]).stem+'.log')).open('w')as log:res=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if res.returncode:
    if 'audit_adaptive_positivity_controls' in command[0] and(report/(case+'_audit.json')).exists():
     print('NUMERICAL GATE FINDING RETAINED',case,flush=True)
    else:raise RuntimeError('Postprocess failed; inspect '+str(command))
  print('PROCESSED',case,flush=True)
 audits=[json.loads((report/(c+'_audit.json')).read_text())for c in CASES];checks=[c for d in audits for c in d['checks']];dump(report/'audit.json',{'cases':{k:v for d in audits for k,v in d['cases'].items()},'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'training':0,'batch58':0})
 for command in ['src/analyses/compare_adaptive_positivity.py','src/analyses/compare_adaptive_boundary_exchange.py','src/analyses/audit_adaptive_rain_quadrature.py','src/analyses/review_adaptive_positivity.py']:
  print('FINAL PROCESS',command,flush=True)
  with(report/(Path(command).stem+'.log')).open('w')as log:subprocess.run([sys.executable,command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('REVIEW COMPLETE',flush=True)
if __name__=='__main__':main()
