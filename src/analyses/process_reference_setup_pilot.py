"""Wait for completed authorized runs, audit each, then compare the matched grids."""
from pathlib import Path
import sys,json,time,subprocess,argparse,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_setup_pilot import OUT,REPORT,CONTROLS,PILOT,ADAPTIVE
from prepare_sfincs_cpu_build import dump

def audit_phase(phase):
 out=ROOT/OUT;report=ROOT/REPORT;manifest=json.loads((out/(phase+'_manifest.json')).read_text())
 for r in manifest['runs']:
  p=ROOT/r['folder'];dest=report/phase/r['case_id'];dest.mkdir(parents=True,exist_ok=True);receipt=dest/'audit.json';print('WAIT',phase,r['case_id'],flush=True)
  while not(p/'execution.json').exists():time.sleep(15)
  ex=json.loads((p/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'],'Preserve failed run'
  if receipt.exists()and json.loads(receipt.read_text())['all_checks_passed']:print('RETAIN AUDIT',r['case_id'],flush=True);continue
  print('AUDIT',phase,r['case_id'],flush=True)
  with(dest/'audit.log').open('w')as log:code=subprocess.run([sys.executable,'src/analyses/audit_reference_setup_pilot.py',phase,r['case_id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  if code and phase=='pilot'and receipt.exists():
   print('PILOT EVENT HOLD RETAINED',r['case_id'],flush=True)
  else:assert code==0,'Inspect saved audit finding; do not relax gates'
 print('ALL AUDITED',phase,flush=True)

def compare_controls():
 import reconstruct_adaptive_positivity_labels as reconstruction
 import compare_adaptive_positivity as comparison
 import compare_adaptive_boundary_exchange as boundary
 out=ROOT/OUT;report=ROOT/REPORT;dest=out/'mesh_comparison';dest.mkdir(exist_ok=True);rd=report/'mesh_comparison';rd.mkdir(exist_ok=True)
 regular={r['case_id']:r for r in json.loads((out/'qualification_manifest.json').read_text())['runs']};ad=json.loads((ROOT/ADAPTIVE/'run_manifest.json').read_text());rows=[]
 for r in ad['runs']:rows.append({**r,'regular_reference_folder':regular[r['case_id']]['folder']})
 dump(dest/'run_manifest.json',{'runs':rows,'new_hydraulic_runs':0,'both_grids_use_exact_rain_integral':True})
 rel=str(dest.relative_to(ROOT));rr=str(rd.relative_to(ROOT))
 reconstruction.OUT=rel;reconstruction.REPORT=rr;reconstruction.PRIOR=OUT+'/qualification';reconstruction.main()
 comparison.OUT=rel;comparison.REPORT=rr;comparison.PRIOR=OUT+'/qualification';comparison.main()
 boundary.OUT=rel;boundary.REPORT=rr;boundary.main()
 dump(report/'primary_grid_qualification.json',{'all_checks_passed':all(json.loads((report/'qualification'/c/'audit.json').read_text())['all_checks_passed']for c in CONTROLS),'controls':CONTROLS,'matched_corrected_adaptive_results_reused':True,'new_hydraulic_runs':2,'pilot8_gates_unlocked':True,'formal_grid_convergence_claim':False,'training':False})
 print('MATCHED CONTROLS COMPLETE; EIGHT-INPUT PILOT GATE UNLOCKED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['qualification','pilot']);a=p.parse_args();audit_phase(a.phase)
 if a.phase=='qualification':compare_controls()
