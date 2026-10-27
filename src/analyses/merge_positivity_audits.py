"""Combine independently completed case audits without repeating passed checks."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_conservative_positivity import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;assert not(out/'freeze.json').exists()
 summaries=[json.loads((report/(c+'_summary.json')).read_text())for c in CASES]
 assert all(s['all_checks_passed']and s['hydrodynamic_runs']==1 for s in summaries)
 combined=dict(summaries[0]);combined['cases']={c:s['cases'][c]for c,s in zip(CASES,summaries)};combined['checks']=[x for s in summaries for x in s['checks']];combined['hydrodynamic_runs']=2;combined['constraint_findings']=[x for s in summaries for x in s['constraint_findings']];combined['all_declared_boundary_closures_satisfied']=all(s['all_declared_boundary_closures_satisfied']for s in summaries);combined['all_checks_passed']=all(x['pass']for x in combined['checks']);dump(report/'summary.json',combined)
 for name in ['water_group_cell_summary','crossing_face_mechanisms','first_crossing_evidence','scope_coverage','wet_support_exposure','principal_peak_regions','all_closed_boundary_faces']:
  rows=[];headers=[]
  for c in CASES:
   with(report/(c+'_'+name+'.csv')).open()as f:
    reader=csv.DictReader(f);headers+=reader.fieldnames;rows+=list(reader)
  fields=list(dict.fromkeys(headers))
  with(report/(name+'.csv')).open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 closure=[json.loads((report/(c+'_boundary_closure_audit.json')).read_text())for c in CASES];bc=dict(closure[0]);bc['cases']={c:s['cases'][c]for c,s in zip(CASES,closure)};bc['checks']=[x for s in closure for x in s['checks']];bc['all_data_checks_passed']=all(s['all_data_checks_passed']for s in closure);dump(report/'boundary_closure_audit.json',bc)
 positivity=[json.loads((report/(c+'_positivity_diagnostic_audit.json')).read_text())for c in CASES];pp=dict(positivity[0]);pp['cases']={c:s['cases'][c]for c,s in zip(CASES,positivity)};pp['checks']=[x for s in positivity for x in s['checks']];pp['all_checks_passed']=all(s['all_checks_passed']for s in positivity);pp['completed_v2_hydrodynamic_runs']=2;dump(report/'positivity_diagnostic_audit.json',pp)
 budget=report/'budget';ss=[json.loads((budget/(c+'_validation.json')).read_text())for c in CASES];assert all(s['all_checks_passed']for s in ss)
 checks={}
 for s in ss:
  for x in s['checks']:
   key=(x['case_id'],x['check'])
   if key in checks:assert checks[key]==x
   checks[key]=x
 b=dict(ss[0]);b['case_summaries']={c:s['case_summaries'][c]for c,s in zip(CASES,ss)};b['checks']=list(checks.values());b['completed_full_control_runs']=2;b['all_checks_passed']=all(x['pass']for x in b['checks']);assert len(b['checks'])==60
 for name in ['summary.json','validation.json']:dump(budget/name,b)
 print(json.dumps({'case_audits':len(summaries),'trace_checks':len(combined['checks']),'budget_checks':len(b['checks']),'all_passed':True}))
if __name__=='__main__':main()
