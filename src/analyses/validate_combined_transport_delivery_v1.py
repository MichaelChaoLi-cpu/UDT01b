"""Independent numerical,policy-ledger and delivery checks for all fixed cells."""
from pathlib import Path
import json,sys,hashlib
from html.parser import HTMLParser
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import run_combined_transport_measures_v1 as m
P=m.P;R=m.R
def main():
 m.check();vals=[json.loads(f.read_text())for f in R.glob('*_fit_validation.json')];paths=[json.loads(f.read_text())for f in R.glob('*_path_validation.json')]
 assert len(vals)==40 and all(v['passed']for v in vals)
 assert len(paths)==4 and all(v['passed']for v in paths)
 assert sum(v['inherited_source']is not None for v in vals)==3
 assert sum(v['seconds']for v in vals)<=5400 and max(v['seconds']for v in vals)<=120
 assert sum(v['seconds']for v in paths)<=900 and max(v['seconds']for v in paths)<=180
 versions={v['code_sha256']for v in vals+paths};assert len(versions)==1
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 assert next(iter(versions))==sha(ROOT/'src/analyses/run_combined_transport_measures_v1.py')
 dem=m.load(m.prev.REP/'multi_routing_demands.npz');q0=m.original_request(dem);pr=m.load(m.prev.M/'road_projection.npz');states={};by={};rule_error=0.;ledger_error=0.
 for v in vals:
  case=v['case_id'];arm=v['arm'];r=v['deferred_fraction'];g=v['guidance'];z=m.load(P/'fits'/(m.key(case,arm,r,g)+'.npz'));by[case,arm,r,g]=z
  if(case,arm)not in states:states[case,arm]=m.state(case,arm)
  closed,slow,tref,capacity=states[case,arm];expected=tref*(1+.15*(z['arc_flow']/capacity)**4);relative=float(np.max(abs(z['arc_cost']-expected)/np.maximum(1,expected)));rule_error=max(rule_error,relative);assert relative<1e-10
  assert np.allclose(z['original_requested'],q0,rtol=0,atol=1e-10);ret=q0*(1-r);assert np.allclose(z['retained_requested'],ret,rtol=0,atol=1e-10)
  deferred=q0-ret;served=ret*z['reachable'];failed=ret*~z['reachable'];error=float(abs(q0-deferred-served-failed).max());ledger_error=max(ledger_error,error);assert error<1e-8
  assert float(z['arc_flow'][closed[pr['arc_to_road']]].max(initial=0))<1e-9
 for (case,arm,r,g),z in by.items():
  if g!='SO':continue
  ue=by[case,arm,r,'UE'];assert np.array_equal(z['retained_requested'],ue['retained_requested']);assert np.array_equal(z['reachable'],ue['reachable']);assert float(z['total_model_cost'])<=float(ue['total_model_cost'])*(1+1e-6)
 df=pd.read_csv(P/'combined_outcomes.csv');expected={(c,a,r,g)for c in m.CASES for a in ['flood','targeted']for r in m.REDUCTIONS for g in ['UE','SO']};actual=set(zip(df.case_id,df.protection,df.deferred_fraction,df.guidance));assert actual==expected
 assert np.allclose(df.original_core_request,df.deferred_core_proxy+df.retained_served_core_proxy+df.retained_unreachable_core_proxy,rtol=0,atol=1e-6)
 assert json.loads((R/'browser_validation.json').read_text())['passed'];assert json.loads((R/'interpretation_validation.json').read_text())['passed']
 class Parser(HTMLParser):
  def __init__(self):super().__init__();self.refs=[]
  def handle_starttag(self,t,attrs):
   for k,v in attrs:
    if k in ['href','src']and not v.startswith(('http:','https:','data:','#')):self.refs.append(v)
 p=Parser();p.feed((R/'review.html').read_text());assert all((R/f).exists()for f in p.refs)
 checks={'40_fit_and4_path_gates':True,'37_new3_exact_inherited_cells':True,'same_source_and_predeclared_budgets':True,'cost_function_all_cells_independent':True,'original_request_ledger_per_OD':True,'same_retained_demand_in_guidance_pairs':True,'all32_intervention_cells_present':True,'SO_network_objective_not_worse_than_UE':True,'browser_images_mobile_and_local_links':True}
 files=[ROOT/'src/analyses/run_combined_transport_measures_v1.py',ROOT/'src/analyses/finalize_combined_transport_v1.py',ROOT/'src/analyses/validate_combined_transport_browser_v1.cjs',Path(__file__),ROOT/'src/analyses/demand_protection_feedback_v1.py',ROOT/'src/analyses/observed_load_common_v1.py']
 manifest=dict(passed=True,checks=checks,cost_function_max_relative_error=rule_error,original_per_OD_ledger_max_error=ledger_error,max_restricted_gap=max(v['gap']for v in vals),active_optimizer_seconds=sum(v['seconds']for v in vals),active_path_seconds=sum(v['seconds']for v in paths),source_sha256={str(f.relative_to(ROOT)):sha(f)for f in files},protocol_sha256=sha(P/'protocol.json'),reproduction=['run_combined_transport_measures_v1.py prepare','run_combined_transport_measures_v1.py paths','run_combined_transport_measures_v1.py fit --case normal','run_combined_transport_measures_v1.py fit --case E20250911__H__U','run_combined_transport_measures_v1.py fit --case E20240830__H__U','run_combined_transport_measures_v1.py report','finalize_combined_transport_v1.py','validate_combined_transport_browser_v1.cjs','validate_combined_transport_delivery_v1.py'],not_done=['new hydraulics','new training','people impacts','real queue delay','observed historical traffic validation'])
 (R/'delivery_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items()if k not in ['source_sha256','reproduction']},indent=2))
if __name__=='__main__':main()
