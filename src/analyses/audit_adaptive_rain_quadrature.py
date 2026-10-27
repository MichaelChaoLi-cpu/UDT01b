"""Explain paired rain-volume differences using existing actual time steps; no new simulation."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_adaptive_positivity import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump
from audit_adaptive_positivity_controls import structured

def main():
 out=ROOT/OUT;report=ROOT/REPORT;area=float(json.loads((ROOT/'data/processed/three_ward_adaptive_connectivity/allocation5_complete.json').read_text())['total_contributing_area_m2']);rows=[];pairs={};checks=[]
 for run in json.loads((out/'run_manifest.json').read_text())['runs']:
  case=run['case_id'];values={};peak=0. if case=='dry_raised'else 60.
  for grid,rel in [('regular',run['regular_reference_folder']),('adaptive',run['folder'])]:
   path=ROOT/rel;assert json.loads((path/'execution.json').read_text())['completion_marker'];d=structured(path/'udt_step_budget.csv');t=d['actual_state_time'];dt=d['actual_step_dt'];knots=np.array([0,3600,3601,7200,7201,14400.]);rates=np.array([0,0,peak,peak,0,0.])*area/3.6e6;continuous=float(np.trapezoid(rates,knots));right=float(np.sum(np.interp(t,knots,rates)*dt));actual=float(d['rain_applied_cumulative'][-1]);roundoff=actual-right;bound=float(32*np.finfo('f4').eps*max(1.,actual));ok=abs(roundoff)<=bound
   row={'case':case,'grid':grid,'continuous_original_allocation_m3':continuous,'actual_step_right_endpoint_m3':right,'native_applied_m3':actual,'right_endpoint_minus_continuous_m3':right-continuous,'native_minus_right_endpoint_m3':roundoff,'float32_consistency_bound_m3':bound,'same_original_allocation_area_m2':area,'within_float32_consistency_bound':bool(ok)};rows.append(row);values[grid]=row;checks.append({'case':case,'check':grid+'_native_rain_explained_by_actual_step_quadrature_and_float32_arithmetic','pass':bool(ok)})
  r=values['regular'];a=values['adaptive'];delta=a['native_applied_m3']-r['native_applied_m3'];qdelta=a['actual_step_right_endpoint_m3']-r['actual_step_right_endpoint_m3'];remainder=a['native_minus_right_endpoint_m3']-r['native_minus_right_endpoint_m3'];pairs[case]={'applied_rain_difference_m3':delta,'actual_step_quadrature_difference_m3':qdelta,'native_arithmetic_remainder_difference_m3':remainder,'decomposition_error_m3':delta-qdelta-remainder,'applied_difference_pct':delta/r['native_applied_m3']*100 if r['native_applied_m3']else None};checks.append({'case':case,'check':'paired_rain_difference_decomposes_without_budget_or_forcing_repair','pass':bool(abs(delta-qdelta-remainder)<1e-8)})
 result={'rows':rows,'cases':pairs,'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'interpretation':'Original rain knots and nested roof allocation are fixed. Different actual CFL steps change right-endpoint integration on one-second onset/cessation ramps. Native single-precision interpolation and source arithmetic leave a separately reported remainder. This explanatory check does not alter the frozen1e-6 final rain admission gate, repair an input curve or certify a bound on propagated local-depth error.','new_simulations':0,'training':0};dump(report/'rain_quadrature_audit.json',result);print(json.dumps(pairs,indent=2));assert result['all_checks_passed']
if __name__=='__main__':main()
