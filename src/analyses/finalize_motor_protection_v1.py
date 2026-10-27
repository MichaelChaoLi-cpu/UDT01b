"""Bind new deliverables and verify frozen intervention/source contracts."""
from pathlib import Path
import sys,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from compare_motor_road_protection_v1 import P,R,SOURCE,read,dump,load,sha

def main():
 checks=[]
 def check(name,value):checks.append(dict(check=name,passed=bool(value)));assert value,name
 for f in ['validation.json','path_validation.json','browser_validation.json']:check(f,json.loads((R/f).read_text())['all_passed'])
 for f in ['source_bindings.json','frozen_selection_binding.json']:
  check(f,all(sha(ROOT/name)==h for name,h in json.loads((P/f).read_text()).items()))
 summary=json.loads((R/'summary.json').read_text());contract=json.loads((P/'protocol.json').read_text());rows=read(R/'classification_summary.csv');check('all1872same_network_states_complete',len(rows)==3*12*13*2 and summary['new_intervention_evaluations']==1248)
 check('no_hydraulics_training_sealedtests_people',all(contract[k]is False for k in ['new_hydraulics','training','sealed_tests_opened','people_counted']))
 check('no_claimed_engineering_validation',summary['engineering_feasibility_demonstrated']is False)
 check('equal_length_tolerance',summary['length_difference_m']<=contract['length_match_tolerance_m'])
 check('unchanged_unknown_and_support_rules',summary['outward_common_probes']==93 and summary['two_direction_common_probes']==89 and all(int(r['insufficient_evidence'])==(116 if r['direction']=='outward' else 120)for r in rows))
 text=(R/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',text);missing=[s for s in links if not s.startswith(('http:','https:','#'))and not(R/s).resolve().exists()and s not in ['delivery_validation.json','delivery_manifest.json']];check('all_local_report_assets_resolve',not missing)
 feature=json.loads((P/'protected_roads_4326.geojson').read_text())['features'];check('gis_inventory_agrees',len(feature)==len(read(P/'protected_road_inventory.csv')))
 check('frozen_selection_no_test_outcome_search',contract['planning_index']==282 and contract['selection_case']['case_id']=='R1__L__U')
 check('non_simultaneous_peak_explicit',all(r['simultaneous']==('False'if r['frame_id']=='peak_envelope'else'True')for r in rows))
 dump(R/'delivery_validation.json',dict(all_passed=True,count=len(checks),checks=checks))
 state=json.loads((R/'execution_state.json').read_text());state['status']='qualified_complete';state['delivery_validation']='delivery_validation.json';dump(R/'execution_state.json',state)
 files=[f for f in P.rglob('*')if f.is_file()and'superseded_selection_before_outcome_review'not in f.parts]+[f for f in R.rglob('*')if f.is_file()and f.name!='delivery_manifest.json']+[ROOT/'src/analyses'/n for n in ['compare_motor_road_protection_v1.py','report_motor_road_protection_v1.py','validate_motor_protection_paths_v1.py','validate_motor_protection_browser_v1.cjs','finalize_motor_protection_v1.py']]
 dump(R/'delivery_manifest.json',dict(files={str(f.relative_to(ROOT)):sha(f)for f in files},original_source_binding=str((P/'source_bindings.json').relative_to(ROOT)),frozen_selection_binding=str((P/'frozen_selection_binding.json').relative_to(ROOT)),new_hydraulics=False,people_counted=False))
 print('PROTECTION DELIVERY COMPLETE',len(checks),'checks;',len(files),'bound files')
if __name__=='__main__':main()
