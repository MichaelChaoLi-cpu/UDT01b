"""Final regional vehicle-result delivery checks and version binding."""
from pathlib import Path
import sys,json
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_motor_flood_access_v1 import P,R,read,dump,sha

def main():
 checks=[]
 def check(n,v):
  checks.append(dict(check=n,passed=bool(v)))
  if not v:raise AssertionError(n)
 for name in ['preparation_validation','normal_route_reuse_validation','validation','browser_validation']:check(name,json.loads((R/(name+'.json')).read_text())['all_passed'])
 agg=read(R/'classification_summary.csv');scenarios=read(P/'scenario_index.csv');classes=read(R/'probe_classification.csv')
 check('624_network_states_and32604_cell_classifications',len(scenarios)==624 and len(classes)==32604)
 check('source_cells_always_partitioned',all(sum(int(x[k])for k in ['robust_modeled_loss','conditional_sensitivity','insufficient_evidence','retained_access'])==209 for x in agg))
 check('outward93_and_bidirectional89_evidence_cohorts_fixed',all(int(x['insufficient_evidence'])==(116 if x['direction']=='outward'else 120)for x in agg))
 check('peak_envelope_never_labeled_synchronous',all((x['frame_id']=='peak_envelope')==(x['simultaneous']=='False')for x in scenarios+agg+classes))
 witness=json.loads((R/'route_witness_summary.json').read_text());check('1672_directional_path_records_qualified',witness['records']==1672 and witness['all_direction_and_turn_rules_passed'])
 impacts=json.loads((R/'road_impact_summary.json').read_text());check('road_list_and_gis_feature_count_agree',impacts['core_intersecting_source_segments_triggered']==impacts['gis_features'])
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.paths=[]
  def handle_starttag(self,t,a):
   for k,v in a:
    if k in ['href','src']and v and not v.startswith(('http:','https:','#','data:')):self.paths.append(v)
 parser=Links();parser.feed((R/'review.html').read_text());future={'delivery_validation.json','delivery_manifest.json'};check('local_report_links_resolve',all((R/p).exists()or p in future for p in parser.paths))
 check('all_bound_original_inputs_unchanged',all(sha(ROOT/k)==v for k,v in json.loads((P/'source_bindings.json').read_text()).items()))
 summary=json.loads((R/'analysis_summary.json').read_text());check('scope_no_new_hydraulics_people_or_queue_simulation',all(summary[k]is False for k in ['new_hydraulics','people_counted','traffic_flow_or_queue_simulated']))
 dump(R/'delivery_validation.json',dict(all_passed=True,checks=checks));dump(R/'execution_state.json',dict(status='qualified_complete',network_cases_done=624,network_cases_total=624,new_hydraulics=False,people_counted=False,report=str((R/'review.html').relative_to(ROOT))))
 scripts=[ROOT/'src/preprocessing/prepare_motor_flood_access_v1.py']+[ROOT/'src/analyses'/name for name in ['run_motor_flood_access_v1.py','validate_motor_flood_access_v1.py','report_motor_flood_access_v1.py','export_motor_road_impacts_v1.py','validate_motor_flood_browser_v1.cjs','finalize_motor_flood_access_v1.py']]
 files=[p for p in P.rglob('*')if p.is_file()]+[p for p in R.iterdir()if p.is_file()and p.name!='delivery_manifest.json']+scripts;dump(R/'delivery_manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p)for p in files},new_hydraulics=False,people_counted=False));print('CAR DELIVERY QUALIFIED',len(checks),'CHECKS',len(files),'FILES')
if __name__=='__main__':main()
