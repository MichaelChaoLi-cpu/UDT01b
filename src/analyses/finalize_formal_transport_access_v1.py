"""Final evidence gate, local asset checks and reproducible delivery manifest."""
from pathlib import Path
import os,sys,json,csv
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_formal_transport_access_v1 import P,R,read,dump,sha

def main():
 checks=[]
 def check(name,ok):
  checks.append(dict(check=name,passed=bool(ok)))
  if not ok:raise AssertionError(name)
 for name in ['preparation_validation','independent_validation','route_witness_validation','projection_direct_validation','browser_validation']:
  j=json.loads((R/(name+'.json')).read_text());check(name,j['all_passed'])
 scenarios=read(P/'scenario_index.csv');contexts=read(P/'road_context_index.csv');aggregates=read(R/'classification_summary.csv');changes=read(R/'geometry_classification_changes.csv')
 check('all9984_scenario_indices_sequential',len(scenarios)==9984 and all(int(x['scenario_index'])==i for i,x in enumerate(scenarios)))
 check('all1248_contexts_sequential',len(contexts)==1248 and all(int(x['road_context_index'])==i for i,x in enumerate(contexts)))
 check('synchronous_vs_peak_labels_consistent',all((x['frame_id']=='peak_envelope')==(x['simultaneous']=='False')for x in scenarios+contexts+aggregates+changes))
 check('each_aggregate_partitions_its_probe_cohort',all(sum(int(x[k])for k in ['robust_modeled_disconnection','conditional_sensitivity','insufficient_evidence','stable_modeled_alternative'])==int(x['origin_probes'])for x in aggregates))
 check('outward_and_two_direction_summaries_complete',len(aggregates)==12480 and len(changes)==3744)
 check('unchanged_normal_support_and_evidence_gaps',all(int(x['insufficient_evidence'])==(94 if x['projection']=='strict_classified_road'else 65)for x in aggregates if x['ward']=='all'))
 with(R/'probe_classification.csv').open(encoding='utf-8-sig')as f:
  n=sum(1 for _ in csv.DictReader(f))
 check('all260832_probe_state_classifications_present',n==260832)
 payload=json.loads((R/'map_states.json').read_text());check('interactive_state_payload_complete',len(payload['states'])==1248 and len(payload['points'])==209 and all(len(v)==209 for state in payload['states'].values()for v in state.values()))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.paths=[]
  def handle_starttag(self,t,attrs):
   for key,value in attrs:
    if key in ['href','src']and value and not value.startswith(('http:','https:','#','data:')):self.paths.append(value)
 parser=Links();parser.feed((R/'review.html').read_text());future={'delivery_validation.json','delivery_manifest.json'}
 check('all_local_report_links_resolve',all((R/p).exists()or p in future for p in parser.paths))
 bindings=json.loads((P/'source_bindings.json').read_text());check('all_original_bound_sources_still_unchanged',all(sha(ROOT/k)==v for k,v in bindings.items()))
 s=json.loads((R/'analysis_summary.json').read_text());check('no_hydraulics_population_training_or_sealed_tests',all(s[k]is False for k in ['new_hydraulics','people_counted','training','sealed_tests_opened']))
 dump(R/'delivery_validation.json',dict(all_passed=True,checks=checks))
 state=json.loads((R/'execution_state.json').read_text());state.update(status='qualified_complete',independent_network_checks_passed=True,direct_projection_checks_passed=True,route_witnesses_passed=True,browser_checks_passed=True,report='data/exp/estimation/three_ward_formal_transport_access_v1/review.html');dump(R/'execution_state.json',state)
 scripts=[ROOT/'src/preprocessing/prepare_formal_transport_access_v1.py']+[ROOT/'src/analyses'/n for n in ['compare_formal_transport_access_v1.py','validate_formal_transport_access_v1.py','audit_formal_transport_projection_v1.py','export_formal_transport_routes_v1.py','report_formal_transport_access_v1.py','validate_formal_transport_browser_v1.cjs','finalize_formal_transport_access_v1.py']]
 files=[p for p in P.rglob('*')if p.is_file()]+[p for p in R.iterdir()if p.is_file()and p.name!='delivery_manifest.json']+scripts
 dump(R/'delivery_manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p)for p in files},source_bindings='data/processed/three_ward_formal_transport_access_v1/source_bindings.json',network_evaluations=9984,people_counted=False,new_hydraulics=False))
 print('DELIVERY QUALIFIED',len(checks),'CHECKS',len(files),'FILES')
if __name__=='__main__':main()
