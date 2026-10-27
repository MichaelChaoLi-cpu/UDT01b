"""Final delivery checks after actual8case numerical and motor-access qualification."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_cross_event_protection_v1 import P,R,D,MEASURE,read,dump,sha

def main():
 checks=[]
 def check(name,v):checks.append(dict(check=name,passed=bool(v)));assert v,name
 manifest=json.loads((P/'run_manifest.json').read_text());state=json.loads((R/'campaign_state.json').read_text());check('8numericallyqualified_cases',len(state['completed_new_cases'])==8 and set(state['completed_new_cases'])==set(manifest['new_execution_order']))
 for run in manifest['runs']:
  a=json.loads((R/'execution'/run['case_id']/'audit.json').read_text());check(run['case_id']+'_original_gates',a['all_checks_passed']and not a['training']and not a['formal_reference_release']);check(run['case_id']+'_protected_input_identity',all(sha(ROOT/run['folder']/f)==h for f,h in run['input_sha256'].items()))
 for name in ['input_validation.json','plan_validation.json','motor_validation.json']:check(name,json.loads((R/name).read_text())['all_passed'])
 for name in ['protected_sources.json','execution_code_binding.json']:check(name,all(sha(ROOT/f)==h for f,h in json.loads((P/name).read_text()).items()))
 check('fixed_measures_unchanged',all(sha(ROOT/f)==h for f,h in json.loads((MEASURE/'frozen_selection_binding.json').read_text()).items()))
 summary=json.loads((R/'motor_summary.json').read_text());check('same_original_support_upper_bound',summary['eligible_outward']<=93 and summary['eligible_two_direction']<=89);check('1248network_states',summary['network_evaluations']==1248 and summary['hydraulic_cases']==8)
 rows=read(R/'motor_classification_summary.csv');check('209points_partition_all624armcaseframe_directions',len(rows)==3*8*13*2 and all(sum(int(r[k])for k in ['robust_modeled_loss','conditional_sensitivity','retained_access','insufficient_evidence'])==209 for r in rows));check('no_non_simultaneous_peak_as_duration',all(r['simultaneous']==('False'if r['frame_id']=='peak_envelope'else'True')for r in rows))
 text=(R/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',text);check('all_local_report_links_resolve',all((R/s).resolve().exists()for s in links if not s.startswith(('http:','https:','#'))))
 dump(R/'delivery_validation.json',dict(all_passed=True,count=len(checks),checks=checks))
 files=[D/'protocol.json',D/'rainfall_selection_protocol.json',P/'event_selection.json',P/'run_manifest.json',P/'protected_sources.json',P/'execution_code_binding.json',P/'audit_metadata_adapter.json',P/'motor_common_support.npz',P/'motor_accessibility.npz',P/'motor_scenario_index.csv',P/'restored_route_witnesses.jsonl']+[f for f in R.rglob('*')if f.is_file()and f.name not in ['delivery_manifest.json','campaign.log','live_report.log']]
 dump(R/'delivery_manifest.json',dict(files={str(f.relative_to(ROOT)):sha(f)for f in files},solver_outputs=[dict(case_id=run['case_id'],folder=run['folder'],files={f.name:dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)for f in(ROOT/run['folder']).iterdir()if f.is_file()and f.name.startswith('sfincs_')})for run in manifest['runs']],source_products='Original radar bytes bound in per-event acquisition manifests; native map numerical state audited with original28checks.',fixed_measures=True,new_hydraulic_cases=8,training=False,people_counted=False,sealed_tests_opened=False))
 print('CROSS-EVENT DELIVERY VALIDATED',len(checks))
if __name__=='__main__':main()
