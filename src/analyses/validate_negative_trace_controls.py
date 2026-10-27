"""Delivery/provenance checks for exactly two actual crossing-diagnostic replays."""
from pathlib import Path
import sys,json,re,csv,argparse,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_negative_trace_controls import OUT,REPORT,CASES,PRIOR,INTERFACE
from prepare_sfincs_cpu_build import sha,dump
from reference_label_policy import event_admission
from load_three_ward_reference_labels import load_reference
from audit_three_ward_negative_volumes import readnp

def main(freeze=False):
 out=ROOT/OUT;report=ROOT/REPORT;checks=[];cache={}
 def h(p):
  st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 summary=json.loads((report/'summary.json').read_text());budget=json.loads((report/'budget/summary.json').read_text());manifest=json.loads((out/'run_manifest.json').read_text());solver=json.loads((ROOT/INTERFACE/'solver_binding.json').read_text())
 check('exactly_two_authorized_legacy_controls_no_training_or_batch',list(summary['cases'])==CASES and summary['hydrodynamic_runs']==2 and summary['new_parent_events']==summary['training']==summary['batch_58']==0 and len(manifest['runs'])==2)
 check('all_actual_transition_and_label_support_checks_pass',summary['all_checks_passed']and len(summary['checks'])>=24 and all(x['pass']for x in summary['checks']))
 check('all_60_actual_clock_same_step_budget_and_output_checks_pass',budget['all_checks_passed']and len(budget['checks'])==60 and all(x['pass']for x in budget['checks']))
 rows=list(csv.DictReader((report/'first_crossing_evidence.csv').open()));codes=json.loads((ROOT/INTERFACE/'output_schema.json').read_text())['groups']
 for run in manifest['runs']:
  case=run['case_id'];folder=ROOT/run['folder'];previous=ROOT/run['previous_e2e_folder'];ex=json.loads((folder/'execution.json').read_text());ident=run['identity']
  check(case+'_completed_4h_bound_to_frozen_trace_image',ex['returncode']==0 and ex['completion_marker']and ex['solver_image_id']==solver['image_id']==ident['solver_digest']and ident['patch_sha256']==solver['patch_sha256'])
  check(case+'_all_original_scientific_input_bytes_unchanged',all(h(folder/name)==h(previous/name)==value for name,value in run['scientific_input_sha256'].items()))
  check(case+'_trace_metadata_required_and_unchanged',run['trace_required']and all(h(folder/name)==run['input_sha256'][name]for name in ['udt_negative_native_map.csv','udt_negative_trace.request'])and'schema=negative-cell-face-v1'in(folder/'udt_negative_trace_receipt.txt').read_text())
  check(case+'_lineage_and_partition_remain_existing_open_control',ident['parent_id']==ident['family_id']=='CONTROL-LEGACY'and ident['split']=='open_control'and not run['library_descendant'])
  r=[r for r in rows if r['case']==case];x=summary['cases'][case]
  check(case+'_one_first_transition_per_ever_negative_cell',len(r)==x['ever_negative_cells']and len({int(a['native_cell_index'])for a in r})==len(r)and all(set(a['water_group_links'].split(';'))<=set(codes)for a in r))
  check(case+'_first_transition_equation_independently_reproduces_new_volume',all(abs(float(a['after_volume_m3'])-(float(a['before_volume_m3'])+float(a['horizontal_applied_m3'])+float(a['rain_postadd_increment_m3'])+float(a['storage_update_remainder_m3'])))<1e-9 for a in r))
  assessment=json.loads((report/'reference_admission_assessment.json').read_text())[case];masks=readnp(out/(case+'_candidate_reference.npz'));preview=event_admission(assessment['required_gates'],masks)
  check(case+'_candidate_admission_uses_unchanged_policy_and_complete_gates',preview==assessment['frozen_policy_function_preview']and not assessment['formal_registry_changed']and not assessment['training_release'])
  try:load_reference(case,root=ROOT)
  except PermissionError:closed=True
  else:closed=False
  check(case+'_historical_default_loader_stays_closed',closed)
 closure=json.loads((report/'boundary_closure_audit.json').read_text())
 check('all_closed_boundary_history_audits_pass_with_model_failures_explicit',closure['all_data_checks_passed']and not summary['all_declared_boundary_closures_satisfied']and len(summary['constraint_findings'])==2)
 assessments=json.loads((report/'reference_admission_assessment.json').read_text())
 check('both_controls_held_by_geometry_no_spatial_mask_rescue',all(not v['required_gates']['geometry']and not v['candidate_partial_numerical_reference']and not closure['cases'][c]['geometry_admission_gate']for c,v in assessments.items()))
 # A zero rate or small total budget cannot establish local nonnegativity or error bounds.
 check('same_output_comparison_not_claimed_as_physical_or_corrected_grid_convergence',all(not v['propagated_error_bounded']and not v['exact_continuous_crossing_time']for v in summary['cases'].values())and not summary['real_world_flood_accuracy_verified'])
 for rel in [PRIOR,'data/processed/three_ward_negative_volume_audit_v1',INTERFACE]:
  frozen=json.loads((ROOT/rel/'freeze.json').read_text());check(Path(rel).name+'_all_frozen_files_unchanged',all(h(ROOT/p)==value for p,value in frozen['files_sha256'].items()))
 prov=json.loads((report/'provenance.json').read_text());check('all_bound_source_inputs_and_reference_policies_unchanged',all(h(ROOT/p)==value for p,value in prov['protected_sha256'].items()))
 pending=json.loads((ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs'];check('all_58_future_cases_remain_unrun',len(pending)==58 and all(r['status']=='unrun'and r['response_path']is None for r in pending))
 before=(report/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text();split=lambda t:re.split(r'(?=^## [1-8]\. )',t,flags=re.M)
 aa,bb=split(before),split(after);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(aa,bb)if a!=b}
 check('AnaSOP_only_skill_routed_4_6_7_no_new_final_variables',changed=={4,6,7}and before.count('| yes |')==after.count('| yes |')and all(b.isascii()for b in bb if b.startswith(('## 4. ','## 6. ','## 7. '))))
 page=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',page);check('all_review_links_and_figure_exist',all((report/p).is_file()for p in links))
 inspect=subprocess.run(['docker','image','inspect',solver['image_id']],capture_output=True,text=True);check('actual_frozen_CPU_image_available',inspect.returncode==0 and json.loads(inspect.stdout)[0]['Architecture']=='amd64')
 if freeze:
  assert all(x['pass']for x in checks)
  paths=[p for base in [out,report]for p in base.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery-validation.json']]
  paths+=[ROOT/'src/preprocessing/prepare_negative_trace_controls.py']
  paths+=[ROOT/'src/analyses'/name for name in ['run_negative_trace_controls.py','audit_negative_trace_controls.py','review_negative_trace_controls.py','validate_negative_trace_controls.py','audit_trace_boundary_closures.py']]
  dump(out/'freeze.json',{'state':'two_actual_controls_diagnostics_audited_both_geometry_hold','files_sha256':{str(p.relative_to(ROOT)):h(p)for p in sorted(paths)},'image_id':solver['image_id'],'hydrodynamic_runs':2,'new_parent_events':0,'training':0,'batch_58':0,'formal_reference_release':False})
 if(out/'freeze.json').exists():
  f=json.loads((out/'freeze.json').read_text());check('all_current_frozen_delivery_hashes_match',all(h(ROOT/p)==value for p,value in f['files_sha256'].items()))
 result={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'hydrodynamic_runs':2,'training':0,'batch_58':0,'formal_reference_release':False};dump(report/'delivery-validation.json',result);print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');main(a.parse_args().freeze)
