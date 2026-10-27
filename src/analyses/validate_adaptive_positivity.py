"""Provenance, evidence, scope and delivery checks for the bounded conservative grid comparison."""
from pathlib import Path
import sys,json,re,csv,argparse,subprocess,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_adaptive_positivity import OUT,REPORT,PRIOR,CASES
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
 manifest=json.loads((out/'run_manifest.json').read_text());summary=json.loads((report/'summary.json').read_text());assessment=json.loads((report/'reference_admission_assessment.json').read_text());audit=json.loads((report/'audit.json').read_text());qual=json.loads((out/'qualification.json').read_text());solver=json.loads((out/'solver_binding.json').read_text());build=json.loads((report/'build_validation.json').read_text());binding=json.loads((out/'source_binding.json').read_text());geom=json.loads((report/'matched_boundary_geometry.json').read_text());exchange=json.loads((report/'boundary_comparison.json').read_text());spatial=json.loads((report/'spatial_comparison.json').read_text())
 check('two_existing_open_control_companions_only',len(manifest['runs'])==2 and[run['case_id']for run in manifest['runs']]==CASES and not manifest['pending_58_scenarios_authorized']and manifest['new_parent_count']==0 and not manifest['training'])
 check('source_and_fixture_binding_complete',all(h(out/'context'/name)==value for name,value in binding['source_sha256'].items())and h(out/'full_solver.patch')==binding['full_patch_sha256']==solver['patch_sha256'])
 bounds=json.loads((report/'depth_bound_fixture_validation.json').read_text());check('24_brute_force_asynchronous_and_shallow_depth_bound_fixtures_pass',bounds['all_checks_passed']and len(bounds['checks'])==24 and all(c['censored_local_interval_retained']for c in bounds['checks']))
 check('production_regular_and_both_adaptive_fixture_families_pass',build['all_checks_passed']and all(c['pass']for c in build['checks'])and any(c['check']=='missing_child_production_fixture'for c in build['checks']))
 for kind in ['regular','adaptive','missing_child']:
  path=out/'link_tests_v2'/kind;parallel=out/'link_tests_v2'/(kind+'_parallel');check(kind+'_serial_parallel_results_identical',(path/'stdout.txt').read_bytes()==(parallel/'stdout.txt').read_bytes()and'PASS'in(path/'stdout.txt').read_text())
 check('qualified_regular_source_branch_reused_without_changed_results',(out/'link_tests_v2/regular/stdout.txt').read_bytes()==(ROOT/PRIOR/'link_tests/normal/stdout.txt').read_bytes())
 for case in CASES:
  run=next(r for r in manifest['runs']if r['case_id']==case);path=ROOT/run['folder'];ex=json.loads((path/'execution.json').read_text());masks=readnp(out/(case+'_adaptive_candidate_reference.npz'));a=assessment[case]
  check(case+'_full_control_completed_with_bound_image',ex['returncode']==0 and ex['completion_marker']and ex['solver_image_id']==solver['image_id'])
  check(case+'_all_staged_input_bytes_preserved',all(h(path/name)==v for name,v in run['input_sha256'].items()))
  source=ROOT/run['source_folder'];check(case+'_historical_adaptive_inputs_unchanged',all(h(source/name)==v for name,v in run['source_input_sha256'].items()))
  check(case+'_lineage_is_existing_control_not_new_library_parent',run['original_parent_family']=='CONTROL-LEGACY'and run['split']=='open_control'and not run['new_parent_event'])
  check(case+'_all_numerical_findings_transparently_retained',a['all_numerical_checks_passed']==all(c['pass']for c in audit['checks']if c['case']==case)and a['failed_checks']==sorted(c['check']for c in audit['checks']if c['case']==case and not c['pass']))
  check(case+'_strict_every_step_nonnegativity_verified',a['strict_positivity_verified']and summary['cases'][case]['numerics']['minimum_native_volume_m3']>=0 and summary['cases'][case]['numerics']['negative_native_cells']==0)
  check(case+'_admission_reuses_frozen_policy_without_mask_rescue',event_admission(a['required_gates'],masks)==a['preview']and not a['formal_registry_changed']and not a['training_release'])
  try:load_reference(case,root=ROOT)
  except PermissionError:closed=True
  else:closed=False
  check(case+'_default_reference_loader_remains_closed',closed)
  rec=json.loads((report/(case+'_label_reconstruction.json')).read_text());check(case+'_ground_reconstruction_reproduces_frozen_regular_bounds',rec['quarter_ground_area_reaggregation_exact']and rec['cases'][0]['regular_depth_reconstruction_max_difference_m']<1e-12 and rec['cases'][0]['common_ground_area_m2']==rec['cases'][0]['local_ground_area_m2']==25936197.)
 check('open_receiving_geometry_identical_at_original0_50m',geom['all_physical_open_receiving_segments_equal']and geom['regular_only_open_segments']==geom['adaptive_only_open_segments']==0)
 rainq=json.loads((report/'rain_quadrature_audit.json').read_text());check('actual_clock_rain_difference_explained_without_relaxing_original_admission',rainq['all_checks_passed']and all(c['pass']for c in rainq['checks']))
 check('directional_exchange_and_spatial_interfaces_verified',exchange['all_checks_passed']and spatial['all_checks_passed'])
 frozen=json.loads((ROOT/PRIOR/'freeze.json').read_text());check('prior_qualified_regular_delivery_unchanged',all(h(ROOT/p)==value for p,value in frozen['files_sha256'].items()))
 early=ROOT/'data/processed/three_ward_adaptive_positivity_v1';ef=json.loads((early/'early-attempt-freeze.json').read_text());status=json.loads((early/'attempt_status.json').read_text());check('stopped_unqualified_attempt_and_failure_evidence_frozen',not status['qualified']and status['completed_controls']==0 and not status['rain_started']and'negative native updated storage'in(early/'missing_child_tests/serial/stderr.txt').read_text()and all(h(ROOT/p)==v for p,v in ef['files_sha256'].items()))
 prov=json.loads((report/'provenance.json').read_text());check('reference_registry_and_prospective_manifest_unchanged',h(ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json')==prov['registry_sha256']and h(ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json')==prov['pending58_manifest_sha256'])
 pending=json.loads((ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs'];check('all58_future_runs_still_unrun',len(pending)==58 and all(r['status']=='unrun'and r['response_path']is None for r in pending))
 check('no_training_no_new_parent_no_accuracy_or_convergence_claim',summary['training']==summary['batch58']==0 and not qual['training_release']and qual['new_parent_events']==0 and not qual['formal_grid_convergence_verified']and not qual['physical_accuracy_verified'])
 image=subprocess.run(['docker','image','inspect',solver['image_id']],text=True,capture_output=True);check('immutable_CPU_image_available',image.returncode==0 and json.loads(image.stdout)[0]['Architecture']=='amd64')
 before=(report/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text();old=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);new=re.split(r'(?=^## [1-8]\. )',after,flags=re.M);changed=[int(re.match(r'## (\d)',a).group(1))for a,b in zip(old,new)if a!=b];check('AnaSOP_only_responsible_sections_changed',len(old)==len(new)and set(changed)=={4,6,7}and old[0]==new[0]and after.isascii())
 update=json.loads((report/'anasop-update.json').read_text());check('no_new_final_variable_or_formula_and_skills_recorded',update['new_final_variables']==update['new_formulas']==0 and set(update['skills'])=={'data-preprocessing','estimation-framework-planning'})
 page=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',page);check('all_report_local_links_resolve',all((report/p).is_file()for p in links if not p.startswith(('http:','https:','#'))))
 check('two_scientific_figures_present',all((report/n).stat().st_size>30000 for n in ['whole_domain_grid_comparison.png','boundary_directional_exchange.png']))
 scripts=list((ROOT/'src/analyses').glob('*adaptive*positivity*.py'))+[ROOT/'src/analyses/compare_adaptive_boundary_exchange.py',ROOT/'src/preprocessing/prepare_adaptive_positivity.py',ROOT/'src/preprocessing/reconstruct_adaptive_positivity_labels.py',ROOT/'src/analyses/verify_adaptive_depth_bounds.py',ROOT/'src/analyses/qualify_adaptive_missing_child.py',ROOT/'src/analyses/audit_adaptive_rain_quadrature.py']
 for p in scripts:ast.parse(p.read_text())
 check('all_new_reproducible_Python_sources_parse',True)
 if(out/'freeze.json').exists():check('current_delivery_frozen_hashes_match',all(h(ROOT/p)==v for p,v in json.loads((out/'freeze.json').read_text())['files_sha256'].items()))
 result={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'numerical_reference_gate_pass':audit['all_checks_passed'],'failed_numerical_gates_reported_not_rescued':True,'training':0,'batch58':0,'new_parent_events':0};dump(report/'delivery_validation.json',result)
 assert result['all_checks_passed'],[c for c in checks if not c['pass']]
 if freeze:
  paths=set()
  for folder in [out,report]:
   paths.update(p for p in folder.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery_validation.json'])
  paths.update(scripts);paths.update(ROOT/'src/preprocessing'/name for name in ['sfincs_adaptive_positivity.f90.inc','sfincs_adaptive_diagnostics.f90.inc','adaptive_positivity_fixture.f90','adaptive_missing_child_fixture.f90'])
  dump(out/'freeze.json',{'state':'two_completed_conservative_grid_controls_one_stopped_attempt_retained_no_training_release','files_sha256':{str(p.relative_to(ROOT)):h(p)for p in sorted(paths)}})
 print(json.dumps({'checks':len(checks),'pass':True,'numerical_reference_gate_pass':audit['all_checks_passed']}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');main(a.parse_args().freeze)
