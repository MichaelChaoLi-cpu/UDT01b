"""Validate reproducibility, original-policy admission, unchanged inputs and bounded execution."""
from pathlib import Path
import sys,json,re,ast,argparse,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_rain_step_integral import OUT,REPORT,PRIOR,CASES
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
 m=json.loads((out/'run_manifest.json').read_text());binding=json.loads((out/'source_binding.json').read_text());solver=json.loads((out/'solver_binding.json').read_text());prov=json.loads((report/'provenance.json').read_text());build=json.loads((report/'build_validation.json').read_text());audit=json.loads((report/'audit.json').read_text());ra=json.loads((report/'rain_integral_audit.json').read_text());ad=json.loads((report/'reference_admission_assessment.json').read_text());summary=json.loads((report/'summary.json').read_text());boundary=json.loads((report/'boundary_comparison.json').read_text())
 check('exactly_two_existing_controls_no_new_parent',len(m['runs'])==2 and[r['case_id']for r in m['runs']]==CASES and m['new_parent_count']==0 and not m['pending_58_scenarios_authorized']and not m['training'])
 check('all_original_event_numerical_checks_pass',audit['all_checks_passed']and all(c['pass']for c in audit['checks']))
 check('all_exact_rain_integral_checks_pass',ra['all_checks_passed']and not ra['input_normalization']and not ra['tolerance_relaxation'])
 check('source_full_patch_and_production_fixture_bound',h(out/'full_solver.patch')==binding['full_patch_sha256']==solver['patch_sha256']and all(h(out/'context'/n)==v for n,v in binding['source_sha256'].items())and h(out/'context/rain_integral_fixture.f90')==solver['production_rain_fixture_sha256'])
 links=(report/'runtime_links.txt').read_text();check('Fortran_NetCDF_and_loader_links_verified','not found'not in links and 'PASS: production Fortran NetCDF'in links and 'zero solver entry-point calls'in links)
 check('all_compiled_fixtures_pass',build['all_checks_passed']and all(c['pass']for c in build['checks']))
 for kind in ['regular','adaptive','missing_child','rain']:
  a=out/'link_tests'/(kind+'1');b=out/'link_tests'/(kind+'4')
  check(kind+'_serial_parallel_identical_current_image', (a/'stdout.txt').read_bytes()==(b/'stdout.txt').read_bytes()and all(json.loads((p/'execution.json').read_text())['command'][-1]==solver['image_id']for p in [a,b]))
 check('closure_positivity_budget_decoder_unchanged',all(h(out/'context'/n)==h(ROOT/PRIOR/'context'/n)for n in ['sfincs_boundaries.f90','sfincs_continuity.f90','udt_clock_budget.f90','udt_rain_clock.f90']))
 for r in m['runs']:
  case=r['case_id'];p=ROOT/r['folder'];source=ROOT/r['source_folder'];ex=json.loads((p/'execution.json').read_text());masks=readnp(out/(case+'_adaptive_candidate_reference.npz'))
  check(case+'_input_bytes_and_original_source_identical',all(h(p/n)==v==h(source/n)for n,v in r['input_sha256'].items())and r['input_config_differences']=={})
  check(case+'_bound_completed_image_and_identity',ex['returncode']==0 and ex['completion_marker']and ex['solver_image_id']==solver['image_id']==r['identity']['solver_digest']and r['identity']['patch_sha256']==solver['patch_sha256'])
  check(case+'_lineage_and_split_preserved',r['original_parent_family']=='CONTROL-LEGACY'and r['split']=='open_control'and not r['new_parent_event'])
  check(case+'_strict_nonnegativity_and_closed_faces',summary['cases'][case]['numerics']['negative_native_cells']==0 and summary['cases'][case]['numerics']['minimum_native_volume_m3']>=0 and summary['cases'][case]['numerics']['closed_receiving_faces']==449)
  check(case+'_original_policy_preview_reproduces_without_release',ad[case]['preview']==event_admission(ad[case]['required_gates'],masks)and all(ad[case]['required_gates'].values())and ad[case]['preview']['technical_reference']and not ad[case]['training_release']and not ad[case]['formal_registry_changed'])
  try:load_reference(case,root=ROOT)
  except PermissionError:closed=True
  else:closed=False
  check(case+'_default_loader_stays_closed',closed)
  rec=json.loads((report/(case+'_label_reconstruction.json')).read_text());check(case+'_original_ground_bounds_reconstructed',rec['quarter_ground_area_reaggregation_exact']and rec['cases'][0]['all_bounds_finite']and rec['cases'][0]['intervals_ordered']and rec['cases'][0]['regular_depth_reconstruction_max_difference_m']<1e-12)
 check('boundary_exchange_interface_verified',boundary['all_checks_passed'])
 check('old_adaptive_delivery_and_all_its_frozen_sources_unchanged',all(h(ROOT/p)==v for p,v in json.loads((ROOT/PRIOR/'freeze.json').read_text())['files_sha256'].items()))
 check('policy_registry_and_58_manifest_unchanged',all(h(ROOT/p)==prov[k]for p,k in [('src/preprocessing/reference_label_policy.py','policy_sha256'),('data/processed/three_ward_reference_contract_v1/reference_registry.json','registry_sha256'),('data/processed/three_ward_minimum_scenarios_v1/run_manifest.json','pending58_manifest_sha256')]))
 pending=json.loads((ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs'];check('all58_remain_unrun',len(pending)==58 and all(r['status']=='unrun'and r['response_path']is None for r in pending))
 original_inputs=json.loads((ROOT/'data/exp/data-preprocessing/three_ward_cpu_solver_build/provenance.json').read_text())['protected_sha256'];library={p:v for p,v in original_inputs.items()if p.startswith('data/processed/three_ward_minimum_scenarios_v1/')}
 check('all50_original_forcing_curves_allocation_lineages_and_partitions_frozen',len(library)>=57 and all(h(ROOT/p)==v for p,v in library.items()))
 check('no_training_or_reference_release',summary['training']==summary['batch58']==summary['new_parents']==0 and not summary['formal_reference_release']and not summary['real_event_accuracy_verified'])
 old=(report/'previous-AnaSOP.md').read_text();new=(ROOT/'docs/AnaSOP.md').read_text();aa=re.split(r'(?=^## [1-8]\. )',old,flags=re.M);bb=re.split(r'(?=^## [1-8]\. )',new,flags=re.M);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(aa,bb)if a!=b};check('only_skill_routed_AnaSOP_sections4_6_7_updated',changed=={4,6,7}and len(aa)==len(bb)and new.isascii())
 page=(report/'review.html').read_text();check('report_links_resolve',all((report/p).is_file()for p in re.findall(r'(?:href|src)="([^"]+)"',page)))
 check('scientific_figure_created',(report/'rain_integral_controls.png').stat().st_size>30000)
 scripts=list((ROOT/'src/analyses').glob('*rain_step_integral*.py'))+[ROOT/'src/preprocessing/prepare_rain_step_integral.py']
 for p in scripts:ast.parse(p.read_text())
 check('reproducible_scripts_parse',True)
 if(out/'freeze.json').exists():check('current_frozen_delivery_unchanged',all(h(ROOT/p)==v for p,v in json.loads((out/'freeze.json').read_text())['files_sha256'].items()))
 dump(report/'delivery_validation.json',{'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'training':0,'batch58':0,'formal_reference_release':False});assert all(c['pass']for c in checks)
 if freeze:
  paths=set(scripts+[ROOT/'src/preprocessing/sfincs_rain_step_integral.f90.inc',ROOT/'src/preprocessing/rain_integral_fixture.f90'])
  for folder in [out,report]:paths.update(p for p in folder.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery_validation.json'])
  dump(out/'freeze.json',{'state':'two_completed_control_replays_exact_rain_integral_original_policy_no_training_release','files_sha256':{str(p.relative_to(ROOT)):h(p)for p in sorted(paths)}})
 print(json.dumps({'checks':len(checks),'all_pass':True}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)
