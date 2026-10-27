"""Verify the finite response pilot and immutable source/partition custody, then freeze delivery."""
from pathlib import Path
import sys,json,re,ast,argparse,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_setup_pilot import OUT,REPORT,BASE,SOLVER,CONTROLS,PILOT
from prepare_sfincs_cpu_build import dump,sha
from reference_label_policy import classify_labels,event_admission
from load_three_ward_minimum_scenarios import find_case,response_access
from load_three_ward_reference_labels import load_reference
from audit_three_ward_negative_volumes import readnp

def main(freeze=False):
 out=ROOT/OUT;report=ROOT/REPORT;checks=[];cache={}
 def h(p):
  st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 s=json.loads((report/'summary.json').read_text());prov=json.loads((report/'provenance.json').read_text());plan=json.loads((out/'authorized_plan.json').read_text());overlay=json.loads((out/'execution_overlay.json').read_text());solver=json.loads((ROOT/SOLVER/'solver_binding.json').read_text());support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask']
 check('authorized_selection_frozen_two_legacy_plus_eight_regular',plan['qualification_cases']==CONTROLS and plan['pilot_case_ids']==PILOT and plan['maximum_new_hydraulic_runs']==10 and plan['adaptive_pilot_runs']==0 and not plan['training'])
 for phase,cases in [('qualification',CONTROLS),('pilot',PILOT)]:
  m=json.loads((out/(phase+'_manifest.json')).read_text());check(phase+'_only_authorized_case_ids',[r['case_id']for r in m['runs']]==cases)
  for r in m['runs']:
   c=r['case_id'];p=ROOT/r['folder'];ex=json.loads((p/'execution.json').read_text());a=json.loads((report/phase/c/'audit.json').read_text());labels=readnp(out/phase/(c+'_candidate_reference.npz'));original=find_case(c)
   check(c+'_completed_with_bound_image',ex['returncode']==0 and ex['completion_marker']and r['image_id']==ex['solver_image_id']==solver['image_id']and r['identity']['patch_sha256']==solver['patch_sha256'])
   check(c+'_all_staged_input_bytes_preserved',all(h(p/n)==v for n,v in r['input_sha256'].items()))
   check(c+'_original_forcing_family_and_split',r['identity']['forcing_sha256']==original['forcing_sha256']and r['identity']['parent_id']==original['parent_id']and r['identity']['family_id']==original['family_id']and r['identity']['split']==original['split'])
   policy=event_admission(a['required_gates'],labels);check(c+'_original_event_policy_reproduces',policy==a['admission']and not policy['training_release'])
   rebuilt=classify_labels(labels['Peak Mean Depth Lower'],labels['Peak Mean Depth Upper'],valid,labels['Online Ever Negative'],labels['Assumed Barrier Contact'],interval_source_known=next(x['pass']for x in a['checks']if x['check']=='censoring'))
   check(c+'_original_cell_masks_reproduce',all(np.array_equal(rebuilt[k],labels[k])for k in rebuilt))
   check(c+'_all_findings_transparent',a['all_checks_passed']==all(x['pass']for x in a['checks'])and not a['training']and not a['formal_reference_release'])
   if phase=='qualification':check(c+'_legacy_primary_profile_passed',a['all_checks_passed']and a['admission']['technical_reference']and all(h(p/n)==h(ROOT/r['source_folder']/n)for n in r['input_sha256']))
   else:
    base=ROOT/json.loads((out/'qualification_manifest.json').read_text())['runs'][0]['folder'];check(c+'_unchanged_geometry_options_and_own_forcing',all(h(p/n)==h(base/n)for n in r['input_sha256']if n not in ['roof_surface_precip.nc','sfincs.bzs'])and h(ROOT/r['source_forcing_file'])==r['source_forcing_file_sha256']and r['original_forcing_sha256']==original['forcing_sha256'])
    try:response_access(c,purpose={'train':'train','validation':'validation','open_control':'open_control'}[original['split']],root=ROOT)
    except PermissionError:closed=True
    else:closed=False
    check(c+'_training_response_access_still_closed',closed)
 for package in [BASE,SOLVER]:
  frozen=json.loads((ROOT/package/'freeze.json').read_text())['files_sha256'];check(package.split('/')[-1]+'_old_delivery_unchanged',all(h(ROOT/p)==v for p,v in frozen.items()))
 check('shared_inputs_policy_registry_and_entire_original_library_unchanged',all(h(ROOT/p)==v for p,v in prov['protected_sha256'].items()))
 future=overlay['runs'];done=[r for r in future if r['status']!='unrun'];check('eight_executed_from58_fifty_remain_unrun',len(future)==58 and len(done)==8 and{r['case_id']for r in done}==set(PILOT)and overlay['completed_subset_of58']==8 and sum(r['status']=='unrun'for r in future)==50)
 check('sealed_and_other_responses_absent',all(r['response_path']is None and not r['authorized_in_current_pilot']for r in future if r['case_id']not in PILOT or r['grid']!='regular_10m')and all(r['status']=='unrun'and r['response_path']is None for r in future if r['split']in ['sealed_test','challenge']))
 check('six_new_parent_candidates_two_legacy_controls_no_training',s['new_independent_parent_candidates']==6 and s['new_open_control_responses']==2 and s['training']==0 and not s['formal_reference_release']and not s['real_world_accuracy_verified'])
 primary=json.loads((report/'primary_grid_qualification.json').read_text());mesh=json.loads((report/'mesh_comparison/spatial_comparison.json').read_text());boundary=json.loads((report/'mesh_comparison/boundary_comparison.json').read_text());check('two_corrected_grids_compared_without_new_adaptive_runs',primary['all_checks_passed']and primary['new_hydraulic_runs']==2 and primary['matched_corrected_adaptive_results_reused']and mesh['all_checks_passed']and boundary['all_checks_passed'])
 before=(report/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text();aa=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);bb=re.split(r'(?=^## [1-8]\. )',after,flags=re.M);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(aa,bb)if a!=b};check('only_skill_routed_sections3_4_5_6_7_changed',changed=={3,4,5,6,7}and len(aa)==len(bb)and aa[0]==bb[0]and after.isascii())
 update=json.loads((report/'anasop-update.json').read_text());check('no_final_variable_or_formula_added',update['new_final_variables']==update['new_formulas']==0 and set(update['skills'])=={'data-briefing','data-preprocessing','estimation-framework-planning'})
 page=(report/'review.html').read_text();check('review_local_links_resolve',all((report/p).is_file()for p in re.findall(r'(?:href|src)="([^"]+)"',page)))
 check('whole_domain_and_distribution_figures_present',all((report/p).stat().st_size>30000 for p in ['pilot_whole_domain.png','pilot_depth_and_support.png','mesh_comparison/whole_domain_grid_comparison.png']))
 scripts=list((ROOT/'src/analyses').glob('*reference_setup_pilot*.py'))+[ROOT/'src/preprocessing/prepare_reference_setup_pilot.py']
 for p in scripts:ast.parse(p.read_text())
 check('all_reproducible_sources_parse',True)
 if(out/'freeze.json').exists():check('current_frozen_delivery_matches',all(h(ROOT/p)==v for p,v in json.loads((out/'freeze.json').read_text())['files_sha256'].items()))
 dump(report/'delivery_validation.json',{'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'numerically_eligible_pilot_count':len(s['numerically_eligible_pilot_cases']),'held_pilot_cases':s['held_pilot_cases'],'training':0,'sealed_responses_generated':0});assert all(c['pass']for c in checks)
 if freeze:
  paths=set(scripts)
  for folder in [out,report,ROOT/'data/exp/data-briefing/reference_setup_pilot']:paths.update(p for p in folder.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery_validation.json'])
  dump(out/'freeze.json',{'state':'two_qualified_primary_controls_eight_finite_pilot_responses_no_training_release','files_sha256':{str(p.relative_to(ROOT)):h(p)for p in sorted(paths)}})
 print(json.dumps({'checks':len(checks),'all_pass':True,'eligible_pilot':len(s['numerically_eligible_pilot_cases'])}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)
