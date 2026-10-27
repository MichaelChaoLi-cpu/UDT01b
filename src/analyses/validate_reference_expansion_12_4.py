"""Validate custody, unchanged admission and bounded 12/4 delivery; optionally freeze it."""
from pathlib import Path
import sys,json,re,ast,argparse,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_expansion_12_4 import OUT,REPORT,PRIOR,PRIOR_REPORT,LIB,EXISTING,SHAPES,HUMAN
from prepare_sfincs_cpu_build import dump,sha
from reference_label_policy import classify_labels,event_admission
from load_three_ward_minimum_scenarios import find_case,response_access
from generate_three_ward_minimum_scenarios import arrays,semantic
from audit_three_ward_negative_volumes import readnp

def main(freeze=False):
 o=ROOT/OUT;r=ROOT/REPORT;s=json.loads((r/'summary.json').read_text());plan=json.loads((o/'selection.json').read_text());m=json.loads((o/'expansion_manifest.json').read_text());b=json.loads((o/'pre_execution_binding.json').read_text());checks=[];cache={}
 def h(p):
  st=p.stat();key=st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 def check(k,v):checks.append({'check':k,'pass':bool(v)});print(('PASS 'if v else'FAIL ')+k,flush=True)
 check('frozen_input_selection_and_execution_sources',h(o/'selection.json')==b['selection_sha256']==m['selection_sha256']and h(o/'expansion_manifest.json')==b['manifest_sha256']and h(ROOT/'src/analyses/run_reference_expansion_12_4.py')==b['runner_source_sha256']and h(ROOT/'src/preprocessing/prepare_reference_expansion_12_4.py')==b['preparation_source_sha256']and h(ROOT/'src/analyses/audit_reference_setup_pilot.py')==b['original_auditor_sha256'])
 check('exactly_six_reused_and_ten_authorized_new',plan['reused_case_ids']==EXISTING and len(m['reused'])==6 and len(m['runs'])==10 and [x['case_id']for x in m['runs']]==plan['new_case_ids']and len(plan['selected_case_ids'])==16 and set(plan['selected_case_ids'])==set(EXISTING+plan['new_case_ids']))
 selected=[find_case(k)for k in plan['selected_case_ids']];check('original_partitions_and_unique_parent_families',sum(x['split']=='train'for x in selected)==12 and sum(x['split']=='validation'for x in selected)==4 and len({x['parent_id']for x in selected})==len({x['family_id']for x in selected})==16 and all(x['split']in ['train','validation']for x in selected))
 check('fixed_shape_quotas_preserved',all(sum(x['split']==split and x['parameters']['shape_class']==shape for x in selected)==q for split,q in [('train',3),('validation',1)]for shape in SHAPES))
 template=json.loads((ROOT/PRIOR/'pilot_manifest.json').read_text())['runs'][2];base=ROOT/template['folder'];support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];old={x['case_id']:x for x in m['reused']};new={x['case_id']:x for x in m['runs']}
 for k in plan['selected_case_ids']:
  c=find_case(k);ap=ROOT/old[k]['audit_path']if k in old else r/'expansion'/k/'audit.json';lp=ROOT/old[k]['label_path']if k in old else o/'expansion'/(k+'_candidate_reference.npz');a=json.loads(ap.read_text());d=readnp(lp)
  check(k+'_original_numerical_checks_and_policy',a['all_checks_passed']and all(x['pass']for x in a['checks'])and event_admission(a['required_gates'],d)==a['admission']and a['admission']['technical_reference']and not a['admission']['training_release'])
  rebuilt=classify_labels(d['Peak Mean Depth Lower'],d['Peak Mean Depth Upper'],valid,d['Online Ever Negative'],d['Assumed Barrier Contact'],interval_source_known=next(x['pass']for x in a['checks']if x['check']=='censoring'));check(k+'_original_quality_masks_reproduce',all(np.array_equal(v,d[name])for name,v in rebuilt.items()))
  if k in new:
   x=new[k];p=ROOT/x['folder'];ex=json.loads((p/'execution.json').read_text());f=ROOT/x['source_forcing_file'];check(k+'_bound_solver_and_original_forcing',ex['returncode']==0 and ex['completion_marker']and ex['solver_image_id']==x['image_id']==b['image_id']and x['identity']['patch_sha256']==b['patch_sha256']and semantic(arrays(f))==c['forcing_sha256']==x['identity']['forcing_sha256']and h(f)==x['source_forcing_file_sha256']and x['identity']['parent_id']==c['parent_id']and x['identity']['split']==c['split'])
   check(k+'_input_bytes_and_geometry_preserved',all(h(p/n)==v for n,v in x['input_sha256'].items())and all(h(p/n)==h(base/n)for n in x['input_sha256']if n not in ['roof_surface_precip.nc','sfincs.bzs']))
  try:response_access(k,purpose=c['split'],root=ROOT)
  except PermissionError:closed=True
  else:closed=False
  check(k+'_default_training_access_closed',closed)
 for n,batch in enumerate(plan['batches'],1):
  receipt=json.loads((r/f'batch_{n}_receipt.json').read_text());check('batch_'+str(n)+'_audited_before_advancing',receipt['cases']==batch and receipt['all_original_gates_passed'])
 end=json.loads((r/'execution_complete.json').read_text());check('all_ten_runs_finished_no_training_or_sealed',end['new_runs']==10 and end['all_batches_passed']and not end['training']and not end['sealed_responses']and s['all16_numerically_qualified']and not s['held']and not s['pending'])
 idx=json.loads((o/'execution_overlay.json').read_text());rows=idx['runs'];check('combined58_index_eighteen_complete_forty_unrun',len(rows)==58 and idx['cumulative_completed_subset_of58']==18 and idx['remaining_unrun']==40 and idx['current_expansion_queued_or_running']==0 and sum(x['status']=='unrun'and x['grid']=='regular_10m'for x in rows)==32 and sum(x['status']=='unrun'and x['grid']=='adaptive_10_5m'for x in rows)==8)
 check('sealed_and_challenge_response_custody',all(x['status']=='unrun'and x['response_path']is None for x in rows if x['split']in ['sealed_test','challenge'])and not idx['training']and not idx['sealed_responses_generated'])
 prov=json.loads((r/'provenance.json').read_text());check('original_forcing_policy_registry_preserved',all(h(ROOT/k)==v for k,v in prov['protected_sha256'].items()));prior=json.loads((ROOT/PRIOR/'freeze.json').read_text())['files_sha256'];check('entire_prior_frozen_delivery_preserved',all(h(ROOT/k)==v for k,v in prior.items()))
 before=(r/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text();aa=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);bb=re.split(r'(?=^## [1-8]\. )',after,flags=re.M);changed={int(re.match(r'## (\d)',a).group(1))for a,z in zip(aa,bb)if a!=z};check('only_skill_routed_sections3_5_6_7_updated',changed=={3,5,6,7}and len(aa)==len(bb)and aa[0]==bb[0]and after.isascii());u=json.loads((r/'anasop-update.json').read_text());check('no_new_final_variables_or_formulas',u['new_final_variables']==u['new_formulas']==0 and set(u['skills'])=={'data-briefing','estimation-framework-planning'})
 resume=json.loads((r/'resume_receipt.json').read_text());pause=json.loads((r/'pause_receipt.json').read_text());check('resumed_original_states_without_hydraulic_restart',resume['hydraulic_restarts']==0 and resume['bound_sources_unchanged']and set(resume['containers'])==set(pause['paused_containers'])and all(v=='false'for v in resume['verified_not_paused'].values()))
 with(r/'response_numerical_summary.csv').open()as f:timing=list(csv.DictReader(f))
 check('pause_affected_recorded_times_preserved_and_excluded_from_comparisons',{x['case_id']for x in timing if x['interrupted_by_user_pause']=='True'}=={'train-19','train-16'}and all(x['runtime_comparison_status']=='excluded_user_pause'for x in timing if x['case_id']in ['train-19','train-16'])and len(timing)==16)
 page=(r/'review.html').read_text();check('report_links_and_input_distribution_present',all((r/p).is_file()for p in re.findall(r'(?:href|src)="([^"]+)"',page))and(r/'forcing_coverage.png').stat().st_size>30000)
 scripts=[ROOT/'src/preprocessing/prepare_reference_expansion_12_4.py',*sorted((ROOT/'src/analyses').glob('*reference_expansion_12_4.py'))]
 for p in scripts:ast.parse(p.read_text())
 check('reproducible_sources_parse',True);check('human_scope_record_present',HUMAN in(ROOT/'docs/framelog/mili.md').read_text())
 dump(r/'delivery_validation.json',{'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'qualified_training_parents':12,'qualified_validation_parents':4,'training':False});assert all(x['pass']for x in checks)
 if freeze:
  paths=set(scripts)
  for folder in [o,r,ROOT/'data/exp/data-briefing/reference_expansion_12_4']:paths.update(p for p in folder.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery_validation.json'])
  dump(o/'freeze.json',{'state':'twelve_training_four_validation_parent_candidates_qualified_no_training_release','human_record':HUMAN,'files_sha256':{str(p.relative_to(ROOT)):h(p)for p in sorted(paths)}})
 print(json.dumps({'all_checks_passed':True,'checks':len(checks),'train':12,'validation':4,'new_runs':10,'training':False}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');main(p.parse_args().freeze)
