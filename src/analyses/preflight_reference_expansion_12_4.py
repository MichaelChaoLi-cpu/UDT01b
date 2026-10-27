"""Read-only checks before the bounded response expansion starts."""
from pathlib import Path
import sys,json,itertools,ast,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_reference_expansion_12_4 import OUT,REPORT,PRIOR,PRIOR_REPORT,LIB,EXISTING,FEATURES,SHAPES
from prepare_sfincs_cpu_build import sha,dump
from load_three_ward_minimum_scenarios import find_case
from generate_three_ward_minimum_scenarios import semantic,arrays

def main():
 o=ROOT/OUT;r=ROOT/REPORT;s=json.loads((o/'selection.json').read_text());m=json.loads((o/'expansion_manifest.json').read_text());checks=[];cache={}
 def h(p):
  st=p.stat();key=st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 def check(k,b):print(('PASS 'if b else'FAIL ')+k,flush=True);checks.append({'check':k,'pass':bool(b)})
 selected=[find_case(k)for k in s['selected_case_ids']];check('cumulative_twelve_train_four_validation',len(selected)==16 and sum(x['split']=='train'for x in selected)==12 and sum(x['split']=='validation'for x in selected)==4 and len({x['parent_id']for x in selected})==16)
 check('shape_quotas_three_train_one_validation',all(sum(x['split']==split and x['parameters']['shape_class']==shape for x in selected)==q for split,q in [('train',3),('validation',1)]for shape in SHAPES))
 check('retain_six_previous_parents_ten_new_only',s['reused_case_ids']==EXISTING and len(s['new_case_ids'])==10 and set(s['new_case_ids']).isdisjoint(EXISTING)and len(m['runs'])==10 and {r['case_id']for r in m['runs']}==set(s['new_case_ids']))
 check('selection_frozen_before_new_responses',m['selection_sha256']==h(o/'selection.json')and all(not(ROOT/x['folder']/'launch.json').exists()for x in m['runs']))
 check('five_finite_two_run_batches',[x for batch in s['batches']for x in batch]==s['new_case_ids']and len(s['batches'])==5 and all(len(x)==2 for x in s['batches']))
 check('no_sealed_or_training_authorized',not s['training_authorized']and not s['sealed_test_challenge_authorized']and all(x['split']in ['train','validation']for x in selected))
 prov=json.loads((r/'provenance.json').read_text());check('original_inputs_policy_and_registry_unchanged',all(h(ROOT/k)==v for k,v in prov['protected_sha256'].items()))
 prior=json.loads((ROOT/PRIOR/'freeze.json').read_text())['files_sha256'];check('entire_prior_frozen_delivery_unchanged',all(h(ROOT/k)==v for k,v in prior.items()))
 template=json.loads((ROOT/PRIOR/'pilot_manifest.json').read_text())['runs'][2];source=ROOT/template['folder'];solver=json.loads((ROOT/'data/processed/three_ward_rain_step_integral_v1/solver_binding.json').read_text())
 for x in m['runs']:
  p=ROOT/x['folder'];c=find_case(x['case_id']);f=ROOT/x['source_forcing_file'];check(x['case_id']+'_original_input_identity',semantic(arrays(f))==c['forcing_sha256']==x['identity']['forcing_sha256']and h(f)==x['source_forcing_file_sha256']and x['identity']['parent_id']==c['parent_id']and x['split']==c['split']==x['identity']['split'])
  check(x['case_id']+'_unchanged_geometry_and_bound_solver',x['image_id']==solver['image_id']and x['identity']['patch_sha256']==solver['patch_sha256']and x['identity']['geometry_sha256']==template['identity']['geometry_sha256']and all(h(p/n)==v for n,v in x['input_sha256'].items())and all(h(p/n)==h(source/n)for n in x['input_sha256']if n not in ['roof_surface_precip.nc','sfincs.bzs']))
 for x in m['reused']:
  a=json.loads((ROOT/x['audit_path']).read_text());check(x['case_id']+'_reuse_original_qualified_response',a['all_checks_passed']and a['admission']['technical_reference']and not a['admission']['training_release']and(ROOT/x['label_path']).exists())
 for p in [ROOT/'src/preprocessing/prepare_reference_expansion_12_4.py',ROOT/'src/analyses/run_reference_expansion_12_4.py',Path(__file__)]:ast.parse(p.read_text())
 check('sources_parse',True);dump(r/'preflight.json',{'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'new_run_limit':10,'training':False});assert all(c['pass']for c in checks)
 dump(o/'pre_execution_binding.json',{'selection_sha256':sha(o/'selection.json'),'manifest_sha256':sha(o/'expansion_manifest.json'),'runner_source_sha256':sha(ROOT/'src/analyses/run_reference_expansion_12_4.py'),'preparation_source_sha256':sha(ROOT/'src/preprocessing/prepare_reference_expansion_12_4.py'),'original_auditor_sha256':sha(ROOT/'src/analyses/audit_reference_setup_pilot.py'),'image_id':solver['image_id'],'patch_sha256':solver['patch_sha256'],'new_run_limit':10});print('PREFLIGHT COMPLETE',len(checks),flush=True)
if __name__=='__main__':main()
