"""Validate the bounded two-control qualification and its immutable provenance."""
from pathlib import Path
import sys,json,csv,re,hashlib,tarfile,subprocess,argparse,tempfile,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,BUILD,CASES
from prepare_sfincs_cpu_build import sha,dump

def main(root=ROOT,freeze=False):
 root=Path(root);out=root/OUT;report=root/REPORT;checks=[]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 s=json.loads((report/'summary.json').read_text());manifest=json.loads((out/'run_manifest.json').read_text());runs=manifest['runs'];compat=out/'compatibility_v2'
 check('only_two_existing_open_control_cases_completed',s['completed_full_control_runs']==2 and [r['case_id']for r in runs]==CASES and s['new_parent_events']==s['pending_58_executed']==s['training']==s['sealed_responses_read']==0)
 check('all_actual_clock_budget_output_checks_pass',s['all_checks_passed']and all(x['pass']for x in s['checks']))
 regression=json.loads((report/'compatibility_v2_validation.json').read_text())
 check('57_compatibility_and_original_input_regressions_pass',regression['all_passed']and len(regression['tests'])==57 and all(x['pass']for x in regression['tests']))
 image=regression['new_image_id'];metadata=json.loads((compat/'build-metadata.json').read_text())
 check('all_runs_bound_to_exact_compatible_image_and_patch',all(r['image_id']==image==r['identity']['solver_digest']and r['identity']['patch_sha256']==sha(compat/'sfincs_second_clock_budget_v2.patch')for r in runs)and image==metadata['containerimage.digest'])

 source_binding=json.loads((compat/'source_binding.json').read_text())
 material_ids={'sha256:'+m['digest']['sha256']for m in metadata['buildx.build.provenance']['materials']}
 check('v2_build_materials_match_frozen_runtime_and_builder',material_ids=={source_binding['base_runtime_image_id'],source_binding['base_builder_image_id']})
 prior=root/'data/processed/three_ward_second_clock_budget_v1'
 original_manifest=json.loads((prior/'source_manifest.json').read_text())
 expected=dict(original_manifest['patched_sha256']);expected['udt_rain_clock.f90']=source_binding['rain_module_sha256']
 with tempfile.TemporaryDirectory(prefix='udt01b-v2-patch-check-',dir='/private/tmp')as tmp:
  base=Path(tmp);(base/'source/src').mkdir(parents=True)
  for p in(prior/'base').iterdir():shutil.copyfile(p,base/'source/src'/p.name)
  r=subprocess.run(['/usr/bin/patch','--batch','-p1','-i',str(compat/'sfincs_second_clock_budget_v2.patch')],cwd=base,capture_output=True,text=True)
  check('v2_combined_patch_applies_and_only_origin_module_differs',r.returncode==0 and all(sha(base/'source/src'/p)==h for p,h in expected.items()))
 inspect=subprocess.run(['docker','image','inspect',image],capture_output=True,text=True)
 check('compatible_local_CPU_image_still_available',inspect.returncode==0 and json.loads(inspect.stdout)[0]['Architecture']=='amd64')
 raw_blobs={};count=0;all_hashes=True
 with tarfile.open(compat/'solver_image_v2.tar')as archive:
  for member in archive.getmembers():
   if member.isfile()and member.name.startswith('blobs/sha256/'):
    h=hashlib.sha256();data=bytearray()
    with archive.extractfile(member)as f:
     for block in iter(lambda:f.read(2**20),b''):
      h.update(block)
      if member.size<2**20:data.extend(block)
    digest='sha256:'+h.hexdigest();all_hashes &= digest.split(':')[1]==member.name.split('/')[-1];count+=1
    if data:raw_blobs[digest]=bytes(data)
 index=json.loads(raw_blobs[image]);chosen=[x for x in index['manifests']if x.get('platform',{}).get('architecture')=='amd64'and x.get('platform',{}).get('os')=='linux']
 plat=json.loads(raw_blobs[chosen[0]['digest']]);config=json.loads(raw_blobs[plat['config']['digest']])
 check('offline_image_all_OCI_blobs_and_platform_verified',all_hashes and len(chosen)==1 and config['architecture']=='amd64')
 record={'index_digest':image,'platform_manifest_digest':chosen[0]['digest'],'config_digest':plat['config']['digest'],'archive_sha256':sha(compat/'solver_image_v2.tar'),'verified_blobs':count,'platform':'linux/amd64','source_binding_sha256':sha(compat/'source_binding.json'),'compatibility_patch_sha256':sha(compat/'sfincs_second_clock_budget_v2.patch'),'no_registry_push':True}
 if(compat/'oci_image_binding.json').exists():assert json.loads((compat/'oci_image_binding.json').read_text())==record
 else:dump(compat/'oci_image_binding.json',record)
 for run in runs:
  folder=root/run['folder'];source=root/run['source_folder']
  check(run['case_id']+'_original_rain_and_stage_bytes_preserved',all(sha(source/k)==sha(folder/k)==run['original_source_input_sha256'][k]for k in ['roof_surface_precip.nc','sfincs.bzs']))
  original=(source/'sfincs.inp').read_text();staged=(folder/'sfincs.inp').read_text()
  check(run['case_id']+'_only_recognized_interpolation_key_added',staged==original+'amprblock                = 0\n')
  check(run['case_id']+'_open_control_family_partition_preserved',run['identity']['family_id']==run['identity']['parent_id']=='CONTROL-LEGACY'and run['identity']['split']=='open_control'and not run['library_descendant'])
 failure1=root/manifest['initialization_failure_retained']['folder'];failure2=root/manifest['second_failed_attempt']['folder']
 check('both_failed_attempts_retained_and_not_admitted',json.loads((failure1/'execution.json').read_text())['returncode']!=0 and not(failure1/'udt_step_budget.csv').exists()and json.loads((failure2/'execution.json').read_text())['returncode']!=0 and len((failure2/'udt_step_budget.csv').read_text().splitlines())==2)
 policy=root/'data/processed/three_ward_reference_contract_v1/reference_registry.json';registry=json.loads(policy.read_text())
 check('existing_reference_holds_and_training_denial_unchanged',all(v['state']=='hold_event'and not v['technical_reference']for v in registry['controls'].values()))
 original_manifest=json.loads((root/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())
 check('original_58_prospective_cases_still_unrun',len(original_manifest['pending_runs'])==58 and all(r['status']=='unrun'and r['response_path']is None for r in original_manifest['pending_runs']))
 prov=json.loads((report/'provenance.json').read_text());check('all_original_libraries_curves_labels_and_policies_unchanged',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('all_old_hydrodynamic_outputs_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==v for p,v in prov['existing_result_stat_fingerprints'].items()))
 old_freeze=json.loads((root/BUILD/'freeze.json').read_text());check('prior_frozen_CPU_image_source_and_receipts_unchanged',all(sha(root/p)==h for p,h in old_freeze['files_sha256'].items()))
 astro=json.loads((report/'astronomical_tide_audit.json').read_text());check('current_astronomical_tide_branch_unreachable',all(not x['nodal_table_call_reachable']and x['tidal_component_count_from_initialization_branch']==0 for x in astro['cases']))
 backup=(report/'previous-AnaSOP.md').read_text();current=(root/'docs/AnaSOP.md').read_text();parts=lambda x:re.split(r'(?=^## [1-8]\. )',x,flags=re.M);before,after=parts(backup),parts(current)
 changed={0 if not a.startswith('## ')else int(re.match(r'## (\d)',a).group(1))for a,b in zip(before,after)if a!=b}
 check('AnaSOP_only_authorization_metadata_and_4_6_7_updated',changed=={0,4,6,7}and all(a==b for a,b in zip(before,after)if a.startswith(('## 1. ','## 2. ','## 3. ','## 5. ','## 8. '))))
 check('AnaSOP_English_ASCII_no_new_final_variables',all(s.isascii()for s in after if s.startswith(('## 4. ','## 6. ','## 7. ')))and backup.count('| yes |')==current.count('| yes |'))
 page=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',page)
 check('all_review_links_and_figure_exist',all((report/p).resolve().is_file()for p in links if not p.startswith(('http','#'))))
 check('qualification_does_not_release_labels_training_or_batch',not json.loads((out/'qualification.json').read_text())['future_library_released']and s['no_reference_release'])
 if freeze:
  assert all(x['pass']for x in checks)
  paths=[p for base in [out,report]for p in base.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery-validation.json']]
  paths+=[root/'src/preprocessing'/p for p in ['prepare_three_ward_cpu_e2e.py','prepare_cpu_e2e_compatibility.py','repair_e2e_interpolation_key.py']]
  paths+=[root/'src/analyses'/p for p in ['run_three_ward_cpu_e2e.py','qualify_cpu_e2e_compatibility.py','audit_three_ward_cpu_e2e.py','review_three_ward_cpu_e2e.py','validate_three_ward_cpu_e2e_delivery.py']]
  paths+=list((root/'src/preprocessing/sfincs_second_clock_budget_v2').glob('*.f90'))
  cache={};hashes={}
  for p in sorted(paths):
   stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
   if key not in cache:cache[key]=sha(p)
   hashes[str(p.relative_to(root))]=cache[key]
  dump(out/'freeze.json',{'state':'two_legacy_controls_e2e_verified_no_library_or_training_release','files_sha256':hashes,'image_id':image,'completed_controls':2,'new_parent_events':0,'pending_58_executed':0,'training':0})
 if(out/'freeze.json').exists():
  f=json.loads((out/'freeze.json').read_text());cache={};valid=True
  for path,h in f['files_sha256'].items():
   p=root/path;st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
   if key not in cache:cache[key]=sha(p)
   valid &=cache[key]==h
  check('all_frozen_delivery_hashes_match',valid)
 result={'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'training':0,'pending_58_executed':0,'reference_release':False};dump(report/'delivery-validation.json',result)
 print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();main(freeze=a.freeze)
