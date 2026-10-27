"""Verify frozen source/image bindings and original research-data preservation."""
from pathlib import Path
import sys,json,subprocess,csv,re,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import OUT,REPORT,PREVIOUS,COMMIT,TAG,sha,dump

def main(root=ROOT,freeze=False):
 root=Path(root);out=root/OUT;report=root/REPORT;checks=[]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name)
 source=json.loads((out/'source_binding.json').read_text());binding=json.loads((out/'solver_image_binding.json').read_text());summary=json.loads((report/'summary.json').read_text());metadata=json.loads((out/'build-metadata.json').read_text())
 check('source_archive_and_git_tree_hashes_bound',sha(out/'upstream_source.tar.gz')==source['archive_sha256']and sha(out/'upstream_git_tree.json')==source['git_tree_sha256'])
 check('complete_verified_source_matches_image_binding',source['verified_source_blobs']==424 and source['commit']==binding['source_commit']==COMMIT and sha(out/'source_binding.json')==binding['source_binding_sha256']and all(sha(out/'context'/p)==h for p,h in source['final_source_sha256'].items()))
 check('previous_frozen_rain_and_budget_patch_unmodified',sha(root/PREVIOUS/'sfincs_second_clock_budget.patch')==source['frozen_interface_patch_sha256']==binding['interface_patch_sha256'])
 check('complete_build_recipe_hashes_bound',all(sha(out/'context'/p)==h for p,h in binding['build_recipe_sha256'].items()))
 r=subprocess.run(['docker','image','inspect',binding['local_image_id']],capture_output=True,text=True)
 image=json.loads(r.stdout)[0]if r.returncode==0 else{}
 check('immutable_local_image_present_and_platform_revision_match',r.returncode==0 and image.get('Id')==binding['local_image_id']and image['Architecture']=='amd64'and image['Config']['Labels']['org.opencontainers.image.revision']==COMMIT)
 archive=json.loads((out/'oci_archive_receipt.json').read_text())
 check('OCI_index_manifest_config_and_offline_archive_bound',binding['oci_manifest_digest']==metadata['containerimage.digest']==archive['oci_index_digest']and binding['oci_config_digest']==archive['config_digest']and binding['oci_platform_manifest_digest']==archive['platform_manifest_digest']and sha(out/'solver_image.tar')==archive['archive_sha256']and archive['all_blob_hashes_verified']and archive['all_runtime_layers_present'])
 check('production_checks_all_passed',summary['all_checks_passed']and json.loads((report/'validation.json').read_text())['all_checks_passed'])
 rows=list(csv.DictReader((report/'50_production_reader_checks.csv').open()))
 check('50_frozen_inputs_preserved_in_real_production_reader',len(rows)==50 and all(float(x['time_error_s'])==float(x['time_integral_error_mm'])==0 and x['rain_values_match']=='True'for x in rows))
 bindings=json.loads((out/'run_bindings.json').read_text())['runs'];old=json.loads((root/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs'];lookup={x['run_id']:x for x in bindings}
 check('58_descendant_lineages_splits_and_curve_ids_unchanged',len(bindings)==len(old)==58 and all(all(lookup[x['run_id']][k]==x[k]for k in ['case_id','parent_id','family_id','split','grid','forcing_sha256'])for x in old))
 check('58_descendants_bound_without_run_or_reference_release',all(x['solver_digest']==binding['oci_manifest_digest']and not x['execution_release']and not x['reference_release']and not x['hydrodynamic_end_to_end_verified']for x in bindings))
 prov=json.loads((report/'provenance.json').read_text())
 check('all_protected_previous_inputs_patches_policies_unchanged',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('old_hydrodynamic_outputs_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==v for p,v in prov['existing_result_stat_fingerprints'].items()))
 sop=json.loads((report/'anasop-update.json').read_text());before=(report/'previous-AnaSOP.md').read_text();after=(root/'docs/AnaSOP.md').read_text();parts=lambda s:re.split(r'(?=^## [1-8]\. )',s,flags=re.M)
 a,b=parts(before),parts(after);changed={int(re.match(r'## (\d)',x).group(1))for x,y in zip(a,b)if x!=y and x.startswith('## ')}
 check('AnaSOP_only_4_6_7_and_evidence_receipt_match',changed=={4,6,7}and sop['current_sha256']==sha(root/'docs/AnaSOP.md')and all(x==y for x,y in zip(a,b)if not x.startswith(('## 4. ','## 6. ','## 7. '))))
 check('AnaSOP_English_ASCII_and_final_variable_status_unchanged',all(x.isascii()for x in b if x.startswith(('## 4. ','## 6. ','## 7. ')))and before.count('| yes |')==after.count('| yes |'))
 html=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',html)
 check('review_local_links_exist',all((report/p).resolve().is_file()for p in links if not p.startswith(('http','#'))))
 check('zero_solver_initialize_update_training_and_new_hydrodynamic_runs',binding['solver_initialize_calls']==binding['solver_update_calls']==binding['training']==summary['hydrodynamic_runs']==summary['sealed_responses_read']==0 and not binding['hydrodynamic_end_to_end_qualified']and not binding['reference_release'])
 if freeze:
  assert all(x['pass']for x in checks)
  files=[p for base in [out,report]for p in base.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery-validation.json']]
  files+=[root/'src/preprocessing/prepare_sfincs_cpu_build.py',root/'src/analyses/verify_sfincs_cpu_image.py',root/'src/analyses/bind_sfincs_cpu_oci_archive.py',root/'src/analyses/review_sfincs_cpu_build.py',root/'src/analyses/validate_sfincs_cpu_build_delivery.py']
  files+=list((root/'src/preprocessing/sfincs_cpu_build_v1').iterdir())
  dump(out/'freeze.json',{'state':'complete_CPU_build_link_qualified_hydrodynamic_unrun','files_sha256':{str(p.relative_to(root)):sha(p)for p in sorted(files)if p.is_file()},'image_digest':binding['oci_manifest_digest'],'execution_release':False,'reference_release':False})
 if(out/'freeze.json').exists():
  record=json.loads((out/'freeze.json').read_text());check('frozen_delivery_hashes_match',all(sha(root/p)==h for p,h in record['files_sha256'].items()))
 result={'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'hydrodynamic_runs':0,'training':0,'reference_release':False};dump(report/'delivery-validation.json',result)
 print(json.dumps({'checks':len(checks),'all_checks_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();main(freeze=a.freeze)
