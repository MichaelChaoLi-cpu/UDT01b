"""Read-only delivery checks; apply patch only to a disposable source tree."""
from pathlib import Path
import sys,json,csv,hashlib,re,tempfile,subprocess,shutil,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_second_clock_budget import OUT,REPORT,sha
from stage_second_clock_budget_identity import expected_bindings
from generate_three_ward_minimum_scenarios import semantic

def main(root=ROOT,write_receipt=False):
 root=Path(root);out=root/OUT;report=root/REPORT;checks=[]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name)
 summary=json.loads((report/'summary.json').read_text());validation=json.loads((report/'validation.json').read_text())
 check('34_component_checks_passed',summary['checks']==34 and validation['all_checks_passed']and all(x['pass']for x in validation['checks']))
 manifest=json.loads((out/'source_manifest.json').read_text())
 verified=True
 for item in manifest['base_sources']:
  p=out/'base'/item['file'];b=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
  verified &= sha(p)==item['sha256']and blob==item['git_blob_sha1']
 check('13_pinned_official_source_blobs_verified',verified and len(manifest['base_sources'])==13)
 check('patched_sources_match_tested_manifest',all(sha(out/'patched'/p)==h for p,h in manifest['patched_sha256'].items()))
 with tempfile.TemporaryDirectory(prefix='udt01b-patch-apply-',dir='/private/tmp')as tmp:
  path=Path(tmp);(path/'source/src').mkdir(parents=True)
  for p in (out/'base').iterdir():shutil.copyfile(p,path/'source/src'/p.name)
  result=subprocess.run(['/usr/bin/patch','--batch','-p1','-i',str(out/'sfincs_second_clock_budget.patch')],cwd=path,capture_output=True,text=True)
  check('unified_patch_applies_and_reproduces_all_patched_sources',result.returncode==0 and all(sha(path/'source/src'/p)==h for p,h in manifest['patched_sha256'].items()))
 prov=json.loads((report/'provenance.json').read_text())
 check('all_older_protected_files_unchanged',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('native_hydraulic_results_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==fp for p,fp in prov['existing_result_stat_fingerprints'].items()))
 lib=root/'data/processed/three_ward_minimum_scenarios_v1';frozen=json.loads((lib/'freeze.json').read_text())
 check('original_forcing_freeze_file_hashes_valid',all(sha(lib/p)==h for p,h in frozen['files_sha256'].items()))
 rows=list(csv.DictReader((report/'50_case_transport_receipt.csv').open()));catalog=json.loads((lib/'catalog.json').read_text())
 cases={c['case_id']:c for c in catalog['cases']if not c.get('existing_completed')};ok=len(rows)==len(cases)==50
 for row in rows:
  c=cases[row['case_id']]
  with np.load(lib/c['forcing_file'],allow_pickle=False)as z:semantic_hash=semantic({k:z[k]for k in z.files})
  ok &= semantic_hash==c['forcing_sha256']==row['original_forcing_sha256']and sha(lib/c['forcing_file'])==row['original_file_sha256']
  ok &= all(row[k]==c[k]for k in ['parent_id','family_id','split'])and float(row['time_error_s'])==0 and float(row['time_integral_error_mm'])==0
 check('all_50_time_curves_file_bytes_semantic_hashes_lineage_and_splits_unchanged',ok)
 bindings=json.loads((out/'pending_run_bindings.json').read_text())['bindings'];expected=expected_bindings(root)
 check('58_descendants_bound_to_current_patch_and_original_manifests',bindings==expected and len(bindings)==58 and all(not x['execution_release']for x in bindings))
 # Schema and source only: a diagnostic source hook must not be mistaken for admission.
 schema=json.loads((out/'diagnostic_export_schema.json').read_text());status=json.loads((out/'readiness.json').read_text())
 check('production_qualification_and_reference_release_still_held',not schema['reference_admission_release']and not status['execution_release']and not status['reference_release']and not summary['production_image_qualified'])
 before=(report/'previous-AnaSOP.md').read_text();after=(root/'docs/AnaSOP.md').read_text()
 parts=lambda text:re.split(r'(?=^## [1-8]\. )',text,flags=re.M)
 old,new=parts(before),parts(after);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(old,new)if a!=b and a.startswith('## ')}
 check('anasop_only_sections_4_6_7_changed',len(old)==len(new)and changed=={4,6,7}and all(a==b for a,b in zip(old,new)if not a.startswith(('## 4. ','## 6. ','## 7. '))))
 check('anasop_changed_sections_ascii_english_and_no_new_final_variables',all(s.isascii()for s in new if s.startswith(('## 4. ','## 6. ','## 7. ')))and after.count('| yes |')==before.count('| yes |'))
 sop=json.loads((report/'anasop-update.json').read_text());check('anasop_update_sha_receipt_matches',sop['current_sha256']==sha(root/'docs/AnaSOP.md'))
 page=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',page)
 check('local_review_links_and_image_exist',all((report/p).resolve().is_file()for p in links if not p.startswith(('http','#'))))
 compile_receipt=json.loads((report/'compile_receipt.json').read_text())
 check('no_production_link_hydrodynamic_run_training_or_sealed_response_read',not compile_receipt['production_solver_linked']and not compile_receipt['hydrodynamic_solver_executed']and summary['new_hydrodynamic_runs']==summary['training']==summary['sealed_responses_read']==0)
 if(out/'freeze.json').exists():
  freeze=json.loads((out/'freeze.json').read_text())
  check('frozen_delivery_files_and_source_code_hashes_valid',all(sha(root/p)==h for p,h in freeze['files_sha256'].items()))
 result={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'no_hydrodynamic_model_run':True,'no_training':True,'no_reference_release':True,'read_only_validation':not write_receipt}
 if write_receipt:(report/'delivery-validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--write-receipt',action='store_true');a=p.parse_args();main(write_receipt=a.write_receipt)
