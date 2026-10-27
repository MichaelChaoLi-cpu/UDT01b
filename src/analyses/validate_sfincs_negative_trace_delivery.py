"""Validate compact-diagnostic deployment evidence without running a hydraulic solver."""
from pathlib import Path
import sys,json,csv,hashlib,tarfile,tempfile,shutil,subprocess,re,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_negative_trace import OUT,REPORT
from prepare_sfincs_cpu_build import sha,dump
sys.path.insert(0,str(ROOT/'src/analyses'))
from summarize_sfincs_negative_trace import summarize,read

def main(freeze=False):
 out=ROOT/OUT;report=ROOT/REPORT;checks=[];cache={}
 def h(p):
  st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 def check(n,ok):checks.append({'check':n,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+n,flush=True)
 v=json.loads((report/'validation.json').read_text());binding=json.loads((out/'source_binding.json').read_text());solver=json.loads((out/'solver_binding.json').read_text());meta=json.loads((out/'build-metadata.json').read_text())
 check('all_114_production_interface_and_regression_checks_pass',len(v['checks'])==114 and v['all_checks_passed']and all(x['pass']for x in v['checks']))
 check('no_hydraulic_initialization_updates_training_or_batch',v['whole_domain_hydrodynamic_runs']==v['solver_initialize_calls']==v['solver_update_calls']==v['training']==v['batch_58']==0 and not v['old_control_times_backfilled']and not v['formal_reference_release'])
 check('qualified_source_patch_image_bound',solver['image_id']==v['new_image_id']==meta['containerimage.digest']and solver['patch_sha256']==h(out/'negative_trace_full.patch')==v['new_patch_sha256']and solver['production_interface_qualified']and not solver['hydraulic_end_to_end_qualified'])
 materials={'sha256:'+x['digest']['sha256']for x in meta['buildx.build.provenance']['materials']}
 check('exact_cached_builder_and_previously_qualified_runtime',materials=={binding['base_builder_image_id'],binding['base_runtime_image_id']})
 original=ROOT/'data/processed/three_ward_second_clock_budget_v1';old_manifest=json.loads((original/'source_manifest.json').read_text());expected=dict(old_manifest['patched_sha256'])
 expected['udt_clock_budget.f90']=h(ROOT/'src/preprocessing/sfincs_negative_trace_v1/udt_clock_budget.f90');expected['udt_rain_clock.f90']=h(ROOT/'src/preprocessing/sfincs_second_clock_budget_v2/udt_rain_clock.f90')
 with tempfile.TemporaryDirectory(prefix='udt01b-negative-patch-',dir='/private/tmp')as tmp:
  base=Path(tmp);(base/'source/src').mkdir(parents=True)
  for p in(original/'base').iterdir():shutil.copyfile(p,base/'source/src'/p.name)
  result=subprocess.run(['/usr/bin/patch','--batch','-p1','-i',str(out/'negative_trace_full.patch')],cwd=base,capture_output=True,text=True)
  check('full_patch_applies_and_hydraulic_equation_files_remain_frozen',result.returncode==0 and all(h(base/'source/src'/p)==value for p,value in expected.items()))
 # Derived complete module exactly equals old module plus documented read-only hooks/fragments.
 budget=(original/'patched/udt_clock_budget.f90').read_text();src=ROOT/'src/preprocessing/sfincs_negative_trace_v1'
 budget=budget.replace('contains\n',(src/'negative_declarations.inc').read_text()+'contains\n',1)
 budget=budget.replace("  open(newunit=ledger_unit,file='udt_step_budget.csv'","  call negative_trace_initialize()\n  open(newunit=ledger_unit,file='udt_step_budget.csv'",1)
 budget=budget.replace('  in_step=.true.\n end subroutine','  call negative_trace_begin()\n  in_step=.true.\n end subroutine',1)
 budget=budget.replace('  call write_ledger()\n  in_step=.false.','  call negative_trace_finish(actual_time,dt)\n  call write_ledger()\n  in_step=.false.',1)
 budget=budget.replace('  deallocate(rain_term,horizontal_term,rain_increment,ever_negative,cell_seen)','  call negative_trace_close()\n  deallocate(rain_term,horizontal_term,rain_increment,ever_negative,cell_seen)',1)
 budget=budget.replace('end module',(src/'negative_routines.inc').read_text()+'end module')
 check('only_additive_budget_hooks_and_trace_fragments',budget==(src/'udt_clock_budget.f90').read_text()and h(src/'udt_clock_budget.f90')==binding['budget_module_sha256'])
 check('actual_production_fixture_never_calls_hydraulic_entry_points',not re.search(r'call\s+(?:sfincs_initialize|sfincs_update|update_momentum|update_continuity)',(src/'negative_trace_fixture.f90').read_text(),re.I))
 metadata=json.loads((out/'metadata_binding.json').read_text());whole=out/'interface_tests/whole_metadata'
 check('whole_native_metadata_bytes_used_in_production_fixture',h(whole/'udt_negative_native_map.csv')==metadata['map_sha256']==h(out/'whole_domain_metadata/udt_negative_native_map.csv')and h(whole/'udt_negative_trace.request')==metadata['request_sha256'])
 check('whole_topology_fixture_has_true_zero_step_receipt',len(read(whole/'udt_step_budget.csv'))==1 and read(whole/'udt_final_clock.csv')[0]['step_index']=='0'and not read(whole/'udt_negative_crossings.csv'))
 schema=json.loads((out/'output_schema.json').read_text())
 check('schema_distinguishes_endpoint_time_initial_state_group_overlap_and_budget',schema['threshold_m3']==-.001 and not schema['old_control_records_backfilled']and schema['groups']['79']==schema['groups']['111']and 'bracket'in schema['comparison']and'not an event water budget'in( ROOT/'src/analyses/summarize_sfincs_negative_trace.py').read_text())
 # Corruption tests operate on copied fixture tables, never on original evidence.
 normal=out/'interface_tests/normal'
 for mode in ['missing_face','unknown_group','duplicate_direction']:
  with tempfile.TemporaryDirectory(prefix='udt01b-negative-corruption-',dir='/private/tmp')as tmp:
   folder=Path(tmp)
   for p in normal.glob('udt_negative_*.csv'):shutil.copyfile(p,folder/p.name)
   p=folder/'udt_negative_crossing_faces.csv';rows=read(p);fields=list(rows[0])
   if mode=='missing_face':rows.pop()
   if mode=='unknown_group':rows[0]['boundary_group_code']='999'
   if mode=='duplicate_direction':rows[1]['direction']=rows[0]['direction']
   with p.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
   try:summarize(folder,folder/'grouped',schema['groups'])
   except ValueError:rejected=True
   else:rejected=False
   check('aggregator_rejects_'+mode,rejected)
 # Existing controls and the previous spatial audit remain byte-identical.
 for rel in ['three_ward_cpu_e2e_controls_v1','three_ward_negative_volume_audit_v1']:
  f=json.loads((ROOT/'data/processed'/rel/'freeze.json').read_text())
  check(rel+'_entire_frozen_delivery_unchanged',all(h(ROOT/p)==value for p,value in f['files_sha256'].items()))
 prov=json.loads((report/'provenance.json').read_text());check('bound_original_inputs_and_policies_unchanged',all(h(ROOT/p)==value for p,value in prov['protected_sha256'].items()))
 pending=json.loads((ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs'];check('all_58_future_cases_remain_unrun',len(pending)==58 and all(x['status']=='unrun'and x['response_path']is None for x in pending))
 registry=json.loads((ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json').read_text());check('default_reference_access_and_training_remain_closed',registry['default_response_access']=='closed'and all(not x['technical_reference']and not x['training_release']for x in registry['controls'].values()))
 # Verify archive content addressing and actual linux/amd64 config, not only a tag.
 blobs={};ok=True;count=0
 with tarfile.open(out/'solver_image.tar')as archive:
  for member in archive.getmembers():
   if member.isfile()and member.name.startswith('blobs/sha256/'):
    digest=hashlib.sha256();small=bytearray()
    with archive.extractfile(member)as stream:
     for chunk in iter(lambda:stream.read(2**20),b''):
      digest.update(chunk)
      if member.size<2**20:small.extend(chunk)
    key='sha256:'+digest.hexdigest();ok &=key.split(':')[1]==member.name.split('/')[-1];count+=1
    if small:blobs[key]=bytes(small)
 index=json.loads(blobs[solver['image_id']]);platform=[r for r in index['manifests']if r.get('platform',{}).get('os')=='linux'and r.get('platform',{}).get('architecture')=='amd64'];manifest=json.loads(blobs[platform[0]['digest']]);config=json.loads(blobs[manifest['config']['digest']])
 check('offline_image_all_blobs_exact_qualified_index_and_amd64_config',ok and len(platform)==1 and config['architecture']=='amd64')
 archive_binding={'image_id':solver['image_id'],'platform_manifest':platform[0]['digest'],'config_digest':manifest['config']['digest'],'archive_sha256':h(out/'solver_image.tar'),'verified_blobs':count}
 if(out/'oci_archive_binding.json').exists():assert json.loads((out/'oci_archive_binding.json').read_text())==archive_binding
 else:dump(out/'oci_archive_binding.json',archive_binding)
 inspect=subprocess.run(['docker','image','inspect',solver['image_id']],capture_output=True,text=True)
 check('qualified_immutable_image_available_locally',inspect.returncode==0 and json.loads(inspect.stdout)[0]['Architecture']=='amd64')
 check('superseded_sentinel_assumption_retained_and_not_used', (out/'attempts/initial-sentinel-assumption/NOT_QUALIFIED.json').exists()and json.loads((out/'attempts/initial-sentinel-assumption/build-metadata.json').read_text())['containerimage.digest']!=solver['image_id'])
 before=(report/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text();split=lambda x:re.split(r'(?=^## [1-8]\. )',x,flags=re.M)
 aa,bb=split(before),split(after);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(aa,bb)if a!=b}
 check('AnaSOP_only_skill_routed_4_6_7_and_no_final_variable_changes',changed=={4,6,7}and before.count('| yes |')==after.count('| yes |')and all(x.isascii()for x in bb if x.startswith(('## 4. ','## 6. ','## 7. '))))
 page=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',page)
 check('review_links_resolve_and_fixture_origin_explicit',all((report/p).is_file()for p in links)and'预设记账算例'in page and'旧两组真实触发历史没有回填'in page)
 if freeze:
  assert all(x['pass']for x in checks)
  paths=[p for base in [out,report]for p in base.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery-validation.json']]
  paths+=list(src.glob('*'))
  paths+=[ROOT/'src/preprocessing/prepare_sfincs_negative_trace.py']
  paths+=[ROOT/'src/analyses'/name for name in ['qualify_sfincs_negative_trace.py','summarize_sfincs_negative_trace.py','review_sfincs_negative_trace.py','validate_sfincs_negative_trace_delivery.py']]
  dump(out/'freeze.json',{'state':'compact_negative_trace_production_interface_qualified_no_hydraulic_replay','files_sha256':{str(p.relative_to(ROOT)):h(p)for p in sorted(paths)},'image_id':solver['image_id'],'hydraulic_initialize_calls':0,'hydraulic_update_calls':0,'training':0,'batch_58':0,'formal_reference_release':False})
 if(out/'freeze.json').exists():
  f=json.loads((out/'freeze.json').read_text());check('all_frozen_interface_hashes_match',all(h(ROOT/p)==value for p,value in f['files_sha256'].items()))
 result={'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'hydraulic_runs':0,'training':0,'batch_58':0,'formal_reference_release':False};dump(report/'delivery-validation.json',result)
 print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');main(a.parse_args().freeze)
