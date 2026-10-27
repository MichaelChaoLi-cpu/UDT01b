"""Full-image load and isolated production I/O/ledger checks. No solver entry calls."""
from pathlib import Path
import json,csv,sys,subprocess,shutil,os
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import OUT,REPORT,PREVIOUS,COMMIT,TAG,sha,dump
from stage_second_clock_budget_identity import expected_bindings,stage_identity

def read_csv(p):return list(csv.DictReader(Path(p).open()))
def write_csv(p,rows):
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def docker(args,report):
 r=subprocess.run(['docker',*map(str,args)],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout)
 return r.stdout

def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen; use delivery validation')
 metadata=json.loads((out/'build-metadata.json').read_text())
 digest=metadata['containerimage.digest']
 inspect=json.loads(docker(['image','inspect',TAG],report))[0]
 dump(out/'image-inspect.json',inspect)
 # Both digests are preserved explicitly: local image identity and OCI manifest identity.
 image=inspect['Id'];platform=inspect['Os']+'/'+inspect['Architecture']
 assert platform=='linux/amd64'
 outputs=out/'isolated_probe_outputs';outputs.mkdir(exist_ok=True)
 shutil.copytree(out/'context',report/'build-inputs',ignore=shutil.ignore_patterns('source'),dirs_exist_ok=True)
 cases=[c for c in json.loads((root/'data/processed/three_ward_minimum_scenarios_v1/catalog.json').read_text())['cases']if not c.get('existing_completed')]
 assert len(cases)==50
 script=['#!/bin/sh','set -eu','/opt/udt-build/netcdf_link_probe > /audit/netcdf-link-probe.txt','/opt/udt-build/loader_probe > /audit/loader-probe.txt','mkdir -p /audit/rain /audit/budget',"cp -R /opt/udt-build/receipts /audit/build-receipts",'cp /opt/udt-build/runtime-*-ldd.txt /audit/','cp /opt/udt-build/SOURCE_BINDING.json /audit/embedded-source-binding.json','nf-config --all > /audit/runtime-nf-config.txt','sha256sum /usr/local/bin/sfincs /usr/local/lib/libsfincs.so.0.0.0 /usr/local/lib/libnetcdff.so.7.2.0 > /audit/runtime-binaries.sha256']
 for c in cases:
  cid=c['case_id'];assert cid.replace('-','').replace('_','').isalnum()
  script.extend(['mkdir -p /audit/rain/'+cid,'(cd /audit/rain/'+cid+' && /opt/udt-build/production_rain_reader /inputs/'+cid+'.nc > stdout.txt 2>stderr.txt)'])
 # Precomputed bookkeeping fixtures only; no continuity, BMI or library update call.
 modes=['normal','dry','negative','unsupported','infiltration','spinup','storage','waves','block','clock_mismatch','nonfinite','missing_cell','missing_identity','wrong_identity']
 for mode in modes:
  folder=outputs/'budget'/mode;folder.mkdir(parents=True,exist_ok=True)
  if mode!='missing_identity':
   identity={'schema_version':'second-clock-budget-interface-v1','run_id':'isolated-fixture','parent_id':'fixture','family_id':'fixture','split':'fixture','forcing_sha256':'fixture','geometry_sha256':'fixture','source_commit':COMMIT,'patch_sha256':sha(root/PREVIOUS/'sfincs_second_clock_budget.patch'),'solver_digest':digest}
   if mode=='wrong_identity':identity['source_commit']='wrong'
   (folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in identity.items())+'\n')
  script.extend(['set +e','(cd /audit/budget/'+mode+' && /opt/udt-build/production_budget_fixture '+mode+' > stdout.txt 2>stderr.txt)','status=$?','set -e','printf "%s\\n" "$status" > /audit/budget/'+mode+'/exit_code.txt'])
 path=out/'run-isolated-probes.sh';path.write_text('\n'.join(script)+'\n')
 args=['run','--rm','--platform','linux/amd64','--network','none','--env','OMP_NUM_THREADS=2','--mount','type=bind,source='+str(root/PREVIOUS/'transport_test_inputs')+',target=/inputs,readonly','--mount','type=bind,source='+str(outputs)+',target=/audit','--mount','type=bind,source='+str(path)+',target=/run-isolated-probes.sh,readonly','--entrypoint','/bin/sh',image,'/run-isolated-probes.sh']
 receipt=docker(args,report);(report/'docker-probe-stdout.txt').write_text(receipt)
 checks=[]
 def check(name,ok,evidence=None):checks.append({'check':name,'pass':bool(ok),'evidence':evidence});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 check('embedded_source_binding_matches_host_verified_sources',sha(outputs/'embedded-source-binding.json')==sha(out/'source_binding.json'))
 check('full_image_linux_amd64_cpu_build',platform=='linux/amd64')
 check('real_Fortran_NetCDF_module_link_and_io_roundtrip','PASS:'in(outputs/'netcdf-link-probe.txt').read_text())
 loader=(outputs/'loader-probe.txt').read_text();check('complete_library_RTLD_NOW_and_ten_required_symbols_resolve',loader.count('RESOLVED')==10 and 'zero solver entry-point calls'in loader)
 check('runtime_executable_and_library_no_missing_dependencies',all('not found'not in(outputs/name).read_text()for name in ['runtime-executable-ldd.txt','runtime-library-ldd.txt']))
 symbols=(outputs/'build-receipts/exported-symbols.txt').read_text();check('new_rain_clock_and_budget_symbols_present',all(x in symbols for x in ['udt_decode_rain_seconds','udt_budget_begin','udt_budget_source_added','udt_budget_finish','sfincs_update']))
 rows=[]
 weights=np.array([[0,.25,1.5],[1,2,0]],dtype='f8')
 for c in cases:
  cid=c['case_id'];v=np.fromstring((outputs/'rain'/cid/'stdout.txt').read_text(),sep=' ');nt,ny,nx=v[:3].astype(int);seconds=v[3:3+nt];rain=v[3+nt:3+nt+nt*ny*nx].reshape(nt,ny,nx)
  with np.load(root/'data/processed/three_ward_minimum_scenarios_v1'/c['forcing_file'],allow_pickle=False)as z:t=z['Event Time'];r=z['Physical Rainfall Rate']
  expected=(r[:,None,None]*weights).astype('f4')[:,::-1,:].astype('f8')
  rows.append({'case_id':cid,'parent_id':c['parent_id'],'family_id':c['family_id'],'split':c['split'],'time_error_s':float(max(abs(seconds-t))),'time_integral_error_mm':float(abs(np.trapezoid(r,seconds)-np.trapezoid(r,t))/3600),'rain_values_match':bool(np.array_equal(rain,expected)),'forcing_file_sha256':sha(root/'data/processed/three_ward_minimum_scenarios_v1'/c['forcing_file']),'production_image_digest':digest})
 write_csv(report/'50_production_reader_checks.csv',rows)
 check('all_50_inputs_through_complete_production_reader_and_Fortran_NetCDF',len(rows)==50 and all(x['time_error_s']==0 and x['time_integral_error_mm']==0 and x['rain_values_match']for x in rows))
 fixture=[]
 for mode in modes:
  actual=int((outputs/'budget'/mode/'exit_code.txt').read_text());success=mode in ['normal','dry','negative']
  fixture.append({'fixture':mode,'expected_success':success,'exit_code':actual,'pass':(actual==0)==success})
 write_csv(report/'production_budget_fixture_checks.csv',fixture)
 check('production_linked_budget_fixtures_and_profile_rejections',all(x['pass']for x in fixture))
 ledger=read_csv(outputs/'budget/normal/udt_step_budget.csv');a=lambda k:np.array([float(x[k])for x in ledger]);mapping=read_csv(outputs/'budget/normal/udt_output_clock.csv')
 check('true_initial_actual_dt_clock_and_all_step_rows',len(ledger)==5 and a('actual_state_time')[0]==a('actual_step_dt')[0]==0 and np.allclose(np.diff(a('actual_state_time')),a('actual_step_dt')[1:],atol=1e-12,rtol=0))
 check('same_step_gross_exchange_rain_and_roundoff_retained',a('boundary_out_cumulative')[-1]>.99 and a('boundary_in_cumulative')[-1]>.99 and a('rain_applied_cumulative')[-1]>0 and max(abs(a('budget_residual')))<5e-5,{'fixture_max_residual_m3':float(max(abs(a('budget_residual'))))})
 bystep={int(x['step_index']):x for x in ledger}
 check('all_output_records_bound_to_same_accounted_state',all(float(x['actual_state_time'])==float(bystep[int(x['step_index'])]['actual_state_time'])for x in mapping))
 negative=read_csv(outputs/'budget/negative/udt_native_ever_negative.csv');check('production_online_negative_mask_kept',negative[1]['ever_volume_below_minus_0_001_m3']=='T')
 expected=expected_bindings(root);bindings=[dict(b,solver_digest=digest,local_image_id=image,image_tag=TAG,build_link_verified=True,hydrodynamic_end_to_end_verified=False,execution_release=False,reference_release=False)for b in expected]
 dump(out/'run_bindings.json',{'runs':bindings,'case_count':50,'descendant_count':58,'all_unrun':True})
 # Exercise the existing staging helper with the actual new manifest digest, only in a separate new directory.
 identity=stage_identity(bindings[0]['run_id'],out/'identity_staging_example',digest,root)
 check('58_unchanged_descendants_bound_to_real_new_image',len(bindings)==58 and identity['solver_digest']==digest and all(not b['execution_release']for b in bindings))
 prov=json.loads((report/'provenance.json').read_text())
 check('original_curves_lineage_splits_interface_sources_and_policies_unchanged',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('previous_hydrodynamic_outputs_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==fp for p,fp in prov['existing_result_stat_fingerprints'].items()))
 binding={'source_commit':COMMIT,'interface_patch_sha256':sha(root/PREVIOUS/'sfincs_second_clock_budget.patch'),'source_binding_sha256':sha(out/'source_binding.json'),'build_recipe_sha256':{p.name:sha(p)for p in(out/'context').iterdir()if p.is_file()},'image_tag':TAG,'local_image_id':image,'oci_manifest_digest':digest,'oci_config_digest':metadata.get('containerimage.config.digest'),'platform':platform,'base_image':json.loads((out/'source_binding.json').read_text())['base_image'],'runtime_binaries_sha256':(outputs/'runtime-binaries.sha256').read_text(),'Fortran_NetCDF':(outputs/'runtime-nf-config.txt').read_text(),'build_link_qualified':True,'hydrodynamic_end_to_end_qualified':False,'solver_initialize_calls':0,'solver_update_calls':0,'training':0,'reference_release':False}
 dump(out/'solver_image_binding.json',binding)
 summary={'checks':len(checks),'all_checks_passed':all(x['pass']for x in checks),'inputs_tested':50,'source_commit':COMMIT,'image_tag':TAG,'oci_manifest_digest':digest,'local_image_id':image,'build_link_qualified':True,'hydrodynamic_end_to_end_qualified':False,'hydrodynamic_runs':0,'training':0,'sealed_responses_read':0,'pending_descendants':58,'reference_release':False}
 dump(report/'validation.json',{'checks':checks,'all_checks_passed':summary['all_checks_passed']});dump(report/'summary.json',summary)
 print(json.dumps(summary,indent=2))
 if not summary['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
