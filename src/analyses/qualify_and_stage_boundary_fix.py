"""Link-test the closure fix against its predecessor, then stage exactly two controls."""
from pathlib import Path
import sys,json,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_boundary_closure_fix import OUT,REPORT,PRIOR,INTERFACE,CASES,TAG
from prepare_sfincs_cpu_build import sha,dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;binding=json.loads((out/'source_binding.json').read_text());meta=json.loads((out/'build-metadata.json').read_text());image=meta['containerimage.digest']
 assert not(out/'run_manifest.json').exists()
 tests=out/'link_tests';tests.mkdir(exist_ok=True)
 command=['docker','run','--rm','--platform','linux/amd64','--network','none','--mount','type=bind,source='+str(tests)+',target=/audit','--entrypoint','/bin/sh',image,'-ec','/opt/udt-build/closed_boundary_fixture > /audit/corrected.txt; cp /opt/udt-build/closed_boundary_fixture /audit/fixture; /opt/udt-build/netcdf_link_probe > /audit/netcdf.txt; /opt/udt-build/loader_probe > /audit/loader.txt; ldd /usr/local/lib/libsfincs.so > /audit/ldd.txt; sha256sum /usr/local/bin/sfincs /usr/local/lib/libsfincs.so.0.0.0 /usr/local/lib/libnetcdff.so.7.2.0 > /audit/binaries.sha256; cp /opt/udt-build/CLOSURE_BINDING.json /audit/embedded_binding.json; cp /opt/udt-build/receipts/closure-* /audit/']
 subprocess.run(command,check=True)
 old=subprocess.run(['docker','run','--rm','--platform','linux/amd64','--network','none','--mount','type=bind,source='+str(tests/'fixture')+',target=/fixture,readonly','--entrypoint','/fixture',binding['prior_runtime_image_id']],capture_output=True,text=True,check=True);(tests/'previous.txt').write_text(old.stdout)
 a=np.loadtxt(tests/'previous.txt');b=np.loadtxt(tests/'corrected.txt');checks=[]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)})
 check('production_boundary_fixture_matches_modes_and_calls',a.shape==b.shape==(6,11)and np.array_equal(a[:,:2],b[:,:2]))
 check('closed_flux_velocity_memory_zero_in_regular_and_subgrid',np.all(b[:,[2,5,8]]==0))
 check('previous_image_exhibits_closed_face_exchange',np.all(a[:,2]!=0))
 check('open_boundary_exchange_velocity_memory_unchanged',np.array_equal(a[:,[3,6,9]],b[:,[3,6,9]]))
 check('existing_structure_zero_exchange_unchanged',np.array_equal(a[:,[4,7,10]],b[:,[4,7,10]])and np.all(b[:,[4,7,10]]==0))
 check('Fortran_NetCDF_link_and_solver_loader_pass','PASS:'in(tests/'netcdf.txt').read_text()and'zero solver entry-point calls'in(tests/'loader.txt').read_text()and'not found'not in(tests/'ldd.txt').read_text())
 check('embedded_source_binding_matches',sha(tests/'embedded_binding.json')==sha(out/'source_binding.json'))
 materials={'sha256:'+x['digest']['sha256']for x in meta['buildx.build.provenance']['materials']};check('local_base_image_ids_bound',materials=={binding['base_builder_image_id'],binding['prior_runtime_image_id']})
 before=(ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src/sfincs_boundaries.f90').read_text();after=(out/'context/sfincs_boundaries.f90').read_text();check('only_closed_flag_condition_added',after==before.replace('if (hnmb < huthresh + 1.0e-6 .or. kcuv(ip) == 3) then','if (hnmb < huthresh + 1.0e-6 .or. kcuv(ip) == 3 .or. kcuv(ip) == 0) then'))
 for name in ['udt_clock_budget.f90','udt_rain_clock.f90']:check(name+'_unchanged',sha(out/'context'/name)==sha(ROOT/INTERFACE/'context'/name))
 summary={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'hydrodynamic_runs':0,'production_boundary_calls_only':True,'training':0,'batch_58':0};dump(report/'build_validation.json',summary)
 assert summary['all_checks_passed'],checks
 dump(out/'solver_binding.json',{'image_tag':TAG,'image_id':image,'patch_sha256':binding['full_patch_sha256'],'source_binding_sha256':sha(out/'source_binding.json'),'production_interface_qualified':True,'hydraulic_end_to_end_qualified':False,'trace_activation_requires_manifest_metadata':True,'boundary_only_source_change':True})
 manifest=json.loads((ROOT/PRIOR/'run_manifest.json').read_text());runs=[]
 for old in manifest['runs']:
  source=ROOT/old['folder'];dest=out/'runs'/(old['case_id']+'__closed_boundary_10m');dest.mkdir(parents=True);inputs={}
  for name,h in old['input_sha256'].items():
   assert sha(source/name)==h,name
   if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(source/name)
   else:shutil.copyfile(source/name,dest/name)
   inputs[name]=h
  ident=dict(old['identity']);ident['run_id']=dest.name;ident['solver_digest']=image;ident['patch_sha256']=binding['full_patch_sha256']
  (dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident)
  runs.append({**old,'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'source_folder':old['folder'],'previous_e2e_folder':old['folder'],'image_id':image,'image_digest':image,'identity':ident,'input_sha256':inputs,'large_readonly_source':old['large_readonly_source'],'library_descendant':False,'trace_required':True,'new_parent_event':False})
 assert [r['case_id']for r in runs]==CASES
 dump(out/'run_manifest.json',{'runs':runs,'case_count':2,'pending_58_scenarios_authorized':False,'training':False,'control_family':'CONTROL-LEGACY','reference_release':False,'human_record':'MILI-D-20261015-013'})
 print(json.dumps({'image_id':image,'checks':len(checks),'all_passed':True,'staged':CASES}))
if __name__=='__main__':main()
