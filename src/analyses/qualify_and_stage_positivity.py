"""Production-linked positivity/conservation fixtures, then exactly two matched controls."""
from pathlib import Path
import sys,json,subprocess,shutil,io
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_conservative_positivity import OUT,REPORT,PRIOR,INTERFACE,CASES,TAG
from prepare_sfincs_cpu_build import sha,dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;binding=json.loads((out/'source_binding.json').read_text());meta=json.loads((out/'build-metadata.json').read_text());image=meta['containerimage.digest'];assert not(out/'run_manifest.json').exists()
 tests=out/'link_tests';tests.mkdir(exist_ok=True);modes=['normal','parallel','negative_initial','negative_rain','adaptive','bad_dummy','nonfinite'];script=['set -eu','/opt/udt-build/netcdf_link_probe > /audit/netcdf.txt','/opt/udt-build/loader_probe > /audit/loader.txt','ldd /usr/local/lib/libsfincs.so > /audit/ldd.txt','sha256sum /usr/local/bin/sfincs /usr/local/lib/libsfincs.so.0.0.0 /usr/local/lib/libnetcdff.so.7.2.0 > /audit/binaries.sha256','cp /opt/udt-build/POSITIVITY_BINDING.json /audit/embedded_binding.json','cp /opt/udt-build/receipts/positivity-* /audit/']
 for mode in modes:
  (tests/mode).mkdir(exist_ok=True)
  script+=['set +e',f'(cd /audit/{mode} && OMP_NUM_THREADS={4 if mode=="parallel"else 1} /opt/udt-build/positivity_fixture {mode} >stdout.txt 2>stderr.txt)','code=$?','set -e',f'printf "%s\\n" "$code" > /audit/{mode}/exit_code.txt']
 path=out/'run-fixtures.sh';path.write_text('\n'.join(script)+'\n');subprocess.run(['docker','run','--rm','--platform','linux/amd64','--network','none','--mount','type=bind,source='+str(tests)+',target=/audit','--mount','type=bind,source='+str(path)+',target=/probe.sh,readonly','--entrypoint','/bin/sh',image,'/probe.sh'],check=True)
 checks=[]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 for mode in modes:
  code=int((tests/mode/'exit_code.txt').read_text());text=(tests/mode/'stdout.txt').read_text()+(tests/mode/'stderr.txt').read_text();check('production_limiter_'+mode,(code==0 and'PASS minimum_updated_volume='in text)if mode in ['normal','parallel']else code!=0 and'UDT positivity:'in text)
 normal=(tests/'normal/stdout.txt').read_text();parallel=(tests/'parallel/stdout.txt').read_text();check('serial_parallel_updated_states_and_fluxes_identical',normal==parallel)
 if int((tests/'normal/exit_code.txt').read_text())==0:
  values=np.loadtxt(io.StringIO('\n'.join(line for line in normal.splitlines()if not line.startswith('PASS'))));check('all_205_prescribed_updates_nonnegative_including_subnormals',values.shape==(205,13)and np.all(values[:,4:7]>=0))
  # Independent first four prescribed cases, in scalar native single precision.
  qq=values[:4,7:12].astype('f4');before=values[:4,1:4];rain=np.zeros((4,3),dtype='f4');rain[3]=np.array([.001,.0001,0],dtype='f4')
  ptr=[[0,1,2,3],[1,5,5,5],[5,5,3,4]];expected=before.copy()
  for n in range(4):
   q=np.r_[qq[n],np.float32(0)]
   for j,(w,e,s,no)in enumerate(ptr):
    h=np.float32(np.float32(np.float32(q[w]-q[e])*np.float32(10))+np.float32(np.float32(q[s]-q[no])*np.float32(10)))*np.float32(.4);source=np.float32(np.float32(rain[n,j]*np.float32(100))*np.float32(.4));expected[n,j]+=float(np.float32(h+source))
  check('independent_native_arithmetic_matches_prescribed_dry_multiface_rain_updates',np.array_equal(expected,values[:4,4:7]))
  check('external_boundary_inflows_are_not_artificially_limited',values[1,7]==float(np.float32(.2))and values[1,9]==float(np.float32(.3)))
  check('inactive_limiter_preserves_all_requested_fluxes',np.array_equal(values[0,7:12],np.full(5,np.float32(.01),dtype='f4').astype('f8')))
 check('Fortran_NetCDF_link_and_solver_loader_pass','PASS:'in(tests/'netcdf.txt').read_text()and'zero solver entry-point calls'in(tests/'loader.txt').read_text()and'not found'not in(tests/'ldd.txt').read_text())
 check('embedded_source_binding_matches',sha(tests/'embedded_binding.json')==sha(out/'source_binding.json'))
 materials={'sha256:'+x['digest']['sha256']for x in meta['buildx.build.provenance']['materials']};check('exact_local_builder_and_closed_boundary_runtime_bound',materials=={binding['base_builder_image_id'],binding['prior_runtime_image_id']})
 for name in ['sfincs_boundaries.f90','udt_clock_budget.f90','udt_rain_clock.f90']:check(name+'_unchanged',sha(out/'context'/name)==sha(ROOT/PRIOR/'context'/name))
 result={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'hydrodynamic_runs':0,'prescribed_production_limiter_only':True,'training':0,'batch_58':0};dump(report/'build_validation.json',result);assert result['all_checks_passed']
 dump(out/'solver_binding.json',{'image_tag':TAG,'image_id':image,'patch_sha256':binding['full_patch_sha256'],'source_binding_sha256':sha(out/'source_binding.json'),'production_interface_qualified':True,'hydraulic_end_to_end_qualified':False,'trace_activation_requires_manifest_metadata':True,'conservative_donor_limit':True})
 previous=json.loads((ROOT/PRIOR/'run_manifest.json').read_text());runs=[]
 for old in previous['runs']:
  source=ROOT/old['folder'];dest=out/'runs'/(old['case_id']+'__conservative_positivity_10m');dest.mkdir(parents=True)
  for name,h in old['input_sha256'].items():
   assert sha(source/name)==h
   if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(source/name)
   else:shutil.copyfile(source/name,dest/name)
  ident=dict(old['identity']);ident.update(run_id=dest.name,solver_digest=image,patch_sha256=binding['full_patch_sha256']);(dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident)
  runs.append({**old,'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'source_folder':old['folder'],'previous_e2e_folder':old['folder'],'image_id':image,'image_digest':image,'identity':ident,'large_readonly_source':old['large_readonly_source'],'library_descendant':False,'new_parent_event':False})
 assert [r['case_id']for r in runs]==CASES;dump(out/'run_manifest.json',{'runs':runs,'case_count':2,'pending_58_scenarios_authorized':False,'training':False,'control_family':'CONTROL-LEGACY','reference_release':False,'human_record':'MILI-D-20261015-014'});print(json.dumps({'checks':len(checks),'image':image,'staged_cases':CASES}))
if __name__=='__main__':main()
