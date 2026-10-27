"""Production fixtures and byte-preserving staging of the two existing adaptive controls."""
from pathlib import Path
import sys,json,subprocess,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_rain_step_integral import OUT,REPORT,PRIOR,TAG,CASES
from prepare_sfincs_cpu_build import sha,dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;old=ROOT/PRIOR;checks=[]
 image=json.loads((out/'build-metadata.json').read_text())['containerimage.digest'];binding=json.loads((out/'source_binding.json').read_text())
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 for kind,binary in [('regular','positivity_fixture'),('adaptive','adaptive_positivity_fixture'),('missing_child','adaptive_missing_child_fixture'),('rain','rain_integral_fixture')]:
  for threads in [1,4]:
   dest=out/'link_tests'/(kind+str(threads));dest.mkdir(parents=True,exist_ok=True)
   cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--env',f'OMP_NUM_THREADS={threads}','--mount',f'type=bind,source={dest},target=/data','--entrypoint','/opt/udt-build/'+binary,image]
   if (dest/'execution.json').exists():
    code=json.loads((dest/'execution.json').read_text())['returncode'];stdout=(dest/'stdout.txt').read_text()
   else:
    res=subprocess.run(cmd,text=True,capture_output=True);code=res.returncode;stdout=res.stdout;(dest/'stdout.txt').write_text(res.stdout);(dest/'stderr.txt').write_text(res.stderr);dump(dest/'execution.json',{'returncode':code,'command':cmd})
   check(f'{kind}_{threads}_production_fixture',code==0 and'PASS'in stdout)
  check(kind+'_serial_parallel_identical',(out/'link_tests'/(kind+'1')/'stdout.txt').read_bytes()==(out/'link_tests'/(kind+'4')/'stdout.txt').read_bytes())
  if kind!='rain':check(kind+'_frozen_positivity_regression_identical',(out/'link_tests'/(kind+'1')/'stdout.txt').read_bytes()==(old/'link_tests_v2'/kind/'stdout.txt').read_bytes())
 check('closure_limiter_budget_and_seconds_decoder_unchanged',all(sha(out/'context'/n)==sha(old/'context'/n)for n in ['sfincs_boundaries.f90','sfincs_continuity.f90','udt_clock_budget.f90','udt_rain_clock.f90']))
 check('source_and_full_patch_bound',sha(out/'full_solver.patch')==binding['full_patch_sha256']and all(sha(out/'context'/n)==v for n,v in binding['source_sha256'].items()))
 dump(report/'build_validation.json',{'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'hydrodynamic_runs':0,'training':False});assert all(c['pass']for c in checks)
 dump(out/'solver_binding.json',{'image_id':image,'image_tag':TAG,'patch_sha256':binding['full_patch_sha256'],'production_rain_fixture_sha256':sha(out/'context/rain_integral_fixture.f90'),'hydraulic_end_to_end_qualified':False})
 if(out/'run_manifest.json').exists():print('RETAIN existing staged inputs');return
 runs=[]
 for previous in json.loads((old/'run_manifest.json').read_text())['runs']:
  case=previous['case_id'];source=ROOT/previous['folder'];dest=out/'runs'/(case+'__rain_step_integral_10_5m');dest.mkdir(parents=True)
  for name,value in previous['input_sha256'].items():
   assert sha(source/name)==value
   if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(source/name)
   else:shutil.copyfile(source/name,dest/name)
  for name in ['scenario_contract.json','flux_sections.json']:shutil.copyfile(source/name,dest/name)
  ident=dict(previous['identity']);ident.update(run_id=dest.name,solver_digest=image,patch_sha256=binding['full_patch_sha256'])
  (dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident)
  runs.append({**previous,'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'source_folder':previous['folder'],'source_input_sha256':previous['input_sha256'],'identity':ident,'image_id':image,'image_digest':image,'input_config_differences':{},'input_sha256':dict(previous['input_sha256'])})
 dump(out/'run_manifest.json',{'runs':runs,'pending_58_scenarios_authorized':False,'training':False,'new_parent_count':0,'human_record':'MILI-D-20261015-016','prior_manifest_sha256':sha(old/'run_manifest.json')});print('STAGED exactly two byte-identical existing control inputs')
if __name__=='__main__':main()
