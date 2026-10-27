"""Production-linked grid fixtures and exactly two matched adaptive control companions."""
from pathlib import Path
import sys,json,subprocess,shutil,csv
import numpy as np
import netCDF4
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_adaptive_positivity import OUT,REPORT,PRIOR,TAG,CASES
from prepare_sfincs_cpu_build import sha,dump

def main():
 out=ROOT/OUT;report=ROOT/REPORT;tests=out/'link_tests_v2';tests.mkdir(exist_ok=True)
 binding=json.loads((out/'source_binding.json').read_text());image=json.loads((out/'build-metadata.json').read_text())['containerimage.digest'];checks=[]
 def check(name,ok):checks.append({'check':name,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 for kind,threads in [('regular',1),('regular_parallel',4),('adaptive',1),('adaptive_parallel',4),('missing_child',1),('missing_child_parallel',4)]:
  folder=tests/kind;folder.mkdir(exist_ok=True);binary='adaptive_missing_child_fixture'if kind.startswith('missing_child')else'adaptive_positivity_fixture'if kind.startswith('adaptive')else'positivity_fixture'
  cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--env',f'OMP_NUM_THREADS={threads}','--mount',f'type=bind,source={folder},target=/data','--entrypoint','/opt/udt-build/'+binary,image]
  
  if (folder/'execution.json').exists() and json.loads((folder/'execution.json').read_text())['returncode']==0:
   result_code=0;result_stdout=(folder/'stdout.txt').read_text()
  else:
   res=subprocess.run(cmd,text=True,capture_output=True);result_code=res.returncode;result_stdout=res.stdout;(folder/'stdout.txt').write_text(res.stdout);(folder/'stderr.txt').write_text(res.stderr);dump(folder/'execution.json',{'returncode':res.returncode,'command':cmd})
  check(kind+'_production_fixture',result_code==0 and'PASS'in result_stdout)
 check('regular_205_native_updates_byte_identical_to_qualified_v2',(tests/'regular/stdout.txt').read_bytes()==(ROOT/PRIOR/'link_tests/normal/stdout.txt').read_bytes())
 for kind in ['regular','adaptive','missing_child']:check(kind+'_serial_parallel_identical',(tests/kind/'stdout.txt').read_bytes()==(tests/(kind+'_parallel')/'stdout.txt').read_bytes())
 check('closure_and_seconds_reader_unchanged',all(sha(out/'context'/n)==sha(ROOT/PRIOR/'context'/n)for n in ['sfincs_boundaries.f90','udt_rain_clock.f90']))
 validation={'all_checks_passed':all(c['pass']for c in checks),'checks':checks,'training':False,'hydrodynamic_runs':0};dump(report/'build_validation.json',validation);assert validation['all_checks_passed']
 dump(out/'solver_binding.json',{'image_id':image,'image_tag':TAG,'patch_sha256':binding['full_patch_sha256'],'regular_fixtures_byte_identical':True,'adaptive_fixtures_pass':True,'hydraulic_end_to_end_qualified':False})
 if(out/'run_manifest.json').exists():raise FileExistsError('Preserve staged controls')
 geom=ROOT/'data/processed/three_ward_adaptive_connectivity';g=dict(np.load(geom/'adaptive_cell_geometry.npz'))
 # One code per native prescribed-stage cell, inherited from its original 20 m parent.
 with(ROOT/'data/processed/three_ward_sfincs_scenarios/boundary_points.csv').open()as f:rows=list(csv.DictReader(f))
 print('boundary metadata keys',list(rows[0]),flush=True)
 # Codes come from the frozen regular trace map; no inference from names or new stage values.
 oldruns=json.loads((ROOT/PRIOR/'run_manifest.json').read_text())['runs'];mapping={}
 oldmap=ROOT/oldruns[0]['folder']/'udt_negative_native_map.csv'
 with oldmap.open()as f:
  for row in csv.DictReader(f):
   if int(row['kcs'])==2:mapping[((1012-int(row['row_south_one_based']))//2,(int(row['col_east_one_based'])-1)//2)]=int(row['water_group_code'])
 groups=np.zeros(len(g['mask']),np.int32)
 for j in np.flatnonzero(g['mask']==2):groups[j]=mapping[(int(g['row_fine'][j]//40),int(g['col_fine'][j]//40))]
 assert set(groups[g['mask']==2])=={17,51,71,79,106,110,111}
 runs=[]
 for old in oldruns:
  case=old['case_id'];source=ROOT/'data/processed/three_ward_sfincs_framework/adaptive_sensitivity'/(case+'_adaptive_10_5m');ex=json.loads((source/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker']
  dest=out/'runs'/(case+'__conservative_positivity_10_5m');dest.mkdir(parents=True,exist_ok=True)
  for name,h in ex['input_sha256'].items():
   assert sha(source/name)==h
   if name=='sfincs_subgrid.nc':None if (dest/name).exists() else (dest/name).hardlink_to(source/name)
   else:shutil.copyfile(source/name,dest/name)
  shutil.copyfile(source/'flux_sections.json',dest/'flux_sections.json');shutil.copyfile(source/'scenario_contract.json',dest/'scenario_contract.json')
  # Preserve all scientific settings; enable the same complete peak-window clock and outputs as 10 m.
  regular={line.split('=',1)[0].strip():line.split('=',1)[1].strip()for line in (ROOT/old['folder']/'sfincs.inp').read_text().splitlines()if'='in line}
  adaptive={line.split('=',1)[0].strip():line.split('=',1)[1].strip()for line in (dest/'sfincs.inp').read_text().splitlines()if'='in line}
  diff={k:[regular.get(k),adaptive.get(k)]for k in regular.keys()|adaptive.keys()if regular.get(k)!=adaptive.get(k)}
  # Geometry input file selectors are the only allowed scientific differences.
  allowed={'qtrfile','depfile','indexfile','mskfile','dtmax','dtout','dtmaxout','store_t_zsmax','storetzsmax','amprblock'}
  assert set(diff)<=allowed,diff
  for k in ['dtmax','dtout','dtmaxout','store_t_zsmax','storetzsmax','amprblock']:
   if k in regular:adaptive[k]=regular[k]
  (dest/'sfincs.inp').write_text(''.join(f'{k:24s} = {v}\n'for k,v in adaptive.items()))
  np.savetxt(dest/'udt_adaptive_groups.txt',np.c_[np.arange(1,len(groups)+1),groups],fmt='%d')
  inputs={p.name:sha(p)for p in dest.iterdir()if p.name not in ['scenario_contract.json','flux_sections.json']}
  av=np.loadtxt(dest/'sfincs.bzs');bv=np.loadtxt(ROOT/old['folder']/'sfincs.bzs');ab=np.loadtxt(dest/'sfincs.bnd');bb=np.loadtxt(ROOT/old['folder']/'sfincs.bnd');assert np.array_equal(av[:,0],bv[:,0])
  lookup={((int(np.floor((xy[0]+11360)/20))),int(np.floor((-32320-xy[1])/20))):bv[:,j+1]for j,xy in enumerate(bb)}
  max_stage_error=max(float(np.max(abs(av[:,j+1]-lookup[(int(np.floor((xy[0]+11360)/20)),int(np.floor((-32320-xy[1])/20)))])))for j,xy in enumerate(ab))
  assert max_stage_error<1e-8,max_stage_error
  with netCDF4.Dataset(dest/'roof_surface_precip.nc')as a,netCDF4.Dataset(ROOT/old['folder']/'roof_surface_precip.nc')as b:
   assert np.array_equal(a['time'][:],b['time'][:]);assert a['time'].units.startswith('minutes since');assert np.allclose(a['time'][:]*60,np.array([0,3600,3601,7200,7201,14400],dtype='f8'),rtol=0,atol=1e-9)
  ident=dict(old['identity']);ident.update(run_id=dest.name,solver_digest=image,patch_sha256=binding['full_patch_sha256'],geometry_sha256=sha(dest/'sfincs.nc'),forcing_sha256=sha(dest/'roof_surface_precip.nc'))
  (dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');dump(dest/'udt_run_identity.json',ident)
  runs.append({'case_id':case,'run_id':dest.name,'folder':str(dest.relative_to(ROOT)),'source_folder':str(source.relative_to(ROOT)),'regular_reference_folder':old['folder'],'input_sha256':inputs,'source_input_sha256':ex['input_sha256'],'identity':ident,'image_id':image,'image_digest':image,'large_readonly_source':str(source/'sfincs_subgrid.nc'),'input_config_differences':diff,'original_parent_family':'CONTROL-LEGACY','split':'open_control','new_parent_event':False})
 dump(out/'run_manifest.json',{'runs':runs,'regular_reference_manifest':PRIOR+'/run_manifest.json','pending_58_scenarios_authorized':False,'training':False,'new_parent_count':0,'human_record':'MILI-D-20261015-015'});print('STAGED TWO ADAPTIVE CONTROLS',flush=True)
if __name__=='__main__':main()
