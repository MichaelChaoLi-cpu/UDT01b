"""Execute dry/rain terrain-only controls, reusing certified identical urban controls.
Preserves frozen inputs and historical results; no retry, training or sealed access.
"""
from pathlib import Path
import sys,json,shutil,time,subprocess,argparse,hashlib
import numpy as np
from netCDF4 import Dataset,num2date
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
from load_three_ward_minimum_scenarios import load_forcing
PREP=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1';P=ROOT/'data/processed/three_ward_urban_geometry_pair_controls_v1';R=ROOT/'data/exp/estimation/three_ward_urban_geometry_pair_controls_v1'
CASES=['dry_raised','rain60_raised'];BUILDER='/private/tmp/udt01b-sfincs-builder/bin/python'
def prepare():
 P.mkdir(exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 if (P/'execution_manifest.json').exists():return json.loads((P/'execution_manifest.json').read_text())
 assert json.loads((ROOT/'data/exp/data-preprocessing/three_ward_urban_geometry_pair_inputs_v1/validation.json').read_text())['all_passed']
 source=json.loads((PREP/'prepared_run_manifest.json').read_text());lookup={(x['case_id'],x['arm']):x for x in source['runs']};old=json.loads((ROOT/'data/processed/three_ward_reference_setup_pilot_v1/qualification_manifest.json').read_text());old={x['case_id']:x for x in old['runs']};reuse=[];runs=[];protected={};hashcache={}
 def hashfile(p):
  st=p.stat();k=(st.st_dev,st.st_ino)
  if k not in hashcache:hashcache[k]=sha(p)
  return hashcache[k]
 for case in CASES:
  target=lookup[case,'U'];prior=old[case];folder=ROOT/prior['folder'];new=ROOT/target['folder'];auditpath=ROOT/'data/exp/data-preprocessing/three_ward_reference_setup_pilot_v1/qualification'/case/'audit.json';a=json.loads(auditpath.read_text());ex=json.loads((folder/'execution.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'] and ex['solver_image_id']==target['image_id'] and ex['returncode']==0 and ex['completion_marker']
  identity=json.loads((folder/'udt_run_identity.json').read_text());assert identity['forcing_sha256']==target['physical_forcing_sha256'] and identity['solver_digest']==target['image_id']
  checked={}
  for name,h in target['input_sha256'].items():
   if name in ['udt_run_identity.txt','roof_surface_precip.nc','sfincs.bzs']:continue
   assert hashfile(folder/name)==h,(case,name,'old/new geometry or configuration differs');checked[name]=h
  with Dataset(folder/'roof_surface_precip.nc') as x,Dataset(new/'roof_surface_precip.nc') as y:
   assert all(np.array_equal(np.asarray(x[k][:]),np.asarray(y[k][:])) for k in ['x','y']);assert num2date(0,x['time'].units)==num2date(0,y['time'].units) and x['Precipitation'].getncattr('unit')==y['Precipitation'].getncattr('unit')=='mm.hr-1'
   xt=np.asarray(x['time'][:]);yt=np.asarray(y['time'][:]);xf=np.asarray(x['Precipitation'][:]);yf=np.asarray(y['Precipitation'][:]);union=np.union1d(xt,yt)
   for t in union:
    def interp(tt,ff):
     j=int(np.clip(np.searchsorted(tt,t,side='right')-1,0,len(tt)-2));f=(t-tt[j])/(tt[j+1]-tt[j]);return ff[j].astype('f8')*(1-f)+ff[j+1].astype('f8')*f
    assert np.array_equal(interp(xt,xf),interp(yt,yf)), 'Legacy physical rain function must be unchanged'
   legacy_raw_time=xt;legacy_units=x['time'].units
  stage_a=np.loadtxt(folder/'sfincs.bzs');stage_b=np.loadtxt(new/'sfincs.bzs');knots=np.union1d(stage_a[:,0],stage_b[:,0]);stage_difference=max(float(np.max(abs(np.interp(knots,stage_a[:,0],stage_a[:,j])-np.interp(knots,stage_b[:,0],stage_b[:,j])))) for j in range(1,stage_a.shape[1]));assert stage_difference<=8*np.finfo('f4').eps*max(1.,float(abs(stage_b[:,1:]).max()))
  with Dataset(folder/'sfincs_map.nc') as d:assert len(d['time'])==49 and len(d['timemax'])==1 and np.all(np.asarray(d['status'][:])==0)
  # Original diagnostic identity remains bound to original native geometry;
  # different bundle identity hashes do not alter otherwise identical physics.
  assert (folder/'udt_diagnostic_identity.txt').read_bytes()==(folder/'udt_run_identity.txt').read_bytes()
  reuse.append({'case_id':case,'arm':'U','folder':prior['folder'],'audit_file':str(auditpath.relative_to(ROOT)),'audit_sha256':sha(auditpath),'execution_sha256':sha(folder/'execution.json'),'solver_image_id':target['image_id'],'original_identity':identity,'physics_input_sha256':checked,'rain_piecewise_linear_functions_equal':True,'stage_representation_difference_to_preparation_m':stage_difference,'paired_execution_uses_original_U_stage_bytes':True,'different_metadata_bundle_identity_only':True,'numerical_gates_passed':True,'new_run':False,'labels_file':'data/processed/three_ward_reference_setup_pilot_v1/qualification/'+case+'_candidate_reference.npz'})
  protected[str(auditpath.relative_to(ROOT))]=sha(auditpath);protected[str((folder/'execution.json').relative_to(ROOT))]=sha(folder/'execution.json')
  x=lookup[case,'T'];src=ROOT/x['folder'];dest=P/'runs'/src.name;dest.mkdir(parents=True,exist_ok=True)
  for name,h in x['input_sha256'].items():
   assert hashfile(src/name)==h
   if not (dest/name).exists():
    if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(src/name)
    else:shutil.copyfile(src/name,dest/name)
  ident=src/'udt_run_identity.json';shutil.copyfile(ident,dest/ident.name)
  # Pair the actual historical control representation exactly, in this new
  # execution overlay only. No change to its physical rain function or parents.
  shutil.copyfile(folder/'sfincs.bzs',dest/'sfincs.bzs')
  rainfile=dest/'terrain_surface_precip.nc';rainfile.unlink()
  support=np.load(PREP/'paired_support_and_native_mapping.npz');weight=np.flipud(np.pad(support['Terrain Rainfall Weight'],1));physical=load_forcing(case);rates=np.interp(legacy_raw_time*60,physical['Event Time'],physical['Physical Rainfall Rate'])
  with Dataset(src/'terrain_surface_precip.nc') as template,Dataset(rainfile,'w') as d:
   for k,dim in template.dimensions.items():d.createDimension(k,len(legacy_raw_time) if k=='time' else len(dim))
   for k in template.ncattrs():d.setncattr(k,template.getncattr(k))
   for key,v in template.variables.items():
    vv=d.createVariable(key,v.dtype,v.dimensions,fill_value=v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None,zlib=True if key=='Precipitation' else False)
    for attr in v.ncattrs():
     if attr!='_FillValue':vv.setncattr(attr,v.getncattr(attr))
    if key=='time':vv[:]=legacy_raw_time
    elif key=='Precipitation':
     for j,rate in enumerate(rates):vv[j]=(weight*rate).astype('f4')
    else:vv[:]=v[:]
  executed_hashes={k:sha(dest/k) for k in x['input_sha256']}
  runs.append({**x,'input_sha256':executed_hashes,'folder':str(dest.relative_to(ROOT)),'prepared_source_folder':x['folder'],'identity':json.loads(ident.read_text()),'large_readonly_source':str(src/'sfincs_subgrid.nc'),'execution_forcing_adapter':'Use original U knots and exact U stage bytes; keep physical rainfall function, spatial terrain weight and parent identity. Preparation remains frozen.','stage_max_difference_from_prepared_representation_m':stage_difference})
 for rel in ['src/preprocessing/reference_label_policy.py','data/processed/three_ward_urban_geometry_pair_inputs_v1/common_evaluation_contract.json','data/processed/three_ward_urban_geometry_pair_inputs_v1/prepared_run_manifest.json','data/processed/three_ward_minimum_scenarios_v1/catalog.json','data/processed/three_ward_minimum_scenarios_v1/run_manifest.json','data/exp/data-preprocessing/three_ward_population_block_stability_v1/delivery_manifest.json']:
  protected[rel]=sha(ROOT/rel)
 result={'runs':runs,'reused_U':reuse,'new_run_cap':2,'execution_order':CASES,'audit_between_runs_required':True,'physical_horizon_seconds':14400,'authorization':'User requested continuation of prepared paired dry then rain comparison.','training':False,'sealed_test':False,'additional_parent_events':0,'original_inputs_and_results_preserved':True};dump(P/'execution_manifest.json',result);dump(P/'protected_sources.json',protected);dump(R/'urban_reuse_audit.json',{'all_passed':True,'reused_controls':reuse,'new_U_runs':0});return result

def execute(case):
 m=prepare();r=next(x for x in m['runs'] if x['case_id']==case);folder=ROOT/r['folder']
 if case=='rain60_raised':assert json.loads((R/'dry_raised/T/audit.json').read_text())['all_checks_passed'],'Do not start rain if dry control fails original gates'
 if (folder/'execution.json').exists():
  ex=json.loads((folder/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'];print('RETAIN completed',case,flush=True);return
 assert not any((folder/n).exists() for n in ['launch.json','sfincs.log','sfincs_map.nc']),'Preserve incomplete attempt; no automatic retry'
 assert all(sha(folder/n)==h for n,h in r['input_sha256'].items())
 assert subprocess.run(['docker','image','inspect',r['image_id'],'--format','{{.Id}}'],capture_output=True,text=True,check=True).stdout.strip()==r['image_id']
 cmd=['docker','run','--rm','--name','udt01b-geometry-T-'+case.replace('_','-'),'--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={folder},target=/data','--mount','type=bind,source='+r['large_readonly_source']+',target=/data/sfincs_subgrid.nc,readonly','--entrypoint','/usr/local/bin/sfincs',r['image_id']]
 dump(folder/'launch.json',{'command':cmd,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Two T controls only; reused U baseline; stop if original gates fail.'});print('START',case,flush=True);start=time.monotonic()
 with (folder/'solver_stdout.txt').open('w') as so,(folder/'solver_stderr.txt').open('w') as se:ex=subprocess.run(cmd,stdout=so,stderr=se)
 log=(folder/'sfincs.log').read_text(errors='replace') if (folder/'sfincs.log').exists() else '';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS'])
 dump(folder/'execution.json',{'returncode':ex.returncode,'completion_marker':done,'runtime_seconds':time.monotonic()-start,'solver_image_id':r['image_id'],'input_sha256':r['input_sha256'],'command':cmd,'new_parent':False,'training':False});print('FINISH',case,ex.returncode,done,flush=True);assert ex.returncode==0 and done
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','dry_raised','rain60_raised']);a=ap.parse_args();prepare() if a.action=='prepare' else execute(a.action)
