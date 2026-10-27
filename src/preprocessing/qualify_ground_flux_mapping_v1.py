#!/usr/bin/env python3
"""Use the exact existing runtime library to map crosssections;no time stepping."""
from pathlib import Path
import subprocess,json,hashlib,tarfile,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_ground_flux_replay_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
REQ=ROOT/'data/processed/three_ward_zone_rain_flux_readiness_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 manifest=json.loads((P/'run_manifest.json').read_text());image=manifest['image_id'];build=P/'probe_build';build.mkdir(exist_ok=False)
 archive=build/'runtime_modules.tar'
 with archive.open('wb') as out:subprocess.run(['docker','run','--rm','--network','none','--platform','linux/amd64','--entrypoint','tar',image,'-cf','-','-C','/opt/udt-build/modules','.'],stdout=out,check=True)
 modules=build/'modules';modules.mkdir()
 with tarfile.open(archive) as t:t.extractall(modules,filter='data')
 driver='''program face_mapping_probe
 use sfincs_input,only:read_sfincs_input
 use sfincs_domain,only:initialize_domain
 use sfincs_structures,only:read_structures
 use sfincs_crosssections,only:read_crs_file
 use sfincs_log,only:open_log
 use sfincs_data
 implicit none
 integer::i,ip,u
 call open_log()
 call read_sfincs_input()
 call initialize_domain()
 call read_structures()
 call read_crs_file()
 open(newunit=u,file='crosssection_native_mapping.csv',status='new')
 write(u,'(a)')'Section,Face Count,Native Face,From Native Cell,To Native Cell,Positive Native Sign,Runtime Closed'
 do i=1,nrcrosssections
  if(crs_nr(i)/=1)error stop 'Crosssection does not map to exactly one native face'
  ip=crs_uv_index(1,i)
  write(u,'(a,6(a,i0))')trim(namecrs(i)),',',crs_nr(i),',',ip-1,',',uv_index_z_nm(ip)-1, &
   ',',uv_index_z_nmu(ip)-1,',',crs_idir(1,i),',',merge(1,0,kcuv(ip)==0)
 enddo
 close(u)
 print *,'MAPPING_PROBE_COMPLETE',nrcrosssections
end program
'''
 (build/'face_mapping_probe.f90').write_text(driver)
 cmd=['docker','run','--rm','--network','none','--platform','linux/amd64','--mount',f'type=bind,source={build},target=/work','--entrypoint','gfortran','udt01b/sfincs-cpu:builder-v1','-fopenmp','-I/work/modules','/work/face_mapping_probe.f90','-L/usr/local/lib','-lsfincs','-Wl,-rpath,/usr/local/lib','-o','/work/face_mapping_probe']
 with (R/'probe_compile.log').open('w') as out:subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)
 req=pd.read_parquet(REQ/'ground_face_requests_preprocessed.parquet');checks=[]
 for run in manifest['runs']:
  folder=P/'mapping_probe'/run['case_id'];src=ROOT/run['folder'];name='udt01b-ground-mapping-'+run['stage'].lower()
  cmd=['docker','run','--rm','--name',name,'--network','none','--platform','linux/amd64','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={folder},target=/data','--mount',f'type=bind,source={src}/sfincs_subgrid.nc,target=/data/sfincs_subgrid.nc,readonly','--mount',f'type=bind,source={build}/face_mapping_probe,target=/probe,readonly','--entrypoint','/probe',image]
  with (R/(run['case_id']+'_mapping.log')).open('w') as out:subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=900)
  actual=pd.read_csv(folder/'crosssection_native_mapping.csv');expected=req[req.Case==run['case_id']]
  assert len(actual)==6763+len(expected) and actual['Face Count'].eq(1).all()
  g=actual.iloc[6763:].reset_index(drop=True);e=expected.reset_index(drop=True)
  assert np.array_equal(g.Section,e['Section Name']) and np.array_equal(g['Native Face'],e['Native Face'])
  assert np.array_equal(g['From Native Cell'],e['From Native Cell']) and np.array_equal(g['To Native Cell'],e['To Native Cell'])
  assert g['Positive Native Sign'].eq(1).all() and np.array_equal(g['Runtime Closed'].astype(bool),e['Configured Closed'])
  checks.append({'case':run['case_id'],'passed':True,'native_G_faces':len(g),'all_single_face_mappings':True,'endpoints_sign_and_closed_flags_exact':True,'physical_timesteps_executed':0});print('MAPPING_QUALIFIED',run['case_id'],len(g),flush=True)
 dump(R/'mapping_validation.json',{'passed':True,'existing_runtime_image':image,'probe_binary_sha256':sha(build/'face_mapping_probe'),'source_sha256':sha(build/'face_mapping_probe.f90'),'cases':checks,'no_time_stepping':True,'new_solver_image_created':False})
if __name__=='__main__':main()
