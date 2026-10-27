"""Regression-test repaired date-origin transport before two authorized control runs."""
from pathlib import Path
import sys,json,subprocess,shutil,csv
import numpy as np
from netCDF4 import Dataset,num2date
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,CASES,BUILD
from prepare_sfincs_cpu_build import sha,dump

def main(root=ROOT):
 root=Path(root);out=root/OUT;compat=out/'compatibility_v2';report=root/REPORT
 meta=json.loads((compat/'build-metadata.json').read_text());image=meta['containerimage.digest']
 inputs=compat/'test_inputs';outputs=compat/'test_outputs';inputs.mkdir(exist_ok=True);outputs.mkdir(exist_ok=True)
 old=root/'data/processed/three_ward_second_clock_budget_v1/transport_test_inputs'
 files=sorted(old.glob('*.nc'));assert len(files)==50
 for p in files:shutil.copyfile(p,inputs/p.name)
 t=np.array([0,3600,3601,7200,7201,14400.]);r=np.array([0,0,60,60,0,0.])
 units_list={'legacy_date_only':'minutes since 2020-01-01','full_datetime':'minutes since 2020-01-01 00:00:00','full_utc':'minutes since 2020-01-01 00:00:00.0 +0000','bad_date':'minutes since 2020-02-30','partial_datetime':'minutes since 2020-01-01 00','wrong_unit':'seconds since 2020-01-01','non_utc':'minutes since 2020-01-01 00:00:00 +0900'}
 for name,units in units_list.items():
  with Dataset(inputs/(name+'.nc'),'w')as d:
   for k,n in [('time',6),('x',3),('y',2)]:d.createDimension(k,n)
   v=d.createVariable('time','f8',('time',));v.units=units;v[:]=t/60
   d.createVariable('x','f8',('x',))[:]=[0,10,20];d.createVariable('y','f8',('y',))[:]=[0,10]
   d.createVariable('Precipitation','f4',('time','y','x'))[:]=np.broadcast_to(r[:,None,None],(6,2,3))
 names=[p.stem for p in files]+list(units_list)
 script=['set -eu','/opt/udt-build/netcdf_link_probe > /audit/nc_probe.txt','/opt/udt-build/loader_probe > /audit/loader.txt','cp /opt/udt-build/COMPAT_BINDING.json /audit/embedded_binding.json','cp /opt/udt-build/receipts/compat-* /audit/','ldd /opt/udt-build/production_rain_reader > /audit/reader_ldd.txt']
 for name in names:
  script+=['mkdir -p /audit/'+name,'set +e','(cd /audit/'+name+' && /opt/udt-build/production_rain_reader /inputs/'+name+'.nc >stdout.txt 2>stderr.txt)','status=$?','set -e','printf "%s\\n" "$status" > /audit/'+name+'/exit_code.txt']
 path=compat/'probe.sh';path.write_text('\n'.join(script)+'\n')
 command=['docker','run','--rm','--platform','linux/amd64','--network','none','--mount','type=bind,source='+str(inputs)+',target=/inputs,readonly','--mount','type=bind,source='+str(outputs)+',target=/audit','--mount','type=bind,source='+str(path)+',target=/probe.sh,readonly','--entrypoint','/bin/sh',image,'/probe.sh']
 subprocess.run(command,check=True)
 rows=[]
 for name in names:
  success=name not in ['bad_date','partial_datetime','wrong_unit','non_utc'];exitcode=int((outputs/name/'exit_code.txt').read_text());okay=(exitcode==0)==success
  if success and exitcode==0:
   with Dataset(inputs/(name+'.nc'))as d:raw=np.array(d['time'][:]);rawunits=d['time'].units
   actual=np.fromstring((outputs/name/'stdout.txt').read_text(),sep=' ');n=int(actual[0]);seconds=actual[3:3+n]
   okay &= np.array_equal(seconds,np.rint(raw*60))
  rows.append({'case':name,'expected_success':success,'exit_code':exitcode,'pass':bool(okay)})
 assert all(x['pass']for x in rows),rows
 assert sha(outputs/'embedded_binding.json')==sha(compat/'source_binding.json')
 record={'all_passed':True,'tests':rows,'production_reader_regressions':50,'origin_syntax_tests':7,'date_only_origin_confirmed_by_local_NetCDF_time_api':[str(x)for x in num2date([0,3601/60],units=units_list['legacy_date_only'])],'new_image_id':image,'new_patch_sha256':sha(compat/'sfincs_second_clock_budget_v2.patch'),'no_input_curve_changed':True,'hydrodynamic_steps_in_compatibility_tests':0}
 dump(report/'compatibility_v2_validation.json',record)
 # Preserve failed v1 initialization intact. No hydraulic step was written.
 manifest=json.loads((out/'run_manifest.json').read_text());failed=out/'runs'/manifest['runs'][0]['run_id'];archive=out/'attempts/v1_date_only_rejected'
 assert(failed/'execution.json').exists()and not(failed/'udt_step_budget.csv').exists()
 archive.parent.mkdir(exist_ok=True);shutil.move(failed,archive)
 failed.mkdir()
 for p in archive.iterdir():
  if p.name in manifest['runs'][0]['input_sha256']:
   if p.name=='sfincs_subgrid.nc':(failed/p.name).hardlink_to(p)
   else:shutil.copyfile(p,failed/p.name)
 for run in manifest['runs']:
  run['image_id']=image;run['image_digest']=image;identity=run['identity'];identity['solver_digest']=image;identity['patch_sha256']=record['new_patch_sha256']
  folder=root/run['folder'];(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in identity.items())+'\n');dump(folder/'udt_run_identity.json',identity)
 manifest['initialization_failure_retained']={'folder':str(archive.relative_to(root)),'hydrodynamic_steps':0,'reason':'legacy date-only units not accepted in v1','superseded_for_legacy_controls_by_v2':True}
 dump(out/'run_manifest.json',manifest)
 dump(compat/'active_solver_binding.json',record|{'base_build_binding':str(Path(BUILD)/'solver_image_binding.json'),'build_metadata_sha256':sha(compat/'build-metadata.json'),'source_binding_sha256':sha(compat/'source_binding.json')})
 print(json.dumps({'regressions':len(rows),'all_passed':True,'image':image,'failed_v1_initialization_preserved':True}))
if __name__=='__main__':main()
