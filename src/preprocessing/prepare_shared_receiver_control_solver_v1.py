"""Isolated opt-in allocator build; old solver and source context remain untouched."""
from pathlib import Path
import sys,json,shutil,subprocess,difflib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from shared_receiver_pilot_common_v1 import sha,dump
P=ROOT/'data/processed/three_ward_shared_receiver_controls_v1';R=ROOT/'data/exp/estimation'/P.name
OLD=ROOT/'data/processed/three_ward_barrier_drainage_v1';NEW=ROOT/'src/preprocessing/shared_receiver_controls_v1'
def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);c=P/'context';c.mkdir(exist_ok=True)
 for f in (OLD/'context').iterdir():
  if f.is_file():shutil.copy2(f,c/f.name)
 p=c/'sfincs_continuity.f90';old=p.read_text();s=old
 decl='''\n logical,private::pool_dynamic=.false.
 integer,private::pool_n=0
 integer,allocatable,private::pump_pool(:)
 real*8,allocatable,private::pool_cap(:),pump_unit(:)
'''
 s=s.replace('contains\n',decl+'contains\n'+(NEW/'weighted_pool.inc').read_text(),1)
 anchor='    close(u)\n    open(newunit=pump_log';assert s.count(anchor)==1;s=s.replace(anchor,'    close(u)\n    call udt_initialize_pools()\n    open(newunit=pump_log',1)
 header="     'requested_m3,transferred_m3,source_prior_volume_m3,head_m,depth_m,limited_by_storage,blocked_by_trigger_or_head'"
 assert s.count(header)==1
 s=s.replace("    write(pump_log,'(a)')'actual_end_time_s", "    if(pool_dynamic)then\n     write(pump_log,'(a)')'actual_end_time_s",1)
 s=s.replace(header,header+"// &\n      ',nominal_reserved_m3,standalone_limit_m3,receiving_pool_id,pool_limit_m3'\n    else\n     write(pump_log,'(a)')'actual_end_time_s,dt_s,pump_id,source_native_cell,receiver_native_cell,receiver_kcs,'// &\n"+header+"\n    endif",1)
 anchor='  pump_source_step=0._real64;pump_sink_step=0._real64\n';assert s.count(anchor)==1
 s=s.replace(anchor,'  if(pool_dynamic)then\n   call udt_prepare_dynamic_pumps(dt,actual_end)\n   return\n  endif\n'+anchor,1)
 p.write_text(s);(P/'allocator.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='prior/sfincs_continuity.f90',tofile='controls/sfincs_continuity.f90')))
 shutil.copy2(NEW/'weighted_pool_fixture.f90',c/'weighted_pool_fixture.f90')
 prior=json.loads((OLD/'solver_binding.json').read_text())['image_id']
 subprocess.run(['docker','image','tag',prior,'udt01b/sfincs-cpu:controls-prior-pinned'],check=True)
 assert subprocess.check_output(['docker','image','inspect','udt01b/sfincs-cpu:controls-prior-pinned','--format','{{.Id}}'],text=True).strip()==prior
 docker=(c/'Dockerfile').read_text();docker=docker.replace('pump_fixture.f90 /opt/udt-build/','pump_fixture.f90 weighted_pool_fixture.f90 /opt/udt-build/')
 docker=docker.replace('\nFROM udt01b/sfincs-cpu:galibier-historical-radar-block-v2',' && gfortran -fopenmp -I. /opt/udt-build/weighted_pool_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/weighted_pool_fixture\nFROM '+'udt01b/sfincs-cpu:controls-prior-pinned')
 docker=docker.replace('make -j1 > /opt/udt-build/receipts/positivity-make.log 2>&1', '(make -j1 > /opt/udt-build/receipts/positivity-make.log 2>&1 || (tail -100 /opt/udt-build/receipts/positivity-make.log;exit 1))')
 docker=docker.replace('barrier-drainage-v1','shared-receiver-controls-v1');(c/'Dockerfile').write_text(docker)
 binding={'prior_image':json.loads((OLD/'solver_binding.json').read_text())['image_id'],'allocator_patch_sha256':sha(P/'allocator.patch'),'source_sha256':{f.name:sha(f) for f in c.glob('*.f90')},'unchanged_core_sources':['sfincs_boundaries.f90','sfincs_meteo.f90','udt_clock_budget.f90','udt_rain_clock.f90'],'decision':'MILI-D-20261021-001'}
 for f in binding['unchanged_core_sources']:assert sha(c/f)==sha(OLD/'context'/f)
 dump(P/'source_binding.json',binding)
 with (R/'docker_build.log').open('w') as log:subprocess.run(['docker','build','--platform','linux/amd64','--network','none','-t','udt01b/sfincs-cpu:shared-receiver-controls-v1',str(c)],stdout=log,stderr=subprocess.STDOUT,check=True)
 image=subprocess.check_output(['docker','image','inspect','udt01b/sfincs-cpu:shared-receiver-controls-v1','--format','{{.Id}}'],text=True).strip()
 dump(P/'solver_binding.json',{'image_id':image,'patch_sha256':binding['allocator_patch_sha256'],'prior_image_id':binding['prior_image'],'unit_qualified':False,'hydraulic_prefix_qualified':False})
 print('CONTROL_SOLVER_BUILT',image,flush=True)
if __name__=='__main__':main()
