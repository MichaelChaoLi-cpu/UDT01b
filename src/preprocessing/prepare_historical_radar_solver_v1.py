"""Versioned exact block-intensity integration, preserving existing linear branch."""
from pathlib import Path
import json,shutil,hashlib,difflib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/three_ward_historical_radar_experiment_v1'
OLD=ROOT/'data/processed/three_ward_rain_step_integral_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
 c=OUT/'context';c.mkdir(parents=True,exist_ok=False)
 for p in (OLD/'context').iterdir():
  if p.is_file():shutil.copyfile(p,c/p.name)
 original=(c/'sfincs_meteo.f90').read_text()
 block='''! Exact duration weights for START-stamped, right-held representative intensity.
! Record k applies over [knots(k),knots(k+1)); point-boundary convention has zero volume.
 subroutine udt_rain_block_weights(knots,finish,step,weights)
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  real,intent(in)::knots(:),step
  real(real64),intent(in)::finish
  real(real64),intent(out)::weights(size(knots))
  real(real64)::start,a,b
  integer::k,n
  n=size(knots)
  if(n<2.or.step<=0.or..not.ieee_is_finite(step).or..not.ieee_is_finite(finish)) &
   error stop 'Block rain integral: invalid clock'
  if(any(.not.ieee_is_finite(knots)))error stop 'Block rain integral: nonfinite knots'
  if(any(knots(2:)<=knots(:n-1)))error stop 'Block rain integral: unordered knots'
  start=finish-real(step,real64);weights=0
  weights(1)=max(0._real64,min(finish,real(knots(1),real64))-start)
  weights(n)=max(0._real64,finish-max(start,real(knots(n),real64)))
  do k=1,n-1
   a=max(start,real(knots(k),real64));b=min(finish,real(knots(k+1),real64))
   weights(k)=weights(k)+max(0._real64,b-a)
  enddo
  if(abs(sum(weights)-real(step,real64))>1.e-10_real64*max(1._real64,real(step,real64))) &
   error stop 'Block rain integral: incomplete step'
 end subroutine
'''
 modified=original.replace(' subroutine udt_prepare_rain_integral(finish,step)',block+' subroutine udt_prepare_rain_integral(finish,step)',1)
 modified=modified.replace('if(ampr_block.or.crsgeo.or..not.subgrid.or.','if(crsgeo.or..not.subgrid.or.',1)
 modified=modified.replace("   write(k,'(a)')'Original nonnegative linear AMPR, projected CPU subgrid, no spinup or alternate rain.'", "   if(ampr_block)then\n    write(k,'(a)')'Nonnegative START-stamped block AMPR; exact step overlap; projected CPU subgrid.'\n   else\n    write(k,'(a)')'Original nonnegative linear AMPR, projected CPU subgrid, no spinup or alternate rain.'\n   endif",1)
 oldcall='  call udt_rain_time_weights(ampr_times,finish,step,udt_rain_weights)'
 newcall='  if(ampr_block)then\n   call udt_rain_block_weights(ampr_times,finish,step,udt_rain_weights)\n  else\n'+oldcall+'\n  endif'
 assert modified.count(oldcall)==1;modified=modified.replace(oldcall,newcall,1)
 assert modified!=original and 'ampr_block.or.crsgeo' not in modified
 (c/'sfincs_meteo.f90').write_text(modified)
 patch=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/source/src/sfincs_meteo.f90',tofile='b/source/src/sfincs_meteo.f90'))
 (OUT/'block_rain_increment.patch').write_text(patch)
 (OUT/'full_solver.patch').write_bytes((OLD/'full_solver.patch').read_bytes()+patch.encode())
 shutil.copyfile(ROOT/'src/preprocessing/radar_block_rain_fixture_v1.f90',c/'radar_block_rain_fixture_v1.f90')
 shutil.copyfile(ROOT/'src/preprocessing/radar_reader_fixture_v1.f90',c/'radar_reader_fixture_v1.f90')
 docker=(OLD/'context/Dockerfile').read_text().replace('rain_integral_fixture.f90 /opt/udt-build/','rain_integral_fixture.f90 radar_block_rain_fixture_v1.f90 radar_reader_fixture_v1.f90 /opt/udt-build/')
 compile_line=' && gfortran -fopenmp -I. /opt/udt-build/{name}.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/{name}'
 docker=docker.replace('\nFROM udt01b/sfincs-cpu:galibier-adaptive-positivity-v2',compile_line.format(name='radar_block_rain_fixture_v1')+compile_line.format(name='radar_reader_fixture_v1')+'\nFROM udt01b/sfincs-cpu:galibier-rain-step-integral-v1')
 docker=docker.replace('rain-step-integral-v1"','historical-radar-block-integral-v1"')
 docker=docker.replace('COPY RAIN_INTEGRAL_BINDING.json /opt/udt-build/RAIN_INTEGRAL_BINDING.json','COPY HISTORICAL_RADAR_BINDING.json /opt/udt-build/HISTORICAL_RADAR_BINDING.json')
 (c/'Dockerfile').write_text(docker)
 binding=dict(prior_solver=json.loads((OLD/'solver_binding.json').read_text()),source_commit=json.loads((OLD/'source_binding.json').read_text())['source_commit'],
  patch_sha256=sha(OUT/'full_solver.patch'),increment_patch_sha256=sha(OUT/'block_rain_increment.patch'),
  context_sha256={p.name:sha(p) for p in c.iterdir() if p.is_file()},
  preserved_modules=['sfincs_boundaries.f90','sfincs_continuity.f90','udt_clock_budget.f90','udt_rain_clock.f90'],
  interpolation='amprblock=1 exact start-stamped holds; amprblock=0 original linear branch untouched',
  scope='Two historical-rainfall 9-hour scenarios only; no formal training/test library release.')
 for name in binding['preserved_modules']:assert sha(c/name)==sha(OLD/'context'/name)
 dump(OUT/'source_binding.json',binding);dump(c/'HISTORICAL_RADAR_BINDING.json',binding)
 print(json.dumps(binding,indent=2))

if __name__=='__main__':main()
