"""Opt-in conservative volume transfers;baseline physics and native drains untouched."""
from pathlib import Path
import sys,json,shutil,difflib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import sha,dump
TAG='three_ward_barrier_drainage_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG;PRIOR=ROOT/'data/processed/three_ward_historical_radar_experiment_v2';IMAGE='sha256:5415c73fc6ac335bd225336d0b2055c0ea0345c2eced42ba9cef8e1c44ae1ace';NEWTAG='udt01b/sfincs-cpu:barrier-drainage-v1'
def main():
 c=P/'context';c.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=True)
 for f in(PRIOR/'context').iterdir():
  if f.is_file():shutil.copy2(f,c/f.name)
 src=(c/'sfincs_continuity.f90').read_text();before=src
 decl='''\n logical,private::pump_initialized=.false.,pump_enabled=.false.
 integer,private::pump_n=0,pump_log
 integer,allocatable,private::pump_from(:),pump_to(:)
 real*8,allocatable,private::pump_rate(:),pump_trigger(:),pump_head(:)
 real*8,allocatable::pump_delta(:),pump_out(:)
 real*8::pump_source_step=0.d0,pump_sink_step=0.d0
'''
 src=src.replace('contains\n',decl+'contains\n'+(ROOT/'src/preprocessing/barrier_drainage_v1/pump_routines.inc').read_text(),1)
 src=src.replace('udt_budget_finish,udt_budget_source_added','udt_budget_finish,udt_budget_source_added,udt_budget_measures',1)
 anchor="  invalid=0;donors=0;minfactor=1._real64";assert src.count(anchor)==1;src=src.replace(anchor,'  call udt_prepare_pumps(dt,actual_end)\n'+anchor)
 anchor='   available=z_volume(nm)+real(rain,real64)';assert src.count(anchor)==2
 # Restrict opt-in pumps to regular profile;adaptive guard remains unchanged.
 pos=src.index(' subroutine udt_limit_fluxes');front=src[:pos];tail=src[pos:].replace(anchor,anchor+'\n   if(pump_enabled)available=available-pump_out(nm)',1);src=front+tail
 src=src.replace('   call udt_budget_begin()','   call udt_budget_begin()\n   call udt_budget_measures(pump_source_step,pump_sink_step)',1)
 pos=src.index('   subroutine compute_water_levels_subgrid');front=src[:pos];tail=src[pos:];anchor='         z_volume(nm) = z_volume(nm) + dvol';assert tail.count(anchor)==1;tail=tail.replace(anchor,anchor+'\n         if(pump_enabled)z_volume(nm)=z_volume(nm)+pump_delta(nm)');src=front+tail
 (c/'sfincs_continuity.f90').write_text(src)
 b=(c/'udt_clock_budget.f90').read_text();bb=b;b=b.replace(' public :: udt_budget_initialize',' public :: udt_budget_measures\n real(real64),private::measure_source_total=0._real64,measure_sink_total=0._real64\n public :: udt_budget_initialize',1);b=b.replace(' subroutine udt_budget_cell', ''' subroutine udt_budget_measures(added,removed)
  real(real64),intent(in)::added,removed
  call require(in_step,'measure ledger outside continuity step')
  call require(ieee_is_finite(added).and.ieee_is_finite(removed).and.added>=0.and.removed>=0,'invalid measure volume')
  measure_source_total=measure_source_total+added
  measure_sink_total=measure_sink_total+removed
 end subroutine
 subroutine udt_budget_cell''',1)
 b=b.replace('residual=storage-initial_storage+out_total-in_total-rain_total','residual=storage-initial_storage+out_total-in_total-rain_total-measure_source_total+measure_sink_total');b=b.replace('storage,rain_total,out_total,in_total,0.0_real64,0.0_real64, &','storage,rain_total,out_total,in_total,measure_source_total,measure_sink_total, &')
 (c/'udt_clock_budget.f90').write_text(b)
 shutil.copy2(ROOT/'src/preprocessing/barrier_drainage_v1/pump_fixture.f90',c/'pump_fixture.f90')
 delta=''.join(difflib.unified_diff(before.splitlines(True),src.splitlines(True),fromfile='a/sfincs_continuity.f90',tofile='b/sfincs_continuity.f90'))+''.join(difflib.unified_diff(bb.splitlines(True),b.splitlines(True),fromfile='a/udt_clock_budget.f90',tofile='b/udt_clock_budget.f90'))
 (P/'measure_solver.patch').write_text(delta);binding=dict(prior_image=IMAGE,prior_solver_binding_sha256=sha(PRIOR/'solver_binding.json'),source_commit='1f1d8286520a8709b2e513854dff18a3616583db',measure_patch_sha256=sha(P/'measure_solver.patch'),source_sha256={f.name:sha(f)for f in c.glob('*.f90')},scope='Opt-in regular10m pumps;native qext/drn/infiltration/storage remain rejected. Available prior storage reserves both shared pump demand and outgoing faces;incoming transfers not pre-spent. Interior pairs both entered;prescribed-stage receiver external sink explicitly entered. No-op has exactly unchanged arithmetic.',human_record='MILI-D-20261018-016');dump(P/'source_binding.json',binding);dump(c/'MEASURE_BINDING.json',binding)
 docker=(c/'Dockerfile').read_text().replace('COPY HISTORICAL_RADAR_BINDING.json','COPY MEASURE_BINDING.json /opt/udt-build/MEASURE_BINDING.json\nCOPY HISTORICAL_RADAR_BINDING.json').replace('radar_budget_profile_fixture_v2.f90 /opt/udt-build/','radar_budget_profile_fixture_v2.f90 pump_fixture.f90 /opt/udt-build/');docker=docker.replace('\nFROM udt01b/sfincs-cpu:galibier-rain-step-integral-v1',' && gfortran -fopenmp -I. /opt/udt-build/pump_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/pump_fixture\nFROM '+IMAGE);docker=docker.replace('historical-radar-block-integral-v2','barrier-drainage-v1');(c/'Dockerfile').write_text(docker);print('MEASURE SOURCE PREPARED',binding['measure_patch_sha256'])
if __name__=='__main__':main()
