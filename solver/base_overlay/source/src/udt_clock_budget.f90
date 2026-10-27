! CPU/subgrid baseline diagnostics; state evolution and original output labels unchanged.
module udt_clock_budget
 use, intrinsic :: iso_fortran_env, only: real64,int64
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use sfincs_data
 implicit none
 private
 public :: udt_budget_initialize,udt_budget_begin,udt_budget_cell,udt_budget_finish
 public :: udt_budget_capture,udt_budget_close,udt_budget_capture_final,udt_budget_source_added
 real(real64),allocatable :: rain_term(:),horizontal_term(:),rain_increment(:)
 logical,allocatable :: ever_negative(:),cell_seen(:)
 real(real64) :: rain_total,out_total,in_total,horizontal_total,increment_total,initial_storage
 real(real64) :: live_time,live_dt,origin,minimum_volume,peak_start
 integer(int64) :: step
 integer :: ledger_unit,output_unit
 logical :: initialized=.false.,in_step=.false.
contains
 subroutine require(ok,message)
  logical,intent(in)::ok
  character(*),intent(in)::message
  if(.not.ok)then
   write(*,'(a)')'UDT diagnostic error: '//message
   error stop 1
  endif
 end subroutine
 subroutine udt_budget_initialize(actual_initial_time)
  real(real64),intent(in)::actual_initial_time
  integer::u,v,i,ios
  logical::exists
  character(1024)::line
  character(24),parameter::keys(10)=[character(24)::'schema_version','run_id','parent_id','family_id','split', &
   'forcing_sha256','geometry_sha256','source_commit','patch_sha256','solver_digest']
#ifdef _OPENACC
  error stop 'UDT diagnostics: GPU/OpenACC build not qualified; use CPU build'
#endif
  call require(.not.initialized,'cannot initialize twice')
  call require(subgrid.and..not.crsgeo.and..not.bathtub,'requires projected subgrid surface profile')
  call require(.not.use_qext.and.nsrcdrn==0.and..not.use_storage_volume,'unsupported source/storage process')
  call require(.not.infiltration.and.tspinup==0.0,'infiltration/spinup requires separate accounting')
  call require(.not.wavemaker.and..not.snapwave.and..not.nonhydrostatic,'wave/nonhydro profile unsupported')
  call require(all(kcs(1:np)/=4),'wave cells unsupported')
  call require(np>0.and.ieee_is_finite(actual_initial_time),'invalid initial state')
  call require(count(kcs(1:np)==1)>0,'no active interior cells')
  inquire(file='udt_run_identity.txt',exist=exists)
  call require(exists,'missing manifest-bound run identity')
  open(newunit=u,file='udt_run_identity.txt',status='old',action='read')
  open(newunit=v,file='udt_diagnostic_identity.txt',status='new',action='write')
  ! Ordered binding keys are validated by the staging/receipt validator; preserve exact bytes as text.
  do i=1,10
   read(u,'(a)',iostat=ios)line
   call require(ios==0.and.len_trim(line)>len_trim(keys(i))+1,'incomplete run identity')
   call require(index(line,trim(keys(i))//'=')==1,'wrong identity field order or name')
   if(i==1)call require(trim(line)=='schema_version=second-clock-budget-interface-v1','wrong schema binding')
   if(i==8)call require(trim(line)=='source_commit=1f1d8286520a8709b2e513854dff18a3616583db','wrong source binding')
   write(v,'(a)')trim(line)
  enddo
  close(u);close(v)
  allocate(rain_term(np),horizontal_term(np),rain_increment(np),ever_negative(np),cell_seen(np))
  ever_negative=.false.;rain_term=0;horizontal_term=0;rain_increment=0
  origin=actual_initial_time;live_time=origin;live_dt=0;step=0_int64;peak_start=origin
  rain_total=0;out_total=0;in_total=0;horizontal_total=0;increment_total=0
  call require(all(ieee_is_finite(z_volume(1:np))),'nonfinite initial storage')
  initial_storage=sum(real(z_volume(1:np),real64),mask=kcs(1:np)==1)
  minimum_volume=minval(real(z_volume(1:np),real64),mask=kcs(1:np)==1)
  ever_negative=(kcs(1:np)==1.and.z_volume(1:np)<-0.001_real64)
  open(newunit=ledger_unit,file='udt_step_budget.csv',status='new',action='write')
  write(ledger_unit,'(a)')'step_index,actual_state_time,actual_step_dt,signed_storage,rain_applied_cumulative,'// &
   'boundary_out_cumulative,boundary_in_cumulative,other_source_cumulative,other_sink_cumulative,'// &
   'minimum_native_volume_since_initialization,negative_native_count,negative_volume_sum,'// &
   'horizontal_applied_cumulative,rain_dvol_increment_cumulative,budget_residual,internal_flux_roundoff_residual,'// &
   'meteo_bracket_start,meteo_bracket_end,meteo_update_interval,meteo_bracket_observed'
  open(newunit=output_unit,file='udt_output_clock.csv',status='new',action='write')
  write(output_unit,'(a)')'step_index,nominal_output_time,actual_state_time,actual_step_dt,'// &
   'write_map,write_history,write_peak,write_restart,map_record,history_record,peak_record,actual_peak_start,actual_peak_end'
  initialized=.true.
  call write_ledger()
 end subroutine
 subroutine udt_budget_begin()
  integer::ib,ip,iref
  real::width,dv
  call require(initialized.and..not.in_step,'step begin sequence')
  rain_term=0;horizontal_term=0;rain_increment=0;cell_seen=.false.
  ! q is the same final flux field used by the immediately following continuity call.
  ! Split each native boundary face before reduction; coarse/fine physical widths are native.
  do ib=1,nkcuv2
   ip=index_kcuv2(ib);iref=uv_flags_iref(ip)
   if(uv_flags_dir(ip)==0)then
    width=dyrm(iref)
   else
    width=dxrm(iref)
   endif
   call require(width>0.and.ieee_is_finite(q(ip)),'nonfinite boundary flux or invalid width')
  enddo
  in_step=.true.
 end subroutine
 subroutine udt_budget_cell(nm,horizontal,area,dt)
  integer,intent(in)::nm
  real,intent(in)::horizontal,area,dt
  real::dv
  ! One independent slot per nm: safe in the original OpenMP cell loop.
  cell_seen(nm)=.true.
  horizontal_term(nm)=real(horizontal,real64)
  dv=0.0
  if(precip)dv=netprcp(nm)*area*dt
  rain_term(nm)=real(dv,real64)
  rain_increment(nm)=0.0_real64
 end subroutine
 subroutine udt_budget_source_added(nm,actual_dvol)
  integer,intent(in)::nm
  real,intent(in)::actual_dvol
  rain_increment(nm)=real(actual_dvol,real64)-horizontal_term(nm)
 end subroutine
 subroutine udt_budget_finish(actual_time,dt)
  real(real64),intent(in)::actual_time
  real,intent(in)::dt
  integer::ib,ip,iref
  real::width,dv
  call require(initialized.and.in_step,'step finish sequence')
  call require(ieee_is_finite(actual_time).and.dt>0,'invalid clock')
  call require(abs(actual_time-live_time-real(dt,real64))<1.0e-8_real64,'actual clock/dt disagreement')
  call require(all(ieee_is_finite(z_volume(1:np))).and.all(ieee_is_finite(rain_term)), &
   'nonfinite state or applied rainfall')
  call require(all(cell_seen.or.kcs(1:np)/=1),'missing active-cell continuity contribution')
  call require(all(rain_term>=0),'negative rain: source/sink profile extension required')
  if(precip)then
   call require(ieee_is_finite(meteo_t0).and.ieee_is_finite(meteo_t1).and.meteo_t1>meteo_t0,'invalid meteo bracket')
   call require(.not.ampr_block.and.dtwindupd==1.0,'requires piecewise linear rain and one-second meteo updates')
  endif
  do ib=1,nkcuv2
   ip=index_kcuv2(ib);iref=uv_flags_iref(ip)
   if(uv_flags_dir(ip)==0)then
    width=dyrm(iref)
   else
    width=dxrm(iref)
   endif
   dv=-real(ibuvdir(ib))*q(ip)*width*dt
   out_total=out_total+max(real(dv,real64),0.0_real64)
   in_total=in_total+max(-real(dv,real64),0.0_real64)
  enddo
  rain_total=rain_total+sum(rain_term)
  horizontal_total=horizontal_total+sum(horizontal_term)
  increment_total=increment_total+sum(rain_increment)
  live_time=actual_time;live_dt=real(dt,real64);step=step+1_int64
  minimum_volume=min(minimum_volume,minval(real(z_volume(1:np),real64),mask=kcs(1:np)==1))
  ever_negative=ever_negative.or.(kcs(1:np)==1.and.z_volume(1:np)<-0.001_real64)
  call write_ledger()
  in_step=.false.
 end subroutine
 subroutine write_ledger()
  real(real64)::storage,deficit,residual,values(19),bracket(4)
  integer::negative
  storage=sum(real(z_volume(1:np),real64),mask=kcs(1:np)==1)
  negative=count(kcs(1:np)==1.and.z_volume(1:np)<-0.001_real64)
  deficit=-sum(min(real(z_volume(1:np),real64),0.0_real64),mask=kcs(1:np)==1)
  residual=storage-initial_storage+out_total-in_total-rain_total
  bracket=[origin,origin,0.0_real64,0.0_real64]
  if(step>0.and.precip)bracket=[real(meteo_t0,real64),real(meteo_t1,real64),real(dtwindupd,real64),1.0_real64]
  values=[live_time,live_dt,storage,rain_total,out_total,in_total,0.0_real64,0.0_real64, &
   minimum_volume,real(negative,real64),deficit,horizontal_total,increment_total,residual, &
   horizontal_total+out_total-in_total,bracket]
  write(ledger_unit,'(i0,19(a,es25.17))')step,(',',values(negative),negative=1,19)
 end subroutine
 subroutine udt_budget_capture(nominal,actual,dt,nt,wm,wh,wp,wr,im,ih,ip)
  real(real64),intent(in)::nominal,actual
  real,intent(in)::dt
  integer(int64),intent(in)::nt
  logical,intent(in)::wm,wh,wp,wr
  integer,intent(in)::im,ih,ip
  real(real64)::window_start,window_end
  call require(initialized.and..not.in_step,'capture sequence')
  call require(nt==step.and.actual==live_time.and.real(dt,real64)==live_dt,'output is not the accounted state')
  window_start=peak_start;window_end=actual
  write(output_unit,'(i0,3(a,es25.17),4(a,l1),3(a,i0),2(a,es25.17))') &
   step,',',nominal,',',actual,',',live_dt,',',wm,',',wh,',',wp,',',wr,',',im,',',ih,',',ip,',',window_start,',',window_end
  if(wp)peak_start=actual
 end subroutine
 subroutine udt_budget_capture_final(nominal,peak_record)
  real(real64),intent(in)::nominal
  integer,intent(in)::peak_record
  call udt_budget_capture(nominal,live_time,real(live_dt),step,.false.,.false.,.true.,.false.,0,0,peak_record)
 end subroutine
 subroutine udt_budget_close()
  integer::u,nm
  call require(initialized.and..not.in_step,'close sequence')
  ! Completion receipt always uses the last accounted state, never an instability sentinel.
  open(newunit=u,file='udt_final_clock.csv',status='new',action='write')
  write(u,'(a)')'step_index,actual_initial_time,actual_final_time,actual_last_dt,solver_error'
  write(u,'(i0,3(a,es25.17),a,i0)')step,',',origin,',',live_time,',',live_dt,',',error
  close(u)
  open(newunit=u,file='udt_native_ever_negative.csv',status='new',action='write')
  write(u,'(a)')'native_cell_index,ever_volume_below_minus_0_001_m3'
  do nm=1,np
   write(u,'(i0,a,l1)')nm,',',ever_negative(nm)
  enddo
  close(u);close(ledger_unit);close(output_unit)
  deallocate(rain_term,horizontal_term,rain_increment,ever_negative,cell_seen)
  initialized=.false.
 end subroutine
end module
