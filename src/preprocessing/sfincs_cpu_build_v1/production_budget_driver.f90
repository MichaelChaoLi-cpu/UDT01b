! Interface bookkeeping fixture. Never calls continuity, fluxes, BMI or solver update.
program budget_fixture
 use sfincs_data
 use udt_clock_budget
 use iso_fortran_env,only:real64,int64
 implicit none
 character(100)::mode
 real(real64)::clock
 real::dt,fixture_area(2),horizontal(2),dvol(2)
 integer::i,n
 call get_command_argument(1,mode)
 subgrid=.true.;crsgeo=.false.;bathtub=.false.;use_qext=.false.;nsrcdrn=0
 use_storage_volume=.false.;infiltration=.false.;tspinup=0.0
 wavemaker=.false.;snapwave=.false.;nonhydrostatic=.false.;use_quadtree=.true.
 error=0;precip=.true.;np=2;nkcuv2=2
 meteo_t0=0;meteo_t1=1;dtwindupd=1;ampr_block=.false.
 allocate(kcs(2),z_volume(2),netprcp(2),q(2),uv_flags_iref(2),uv_flags_dir(2),dxrm(2),dyrm(2),index_kcuv2(2),ibuvdir(2))
 kcs=1;z_volume=[100.0,100.0];netprcp=[0.0001,0.0002]
 uv_flags_iref=[1,2];uv_flags_dir=[0,1];dxrm=[10.0,5.0];dyrm=[10.0,5.0]
 index_kcuv2=[1,2];ibuvdir=[1,-1];q=[0.1,0.2];fixture_area=[100.0,25.0]
 if(trim(mode)=='unsupported')use_qext=.true.
 if(trim(mode)=='infiltration')infiltration=.true.
 if(trim(mode)=='spinup')tspinup=1.0
 if(trim(mode)=='storage')use_storage_volume=.true.
 if(trim(mode)=='waves')wavemaker=.true.
 if(trim(mode)=='block')ampr_block=.true.
 if(trim(mode)=='dry')then
  netprcp=0;precip=.false.
 endif
 clock=0
 call udt_budget_initialize(clock)
 do n=1,4
  dt=real(n)*0.1
  if(n==3)q=-q ! Explicit gross return flow: no signed cancellation before reduction.
  call udt_budget_begin()
  horizontal=[q(1)*dyrm(1)*dt,-q(2)*dxrm(2)*dt]
  !$omp parallel do private(i)
  do i=1,2
   if(.not.(trim(mode)=='missing_cell'.and.i==2))call udt_budget_cell(i,horizontal(i),fixture_area(i),dt)
   dvol(i)=horizontal(i)
   if(precip)then
    dvol(i)=dvol(i)+netprcp(i)*fixture_area(i)*dt
    call udt_budget_source_added(i,dvol(i))
   endif
  enddo
  !$omp end parallel do
  ! Preset bookkeeping values, no hydraulic calculation. Retain an explicit negative fixture.
  z_volume=z_volume+dvol
  if(trim(mode)=='negative'.and.n==2)z_volume(2)=-0.25
  if(trim(mode)=='nonfinite'.and.n==2)netprcp(1)=sqrt(-abs(netprcp(1)))
  clock=clock+real(dt,real64)
  if(trim(mode)=='clock_mismatch'.and.n==2)clock=clock+1.0_real64
  call udt_budget_finish(clock,dt)
  call udt_budget_capture(real(n,real64)*0.1_real64,clock,dt,int(n,int64),.true.,.true.,n==2,.false.,n,n,n/2)
 enddo
 call udt_budget_capture_final(1.0_real64,3)
 call udt_budget_close()
end program
