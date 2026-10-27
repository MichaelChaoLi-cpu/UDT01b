! Prescribed bookkeeping arrays only. No solver initialization, momentum or continuity calls.
program negative_trace_fixture
 use sfincs_data
 use udt_clock_budget
 use iso_fortran_env,only:real64,int64
 implicit none
 character(100)::mode
 integer::i,n,unit
 integer,allocatable::fixture_mask(:)
 real::dt,horizontal(3),dvol(3),delta(4)
 real(real64)::clock
 call get_command_argument(1,mode)
 subgrid=.true.;crsgeo=.false.;bathtub=.false.;use_qext=.false.;nsrcdrn=0
 use_storage_volume=.false.;infiltration=.false.;tspinup=0.0
 wavemaker=.false.;snapwave=.false.;nonhydrostatic=.false.;use_quadtree=.false.
 if(trim(mode)=='whole_metadata')then
  nref=1;ncuv=0;nkcuv2=0;dx=10.;dy=10.;error=0;precip=.false.
  open(newunit=unit,file='whole_native_topology.bin',access='stream',form='unformatted',status='old')
  read(unit)np,npuv
  allocate(kcs(np),z_volume(np),netprcp(np),q(npuv+1),fixture_mask(np))
  allocate(z_index_z_n(np),z_index_z_m(np),z_index_uv_md(np),z_index_uv_mu(np))
  allocate(z_index_uv_nd(np),z_index_uv_nu(np),uv_index_z_nm(npuv),uv_index_z_nmu(npuv))
  allocate(uv_flags_dir(npuv),subgrid_uv_zmin(npuv))
  read(unit)z_index_z_n,z_index_z_m,fixture_mask
  kcs=fixture_mask
  read(unit)z_index_uv_md,z_index_uv_mu,z_index_uv_nd,z_index_uv_nu
  read(unit)uv_index_z_nm,uv_index_z_nmu,uv_flags_dir,subgrid_uv_zmin
  close(unit)
  z_volume=0;netprcp=0;q=0;clock=0
  call udt_budget_initialize(clock)
  call udt_budget_close()
  stop
 endif
 nref=1;np=6;npuv=5;ncuv=0;nkcuv2=3;dx=10.;dy=10.;error=0;precip=.true.
 meteo_t0=0;meteo_t1=1;dtwindupd=1;ampr_block=.false.
 allocate(kcs(np),z_volume(np),netprcp(np),q(npuv+1))
 allocate(z_index_z_n(np),z_index_z_m(np),z_index_uv_md(np),z_index_uv_mu(np))
 allocate(z_index_uv_nd(np),z_index_uv_nu(np),uv_index_z_nm(npuv),uv_index_z_nmu(npuv))
 allocate(uv_flags_iref(npuv),uv_flags_dir(npuv),dxrm(1),dyrm(1),index_kcuv2(3),ibuvdir(3))
 allocate(subgrid_uv_zmin(npuv))
 kcs=[1,1,1,2,2,2];z_volume=[.1_real64,.01_real64,.2_real64,0._real64,0._real64,0._real64]
 netprcp=[.001,.0,.002,.0,.0,.0]
 z_index_z_n=[2,2,3,2,1,4];z_index_z_m=[2,3,2,1,2,2]
 z_index_uv_md=[1,2,0,0,0,0];z_index_uv_mu=[2,0,0,1,0,0]
 z_index_uv_nd=[3,0,4,0,0,5];z_index_uv_nu=[4,0,5,0,3,0]
 uv_index_z_nm=[4,1,5,1,3];uv_index_z_nmu=[1,2,1,3,6]
 uv_flags_iref=1;uv_flags_dir=[0,0,1,1,1];dxrm=10.;dyrm=10.
 index_kcuv2=[1,3,5];ibuvdir=[1,1,-1]
 subgrid_uv_zmin=[.1,.2,.3,.4,.5]
 where(z_index_uv_md==0)z_index_uv_md=npuv+1
 where(z_index_uv_mu==0)z_index_uv_mu=npuv+1
 where(z_index_uv_nd==0)z_index_uv_nd=npuv+1
 where(z_index_uv_nu==0)z_index_uv_nu=npuv+1
 delta=[.125,.375,.2,.3];q=0
 if(trim(mode)=='adaptive')use_quadtree=.true.
 if(trim(mode)=='dummy_nonzero')q(npuv+1)=.1
 if(trim(mode)=='raw_zero_pointer')z_index_uv_md(3)=0
 if(trim(mode)=='initial_negative')z_volume(2)=-.2_real64
 if(trim(mode)=='threshold_equal')then
  z_volume(1)=-.001_real64;netprcp=0
 endif
 clock=0
 call udt_budget_initialize(clock)
 do n=1,4
  dt=delta(n)
  select case(n)
  case(1)
   q=[.01,.2,-.02,.005,0.,0.]
  case(2)
   q=[.1,.01,0.,0.,.2,0.]
  case(3)
   q=[0.,.3,0.,0.,-.5,0.]
  case(4)
   q=0
  end select
  if(trim(mode)=='threshold_equal')q=0
  call udt_budget_begin()
  !$omp parallel do private(i)
  do i=1,3
   horizontal(i)=((q(z_index_uv_md(i))-q(z_index_uv_mu(i)))*dy+ &
    (q(z_index_uv_nd(i))-q(z_index_uv_nu(i)))*dx)*dt
   if(.not.(trim(mode)=='missing_cell'.and.i==2))call udt_budget_cell(i,horizontal(i),100.,dt)
   dvol(i)=horizontal(i)+netprcp(i)*100.*dt
   call udt_budget_source_added(i,dvol(i))
   z_volume(i)=z_volume(i)+real(dvol(i),real64)
  enddo
  !$omp end parallel do
  clock=clock+real(dt,real64)
  if(trim(mode)=='clock_mismatch'.and.n==2)clock=clock+1._real64
  call udt_budget_finish(clock,dt)
  call udt_budget_capture(real(n,real64)*.2_real64,clock,dt,int(n,int64),.true.,.true.,.false.,.false.,n,n,0)
 enddo
 call udt_budget_capture_final(clock,1)
 call udt_budget_close()
end program
