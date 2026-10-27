! Prescribed-array production limiter regression; no full solver integration.
program positivity_fixture
 use sfincs_data
 use sfincs_continuity,only:udt_limit_fluxes
 use iso_fortran_env,only:real64
 use ieee_arithmetic,only:ieee_value,ieee_quiet_nan
 implicit none
 character(64)::mode
 integer::i,j,n,seedsize
 integer,allocatable::seed(:)
 real::dt,dvol,horizontal,source,oldq(6),r(8),width(5)
 real(real64)::clock,before(3),after(3),initial(6),boundary,residual,scale,minimum
 call get_command_argument(1,mode)
 np=6;npuv=5;nkcuv2=3;nref=1;subgrid=.true.;crsgeo=.false.;use_quadtree=.false.
 use_qext=.false.;nsrcdrn=0;use_storage_volume=.false.;infiltration=.false.;tspinup=0.
 wavemaker=.false.;snapwave=.false.;nonhydrostatic=.false.;precip=.true.;dx=10.;dy=10.
 allocate(kcs(6),z_volume(6),netprcp(6),z_flags_iref(6),cell_area(1),q(6),uv(5),uvmean(3))
 allocate(z_index_uv_md(6),z_index_uv_mu(6),z_index_uv_nd(6),z_index_uv_nu(6))
 allocate(uv_index_z_nm(5),uv_index_z_nmu(5),uv_flags_dir(5),index_kcuv2(3))
 kcs=[1,1,1,2,2,2];z_flags_iref=1;cell_area=100.
 z_index_uv_md=[1,2,6,6,6,6];z_index_uv_mu=[2,6,6,1,6,6]
 z_index_uv_nd=[3,6,4,6,6,5];z_index_uv_nu=[4,6,5,6,3,6]
 uv_index_z_nm=[4,1,5,1,3];uv_index_z_nmu=[1,2,1,3,6];uv_flags_dir=[0,0,1,1,1]
 index_kcuv2=[1,3,5];width=10.;dt=.4;clock=0.;minimum=huge(1._real64)
 call random_seed(size=seedsize);allocate(seed(seedsize));seed=1729;call random_seed(put=seed)
 do n=1,205
  netprcp=0.;q=0.;z_volume=0.
  select case(n)
  case(1)
   z_volume(1:3)=1000.;q(1:5)=.01
  case(2)
   q(1:5)=[.2,.2,.3,.1,.1]
  case(3)
   z_volume(1:3)=[.1_real64,.01_real64,.2_real64];q(1:5)=[-.02,.2,-.03,.3,.4]
  case(4)
   netprcp(1:3)=[.001,.0001,0.];q(1:5)=[-.02,-.2,-.03,.3,.4]
  case(5)
   z_volume(1:3)=1.e-44_real64;q(1:5)=[0.,.01,0.,.01,0.]
  case default
   call random_number(r);z_volume(1:3)=10._real64**(real(r(1:3),real64)*9._real64-6._real64)
   if(mod(n,7)==0)z_volume(2)=0.
   netprcp(1:3)=r(6:8)*.001
   call random_number(r);q(1:5)=(2.*r(1:5)-1.)*10.**(r(4)*8.-4.)
  end select
  if(trim(mode)=='negative_initial')z_volume(1)=-.01_real64
  if(trim(mode)=='negative_rain')netprcp(1)=-.001
  if(trim(mode)=='adaptive')use_quadtree=.true.
  if(trim(mode)=='bad_dummy')q(6)=1.
  if(trim(mode)=='nonfinite')q(2)=ieee_value(0.,ieee_quiet_nan)
  uv=2.;uvmean=3.;oldq=q;initial=z_volume;before=z_volume(1:3);clock=clock+real(dt,real64)
  call udt_limit_fluxes(dt,clock)
  if(any(initial/=z_volume))error stop 'fixture: limiter modified storage'
  if(n==1.and.any(q/=oldq))error stop 'fixture: unnecessary limit'
  if(n==2.and.(q(1)/=oldq(1).or.q(3)/=oldq(3)))error stop 'fixture: external inflow limited'
  do i=1,3
   horizontal=((q(z_index_uv_md(i))-q(z_index_uv_mu(i)))*dy &
    +(q(z_index_uv_nd(i))-q(z_index_uv_nu(i)))*dx)*dt
   source=netprcp(i)*100.*dt;dvol=horizontal+source
   after(i)=before(i)+real(dvol,real64)
  enddo
  if(any(after<0._real64))error stop 'fixture: negative updated volume'
  boundary=real((q(1)+q(3)-q(5))*10.*dt,real64)
  residual=sum(after)-sum(before)-sum(real(netprcp(1:3)*100.*dt,real64))-boundary
  scale=max(1._real64,sum(abs(before)),sum(abs(after)),abs(boundary))
  if(abs(residual)>16._real64*real(epsilon(1.),real64)*scale)error stop 'fixture: mass imbalance'
  minimum=min(minimum,minval(after))
  write(*,'(i0,1x,12(es25.17,1x))')n,before,after,q(1:5),residual
 enddo
 write(*,'(a,es25.17)')'PASS minimum_updated_volume=',minimum
end program
