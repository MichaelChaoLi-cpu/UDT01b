program fixture
 use sfincs_data
 use sfincs_continuity,only:udt_limit_fluxes
 use iso_fortran_env,only:real64
 implicit none
 integer::i,n,ns
 integer,allocatable::seed(:)
 real::dt,h,src,r(9),oldq(8)
 real(real64)::before(6),after(3),clock,residual,bd,scale
 np=6;npuv=6;ncuv=1;nkcuv2=3;nref=2;subgrid=.true.;crsgeo=.false.;use_quadtree=.true.
 use_qext=.false.;nsrcdrn=0;use_storage_volume=.false.;infiltration=.false.;tspinup=0.
 wavemaker=.false.;snapwave=.false.;nonhydrostatic=.false.;precip=.true.;dx=10.;dy=10.
 allocate(kcs(6),z_volume(6),netprcp(6),z_flags_iref(6),cell_area(2),dxrm(2),dyrm(2),q(8),uv(8),uvmean(3))
 allocate(z_index_uv_md(6),z_index_uv_mu(6),z_index_uv_nd(6),z_index_uv_nu(6))
 allocate(uv_index_z_nm(6),uv_index_z_nmu(6),uv_flags_dir(6),uv_flags_iref(6),index_kcuv2(3))
 allocate(cuv_index_uv(1),cuv_index_uv1(1),cuv_index_uv2(1))
 kcs=[1,1,1,2,2,2];z_flags_iref=[1,2,2,1,2,2];cell_area=[100.,25.];dxrm=[10.,5.];dyrm=dxrm
 z_index_uv_md=[3,1,2,8,8,8];z_index_uv_mu=[7,4,5,8,8,8]
 z_index_uv_nd=[8,8,6,8,8,8];z_index_uv_nu=[8,6,8,8,8,8]
 uv_index_z_nm=[1,1,4,2,3,2];uv_index_z_nmu=[2,3,1,5,6,3];uv_flags_dir=[0,0,0,0,0,1]
 uv_flags_iref=[2,2,1,2,2,2];index_kcuv2=[3,4,5];cuv_index_uv=7;cuv_index_uv1=1;cuv_index_uv2=2
 dt=.4;clock=0.;call random_seed(size=ns);allocate(seed(ns));seed=1729;call random_seed(put=seed)
 do n=1,505
  dt=.4
  if(n==4.or.n==5.or.mod(n,11)==0)dt=4.e-7
  call random_number(r);z_volume=0.;z_volume(1:3)=10._real64**(real(r(1:3),real64)*10._real64-7._real64)
  netprcp=0.;netprcp(1:3)=r(4:6)*.001
  call random_number(r);q=0.;q(1:6)=(2*r(1:6)-1)*10.**(r(7)*8.-4.)
  if(n==1)then
   z_volume=1000.;q(1:6)=.01
  elseif(n==2)then
   z_volume=0.;netprcp=0.;q(1:6)=[.2,.1,.3,.2,.1,-.1]
  elseif(n==3)then
   z_volume=1.e-44_real64;netprcp=0.;q(1:6)=[.01,.01,0.,.01,.01,0.]
  elseif(n==4)then
   z_volume=1.e-43_real64;netprcp=0.;q(1:6)=[-.01,-.02,-.03,-.04,-.05,.06]
  elseif(n==5)then
   z_volume=0.;netprcp=0.;q(1:6)=[1.e-44,-1.e-44,1.e-44,-1.e-44,1.e-44,1.e-44]
  endif
  q(7)=(q(1)+q(2))/2;uv=2.;uvmean=3.;oldq=q;before=z_volume;clock=clock+real(dt,real64)
  call udt_limit_fluxes(dt,clock)
  if(any(z_volume/=before))error stop 'fixture: storage modified'
  if(n==1.and.any(q/=oldq))error stop 'fixture: inactive limiter changed flux'
  if(q(7)/=(q(1)+q(2))/2)error stop 'fixture: stale coarse aggregate'
  do i=1,3
   h=((q(z_index_uv_md(i))-q(z_index_uv_mu(i)))*dyrm(z_flags_iref(i)) &
    +(q(z_index_uv_nd(i))-q(z_index_uv_nu(i)))*dxrm(z_flags_iref(i)))*dt
   src=netprcp(i)*cell_area(z_flags_iref(i))*dt
   after(i)=before(i)+real(h+src,real64)
  enddo
  if(any(after<0.))error stop 'fixture: negative native updated storage'
  bd=real((q(3)*10.-q(4)*5.-q(5)*5.)*dt,real64)
  residual=sum(after)-sum(before(1:3))-sum(real(netprcp(1:3)*[100.,25.,25.]*dt,real64))-bd
  scale=max(1._real64,sum(abs(before(1:3))),sum(abs(after)),abs(bd))
  if(abs(residual)>32*epsilon(1.)*scale)error stop 'fixture: coarse fine mass imbalance'
  write(*,'(i0,1x,10(es25.17,1x))')n,after,q(1:6),residual
 enddo
 print *,'PASS 505 native coarse fine updates, mixed directions, dry donors and subnormals'
end program
