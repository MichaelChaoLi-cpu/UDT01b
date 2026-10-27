module sfincs_continuity
 use udt_clock_budget, only: udt_budget_begin,udt_budget_cell,udt_budget_finish,udt_budget_source_added,udt_budget_measures

 real*8,allocatable,private :: positivity_factor(:)
 real,allocatable,private :: positivity_face_ratio(:)
 integer,private :: positivity_unit
 logical,private :: positivity_started=.false.
 logical,private::ad_started=.false.
 integer,private::ad_unit
 integer,allocatable,private::ad_degree(:),ad_faces(:,:)
 real,allocatable,private::ad_width(:),ad_ratio(:)
 real*8,allocatable,private::ad_factor(:)

 logical,private::pump_initialized=.false.,pump_enabled=.false.
 integer,private::pump_n=0,pump_log
 integer,allocatable,private::pump_from(:),pump_to(:)
 real*8,allocatable,private::pump_rate(:),pump_trigger(:),pump_head(:)
 real*8,allocatable::pump_delta(:),pump_out(:)
 real*8::pump_source_step=0.d0,pump_sink_step=0.d0

 logical,private::pool_dynamic=.false.
 integer,private::pool_n=0
 integer,allocatable,private::pump_pool(:)
 real*8,allocatable,private::pool_cap(:),pump_unit(:)
contains
 subroutine udt_initialize_pools()
  use sfincs_data
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  implicit none
  integer::u,ios,i,j,np_in,id,ipool
  logical::found
  character(len=80)::header
  inquire(file='udt_pool_rules.dat',exist=found)
  pool_dynamic=found
  if(.not.found)return
  open(newunit=u,file='udt_pool_rules.dat',status='old',action='read')
  read(u,'(a)',iostat=ios)header
  if(ios/=0.or.trim(header)/='UDT_WEIGHTED_POOL_V1')error stop 'Pools:invalid schema'
  read(u,*,iostat=ios)pool_n
  if(ios/=0.or.pool_n<1.or.pool_n>100)error stop 'Pools:invalid count'
  allocate(pool_cap(pool_n),pump_pool(pump_n),pump_unit(pump_n))
  do i=1,pool_n
   read(u,*,iostat=ios)id,pool_cap(i)
   if(ios/=0.or.id/=i)error stop 'Pools:invalid group id'
   if(.not.ieee_is_finite(pool_cap(i)).or.pool_cap(i)<=0)error stop 'Pools:invalid cap'
  enddo
  read(u,*,iostat=ios)np_in
  if(ios/=0.or.np_in/=pump_n)error stop 'Pools:intake count mismatch'
  do i=1,pump_n
   read(u,*,iostat=ios)id,ipool,pump_unit(i)
   if(ios/=0.or.id/=i.or.ipool<1.or.ipool>pool_n)error stop 'Pools:invalid intake group'
   if(.not.ieee_is_finite(pump_unit(i)).or.pump_unit(i)<=0)error stop 'Pools:invalid unit cap'
   if(pump_rate(i)>pump_unit(i))error stop 'Pools:weight exceeds unit cap'
   if(kcs(pump_to(i))/=2)error stop 'Pools:only external stage receivers supported'
   do j=1,i-1
    if(pump_from(j)==pump_from(i))error stop 'Pools:shared donor not supported'
   enddo
   pump_pool(i)=ipool
  enddo
  read(u,*,iostat=ios)id
  if(ios>=0)error stop 'Pools:extra rows'
  close(u)
 end subroutine

 subroutine udt_weighted_fill(n,cap,weight,limit,amount)
  use iso_fortran_env,only:real64
  implicit none
  integer,intent(in)::n
  real(real64),intent(in)::cap,weight(n),limit(n)
  real(real64),intent(out)::amount(n)
  logical::active(n),saturated(n)
  real(real64)::remaining,ws,proposal(n),spent,take
  integer::i,iter
  amount=0._real64;active=limit>0._real64;remaining=max(0._real64,cap)
  do iter=1,n+1
   if(.not.any(active).or.remaining<=0._real64)exit
   ws=sum(weight,mask=active)
   if(ws<=0._real64)error stop 'Pools:nonpositive active weight'
   proposal=0._real64
   where(active)proposal=remaining*weight/ws
   saturated=active.and.proposal>=limit
   if(.not.any(saturated))then
    ! Deterministic order retains the exact shared upper bound under roundoff.
    do i=1,n
     if(.not.active(i))cycle
     take=min(limit(i),max(0._real64,remaining),proposal(i))
     amount(i)=take;remaining=max(0._real64,remaining-take)
    enddo
    exit
   endif
   do i=1,n
    if(.not.saturated(i))cycle
    take=min(limit(i),max(0._real64,remaining))
    amount(i)=take;remaining=max(0._real64,remaining-take);active(i)=.false.
   enddo
  enddo
  if(any(amount<0._real64).or.any(amount>limit))error stop 'Pools:invalid allocation'
 end subroutine

 subroutine udt_prepare_dynamic_pumps(dt,actual_end)
  use sfincs_data
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  implicit none
  real,intent(in)::dt
  real(real64),intent(in)::actual_end
  real(real64)::lim(pump_n),head(pump_n),depth(pump_n),v(pump_n),tmp(pump_n),group_limit(pump_n),reserve,request
  integer::i,ipool,a,b
  if(dt<=0..or..not.ieee_is_finite(dt))error stop 'Pools:invalid dt'
  reserve=1._real64+64._real64*real(epsilon(1.),real64)
  pump_source_step=0._real64;pump_sink_step=0._real64;pump_delta=0._real64;pump_out=0._real64
  do i=1,pump_n
   a=pump_from(i);b=pump_to(i)
   if(.not.ieee_is_finite(z_volume(a)).or.z_volume(a)<0)error stop 'Pools:invalid prior volume'
   head(i)=max(0._real64,real(zs(b),real64)-real(zs(a),real64))
   depth(i)=max(0._real64,real(zs(a),real64)-real(subgrid_z_zmin(a),real64))
   lim(i)=0._real64
   if(depth(i)>=pump_trigger(i).and.head(i)<=pump_head(i)) &
    lim(i)=min(pump_unit(i)*real(dt,real64),z_volume(a)/reserve)
  enddo
  v=0._real64
  do ipool=1,pool_n
   group_limit=0._real64
   where(pump_pool==ipool)group_limit=lim
   call udt_weighted_fill(pump_n,pool_cap(ipool)*real(dt,real64),pump_rate,group_limit,tmp)
   v=v+tmp
  enddo
  do i=1,pump_n
   a=pump_from(i);b=pump_to(i);ipool=pump_pool(i);request=pump_unit(i)*real(dt,real64)
   pump_out(a)=v(i);pump_delta(a)=-v(i);pump_sink_step=pump_sink_step+v(i)
   write(pump_log,'(2(es25.17,a),4(i0,a),5(es25.17,a),l1,a,l1,a,2(es25.17,a),i0,a,es25.17)') &
    actual_end,',',real(dt,real64),',',i,',',a,',',b,',',kcs(b),',',request,',',v(i),',',z_volume(a),',', &
    head(i),',',depth(i),',',lim(i)<request.and.depth(i)>=pump_trigger(i).and.head(i)<=pump_head(i),',', &
    depth(i)<pump_trigger(i).or.head(i)>pump_head(i),',',pump_rate(i)*real(dt,real64),',',request,',',ipool,',', &
    pool_cap(ipool)*real(dt,real64)
  enddo
 end subroutine
 subroutine udt_prepare_pumps(dt,actual_end)
  use sfincs_data
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  implicit none
  real,intent(in)::dt
  real(real64),intent(in)::actual_end
  integer::u,ios,i,a,b
  logical::found
  real(real64)::available,dv,head,depth
  if(.not.pump_initialized)then
   inquire(file='udt_pumps.dat',exist=found)
   pump_enabled=found
   if(found)then
    if(use_quadtree.or..not.subgrid.or.crsgeo)error stop 'Pumps:regular projected CPU subgrid required'
    open(newunit=u,file='udt_pumps.dat',status='old',action='read')
    read(u,*,iostat=ios)pump_n
    if(ios/=0.or.pump_n<1.or.pump_n>10000)error stop 'Pumps:invalid count'
    allocate(pump_from(pump_n),pump_to(pump_n),pump_rate(pump_n),pump_trigger(pump_n),pump_head(pump_n))
    allocate(pump_delta(np),pump_out(np))
    do i=1,pump_n
     read(u,*,iostat=ios)a,b,pump_rate(i),pump_trigger(i),pump_head(i)
     if(ios/=0.or.min(a,b)<1.or.max(a,b)>np.or.a==b)error stop 'Pumps:invalid endpoint'
     if(kcs(a)/=1.or.(kcs(b)/=1.and.kcs(b)/=2))error stop 'Pumps:invalid endpoint mask'
     if(.not.ieee_is_finite(pump_rate(i)).or.pump_rate(i)<=0)error stop 'Pumps:invalid capacity'
     if(.not.ieee_is_finite(pump_trigger(i)).or.pump_trigger(i)<0)error stop 'Pumps:invalid trigger'
     if(.not.ieee_is_finite(pump_head(i)).or.pump_head(i)<=0)error stop 'Pumps:invalid head'
     pump_from(i)=a;pump_to(i)=b
    enddo
    read(u,*,iostat=ios)a
    if(ios>=0)error stop 'Pumps:extra input rows'
    close(u)
    call udt_initialize_pools()
    open(newunit=pump_log,file='udt_pump_steps.csv',status='new')
    if(pool_dynamic)then
     write(pump_log,'(a)')'actual_end_time_s,dt_s,pump_id,source_native_cell,receiver_native_cell,receiver_kcs,'// &
     'requested_m3,transferred_m3,source_prior_volume_m3,head_m,depth_m,limited_by_storage,blocked_by_trigger_or_head'// &
      ',nominal_reserved_m3,standalone_limit_m3,receiving_pool_id,pool_limit_m3'
    else
     write(pump_log,'(a)')'actual_end_time_s,dt_s,pump_id,source_native_cell,receiver_native_cell,receiver_kcs,'// &
     'requested_m3,transferred_m3,source_prior_volume_m3,head_m,depth_m,limited_by_storage,blocked_by_trigger_or_head'
    endif
   endif
   pump_initialized=.true.
  endif
  if(pool_dynamic)then
   call udt_prepare_dynamic_pumps(dt,actual_end)
   return
  endif
  pump_source_step=0._real64;pump_sink_step=0._real64
  if(.not.pump_enabled)return
  if(dt<=0..or..not.ieee_is_finite(dt))error stop 'Pumps:invalid dt'
  pump_delta=0._real64;pump_out=0._real64
  do i=1,pump_n
   a=pump_from(i);b=pump_to(i)
   if(.not.ieee_is_finite(z_volume(a)).or.z_volume(a)<0)error stop 'Pumps:invalid prior volume'
   head=max(0._real64,real(zs(b),real64)-real(zs(a),real64))
   depth=max(0._real64,real(zs(a),real64)-real(subgrid_z_zmin(a),real64))
   available=max(0._real64,z_volume(a)-pump_out(a))
   dv=udt_pump_amount(pump_rate(i),real(dt,real64),available,depth,head,pump_trigger(i),pump_head(i))
   pump_out(a)=pump_out(a)+dv;pump_delta(a)=pump_delta(a)-dv;pump_sink_step=pump_sink_step+dv
   if(kcs(b)==1)then
    pump_delta(b)=pump_delta(b)+dv;pump_source_step=pump_source_step+dv
   endif
   write(pump_log,'(2(es25.17,a),4(i0,a),5(es25.17,a),l1,a,l1)')actual_end,',',real(dt,real64),',', &
    i,',',a,',',b,',',kcs(b),',',pump_rate(i)*real(dt,real64),',',dv,',',z_volume(a),',',head,',',depth,',', &
    dv<pump_rate(i)*real(dt,real64).and.depth>=pump_trigger(i).and.head<=pump_head(i),',', &
    depth<pump_trigger(i).or.head>pump_head(i)
  enddo
 end subroutine
 real(kind=8) function udt_pump_amount(rate,dt,available,depth,head,trigger,maxhead)result(dv)
  use iso_fortran_env,only:real64
  implicit none
  real(real64),intent(in)::rate,dt,available,depth,head,trigger,maxhead
  dv=0._real64
  if(depth<trigger.or.head>maxhead)return
  ! Keep a roundoff reserve;no simultaneous face inflow or incoming-pump credit.
  dv=min(rate*dt,max(0._real64,available)/(1._real64+64._real64*real(epsilon(1.),real64)))
 end function
 subroutine udt_limit_adaptive(dt,actual_end)
  use sfincs_data
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  implicit none
  real,intent(in)::dt
  real(real64),intent(in)::actual_end
  integer::nm,ip,j,a,b,donor,ib,ic,u,invalid,donors,faces,slot
  real::rain,qold,qnew,ratio,width
  real(real64)::available,outgoing,factor,held,minfactor,reserve
  real(real64),parameter::guard=64._real64*real(epsilon(1.),real64)
  if(.not.ad_started)then
   if(.not.subgrid.or.crsgeo.or..not.use_quadtree.or.nref/=2)error stop 'UDT adaptive: projected two-level subgrid only'
   if(use_qext.or.nsrcdrn/=0.or.use_storage_volume.or.infiltration.or.tspinup/=0.)error stop 'UDT adaptive: unsupported sources'
   if(wavemaker.or.snapwave.or.nonhydrostatic)error stop 'UDT adaptive: unsupported waves'
   if(any(kcs(1:np)/=1.and.kcs(1:np)/=2))error stop 'UDT adaptive: unsupported mask'
   if(size(q)/=npuv+ncuv+1)error stop 'UDT adaptive: unexpected flux layout'
   allocate(ad_degree(np),ad_faces(8,np),ad_width(npuv),ad_factor(np),ad_ratio(npuv))
   ad_degree=0;ad_faces=0
   do ip=1,npuv
    a=uv_index_z_nm(ip);b=uv_index_z_nmu(ip)
    if(min(a,b)<1.or.max(a,b)>np)error stop 'UDT adaptive: invalid endpoints'
    width=dyrm(uv_flags_iref(ip));if(uv_flags_dir(ip)==1)width=dxrm(uv_flags_iref(ip))
    if(width<=0.)error stop 'UDT adaptive: invalid face width'
    ad_width(ip)=width
    do slot=1,2
     nm=a;if(slot==2)nm=b
     ad_degree(nm)=ad_degree(nm)+1
     if(ad_degree(nm)>8)error stop 'UDT adaptive: too many physical faces'
     ad_faces(ad_degree(nm),nm)=ip
    enddo
   enddo
   open(newunit=ad_unit,file='udt_positivity_steps.csv',status='new')
   write(ad_unit,'(a)')'actual_end_time_s,dt_s,limited_donors,limited_faces,held_requested_transport_m3,minimum_factor'
   open(newunit=u,file='udt_positivity_receipt.txt',status='new')
   write(u,'(a)')'schema=conservative-donor-limit-adaptive-v1; same donor availability and shared-face restriction'
   write(u,'(a)')'available=prior_signed_volume+native_nonnegative_rain; no same-step inflow credit; no storage clipping'
   write(u,'(a)')'physical child faces limited once; combined coarse flux regenerated using native arithmetic'
   write(u,'(a)')'float32_guard=64*epsilon; adaptive absolute retained-volume reserve=8*nearest(0,1)*(1+sum_incident_width*dt)'
   close(u);ad_started=.true.
  endif
  if(dt<=0..or..not.ieee_is_finite(dt))error stop 'UDT adaptive: invalid dt'
  if(.not.all(ieee_is_finite(q)))error stop 'UDT adaptive: nonfinite q'
  if(q(npuv+ncuv+1)/=0.)error stop 'UDT adaptive: nonzero dummy'
  donors=0;invalid=0;minfactor=1._real64
  !$omp parallel do private(nm,rain,available,outgoing,factor,j,ip,reserve) reduction(+:invalid,donors) reduction(min:minfactor)
  do nm=1,np
   ad_factor(nm)=1._real64
   if(kcs(nm)/=1)cycle
   rain=0.;if(precip)rain=netprcp(nm)*cell_area(z_flags_iref(nm))*dt
   if(.not.ieee_is_finite(z_volume(nm)).or.z_volume(nm)<0._real64.or..not.ieee_is_finite(rain).or.rain<0.)then
    invalid=invalid+1;cycle
   endif
   available=z_volume(nm)+real(rain,real64);outgoing=0.;reserve=8._real64*real(nearest(0.,1.),real64)
   do j=1,ad_degree(nm)
    ip=ad_faces(j,nm)
    reserve=reserve+real(ad_width(ip),real64)*real(dt,real64)*8._real64*real(nearest(0.,1.),real64)
    if((q(ip)>0..and.uv_index_z_nm(ip)==nm).or.(q(ip)<0..and.uv_index_z_nmu(ip)==nm)) &
     outgoing=outgoing+abs(real(q(ip),real64))*real(ad_width(ip),real64)*real(dt,real64)
   enddo
   if(outgoing>0.)then
    factor=min(1._real64,max(0._real64,available-reserve)/(outgoing*(1._real64+guard)));ad_factor(nm)=factor
    if(factor<1.)donors=donors+1
    minfactor=min(minfactor,factor)
   endif
  enddo
  !$omp end parallel do
  if(invalid/=0)error stop 'UDT adaptive: invalid prior storage or rain'
  faces=0;held=0.
  !$omp parallel do private(ip,qold,qnew,donor,factor,ratio) reduction(+:faces,held)
  do ip=1,npuv
   ad_ratio(ip)=1.;qold=q(ip);if(qold==0.)cycle
   donor=uv_index_z_nm(ip);if(qold<0.)donor=uv_index_z_nmu(ip)
   if(kcs(donor)/=1)cycle
   factor=ad_factor(donor);if(factor>=1.)cycle
   qnew=real(real(qold,real64)*factor)
   if(abs(real(qnew,real64))>abs(real(qold,real64)*factor))qnew=nearest(qnew,-sign(1.,qnew))
   ratio=qnew/qold;q(ip)=qnew;uv(ip)=uv(ip)*ratio;ad_ratio(ip)=ratio
   if(qnew/=qold)then
    faces=faces+1;held=held+abs(real(qold,real64)-real(qnew,real64))*real(ad_width(ip),real64)*real(dt,real64)
   endif
  enddo
  !$omp end parallel do
  do ib=1,nkcuv2
   ip=index_kcuv2(ib)
   if(ip>npuv)error stop 'UDT adaptive: aggregate boundary face unsupported'
   uvmean(ib)=uvmean(ib)*ad_ratio(ip)
  enddo
  !$omp parallel do private(ic)
  do ic=1,ncuv
   q(cuv_index_uv(ic))=(q(cuv_index_uv1(ic))+q(cuv_index_uv2(ic)))/2
   uv(cuv_index_uv(ic))=(uv(cuv_index_uv1(ic))+uv(cuv_index_uv2(ic)))/2
  enddo
  !$omp end parallel do
  write(ad_unit,'(2(es25.17,","),2(i0,","),es25.17,",",es25.17)')actual_end,dt,donors,faces,held,minfactor
 end subroutine
 subroutine udt_limit_fluxes(dt,actual_end)
  use sfincs_data
  use, intrinsic :: iso_fortran_env,only:real64,int64
  use, intrinsic :: ieee_arithmetic,only:ieee_is_finite
  implicit none
  real,intent(in)::dt
  real(real64),intent(in)::actual_end
  integer::nm,ip,donor,ib,invalid,donors,faces,unit
  real::rain,qold,qnew,ratio,width
  real(real64)::available,outgoing,factor,held,minfactor
  real(real64),parameter::roundoff_guard=64._real64*real(epsilon(1.0),real64)
  if(use_quadtree)then
   call udt_limit_adaptive(dt,actual_end)
   return
  endif
  if(.not.positivity_started)then
#ifdef _OPENACC
   error stop 'UDT positivity: CPU profile only'
#endif
   if(.not.subgrid.or.crsgeo.or.use_quadtree.or.nref/=1)error stop 'UDT positivity: projected regular subgrid only'
   if(use_qext.or.nsrcdrn/=0.or.use_storage_volume.or.infiltration.or.tspinup/=0.) &
    error stop 'UDT positivity: unsupported sources sinks storage or spinup'
   if(wavemaker.or.snapwave.or.nonhydrostatic)error stop 'UDT positivity: unsupported wave profile'
   if(any(kcs(1:np)/=1.and.kcs(1:np)/=2))error stop 'UDT positivity: unsupported mask'
   allocate(positivity_factor(np),positivity_face_ratio(npuv))
   open(newunit=positivity_unit,file='udt_positivity_steps.csv',status='new',action='write')
   write(positivity_unit,'(a)')'actual_end_time_s,dt_s,limited_donors,limited_faces,held_requested_transport_m3,minimum_factor'
   open(newunit=unit,file='udt_positivity_receipt.txt',status='new',action='write')
   write(unit,'(a)')'schema=conservative-donor-limit-v1; projected regular CPU subgrid'
   write(unit,'(a)')'available=prior_signed_volume+native_nonnegative_rain; no same-step inflow credit; no storage clipping'
   write(unit,'(a,es24.16)')'float32_relative_roundoff_guard=',roundoff_guard
   close(unit);positivity_started=.true.
  endif
  if(dt<=0..or..not.ieee_is_finite(dt))error stop 'UDT positivity: invalid dt'
  if(.not.all(ieee_is_finite(q)))error stop 'UDT positivity: nonfinite flux'
  if(q(npuv+1)/=0.)error stop 'UDT positivity: nonzero dummy flux'
  call udt_prepare_pumps(dt,actual_end)
  invalid=0;donors=0;minfactor=1._real64
  !$omp parallel do private(nm,rain,available,outgoing,factor) reduction(+:invalid,donors) reduction(min:minfactor) schedule(static)
  do nm=1,np
   positivity_factor(nm)=1._real64
   if(kcs(nm)/=1)cycle
   rain=0.
   if(precip)rain=netprcp(nm)*cell_area(z_flags_iref(nm))*dt
   if(.not.ieee_is_finite(z_volume(nm)).or.z_volume(nm)<0._real64.or..not.ieee_is_finite(rain).or.rain<0.)then
    invalid=invalid+1;cycle
   endif
   available=z_volume(nm)+real(rain,real64)
   if(pump_enabled)available=available-pump_out(nm)
   outgoing=(real(max(-q(z_index_uv_md(nm)),0.),real64)*real(dy,real64) &
    +real(max(q(z_index_uv_mu(nm)),0.),real64)*real(dy,real64) &
    +real(max(-q(z_index_uv_nd(nm)),0.),real64)*real(dx,real64) &
    +real(max(q(z_index_uv_nu(nm)),0.),real64)*real(dx,real64))*real(dt,real64)
   if(outgoing>0._real64)then
    factor=min(1._real64,available/(outgoing*(1._real64+roundoff_guard)))
    positivity_factor(nm)=factor
    if(factor<1._real64)donors=donors+1
    minfactor=min(minfactor,factor)
   endif
  enddo
  !$omp end parallel do
  if(invalid/=0)error stop 'UDT positivity: negative or nonfinite prior storage or rainfall'
  faces=0;held=0._real64
  !$omp parallel do private(ip,qold,qnew,donor,factor,ratio,width) reduction(+:faces,held) schedule(static)
  do ip=1,npuv
   positivity_face_ratio(ip)=1.
   qold=q(ip)
   if(qold==0.)cycle
   donor=uv_index_z_nm(ip)
   if(qold<0.)donor=uv_index_z_nmu(ip)
   if(kcs(donor)/=1)cycle
   factor=positivity_factor(donor)
   if(factor>=1._real64)cycle
   qnew=real(real(qold,real64)*factor)
   ! Round limited transport toward zero, including subnormal float32 values.
   if(abs(real(qnew,real64))>abs(real(qold,real64)*factor))qnew=nearest(qnew,-sign(1.,qnew))
   ratio=qnew/qold
   q(ip)=qnew;uv(ip)=uv(ip)*ratio;positivity_face_ratio(ip)=ratio
   if(qnew/=qold)then
    faces=faces+1
    width=dy
    if(uv_flags_dir(ip)==1)width=dx
    held=held+abs(real(qold,real64)-real(qnew,real64))*real(width,real64)*real(dt,real64)
   endif
  enddo
  !$omp end parallel do
  do ib=1,nkcuv2
   ip=index_kcuv2(ib)
   uvmean(ib)=uvmean(ib)*positivity_face_ratio(ip)
  enddo
  write(positivity_unit,'(2(es25.17,","),2(i0,","),es25.17,",",es25.17)')actual_end,dt,donors,faces,held,minfactor
 end subroutine udt_limit_fluxes
   
   subroutine compute_water_levels(t, dt, tloop)
   !
   use sfincs_data
   !
   implicit none
   !
   real*4           :: dt
   real*8           :: t
   !
   integer          :: count0
   integer          :: count1
   integer          :: count_rate
   integer          :: count_max
   real             :: tloop
   !
   call system_clock(count0, count_rate, count_max)
   !
   if (subgrid) then
      !
      call compute_water_levels_subgrid(dt,t)
      !
   else
      !
      call compute_water_levels_regular(dt,t)
      !
   endif  
   !
   ! Put non-default store options in a separate subroutine (all but zsmax) to save computation time when not used (both regular and subgrid):
   !
   if ((store_maximum_velocity .eqv. .true.) .or. (store_maximum_flux .eqv. .true.) .or. (store_twet .eqv. .true.)) then    
      ! 
      call compute_store_variables(dt)       
      !    
   endif
   !   
   call system_clock(count1, count_rate, count_max)
   tloop = tloop + 1.0*(count1 - count0)/count_rate
   !
   end subroutine

   
   subroutine compute_water_levels_regular(dt,t)
   !
   use sfincs_data
   !
   implicit none
   !
   real*4           :: dt
   real*8           :: t   
   !
   integer          :: nm
   integer          :: isrc
   !
   integer          :: iwm
   !
   integer          :: nmu
   integer          :: nmd
   integer          :: num
   integer          :: ndm
   !
   real*4           :: qnmu
   real*4           :: qnmd
   real*4           :: qnum
   real*4           :: qndm
   real*4           :: factime
   real*4           :: dvol    
   !
   if (snapwave) then ! need to compute filtered water levels for snapwave
      !
      factime = min(dt / wavemaker_filter_time, 1.0)
      !
   endif
   !
   !$acc parallel present( kcs, zs, zb, netprcp, prcp, q, qext, zsmax, zsm, maxzsm, &
   !$acc                   z_flags_iref, uv_flags_iref, &
   !$acc                   z_index_uv_md, z_index_uv_nd, z_index_uv_mu, z_index_uv_nu, &
   !$acc                   dxm, dxrm, dyrm, dxminv, dxrinv, dyrinv, cell_area_m2, cell_area,  &
   !$acc                   nmindsrc, qtsrc, &
   !$acc                   z_index_wavemaker, wavemaker_uvmean, wavemaker_nmd, wavemaker_nmu, wavemaker_ndm, wavemaker_num )
   !
   ! First discharges (don't do this parallel, as it's probably not worth it)
   !
   if (nsrcdrn > 0) then
      ! 
      !$acc loop
      do isrc = 1, nsrcdrn
         ! 
         nm = nmindsrc(isrc)
         ! 
         if (crsgeo) then
            ! 
            zs(nmindsrc(isrc)) = max(zs(nm) + qtsrc(isrc) * dt / cell_area_m2(nm), zb(nm))
            ! 
         else
            ! 
            zs(nmindsrc(isrc)) = max(zs(nm) + qtsrc(isrc) * dt / cell_area(z_flags_iref(nm)), zb(nm))
            ! 
         endif
         ! 
      enddo
      ! 
   endif
   !
   !$omp parallel &
   !$omp private ( nm,dvol,nmd,nmu,ndm,num,qnmd,qnmu,qndm,qnum,iwm)
   !$omp do schedule ( dynamic, 256 )
   !$acc loop gang vector
   do nm = 1, np
      ! 
      if (kcs(nm) == 1) then ! Regular point
         !
         if (precip) then
            !
            zs(nm) = zs(nm) + netprcp(nm) * dt
            !
         endif
         !
         if (use_qext) then
            !
            ! Add external source (e.g. from XMI coupling) 
            ! 
            zs(nm) = zs(nm) + qext(nm) * dt 
            !
         endif
         !
         nmd = z_index_uv_md(nm)
         nmu = z_index_uv_mu(nm)
         ndm = z_index_uv_nd(nm)
         num = z_index_uv_nu(nm)
         !
         if (crsgeo) then
            !
            ! Use cell width dxm (which varies with latitude)
            !
            zs(nm)   = zs(nm) + ( (q(nmd) - q(nmu)) / dxm(nm) + (q(ndm) - q(num)) * dyrinv(z_flags_iref(nm)) ) * dt
            !
            ! Should really be:
            !
            ! zs(nm)   = zs(nm) + ( (q(nmd) - q(nmu)) / dxm(nm) + (q(ndm) * f - q(num) / f) * dyrinv(z_flags_iref(nm)) ) * dt
            !
            ! Where f if a correction factor for the ratio between cell width at cell centre and cell bottom:
            !
            ! f = cos(y - 0.5*dy) / cos(y) (this holds both in northern and southern hemisphere)
            ! 
         else   
            !
            zs(nm)   = zs(nm) + ( (q(nmd) - q(nmu)) * dxrinv(z_flags_iref(nm)) + (q(ndm) - q(num)) * dyrinv(z_flags_iref(nm)) ) * dt
            !
         endif
         !
      endif
      !
      if (wavemaker) then
         !      
         if (kcs(nm) == 4) then
            !
            ! Wave maker point (seaward of wave maker)
            ! Here we use the mean flux at the location of the wave maker 
            !
            iwm = z_index_wavemaker(nm)
            !
            if (wavemaker_nmd(iwm) > 0) then
               !
               ! Wave paddle on the left
               !
               qnmd = wavemaker_uvmean(wavemaker_nmd(iwm))
               !
            else
               !
               qnmd = q(z_index_uv_md(nm))
               !
            endif   
            !
            if (wavemaker_nmu(iwm) > 0) then
               !
               ! Wave paddle on the right
               !
               qnmu = wavemaker_uvmean(wavemaker_nmu(iwm))
               !
            else
               !
               qnmu = q(z_index_uv_mu(nm))
               !
            endif   
            !
            if (wavemaker_ndm(iwm) > 0) then
               !
               ! Wave paddle below
               !
               qndm = wavemaker_uvmean(wavemaker_ndm(iwm))
               !
            else
               !
               qndm = q(z_index_uv_nd(nm))
               !
            endif   
            !
            if (wavemaker_num(iwm) > 0) then
               !
               ! Wave paddle above
               !
               qnum = wavemaker_uvmean(wavemaker_num(iwm))
               !
            else
               !
               qnum = q(z_index_uv_nu(nm))
               !
            endif   
            !
            zs(nm)   = zs(nm) + (((qnmd - qnmu) * dxrinv(z_flags_iref(nm)) + (qndm - qnum) * dyrinv(z_flags_iref(nm)))) * dt
            !
         endif   
         !
      endif
      !
      if (snapwave) then
         !
         ! Time-averaged water level used for SnapWave using exponential filter
         !
         ! Would double exponential filtering be better?
         !
         zsm(nm) = factime * zs(nm) + (1.0 - factime) * zsm(nm)
         !
         if (store_maximum_waterlevel) then
             !
             maxzsm(nm) = max(maxzsm(nm), zsm(nm))             
             !
         endif         
         !             
      endif 
      !
      ! No continuity update but keeping track of variables         
      ! zsmax used by default, therefore keep in standard continuity loop:
      !
      if (store_maximum_waterlevel) then
         !
         ! Store when the maximum water level changed
         !
         if (store_t_zsmax) then
             if (zs(nm) > zsmax(nm)) then
                 if ( (zs(nm) - zb(nm)) > huthresh) then
                    t_zsmax(nm) = t
                 endif
             endif
         endif
         !
         ! Store the maximum water level itself
         !
         zsmax(nm) = max(zsmax(nm), zs(nm))
         !
      endif
      !
   enddo
   !$omp end do
   !$omp end parallel
   !$acc end parallel
   !         
   end subroutine

   
   subroutine compute_water_levels_subgrid(dt,t)
   !
   use sfincs_data
   !
   implicit none
   !
   real*4           :: dt
   real*8           :: t
   !
   integer          :: nm
   integer          :: isrc
   !
   integer          :: iwm
   integer          :: ind
   !
   integer          :: nmu
   integer          :: nmd
   integer          :: num
   integer          :: ndm
   !
   real*4           :: factime
   real*4           :: dvol  
   real*4           :: dzsdt
   !
   real*4           :: qnmu
   real*4           :: qnmd
   real*4           :: qnum
   real*4           :: qndm
   !
   integer          :: iuv
   real*4           :: dzvol
   real*4           :: facint
   real*4           :: a
   real*4           :: dv
   real*4           :: zs00
   real*4           :: zs11
   !
   call udt_limit_fluxes(dt,t)
   call udt_budget_begin()
   call udt_budget_measures(pump_source_step,pump_sink_step)
   if (wavemaker) then
      !
      factime = min(dt / wavemaker_filter_time, 1.0)
      !
   endif   
   !
   ! First discharges (don't do this parallel, as it's probably not worth it)
   ! NVFORTAN turns this into a sequential loop (!$acc loop seq)
   !
   if (nsrcdrn > 0) then
      !
      !$acc serial present( z_volume, nmindsrc, qtsrc )
      do isrc = 1, nsrcdrn
         !
         nm = nmindsrc(isrc)
         !
         if ((z_volume(nm) >= 0) .or. ((qtsrc(isrc)<0.0) .and. (z_volume(nm) >= 0))) then
            z_volume(nm) = z_volume(nm) + qtsrc(isrc) * dt
         endif
         !
      enddo
      !$acc end serial
      !
   endif
   !
   !$omp parallel &
   !$omp private ( dvol,dzsdt,nmd,nmu,ndm,num,a,iuv,facint,dzvol,ind,iwm,qnmd,qnmu,qndm,qnum,dv,zs00,zs11 )
   !$omp do schedule ( dynamic, 256 )
   !$acc parallel present( kcs, zs, zs0, zb, z_volume, zsmax, zsm, maxzsm, zsderv, &
   !$acc                   subgrid_z_zmin,  subgrid_z_zmax, subgrid_z_dep, subgrid_z_volmax, &
   !$acc                   netprcp, prcp, q, qext, z_flags_iref, uv_flags_iref, &
   !$acc                   z_index_uv_md, z_index_uv_nd, z_index_uv_mu, z_index_uv_nu, &
   !$acc                   dxm, dxrm, dyrm, dxminv, dxrinv, dyrinv, cell_area_m2, cell_area, &   
   !$acc                   z_index_wavemaker, wavemaker_uvmean, wavemaker_nmd, wavemaker_nmu, wavemaker_ndm, wavemaker_num, storage_volume)
   !$acc loop gang vector
   do nm = 1, np
      !
      ! And now water level changes due to horizontal fluxes
      !
      dvol = 0.0
      !
      if (kcs(nm) == 1) then
         !
         nmd = z_index_uv_md(nm)
         nmu = z_index_uv_mu(nm)
         ndm = z_index_uv_nd(nm)
         num = z_index_uv_nu(nm)
         !
         if (crsgeo) then
            !
            ! dxm  = size of cell in x - direction (it varies for all cells)
            ! dyrm = size of cell in y - direction (it varies for all zoom levels)
            !
            dvol = dvol + ( (q(nmd) - q(nmu)) * dyrm(z_flags_iref(nm)) + (q(ndm) - q(num)) * dxm(nm) ) * dt
            !
            ! Should really be:
            !
            ! dvol = dvol + ( (q(nmd) - q(nmu)) * dyrm(z_flags_iref(nm)) + (q(ndm) * f - q(num) / f) * dxm(nm) ) * dt
            !
            ! where f if a correction factor for the ratio between cell width at cell centre and cell bottom:
            !
            ! f = cos(y - 0.5*dy) / cos(y) (this holds both in northern and southern hemisphere)
            !
            ! This assumes that we can use the same factor f for q(ndm) and q(num), i.e.:
            !
            ! cos(y - 0.5*dy) / cos(y) ~= cos(y + 0.5*dy) / cos(y) or: cos(y - 0.5*dy) ~= cos(y + 0.5*dy) which is pretty much true for dy < 1.0 degree
            !
         else
            !
            if (use_quadtree) then   
               !
               ! dxrm = size of cell in x - direction (it varies for all zoom levels)
               ! dyrm = size of cell in y - direction (it varies for all zoom levels)
               !               
               dvol = dvol + ( (q(nmd) - q(nmu)) * dyrm(z_flags_iref(nm)) + (q(ndm) - q(num)) * dxrm(z_flags_iref(nm)) ) * dt
               ! 
            else
               ! 
               dvol = dvol + ( (q(nmd) - q(nmu)) * dy + (q(ndm) - q(num)) * dx ) * dt
               ! 
            endif   
            !
         endif   
      endif ! kcs==1
      !
      if (wavemaker .and. kcs(nm) == 4) then
         !
         ! Wave maker point (seaward of wave maker)
         ! Here we use the mean flux at the location of the wave maker
         !
         iwm = z_index_wavemaker(nm)
         !
         if (wavemaker_nmd(iwm) > 0) then
            !
            ! Wave paddle on the left
            !
            qnmd = wavemaker_uvmean(wavemaker_nmd(iwm))
            !
         else
            !
            qnmd = q(z_index_uv_md(nm))
            !
         endif   
         !
         if (wavemaker_nmu(iwm) > 0) then
            !
            ! Wave paddle on the right
            !
            qnmu = wavemaker_uvmean(wavemaker_nmu(iwm))
            !
         else
            !
            qnmu = q(z_index_uv_mu(nm))
            !
         endif   
         !
         if (wavemaker_ndm(iwm) > 0) then
            !
            ! Wave paddle below
            !
            qndm = wavemaker_uvmean(wavemaker_ndm(iwm))
            !
         else
            !
            qndm = q(z_index_uv_nd(nm))
            !
         endif   
         !
         if (wavemaker_num(iwm) > 0) then
            !
            ! Wave paddle above
            !
            qnum = wavemaker_uvmean(wavemaker_num(iwm))
            !
         else
            !
            qnum = q(z_index_uv_nu(nm))
            !
         endif   
         !
         if (use_quadtree) then
            ! 
            dvol = dvol + ( (qnmd - qnmu) * dyrm(z_flags_iref(nm)) + (qndm - qnum) * dxrm(z_flags_iref(nm)) ) * dt
            ! 
         else
            ! 
            dvol = dvol + ( (qnmd - qnmu) * dy + (qndm - qnum) * dx ) * dt
            !
         endif   
         !
      endif
      !
      ! We got the volume change dvol in each active cell from fluxes
      ! Now first add precip and qext
      ! Then adjust for storage volume
      ! Then update the volume and compute new water level           
      !
      if (kcs(nm) == 1 .or. kcs(nm) == 4) then
         !
         ! Obtain cell area
         !
         if (crsgeo) then
            !
            a = cell_area_m2(nm)
            !
         else   
            !
            a = cell_area(z_flags_iref(nm))
            !
         endif
         !
         call udt_budget_cell(nm,dvol,a,dt)
         if (precip .or. use_qext) then
            !
            dzsdt = 0.0
            !   
            if (precip) then
               !
               ! Add nett rainfall 
               !
               dzsdt = dzsdt + netprcp(nm)
               !
            endif
            !
            if (use_qext) then
               !
               ! Add external source (e.g. from XMI coupling) 
               ! 
               dzsdt = dzsdt + qext(nm)
               !
            endif
            !
            ! dzsdt is still in m/s, so multiply with a * dt to get m^3
            !
            dvol = dvol + dzsdt * a * dt
            call udt_budget_source_added(nm,dvol)         
            !      
         endif
         !
         if (use_storage_volume) then
            !
            ! If water enters the cell through a point discharge, it will NOT end up in storage volume !  
            !   
            if (storage_volume(nm) > 1.0e-6 .and. dvol > 0.0) then
               !
               ! There is still some storage left, and water is entering the cell
               !
               ! Compute remaining storage volume
               !
               dv = storage_volume(nm) - dvol
               !
               ! Update storage volume (it cannot become negative)) 
               ! 
               storage_volume(nm) = max(dv, 0.0)
               !
               if (dv < 0.0) then
                  !
                  ! Overshoot, so add remaining volume to z_volume
                  !
                  dvol = - dv
                  ! 
               else                   
                  !
                  ! Everything went into storage  
                  ! 
                  dvol = 0.0 
                  !
               endif
               !
            endif
            !
         endif
         !
         ! Update volume 
         !
         z_volume(nm) = z_volume(nm) + dvol
         if(pump_enabled)z_volume(nm)=z_volume(nm)+pump_delta(nm)
         !
         if (wiggle_suppression) then
            !
            ! Store previous water level to determine gradient
            ! 
            zs00    = zs0(nm) ! previous time step
            zs11    = zs(nm)  ! current time step before updating
            zs0(nm) = zs11    ! next previous time step
            ! 
         endif
         ! 
         ! Obtain new water level from subgrid tables 
         !
         if (z_volume(nm) >= subgrid_z_volmax(nm) * 0.999) then
            !
            ! Entire cell is wet, no interpolation needed
            !
            zs(nm) = max(subgrid_z_zmax(nm), -20.0) + (z_volume(nm) - subgrid_z_volmax(nm)) / a
            !
         elseif (z_volume(nm) <= 1.0e-6) then
            !
            ! No water in this cell. Set zs to z_zmin.
            !
            zs(nm) = max(subgrid_z_zmin(nm), -20.0)
            !
         else
            !
            ! Interpolation from subgrid tables needed.
            !
            dzvol    = subgrid_z_volmax(nm) / (subgrid_nlevels - 1)
            iuv      = int(z_volume(nm) / dzvol) + 1
            facint   = (z_volume(nm) - (iuv - 1) * dzvol ) / dzvol
            zs(nm)   = subgrid_z_dep(iuv, nm) + (subgrid_z_dep(iuv + 1, nm) - subgrid_z_dep(iuv, nm)) * facint
            !
         endif
         !
         !
         if (wiggle_suppression) then 
            ! 
            zsderv(nm) = zs(nm) - 2 * zs11 + zs00
            ! 
         endif
         !
      endif
      !
      if (snapwave) then
         !
         ! Time-averaged water level used for SnapWave using exponential filter
         !
         ! Would double exponential filtering be better?
         !
         zsm(nm) = factime * zs(nm) + (1.0 - factime) * zsm(nm)
         !
         if (store_maximum_waterlevel) then
             !
             maxzsm(nm) = max(maxzsm(nm), zsm(nm))             
             !
         endif         
         !    
      endif 
      !
      ! No continuity update but keeping track of variables         
      ! zsmax used by default, therefore keep in standard continuity loop:         
      !
      if (store_maximum_waterlevel) then
         !
         ! Store when the maximum water level changed
         !
         if (store_t_zsmax) then
             if (zs(nm) > zsmax(nm)) then
                 if ( (zs(nm) - subgrid_z_zmin(nm)) > huthresh) then
                    t_zsmax(nm) = t
                 endif
             endif
         endif
         !
         ! Store the maximum water level itself
         !
         zsmax(nm) = max(zsmax(nm), zs(nm))
         !
      endif
      !
   enddo
   !$omp end do
   !$omp end parallel
   !         
   !$acc end parallel
   !         
   call udt_budget_finish(t,dt)
   end subroutine
   
   subroutine compute_store_variables(dt)
   !
   use sfincs_data
   !
   implicit none
   !
   real*4           :: dt
   !   
   integer          :: nm
   !   
   integer          :: nmu
   integer          :: nmd
   integer          :: num
   integer          :: ndm
   !
   real*4           :: quz
   real*4           :: qvz
   real*4           :: qz
   real*4           :: uvz      
   !
   !$omp parallel &
   !$omp private ( nmd, nmu, ndm, num, quz, qvz, qz, uvz )
   !$omp do schedule ( dynamic, 256 )
   !$acc parallel present( kcs, zs, zb, subgrid_z_zmin, q, uv, vmax, qmax, twet, &
   !$acc                   z_index_uv_md, z_index_uv_nd, z_index_uv_mu, z_index_uv_nu )
   !$acc loop gang vector 
   do nm = 1, np
      !
      ! And now water level changes due to horizontal fluxes
      !
      qz = 0.0
      uvz = 0.0    
      !
      if (kcs(nm) == 1 .or. kcs(nm) == 4) then ! TL: kcs(nm)==4 also correct for regular?
         !      
         ! Regular point with four surrounding cells of the same size
         !
         nmd = z_index_uv_md(nm)
         nmu = z_index_uv_mu(nm)
         ndm = z_index_uv_nd(nm)
         num = z_index_uv_nu(nm)
         !
         if (store_maximum_velocity) then
            quz = (uv(nmd) + uv(nmu)) / 2   
            qvz = (uv(ndm) + uv(num)) / 2      
            uvz = sqrt(quz**2 + qvz**2)
         endif
         !
         if (store_maximum_flux) then
            quz = (q(nmd) + q(nmu)) / 2   
            qvz = (q(ndm) + q(num)) / 2      
            qz = sqrt(quz**2 + qvz**2)
         endif   
         !
         ! No continuity update but keeping track of variables
         ! 1. store vmax
         if (store_maximum_velocity) then
            !             
            vmax(nm) = max(vmax(nm), uvz)
            !
         endif      
         !
         ! 2. store qmax         
         if (store_maximum_flux) then
            !             
            qmax(nm) = max(qmax(nm), qz)
            !
         endif      
         !
         ! 3. store Twet
         if (store_twet) then
            if (subgrid) then
              !
              if ( (zs(nm) - subgrid_z_zmin(nm)) > twet_threshold) then
                 twet(nm) = twet(nm) + dt
              endif
              !
            else
              !
              if ( (zs(nm) - zb(nm)) > twet_threshold) then
                 !
                 twet(nm) = twet(nm) + dt
                 !
              endif
              !
            endif               
         endif         
         !
      endif   
   enddo   
   !$omp end do
   !$omp end parallel
   !$acc end parallel
   !       
   end subroutine
   
end module
