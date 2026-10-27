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
