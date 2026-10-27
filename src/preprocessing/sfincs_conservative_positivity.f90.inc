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
