! CPU scoped integration of the ORIGINAL spatial, piecewise-linear AMPR forcing.
! Spatial weights and units follow Galibier; time weights use the actual step.
 subroutine udt_rain_time_weights(knots,finish,step,weights)
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  real,intent(in)::knots(:),step
  real(real64),intent(in)::finish
  real(real64),intent(out)::weights(size(knots))
  real(real64)::start,a,b,f,h
  integer::k,n
  n=size(knots)
  if(n<2.or.step<=0.or..not.ieee_is_finite(finish))error stop 'Rain integral: invalid clock'
  if(any(knots(2:)<=knots(:n-1)))error stop 'Rain integral: unordered knots'
  start=finish-real(step,real64);weights=0
  ! Native input semantics hold first/last records outside the source horizon.
  weights(1)=max(0._real64,min(finish,real(knots(1),real64))-start)
  weights(n)=max(0._real64,finish-max(start,real(knots(n),real64)))
  do k=1,n-1
   a=max(start,real(knots(k),real64));b=min(finish,real(knots(k+1),real64))
   if(b<=a)cycle
   h=real(knots(k+1),real64)-real(knots(k),real64)
   f=((a-real(knots(k),real64))+(b-real(knots(k),real64)))/(2*h)
   weights(k)=weights(k)+(b-a)*(1-f);weights(k+1)=weights(k+1)+(b-a)*f
  enddo
  if(abs(sum(weights)-real(step,real64))>1.e-10_real64*max(1._real64,real(step,real64))) &
   error stop 'Rain integral: incomplete step'
 end subroutine
 subroutine udt_prepare_rain_integral(finish,step)
  use sfincs_data
  use iso_fortran_env,only:real64
  use ieee_arithmetic,only:ieee_is_finite
  real(real64),intent(in)::finish
  real,intent(in)::step
  real::yul,xll,di1,dj1,f(4),p
  integer::nm,k,ip,ix,iy,ind1(4),ind2(4)
  if(.not.udt_rain_started)then
   if(ampr_block.or.crsgeo.or..not.subgrid.or.(spinup_meteo.and.tspinup>t0+1.e-3).or.spw_precip.or.prcpfile(1:4)/='none') &
    error stop 'Rain integral: unqualified forcing profile'
   if(any(.not.ieee_is_finite(ampr_pr)).or.any(ampr_pr<0.))error stop 'Rain integral: invalid rain'
   allocate(udt_rain_rates(ampr_nt,np),udt_rain_mean(np),udt_rain_weights(ampr_nt), &
    udt_rain_volume_rates(ampr_nt))
   udt_rain_rates=0.;udt_rain_volume_rates=0
   yul=ampr_y_llcorner+(ampr_nrows-1)*ampr_dy+0.5*ampr_dy
   xll=ampr_x_llcorner+0.5*ampr_dx
   do nm=1,np
    iy=int((yul-z_yz(nm))/ampr_dy)+1;ix=int((z_xz(nm)-xll)/ampr_dx)+1
    if(iy<=1.or.iy>=ampr_nrows.or.ix<=0.or.ix>=ampr_ncols)cycle
    dj1=(yul-(iy-1)*ampr_dy-z_yz(nm))/ampr_dy
    di1=(z_xz(nm)-(xll+(ix-1)*ampr_dx))/ampr_dx
    ind1=[iy,iy,iy+1,iy+1];ind2=[ix,ix+1,ix+1,ix]
    f=[(1-di1)*(1-dj1),di1*(1-dj1),di1*dj1,(1-di1)*dj1]
    do k=1,ampr_nt
     p=0.
     do ip=1,4
      p=p+f(ip)*ampr_pr(k,ind1(ip),ind2(ip))
     enddo
     udt_rain_rates(k,nm)=p/(1000*3600)
     if(kcs(nm)==1)udt_rain_volume_rates(k)=udt_rain_volume_rates(k)+ &
      real(udt_rain_rates(k,nm),real64)*real(cell_area(z_flags_iref(nm)),real64)
    enddo
   enddo
   open(newunit=udt_rain_unit,file='udt_rain_integral.csv',status='new',action='write')
   write(udt_rain_unit,'(a)')'actual_end_time_s,dt_s,integrated_native_rate_volume_m3,sum_weight_seconds'
   open(newunit=k,file='udt_rain_integral_profile.txt',status='new',action='write')
   write(k,'(a)')'Original nonnegative linear AMPR, projected CPU subgrid, no spinup or alternate rain.'
   write(k,'(a)')'prcp remains instantaneous; netprcp and cumulative precipitation use the step mean.'
   write(k,'(a)')'No input, roof weight, hydraulic timestep, stored volume or budget normalization.'
   close(k);udt_rain_started=.true.
  endif
  call udt_rain_time_weights(ampr_times,finish,step,udt_rain_weights)
  !$omp parallel do private(nm)
  do nm=1,np
   udt_rain_mean(nm)=real(dot_product(udt_rain_weights,real(udt_rain_rates(:,nm),real64))/real(step,real64))
  enddo
  !$omp end parallel do
  write(udt_rain_unit,'(es25.17,3(a,es25.17))')finish,',',real(step,real64),',', &
   dot_product(udt_rain_weights,udt_rain_volume_rates),',',sum(udt_rain_weights)
 end subroutine
