program fixture
 use sfincs_data
 use sfincs_meteo,only:udt_rain_time_weights,update_meteo_forcing
 use iso_fortran_env,only:real64
 implicit none
 real::knots(6),rate(6),step,tloop,mean(3),oldcum(3)
 real(real64)::w(6),t,a,b,expected,total(3),scale
 integer::i,j,n
 knots=[0.,3600.,3601.,7200.,7201.,14400.];rate=[0.,0.,60.,60.,0.,0.]
 do i=1,20000
  step=real(mod(i*571,18000)+1)/3.;t=real(mod(i*3719,18000)-1000,real64)
  if(i==1)then
   step=4.e-7;t=real(step,real64)
  endif
  call udt_rain_time_weights(knots,t,step,w)
  a=t-real(step,real64);expected=primitive(t)-primitive(a)
  if(any(w<0).or.abs(sum(w)-real(step,real64))>1.e-9_real64)error stop 'fixture: weight coverage'
  if(abs(dot_product(w,real(rate,real64))-expected)>1.e-8_real64)error stop 'fixture: integral error'
 enddo
 ! Exercise the actual meteo interface, not a second implementation of its clock.
 np=3;ampr_nt=6;ampr_nrows=4;ampr_ncols=4;ampr_dx=5.;ampr_dy=5.
 ampr_x_llcorner=0.;ampr_y_llcorner=0.;netamprfile='original.nc';ampr_block=.false.
 crsgeo=.false.;subgrid=.true.;spinup_meteo=.false.;spw_precip=.false.;prcpfile='none';wndfile='none'
 meteo3d=.true.;precip=.true.;wind=.false.;patmos=.false.;pavbnd=0.;tspinup=0.;t0=0.
 store_cumulative_precipitation=.true.
 allocate(ampr_times(6),ampr_pr(6,4,4),z_xz(3),z_yz(3),kcs(3),z_flags_iref(3),cell_area(2))
 allocate(z_volume(3),zs(3),zb(3),prcp(3),prcp0(3),prcp1(3),netprcp(3),cumprcp(3))
 ampr_times=knots;z_xz=[7.5,10.,12.5];z_yz=[7.5,10.,7.5];kcs=[1,1,2]
 z_flags_iref=[1,2,2];cell_area=[100.,25.];z_volume=0.;zs=0.;zb=0.;cumprcp=0.;total=0
 do i=1,4
  do j=1,4
   ampr_pr(:,i,j)=rate*real(i+j)
  enddo
 enddo
 t=0.;n=0;tloop=0.
 do while(t<14401)
  n=n+1;step=.17+real(mod(n*23,79))*.01
  if(n==1)step=4.e-7
  t=t+real(step,real64);meteo_t0=floor(t);meteo_t1=meteo_t0+1
  ! Keep an independent instantaneous meteo signal to prove it is preserved.
  prcp0=2.e-5;prcp1=2.e-5;oldcum=cumprcp
  call update_meteo_forcing(t,step,tloop)
  if(any(abs(prcp-2.e-5)>2*epsilon(1.)*2.e-5))error stop 'fixture: instantaneous output changed'
  call udt_rain_time_weights(knots,t,step,w)
  expected=dot_product(w,real(rate,real64))/3.6e6_real64
  mean=real(expected/real(step,real64)*[5._real64,5._real64,6._real64])
  if(any(abs(netprcp-mean)>4*epsilon(1.)*max(1.e-30,mean)))error stop 'fixture: native spatial rain'
  if(any(cumprcp/=oldcum+netprcp*step))error stop 'fixture: cumulative rain mismatch'
  total=total+real(netprcp*step,real64)
 enddo
 if(maxval(abs(total-.06_real64*[5._real64,5._real64,6._real64]))>1.e-7_real64) &
  error stop 'fixture: total original rain'
 print *,'PASS 20000 exact-step tests and native meteo/cumulative/spatial interface',n
 print *,'Integrated depths ',total
contains
 function primitive(x)result(v)
  real(real64),intent(in)::x
  real(real64)::v,u
  v=0.
  if(x<=3600)return
  if(x<3601)then
   u=x-3600;v=30*u*u;return
  endif
  v=30
  if(x<=7200)then
   v=v+60*(x-3601);return
  endif
  v=v+60*3599
  if(x<7201)then
   u=x-7200;v=v+60*u-30*u*u;return
  endif
  v=v+30
 end function
end program
