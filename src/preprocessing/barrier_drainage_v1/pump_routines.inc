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
    open(newunit=pump_log,file='udt_pump_steps.csv',status='new')
    write(pump_log,'(a)')'actual_end_time_s,dt_s,pump_id,source_native_cell,receiver_native_cell,receiver_kcs,'// &
     'requested_m3,transferred_m3,source_prior_volume_m3,head_m,depth_m,limited_by_storage,blocked_by_trigger_or_head'
   endif
   pump_initialized=.true.
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
