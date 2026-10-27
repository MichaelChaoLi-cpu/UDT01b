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
