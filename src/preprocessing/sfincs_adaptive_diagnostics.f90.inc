 subroutine adaptive_trace_initialize()
  integer::u,i,nm,ip,a,b,ic,ios,idx,code
  real::width
  if(.not.use_quadtree)return
  call require(nref==2,'adaptive trace requires two-level projected grid')
  allocate(ad_min(np),ad_time(np),ad_previous(np),ad_min_step(np),ad_counts(np),ad_groups(np))
  ad_min=z_volume;ad_previous=z_volume;ad_time=origin;ad_min_step=0;ad_counts=0;ad_groups=0
  open(newunit=u,file='udt_adaptive_groups.txt',status='old',iostat=ios)
  call require(ios==0,'adaptive water groups missing')
  do nm=1,np
   read(u,*,iostat=ios)idx,code
   call require(ios==0.and.idx==nm,'adaptive group native index mismatch')
   if(kcs(nm)==2)then
    call require(any(code==neg_codes),'adaptive stage group invalid')
   else
    call require(code==0,'adaptive interior group invalid')
   endif
   ad_groups(nm)=code
  enddo
  close(u)
  open(newunit=u,file='udt_adaptive_topology.bin',access='stream',form='unformatted',status='new')
  write(u)np,npuv,ncuv,nkcuv2
  write(u)z_index_z_n,z_index_z_m,kcs,z_flags_iref
  write(u)uv_index_z_nm,uv_index_z_nmu,uv_flags_dir,uv_flags_iref,kcuv(1:npuv)
  write(u)index_kcuv2,ibuvdir
  write(u)cuv_index_uv,cuv_index_uv1,cuv_index_uv2
  write(u)z_index_uv_md,z_index_uv_mu,z_index_uv_nd,z_index_uv_nu
  close(u)
  open(newunit=ad_exchange_unit,file='udt_adaptive_exchange.csv',status='new')
  write(ad_exchange_unit,'(a)')'step,time_s,closed_max_abs_q,group17_out,group17_in,group51_out,group51_in,'// &
   'group71_out,group71_in,group79_out,group79_in,group106_out,group106_in,group110_out,group110_in,group111_out,group111_in'
  open(newunit=ad_events_unit,file='udt_adaptive_negative_events.csv',status='new')
  write(ad_events_unit,'(a)')'step,time_s,native_cell,before_m3,after_m3,horizontal_m3,rain_increment_m3'
 end subroutine
 subroutine adaptive_trace_finish()
  integer::nm,ib,ip,boundary,j,ig
  real::width,dv,closed
  real(real64)::ex(14)
  if(.not.use_quadtree)return
  !$omp parallel do private(nm)
  do nm=1,np
   if(kcs(nm)/=1)cycle
   if(z_volume(nm)<ad_min(nm))then
    ad_min(nm)=z_volume(nm);ad_time(nm)=live_time;ad_min_step(nm)=step
   endif
   if(z_volume(nm)<0.)ad_counts(nm)=ad_counts(nm)+1
  enddo
  !$omp end parallel do
  do nm=1,np
   if(kcs(nm)/=1)cycle
   if((ad_previous(nm)<neg_threshold).neqv.(z_volume(nm)<neg_threshold))then
    write(ad_events_unit,'(i0,a,es25.17,a,i0,4(a,es25.17))')step,',',live_time,',',nm, &
     ',',ad_previous(nm),',',z_volume(nm),',',horizontal_term(nm),',',rain_increment(nm)
   endif
  enddo
  ad_previous=z_volume;ex=0.;closed=0.
  do ib=1,nkcuv2
   ip=index_kcuv2(ib)
   call require(ip<=npuv,'adaptive boundary must use physical child faces')
   boundary=uv_index_z_nm(ip)
   if(kcs(boundary)/=2)boundary=uv_index_z_nmu(ip)
   call require(kcs(boundary)==2,'adaptive boundary topology invalid')
   ig=0
   do j=1,7
    if(neg_codes(j)==ad_groups(boundary))ig=j
   enddo
   call require(ig>0,'unmapped adaptive boundary group')
   width=dyrm(uv_flags_iref(ip));if(uv_flags_dir(ip)==1)width=dxrm(uv_flags_iref(ip))
   dv=-real(ibuvdir(ib))*q(ip)*width*real(live_dt)
   ex(2*ig-1)=ex(2*ig-1)+max(real(dv,real64),0._real64)
   ex(2*ig)=ex(2*ig)+max(-real(dv,real64),0._real64)
   if(kcuv(ip)==0)closed=max(closed,abs(q(ip)))
  enddo
  write(ad_exchange_unit,'(i0,16(a,es25.17))')step,',',live_time,',',real(closed,real64),(',',ex(j),j=1,14)
 end subroutine
 subroutine adaptive_trace_close()
  integer::u,nm
  if(.not.use_quadtree)return
  open(newunit=u,file='udt_adaptive_cell_summary.csv',status='new')
  write(u,'(a)')'native_cell_index,kcs,minimum_volume_m3,minimum_time_s,minimum_step,strict_negative_steps,final_volume_m3'
  do nm=1,np
   write(u,'(i0,a,i0,2(a,es25.17),2(a,i0),a,es25.17)')nm,',',kcs(nm),',',ad_min(nm), &
    ',',ad_time(nm),',',ad_min_step(nm),',',ad_counts(nm),',',z_volume(nm)
  enddo
  close(u);close(ad_exchange_unit);close(ad_events_unit)
 end subroutine
