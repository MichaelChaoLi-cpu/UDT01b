 subroutine negative_trace_initialize()
  logical::requested
  integer::u,v,ios,nm,idx,n,m,mask,code,d,ip,neighbor,j,group_codes(4),ng,counts(4)
  real::width
  real(real64)::meta(12)
  character(2048)::line
  inquire(file='udt_negative_trace.request',exist=requested)
  if(.not.requested)return
  open(newunit=u,file='udt_negative_trace.request',status='old')
  read(u,'(a)',iostat=ios)line
  call require(ios==0.and.trim(line)=='schema=negative-cell-face-v1','invalid negative trace request')
  close(u)
  call require(.not.use_quadtree.and.nref==1,'negative trace requires a regular one-level grid')
  call require(dx>0.and.dy>0,'invalid negative trace grid sizes')
  call require(allocated(z_index_z_n).and.allocated(z_index_z_m),'missing native geometry')
  call require(allocated(z_index_uv_md).and.allocated(z_index_uv_mu),'missing native west/east faces')
  call require(allocated(z_index_uv_nd).and.allocated(z_index_uv_nu),'missing native south/north faces')
  call require(allocated(uv_index_z_nm).and.allocated(uv_index_z_nmu),'missing native face endpoints')
  call require(allocated(subgrid_uv_zmin),'missing native face minimum')
  call require(ncuv==0,'combined refinement faces unsupported')
  call require(lbound(q,1)==1.and.ubound(q,1)==npuv+1,'requires native one-based flux layout')
  call require(q(npuv+1)==0,'nonzero native dummy-face flux')
  allocate(neg_previous(np),neg_min(np),neg_min_time(np),neg_first(np),neg_last(np))
  allocate(neg_steps(np),neg_entries(np),neg_exits(np),neg_min_step(np),neg_initial(np))
  allocate(neg_group(np),neg_geometry(12,np),neg_q(lbound(q,1):ubound(q,1)))
  neg_previous=z_volume;neg_min=z_volume;neg_min_time=origin;neg_min_step=0
  neg_steps=0;neg_entries=0;neg_exits=0;neg_first=-1;neg_last=-1
  neg_initial=kcs==1.and.z_volume<neg_threshold
  where(neg_initial)
   neg_first=origin;neg_last=origin
  endwhere
  open(newunit=u,file='udt_negative_native_map.csv',status='old',iostat=ios)
  call require(ios==0,'missing negative native map')
  read(u,'(a)',iostat=ios)line
  call require(ios==0.and.index(line,'native_cell_index,row_south_one_based,col_east_one_based,kcs,')==1, &
   'invalid negative native map header')
  do nm=1,np
   read(u,*,iostat=ios)idx,n,m,mask,code,meta
   call require(ios==0,'incomplete negative native map')
   call require(idx==nm.and.n==z_index_z_n(nm).and.m==z_index_z_m(nm).and.mask==kcs(nm), &
    'negative native map identity mismatch')
   call require(mask==1.or.mask==2,'negative trace supports interior and stage cells only')
   if(mask==2)then
    call require(any(code==neg_codes),'unknown boundary water group')
   else
    call require(code==0,'interior cell cannot own a prescribed water group')
   endif
   neg_group(nm)=code;neg_geometry(:,nm)=meta
  enddo
  read(u,'(a)',iostat=ios)line
  call require(ios<0,'unexpected extra negative native map row')
  close(u)
  ! Verify each existing interior face against native pointers and geometry.
  do nm=1,np
   if(kcs(nm)/=1)cycle
   do d=1,4
    call negative_face(nm,d,ip,neighbor,width)
    if(ip==0)cycle
    call require(all(ieee_is_finite(neg_geometry([d,d+4,d+8],nm))), &
     'missing geometry for an existing negative trace face')
    call require(neg_geometry(d,nm)>=0.and.neg_geometry(d,nm)<=real(width,real64)+1e-5_real64, &
     'invalid open face width')
    call require(abs(neg_geometry(d+8,nm)-real(subgrid_uv_zmin(ip),real64))<1e-5_real64, &
     'mapped face minimum differs from native subgrid face')
   enddo
  enddo
  open(newunit=neg_event_unit,file='udt_negative_crossings.csv',status='new')
  write(neg_event_unit,'(a)')'event_id,native_cell_index,kind,step_index,before_time_s,after_time_s,dt_s,'// &
   'previous_volume_m3,new_volume_m3,horizontal_applied_m3,rain_applied_m3,rain_postadd_increment_m3,'// &
   'four_face_sum_m3,horizontal_minus_four_faces_m3,storage_update_remainder_m3'
  open(newunit=neg_face_unit,file='udt_negative_crossing_faces.csv',status='new')
  write(neg_face_unit,'(a)')'event_id,native_cell_index,direction,face_exists,native_face_index,'// &
   'neighbor_native_cell,neighbor_mask,boundary_group_code,q_m2_s,native_transverse_width_m,'// &
   'signed_inward_volume_m3,open_width_m,road_width_m,mapped_face_minimum_m,native_face_minimum_m'
  open(newunit=v,file='udt_negative_cell_group_links.csv',status='new')
  write(v,'(a)')'native_cell_index,water_group_code,open_receiver_faces'
  do nm=1,np
   if(kcs(nm)/=1)cycle
   group_codes=0;counts=0;ng=0
   do d=1,4
    call negative_face(nm,d,ip,neighbor,width)
    if(ip==0)cycle
    if(kcs(neighbor)/=2.or.neg_geometry(d,nm)<=0)cycle
    code=neg_group(neighbor);j=1
    do while(j<=ng)
     if(group_codes(j)==code)exit
     j=j+1
    enddo
    if(j>ng)then
     ng=ng+1;group_codes(ng)=code
    endif
    counts(j)=counts(j)+1
   enddo
   if(ng==0)then
    write(v,'(i0,a)')nm,',0,0'
   else
    do j=1,ng
     write(v,'(i0,a,i0,a,i0)')nm,',',group_codes(j),',',counts(j)
    enddo
   endif
  enddo
  close(v)
  open(newunit=u,file='udt_negative_trace_receipt.txt',status='new')
  write(u,'(a)')'schema=negative-cell-face-v1'
  write(u,'(a,i0)')'native_cells=',np
  write(u,'(a)')'threshold_comparison=strict volume < -0.001 m3 at actual step endpoints'
  write(u,'(a)')'initial_below=summary only; no fictitious triggering face event'
  write(u,'(a)')'four_face_sum=individually rounded terms; retain difference from grouped continuity'
  write(u,'(a)')'group_links=direct open stage-cell faces; code0 means no direct open receiver'
  close(u)
  neg_event_id=0;neg_enabled=.true.
 end subroutine
 subroutine negative_face(nm,d,ip,neighbor,width)
  integer,intent(in)::nm,d
  integer,intent(out)::ip,neighbor
  real,intent(out)::width
  select case(d)
  case(1)
   ip=z_index_uv_md(nm);width=dy
  case(2)
   ip=z_index_uv_mu(nm);width=dy
  case(3)
   ip=z_index_uv_nd(nm);width=dx
  case(4)
   ip=z_index_uv_nu(nm);width=dx
  end select
  neighbor=0
  if(ip==npuv+1)then
   ip=0
   return
  endif
  call require(ip>=1.and.ip<=npuv,'invalid crossing face pointer')
  call require(uv_flags_dir(ip)==merge(0,1,d<=2),'face direction mismatch')
  if(d==1.or.d==3)then
   call require(uv_index_z_nmu(ip)==nm,'negative face endpoint mismatch')
   neighbor=uv_index_z_nm(ip)
  else
   call require(uv_index_z_nm(ip)==nm,'positive face endpoint mismatch')
   neighbor=uv_index_z_nmu(ip)
  endif
  call require(neighbor>=1.and.neighbor<=np,'invalid crossing neighbor')
 end subroutine
 subroutine negative_trace_begin()
  if(.not.neg_enabled)return
  call require(q(npuv+1)==0,'nonzero native dummy-face flux')
  neg_previous=z_volume;neg_q=q
 end subroutine
 subroutine negative_trace_finish(actual_time,dt)
  real(real64),intent(in)::actual_time
  real,intent(in)::dt
  integer::nm,d,ip,neighbor,nmask,code
  real::width,single_face
  real(real64)::face_values(4),total,horizontal_error,update_error
  logical::old_below,new_below
  character(4)::kind
  character(5),parameter::directions(4)=[character(5)::'west','east','south','north']
  if(.not.neg_enabled)return
  do nm=1,np
   if(kcs(nm)/=1)cycle
   if(z_volume(nm)<neg_min(nm))then
    neg_min(nm)=z_volume(nm);neg_min_time(nm)=actual_time;neg_min_step(nm)=step
   endif
   old_below=neg_previous(nm)<neg_threshold;new_below=z_volume(nm)<neg_threshold
   if(new_below)then
    neg_steps(nm)=neg_steps(nm)+1
    if(neg_first(nm)<0)neg_first(nm)=actual_time
    neg_last(nm)=actual_time
   endif
   if(old_below.eqv.new_below)cycle
   if(new_below)then
    kind='down';neg_entries(nm)=neg_entries(nm)+1
   else
    kind='up';neg_exits(nm)=neg_exits(nm)+1
   endif
   neg_event_id=neg_event_id+1;face_values=0
   do d=1,4
    call negative_face(nm,d,ip,neighbor,width)
    single_face=0.0
    if(ip>0)single_face=neg_q(ip)*width*dt
    if(d==2.or.d==4)single_face=-single_face
    face_values(d)=real(single_face,real64)
   enddo
   total=sum(face_values);horizontal_error=horizontal_term(nm)-total
   update_error=z_volume(nm)-neg_previous(nm)-horizontal_term(nm)-rain_increment(nm)
   write(neg_event_unit,'(i0,a,i0,a,a,a,i0,11(a,es25.17))') &
    neg_event_id,',',nm,',',trim(kind),',',step,',',live_time-real(dt,real64),',',actual_time,',',real(dt,real64), &
    ',',neg_previous(nm),',',real(z_volume(nm),real64),',',horizontal_term(nm),',',rain_term(nm), &
    ',',rain_increment(nm),',',total,',',horizontal_error,',',update_error
   do d=1,4
    call negative_face(nm,d,ip,neighbor,width)
    nmask=0;code=0
    if(neighbor>0)then
     nmask=kcs(neighbor)
     if(nmask==2)code=neg_group(neighbor)
    endif
    if(ip>0)then
     write(neg_face_unit,'(i0,a,i0,3a,l1,4(a,i0),7(a,es25.17))') &
      neg_event_id,',',nm,',',trim(directions(d)),',',ip>0,',',ip,',',neighbor,',',nmask,',',code, &
      ',',real(neg_q(ip),real64),',',real(width,real64),',',face_values(d),',',neg_geometry(d,nm), &
      ',',neg_geometry(d+4,nm),',',neg_geometry(d+8,nm),',',real(subgrid_uv_zmin(ip),real64)
    else
     ! No geometric face: zero sentinel participates in continuity, geometry remains absent.
     write(neg_face_unit,'(i0,a,i0,3a,l1,4(a,i0),3(a,es25.17),a)') &
      neg_event_id,',',nm,',',trim(directions(d)),',',.false.,',',0,',',0,',',0,',',0, &
      ',',0.0_real64,',',real(width,real64),',',face_values(d),',,,,'
    endif
   enddo
  enddo
 end subroutine
 function negative_optional_time(value)result(text)
  real(real64),intent(in)::value
  character(25)::text
  text=''
  if(value>=0)write(text,'(es25.17)')value
 end function
 subroutine negative_trace_close()
  integer::u,nm
  if(.not.neg_enabled)return
  open(newunit=u,file='udt_negative_cell_summary.csv',status='new')
  write(u,'(a)')'native_cell_index,initial_below_threshold,first_below_endpoint_s,last_below_endpoint_s,'// &
   'minimum_volume_m3,minimum_time_s,minimum_step,negative_poststep_count,down_crossings,up_crossings,final_volume_m3'
  do nm=1,np
   if(kcs(nm)/=1)cycle
   write(u,'(i0,a,l1,4a,2(a,es25.17),4(a,i0),a,es25.17)') &
    nm,',',neg_initial(nm),',',trim(negative_optional_time(neg_first(nm))), &
    ',',trim(negative_optional_time(neg_last(nm))),',',neg_min(nm),',',neg_min_time(nm), &
    ',',neg_min_step(nm),',',neg_steps(nm),',',neg_entries(nm),',',neg_exits(nm),',',real(z_volume(nm),real64)
  enddo
  close(u);close(neg_event_unit);close(neg_face_unit)
  deallocate(neg_previous,neg_min,neg_min_time,neg_first,neg_last,neg_geometry,neg_q,neg_group)
  deallocate(neg_steps,neg_entries,neg_exits,neg_min_step,neg_initial)
  neg_enabled=.false.
 end subroutine
