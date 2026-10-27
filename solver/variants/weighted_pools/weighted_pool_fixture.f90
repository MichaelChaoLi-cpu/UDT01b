program weighted_pool_fixture
 use sfincs_data
 use sfincs_continuity,only:udt_weighted_fill,udt_prepare_pumps,udt_limit_fluxes,pump_out,pump_delta,pump_sink_step,pump_source_step
 use iso_fortran_env,only:real64
 implicit none
 real(real64)::w(7),lim(7),a(7),cap,expect(3),before(3),after(3),total
 real::horizontal,dv
 integer::i,j,u
 ! Analytic cap, reallocation, storage saturation, spare capacity and deterministic ties.
 w=1.;lim=.5;call udt_weighted_fill(7,.5_real64,w,lim,a)
 if(maxval(abs(a-.5_real64/7))>1.e-14)error stop 'uniform allocation'
 w(1:3)=[1._real64,2._real64,3._real64];lim=0.;lim(1:3)=.5
 call udt_weighted_fill(7,.5_real64,w,lim,a)
 if(maxval(abs(a(1:3)-[1._real64,2._real64,3._real64]/12))>1.e-14)error stop 'weighted allocation'
 lim(1)=0.;call udt_weighted_fill(7,.5_real64,w,lim,a)
 if(maxval(abs(a(1:3)-[0._real64,.2_real64,.3_real64]))>1.e-14)error stop 'idle reassignment'
 lim(1)=.02_real64;call udt_weighted_fill(7,.5_real64,w,lim,a)
 if(maxval(abs(a(1:3)-[.02_real64,.192_real64,.288_real64]))>1.e-14)error stop 'capped reassignment'
 lim=.001;call udt_weighted_fill(7,.5_real64,w,lim,a)
 if(maxval(abs(a-lim))>1.e-14)error stop 'unused capacity'
 open(newunit=u,file='weighted_fixture_cases.csv',status='new')
 write(u,'(a)')'case,site,cap,weight,limit,amount'
 do j=1,200
  cap=.001_real64+mod(j,17)*.037_real64
  do i=1,7
   w(i)=.01_real64+abs(sin(real(j*7+i,real64)))
   lim(i)=abs(cos(real(j+i*13,real64)))*.4_real64
   if(mod(j+i,5)==0)lim(i)=0.
  enddo
  call udt_weighted_fill(7,cap,w,lim,a)
  if(sum(a)>cap+1.e-14.or.any(a<0).or.any(a>lim))error stop 'random bounds'
  do i=1,7
   write(u,'(i0,a,i0,4(a,es25.17))')j,',',i,',',cap,',',w(i),',',lim(i),',',a(i)
  enddo
 enddo
 close(u)
 ! Exercise the real sidecar parser, two isolated pools and trace/source-sink wiring.
 np=4;subgrid=.true.;crsgeo=.false.;use_quadtree=.false.
 allocate(kcs(4),z_volume(4),zs(4),subgrid_z_zmin(4));kcs=[1,1,1,2];subgrid_z_zmin=0.;zs=1.
 open(newunit=u,file='udt_pumps.dat',status='new');write(u,*)3
 write(u,*)1,4,.1_real64,.05_real64,5._real64
 write(u,*)2,4,.4_real64,.05_real64,5._real64
 write(u,*)3,4,.5_real64,.05_real64,5._real64;close(u)
 open(newunit=u,file='udt_pool_rules.dat',status='new');write(u,'(a)')'UDT_WEIGHTED_POOL_V1'
 write(u,*)2;write(u,*)1,.5_real64;write(u,*)2,.5_real64;write(u,*)3
 write(u,*)1,1,.5_real64;write(u,*)2,1,.5_real64;write(u,*)3,2,.5_real64;close(u)
 z_volume=[.1_real64,10._real64,10._real64,0._real64]
 call udt_prepare_pumps(1.,1._real64)
 if(pump_out(1)>.1_real64.or.abs(pump_out(1)+pump_out(2)-.5_real64)>1.e-14)error stop 'prior storage or pool'
 if(abs(pump_out(3)-.5_real64)>1.e-14.or.abs(pump_sink_step-1._real64)>1.e-14)error stop 'isolated pools'
 if(pump_source_step/=0.or.abs(sum(pump_delta)+pump_sink_step)>1.e-14)error stop 'sink ledger'
 zs(1)=0.;call udt_prepare_pumps(1.,2._real64)
 if(pump_out(1)/=0.or.abs(pump_out(2)-.5_real64)>1.e-14)error stop 'trigger redistribution'
 zs=1.;zs(4)=10.;call udt_prepare_pumps(1.,3._real64)
 if(any(pump_out/=0))error stop 'head blockade'
 zs=1.;z_volume=0.;call udt_prepare_pumps(1.,4._real64)
 if(any(pump_out/=0))error stop 'dry storage'
 npuv=2;nkcuv2=0;nref=1;use_qext=.false.;nsrcdrn=0;use_storage_volume=.false.
 infiltration=.false.;tspinup=0.;wavemaker=.false.;snapwave=.false.;nonhydrostatic=.false.
 precip=.true.;dx=10.;dy=10.
 allocate(netprcp(4),z_flags_iref(4),cell_area(1),q(3),uv(2),uvmean(0))
 allocate(z_index_uv_md(4),z_index_uv_mu(4),z_index_uv_nd(4),z_index_uv_nu(4))
 allocate(uv_index_z_nm(2),uv_index_z_nmu(2),uv_flags_dir(2),index_kcuv2(0))
 z_flags_iref=1;cell_area=100.;z_index_uv_md=[3,1,2,3];z_index_uv_mu=[1,2,3,3]
 z_index_uv_nd=3;z_index_uv_nu=3;uv_index_z_nm=[1,2];uv_index_z_nmu=[2,3];uv_flags_dir=0
 zs=1.;z_volume=[.1_real64,.2_real64,.3_real64,0._real64];netprcp=.001;q=[100.,100.,0.];uv=1.
 before=z_volume(1:3);call udt_limit_fluxes(1.,5._real64)
 do i=1,3
  horizontal=(q(z_index_uv_md(i))-q(z_index_uv_mu(i)))*10.
  dv=horizontal+netprcp(i)*100.;after(i)=before(i)+real(dv,real64)+pump_delta(i)
 enddo
 if(any(after<0))error stop 'dynamic pump/face positivity'
 total=sum(after)-sum(before)-sum(real(netprcp(1:3)*100.,real64))-pump_source_step+pump_sink_step
 if(abs(total)>1.e-6_real64)error stop 'dynamic pump/face conservation'
 print *,'PASS weighted water filling,200 vectors,production parser,trigger,head,storage,isolated pools,external sink'
end program
