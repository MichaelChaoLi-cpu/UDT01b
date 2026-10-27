program pump_fixture
 use sfincs_data
 use sfincs_continuity,only:udt_prepare_pumps,udt_limit_fluxes,pump_delta,pump_out,pump_sink_step,pump_source_step
 use iso_fortran_env,only:real64
 implicit none
 integer::i,u
 real::horizontal,dv
 real(real64)::before(3),after(3),total,dt8
 np=4;npuv=2;nkcuv2=0;nref=1;subgrid=.true.;crsgeo=.false.;use_quadtree=.false.
 use_qext=.false.;nsrcdrn=0;use_storage_volume=.false.;infiltration=.false.;tspinup=0.
 wavemaker=.false.;snapwave=.false.;nonhydrostatic=.false.;precip=.true.;dx=10.;dy=10.
 allocate(kcs(4),z_volume(4),zs(4),subgrid_z_zmin(4),netprcp(4),z_flags_iref(4),cell_area(1),q(3),uv(2),uvmean(0))
 allocate(z_index_uv_md(4),z_index_uv_mu(4),z_index_uv_nd(4),z_index_uv_nu(4))
 allocate(uv_index_z_nm(2),uv_index_z_nmu(2),uv_flags_dir(2),index_kcuv2(0))
 kcs=[1,1,1,2];z_flags_iref=1;cell_area=100.;subgrid_z_zmin=0.;zs=1.
 z_index_uv_md=[3,1,2,3];z_index_uv_mu=[1,2,3,3];z_index_uv_nd=3;z_index_uv_nu=3
 uv_index_z_nm=[1,2];uv_index_z_nmu=[2,3];uv_flags_dir=0
 open(newunit=u,file='udt_pumps.dat',status='new');write(u,*)3
 write(u,*)1,2,1._real64,.05_real64,5._real64
 write(u,*)1,4,1._real64,.05_real64,5._real64
 write(u,*)2,4,2._real64,.05_real64,5._real64
 close(u)
 ! Multiple pumps sharing an almost-dry donor plus simultaneous face outflow.
 z_volume=[.1_real64,.2_real64,0._real64,0._real64];netprcp=.001;q=[100.,100.,0.];uv=1.
 before=z_volume(1:3);call udt_limit_fluxes(1.,1._real64)
 do i=1,3
  horizontal=((q(z_index_uv_md(i))-q(z_index_uv_mu(i)))*10.)
  dv=horizontal+netprcp(i)*100.;after(i)=before(i)+real(dv,real64)+pump_delta(i)
 enddo
 if(any(after<0))error stop 'Pump fixture:negative coupled updated volume'
 if(pump_out(1)>.1_real64.or.pump_out(2)>.2_real64)error stop 'Pump fixture:spent incoming storage'
 total=sum(after)-sum(before)-sum(real(netprcp(1:3)*100.,real64))-pump_source_step+pump_sink_step
 if(abs(total)>1.e-6_real64)error stop 'Pump fixture:water not conserved'
 if(pump_sink_step<=0.or.pump_source_step<=0)error stop 'Pump fixture:transfer not exercised'
 ! Dry donors and blocked head must have exactly zero abstraction.
 z_volume=0.;zs=1.;call udt_prepare_pumps(1.,2._real64)
 if(any(pump_delta/=0))error stop 'Pump fixture:dry abstraction'
 z_volume=10.;zs=0.;zs(4)=10.;call udt_prepare_pumps(1.,3._real64)
 if(pump_out(1)/=0.or.pump_out(2)/=0)error stop 'Pump fixture:trigger ignored'
 zs=1.;zs(4)=10.;call udt_prepare_pumps(1.,4._real64)
 if(pump_out(2)/=0.or.pump_delta(4)/=0)error stop 'Pump fixture:head ignored'
 if(pump_source_step<=0.or.abs(pump_source_step-pump_sink_step)>1.e-12)error stop 'Pump fixture:internal pair ledger'
 print *,'PASS coupled face/pump positivity,shared donor,dry,trigger,head,internal/external conservation'
end program
