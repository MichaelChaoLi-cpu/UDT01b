! Prescribed production boundary-routine regression; no solver initialization or time stepping.
program closed_boundary_fixture
 use sfincs_data
 use sfincs_boundaries,only:update_boundary_fluxes
 implicit none
 integer :: mode,i
 real :: fixture_dt
 real*8 :: fixture_time
 np=6;npuv=3;nkcuv2=3;bndtype=1;btfilter=60.;btrelax=3600.;g=9.81;huthresh=.01
 store_maximum_waterlevel=.false.;store_t_zsmax=.false.;subgrid_nlevels=2
 allocate(index_kcuv2(3),nmikcuv2(3),nmbkcuv2(3),ibkcuv2(3),kcuv(3),kcs(6),ibuvdir(3))
 allocate(zs(6),z_volume(6),q(3),uv(3),uvmean(3),zb(6),zbuv(3),zsb(3),zsb0(3),zsmax(6))
 allocate(subgrid_uv_zmin(3),subgrid_uv_zmax(3),subgrid_uv_havg(2,3),subgrid_uv_havg_zmax(3),subgrid_z_zmin(6))
 index_kcuv2=[1,2,3];nmikcuv2=[1,3,5];nmbkcuv2=[2,4,6];ibkcuv2=[1,2,3]
 kcuv=[0,2,3];kcs=[1,2,1,2,1,2];ibuvdir=1
 zb=0.;zbuv=0.;subgrid_uv_zmin=0.;subgrid_uv_zmax=2.;subgrid_uv_havg(1,:)=0.
 subgrid_uv_havg(2,:)=2.;subgrid_uv_havg_zmax=0.;subgrid_z_zmin=0.
 do mode=0,1
  subgrid=(mode==1);zs=1.;z_volume=100.;zsb=1.2;zsb0=1.2;zsmax=1.;q=3.;uv=4.;uvmean=5.
  do i=1,3
   fixture_dt=.4;fixture_time=real(i,8)*real(fixture_dt,8)
   call update_boundary_fluxes(fixture_dt,fixture_time)
   write(*,'(2(i2,1x),9(es24.16,1x))')mode,i,q,uv,uvmean
  enddo
 enddo
end program
