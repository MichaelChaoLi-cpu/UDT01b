! Real Fortran NetCDF dependency probe; no SFINCS imports or solver calls.
program netcdf_link_probe
 use netcdf
 use iso_fortran_env,only:real64
 implicit none
 integer::nc,dim,var,status
 real(real64)::raw(3),back(3)
 raw=[0.0_real64,3601.0_real64/60.0_real64,14400.0_real64/60.0_real64]
 call check(nf90_create('/tmp/udt_netcdf_link_probe.nc',nf90_clobber,nc))
 call check(nf90_def_dim(nc,'time',3,dim))
 call check(nf90_def_var(nc,'time',nf90_double,[dim],var))
 call check(nf90_put_att(nc,var,'units','minutes since 2020-01-01 00:00:00'))
 call check(nf90_enddef(nc));call check(nf90_put_var(nc,var,raw));call check(nf90_close(nc))
 call check(nf90_open('/tmp/udt_netcdf_link_probe.nc',nf90_nowrite,nc))
 call check(nf90_inq_varid(nc,'time',var));call check(nf90_get_var(nc,var,back));call check(nf90_close(nc))
 if(any(raw/=back))error stop 'Fortran NetCDF real64 roundtrip failed'
 write(*,'(a)')'PASS: production Fortran NetCDF real64 module/link/I/O roundtrip; no solver called'
contains
 subroutine check(status)
  integer,intent(in)::status
  if(status/=nf90_noerr)then
   write(*,*)trim(nf90_strerror(status))
   error stop 1
  endif
 end subroutine
end program
