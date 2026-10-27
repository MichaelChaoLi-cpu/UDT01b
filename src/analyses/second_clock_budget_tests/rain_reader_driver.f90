program rain_reader_test
 use sfincs_data
 use extracted_rain_reader
 implicit none
 integer::i,j,k
 call get_command_argument(1,netamprfile)
 trefstr='20200101 000000'
 call read_netcdf_ampr_data()
 ! All values are read and row-flipped by the actual patched precipitation reader.
 write(*,*)ampr_nt,ampr_nrows,ampr_ncols
 do i=1,ampr_nt
  write(*,'(es25.17)')real(ampr_times(i),8)
 enddo
 do i=1,ampr_nt
  do j=1,ampr_nrows
   do k=1,ampr_ncols
    write(*,'(es25.17)')real(ampr_pr(i,j,k),8)
   enddo
  enddo
 enddo
 write(*,'(4(es25.17,1x))')real(ampr_x_llcorner,8),real(ampr_y_llcorner,8),real(ampr_dx,8),real(ampr_dy,8)
end program
