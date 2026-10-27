program reader_fixture
 use sfincs_data
 use sfincs_ncinput
 use sfincs_log,only:open_log,close_log
 implicit none
 call get_command_argument(1,netamprfile)
 trefstr='20240821 060000'
 call open_log()
 call read_netcdf_ampr_data()
 call close_log()
 if(ampr_nt/=4.or.ampr_nrows/=4.or.ampr_ncols/=4)error stop 'reader: dimensions'
 if(any(ampr_times/=[0.,1.,300.,32400.]))error stop 'reader: exact date/seconds'
 if(ampr_x_llcorner/=-11370..or.ampr_y_llcorner/=-42450..or.ampr_dx/=10..or.ampr_dy/=10.) &
  error stop 'reader: coordinates'
 if(any(ampr_pr(:,1,1)/=[13.,113.,213.,313.]))error stop 'reader: row flip'
 if(any(ampr_pr(:,4,4)/=[4.,104.,204.,304.]))error stop 'reader: x order'
 print *,'PASS actual NetCDF reader, non-midnight epoch, seconds, rows, coordinates, units'
end program
