! TEST ONLY: limited nf90-compatible signatures backed by actual NetCDF C I/O.
! This is not the production Fortran NetCDF library and does not certify its build.
module netcdf
 use iso_c_binding
 implicit none
 integer,parameter::NF90_CLOBBER=0
 interface
  function c_open(path,mode,id)bind(c,name='nc_open')result(e)
   import
   character(c_char)::path(*)
   integer(c_int),value::mode
   integer(c_int)::id,e
  end function
  function c_dim(id,name,dim)bind(c,name='nc_inq_dimid')result(e)
   import
   integer(c_int),value::id
   character(c_char)::name(*)
   integer(c_int)::dim,e
  end function
  function c_var(id,name,var)bind(c,name='nc_inq_varid')result(e)
   import
   integer(c_int),value::id
   character(c_char)::name(*)
   integer(c_int)::var,e
  end function
  function c_len(id,dim,len)bind(c,name='udt_nc_dimlen')result(e)
   import
   integer(c_int),value::id,dim
   integer(c_int)::len,e
  end function
  function c_get_double(id,var,data)bind(c,name='nc_get_var_double')result(e)
   import
   integer(c_int),value::id,var
   real(c_double)::data(*)
   integer(c_int)::e
  end function
  function c_get_float(id,var,data)bind(c,name='nc_get_var_float')result(e)
   import
   integer(c_int),value::id,var
   real(c_float)::data(*)
   integer(c_int)::e
  end function
  function c_slice(id,var,data,start,count)bind(c,name='udt_nc_slice')result(e)
   import
   integer(c_int),value::id,var
   real(c_float)::data(*)
   integer(c_int)::start(*),count(*),e
  end function
  function c_att(id,var,name,text,cap)bind(c,name='udt_nc_att')result(e)
   import
   integer(c_int),value::id,var,cap
   character(c_char)::name(*),text(*)
   integer(c_int)::e
  end function
  function c_close(id)bind(c,name='nc_close')result(e)
   import
   integer(c_int),value::id
   integer(c_int)::e
  end function
 end interface
 interface nf90_get_var
  module procedure get_double,get_float,get_cube
 end interface
contains
 integer function nf90_open(path,mode,id)result(e)
  character(*)::path
  integer::mode,id
  e=c_open(trim(path)//c_null_char,mode,id)
 end function
 integer function nf90_inq_dimid(id,name,dim)result(e)
  integer::id,dim
  character(*)::name
  e=c_dim(id,trim(name)//c_null_char,dim)
 end function
 integer function nf90_inq_varid(id,name,var)result(e)
  integer::id,var
  character(*)::name
  e=c_var(id,trim(name)//c_null_char,var)
 end function
 integer function nf90_inquire_dimension(id,dim,len)result(e)
  integer::id,dim,len
  e=c_len(id,dim,len)
 end function
 integer function get_double(id,var,data)result(e)
  integer::id,var
  real(c_double)::data(:)
  e=c_get_double(id,var,data)
 end function
 integer function get_float(id,var,data)result(e)
  integer::id,var
  real(c_float)::data(:)
  e=c_get_float(id,var,data)
 end function
 integer function get_cube(id,var,data,start,count)result(e)
  integer::id,var,start(3),count(3)
  real(c_float)::data(:,:,:)
  e=c_slice(id,var,data,start,count)
 end function
 integer function nf90_get_att(id,var,name,text)result(e)
  integer::id,var
  character(*)::name,text
  e=c_att(id,var,trim(name)//c_null_char,text,len(text))
 end function
 integer function nf90_close(id)result(e)
  integer::id
  e=c_close(id)
 end function
end module
module sfincs_log
contains
 subroutine write_log(text,level)
  character(*)::text
  integer::level
 end subroutine
end module
