! Scoped Galibier precipitation transport. Does not replace unrelated date readers.
module udt_rain_clock
 use, intrinsic :: iso_fortran_env, only: real32, real64, int64
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use sfincs_date, only: julian_date
 implicit none
 private
 public :: udt_decode_rain_seconds, udt_write_rain_trace
contains
 subroutine parse_date(text, y,m,d,h,n,s)
  character(*),intent(in) :: text
  integer,intent(out) :: y,m,d,h,n,s
  integer :: ios,days(12)
  if(len_trim(text)/=19) error stop 'Rain clock: invalid reference length'
  if(text(5:5)/='-' .or. text(8:8)/='-' .or. text(11:11)/=' ' .or. &
     text(14:14)/=':' .or. text(17:17)/=':') error stop 'Rain clock: invalid reference format'
  read(text,'(i4,1x,i2,1x,i2,1x,i2,1x,i2,1x,i2)',iostat=ios)y,m,d,h,n,s
  if(ios/=0) error stop 'Rain clock: invalid reference fields'
  if(y<1600.or.y>9999.or.m<1.or.m>12.or.h<0.or.h>23.or.n<0.or.n>59.or.s<0.or.s>59) &
   error stop 'Rain clock: invalid reference range'
  days=[31,28,31,30,31,30,31,31,30,31,30,31]
  if(mod(y,4)==0.and.(mod(y,100)/=0.or.mod(y,400)==0))days(2)=29
  if(d<1.or.d>days(m))error stop 'Rain clock: invalid calendar date'
 end subroutine
 subroutine udt_decode_rain_seconds(raw,units,reference,seconds)
  real(real64),intent(in) :: raw(:)
  character(*),intent(in) :: units,reference
  real(real32),intent(out) :: seconds(:)
  integer :: y,m,d,h,n,s,yr,mr,dr,hr,nr,sr,i,ios
  integer(int64) :: offset,k
  real(real64) :: x
  character(19) :: ref_iso
  character(41) :: normalized_units
  character(:),allocatable :: tail
  ! This version deliberately supports the existing minutes/whole-second profile.
  if(size(raw)<2.or.size(raw)/=size(seconds))error stop 'Rain clock: invalid vector size'
  normalized_units=units
  ! Existing control files use a date-only midnight reference; preserve raw attribute in the trace.
  if(len_trim(units)==24)normalized_units=trim(units)//' 00:00:00'
  if(len_trim(normalized_units)<33)error stop 'Rain clock: incomplete time units'
  if(normalized_units(1:14)/='minutes since ')error stop 'Rain clock: only minutes since is qualified'
  tail=trim(normalized_units(34:))
  if(tail/=''.and.tail/='.0'.and.tail/=' +0000'.and.tail/='.0 +0000') &
   error stop 'Rain clock: timezone or fractional epoch unsupported'
  call parse_date(normalized_units(15:33),y,m,d,h,n,s)
  if(len_trim(reference)/=15)error stop 'Rain clock: invalid solver reference'
  read(reference,'(i4,2i2,1x,3i2)',iostat=ios)yr,mr,dr,hr,nr,sr
  if(ios/=0.or.reference(9:9)/=' ')error stop 'Rain clock: invalid solver reference'
  write(ref_iso,'(i4.4,a,i2.2,a,i2.2,a,i2.2,a,i2.2,a,i2.2)')yr,'-',mr,'-',dr,' ',hr,':',nr,':',sr
  call parse_date(ref_iso,yr,mr,dr,hr,nr,sr)
  offset=int(julian_date(y,m,d)-julian_date(yr,mr,dr),int64)*86400_int64 &
         +int(h*3600+n*60+s-hr*3600-nr*60-sr,int64)
  do i=1,size(raw)
   if(.not.ieee_is_finite(raw(i)))error stop 'Rain clock: nonfinite time'
   x=raw(i)*60.0_real64+real(offset,real64)
   ! All integer seconds in this range are exact in the downstream real*4 clock.
   if(abs(x)>16777216.0_real64)error stop 'Rain clock: downstream real32 range exceeded'
   k=nint(x,int64)
   if(abs(x-real(k,real64))>5.0e-7_real64)error stop 'Rain clock: subsecond time not qualified'
   seconds(i)=real(k,real32)
   if(i>1)then
    if(seconds(i)<=seconds(i-1))error stop 'Rain clock: duplicate or reversed decoded time'
   endif
  enddo
 end subroutine
 subroutine udt_write_rain_trace(raw,units,seconds)
  real(real64),intent(in)::raw(:)
  real(real32),intent(in)::seconds(:)
  character(*),intent(in)::units
  integer::u,i
  open(newunit=u,file='udt_rain_decode.csv',status='new',action='write')
  write(u,'(a)')'# raw units: '//trim(units)
  write(u,'(a)')'raw_time_minutes,decoded_solver_seconds'
  do i=1,size(raw)
   write(u,'(es25.17,a,es25.17)')raw(i),',',real(seconds(i),real64)
  enddo
  close(u)
 end subroutine
end module
