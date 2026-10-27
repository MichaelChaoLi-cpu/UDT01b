#!/bin/sh
set -eu
trap 'status=$?; if [ "$status" -ne 0 ]; then tail -n 70 /opt/udt-build/receipts/configure.log /opt/udt-build/receipts/make.log /opt/udt-build/receipts/install.log 2>/dev/null; fi' EXIT
mkdir -p /opt/udt-build/receipts
{ uname -a; gfortran --version; gcc --version; autoconf --version; automake --version; } > /opt/udt-build/receipts/toolchain.txt
cp /usr/src/sfincs/src/Makefile.am /opt/udt-build/receipts/Makefile.am
dpkg-query -W > /opt/udt-build/receipts/packages.tsv
# Full upstream dependency build. 'make check' and its possible flow-model targets are intentionally not called.
autoreconf -ivf > /opt/udt-build/receipts/autoreconf.log 2>&1
./configure --disable-openacc > /opt/udt-build/receipts/configure.log 2>&1
make -j1 > /opt/udt-build/receipts/make.log 2>&1
make install > /opt/udt-build/receipts/install.log 2>&1
ldconfig
cp config.log config.status /opt/udt-build/receipts/
cp third_party_open/netcdf/netcdf-fortran-4.6.1/config.log /opt/udt-build/receipts/netcdf-fortran-config.log
nf-config --all > /opt/udt-build/receipts/nf-config.txt
nc-config --all > /opt/udt-build/receipts/nc-config.txt
# A real Fortran NetCDF module/link/I/O roundtrip, with no solver initialization.
gfortran $(nf-config --fflags) /opt/udt-build/netcdf_link_probe.f90 $(nf-config --flibs) -o /opt/udt-build/netcdf_link_probe
/opt/udt-build/netcdf_link_probe > /opt/udt-build/receipts/netcdf-link-probe.txt
ldd /usr/local/bin/sfincs > /opt/udt-build/receipts/executable-ldd.txt
ldd /usr/local/lib/libsfincs.so > /opt/udt-build/receipts/library-ldd.txt
nm -D --defined-only /usr/local/lib/libsfincs.so > /opt/udt-build/receipts/exported-symbols.txt
sha256sum /usr/local/bin/sfincs /usr/local/lib/libsfincs.so.0.0.0 /usr/local/lib/libnetcdff.so.7.2.0 > /opt/udt-build/receipts/binaries.sha256
! grep -q 'not found' /opt/udt-build/receipts/executable-ldd.txt
! grep -q 'not found' /opt/udt-build/receipts/library-ldd.txt
