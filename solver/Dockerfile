FROM deltares/sfincs-cpu@sha256:4301bffc74f3cb3f334be5d21aa75dfe5742a16c9838942cdfc2f870859fb404 AS builder
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gfortran autoconf automake libtool pkg-config patch && rm -rf /var/lib/apt/lists/*
COPY source /usr/src/sfincs
COPY SOURCE_BINDING.json /opt/udt-build/SOURCE_BINDING.json
COPY build-and-check.sh netcdf_link_probe.f90 /opt/udt-build/
WORKDIR /usr/src/sfincs
ENV FCFLAGS="-cpp -fopenmp -O3 -fallow-argument-mismatch -ffp-contract=off" \
    FFLAGS="-cpp -fopenmp -O3 -fallow-argument-mismatch -ffp-contract=off"
RUN /bin/sh /opt/udt-build/build-and-check.sh
COPY loader_probe.c production_rain_driver.f90 production_budget_driver.f90 /opt/udt-build/
RUN mkdir -p /opt/udt-build/modules && cp src/*.mod /opt/udt-build/modules/ && \
    gcc /opt/udt-build/loader_probe.c -ldl -o /opt/udt-build/loader_probe && \
    gfortran -fopenmp -I/opt/udt-build/modules -I/usr/local/include /opt/udt-build/production_rain_driver.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/production_rain_reader && \
    gfortran -fopenmp -I/opt/udt-build/modules -I/usr/local/include /opt/udt-build/production_budget_driver.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/production_budget_fixture && \
    /opt/udt-build/loader_probe > /opt/udt-build/receipts/loader-probe.txt

FROM deltares/sfincs-cpu@sha256:4301bffc74f3cb3f334be5d21aa75dfe5742a16c9838942cdfc2f870859fb404 AS runtime
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/udt-build /opt/udt-build
RUN ldconfig && ldd /usr/local/bin/sfincs > /opt/udt-build/runtime-executable-ldd.txt && \
    ldd /usr/local/lib/libsfincs.so > /opt/udt-build/runtime-library-ldd.txt && \
    ! grep -q 'not found' /opt/udt-build/runtime-executable-ldd.txt && \
    ! grep -q 'not found' /opt/udt-build/runtime-library-ldd.txt && \
    /opt/udt-build/netcdf_link_probe && /opt/udt-build/loader_probe
LABEL org.opencontainers.image.revision="1f1d8286520a8709b2e513854dff18a3616583db" \
      org.opencontainers.image.title="UDT01b source-equivalent CPU research solver" \
      org.opencontainers.image.version="second-clock-budget-interface-v1" \
      udt.validation="build-link-and-isolated-interfaces-only-no-hydrodynamic-run" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"
WORKDIR /data
CMD ["sfincs"]
