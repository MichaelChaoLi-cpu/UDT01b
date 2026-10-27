"""Version an origin-syntax repair discovered before the first hydraulic step."""
from pathlib import Path
import json,sys,hashlib,difflib,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,BUILD
from prepare_sfincs_cpu_build import sha,dump,COMMIT
TAG='udt01b/sfincs-cpu:galibier-second-clock-budget-v2'
def main(root=ROOT):
 root=Path(root);out=root/OUT/'compatibility_v2';out.mkdir(exist_ok=True);context=out/'context';context.mkdir(exist_ok=True)
 prior=root/'data/processed/three_ward_second_clock_budget_v1';lines=[]
 for p in sorted((prior/'patched').iterdir()):
  old=prior/'base'/p.name;new=p
  if p.name=='udt_rain_clock.f90':new=root/'src/preprocessing/sfincs_second_clock_budget_v2/udt_rain_clock.f90'
  lines+=list(difflib.unified_diff(old.read_text().splitlines(True)if old.exists()else[],new.read_text().splitlines(True),fromfile='a/source/src/'+p.name,tofile='b/source/src/'+p.name))
 patch=out/'sfincs_second_clock_budget_v2.patch';patch.write_text(''.join(lines))
 module=root/'src/preprocessing/sfincs_second_clock_budget_v2/udt_rain_clock.f90';shutil.copyfile(module,context/module.name)
 builder=json.loads(subprocess.run(['docker','image','inspect','udt01b/sfincs-cpu:builder-v1'],capture_output=True,text=True,check=True).stdout)[0]
 base=json.loads((root/BUILD/'solver_image_binding.json').read_text())
 binding={'source_commit':COMMIT,'base_runtime_image_id':base['local_image_id'],'base_builder_image_id':builder['Id'],'base_interface_patch_sha256':base['interface_patch_sha256'],'new_interface_patch_sha256':sha(patch),'rain_module_sha256':sha(module),'change':'Accept exact minutes-since date-only midnight origin internally; retain raw file attributes and all source curves','source_equations_or_geometry_changed':False,'discovery':'first dry initialization rejected before step0; failed logs preserved','target_tag':TAG}
 dump(context/'COMPAT_BINDING.json',binding);dump(out/'source_binding.json',binding)
 builder_ref='udt01b/sfincs-cpu:builder-v1';runtime_ref='udt01b/sfincs-cpu:galibier-second-clock-budget-v1'
 dockerfile=f'''FROM {builder_ref} AS compat_builder
COPY udt_rain_clock.f90 /usr/src/sfincs/src/udt_rain_clock.f90
COPY COMPAT_BINDING.json /opt/udt-build/COMPAT_BINDING.json
RUN cd /usr/src/sfincs/src && make -j1 > /opt/udt-build/receipts/compat-make.log 2>&1 && make install > /opt/udt-build/receipts/compat-install.log 2>&1 && ldconfig && cp udt_rain_clock.mod /opt/udt-build/modules/ && sha256sum /usr/local/bin/sfincs /usr/local/lib/libsfincs.so.0.0.0 /usr/local/lib/libnetcdff.so.7.2.0 > /opt/udt-build/receipts/compat-binaries.sha256
FROM {runtime_ref}
COPY --from=compat_builder /usr/local /usr/local
COPY --from=compat_builder /opt/udt-build /opt/udt-build
RUN ldconfig && /opt/udt-build/netcdf_link_probe && /opt/udt-build/loader_probe
LABEL org.opencontainers.image.version="second-clock-budget-v2-date-origin" udt.compatibility="date-only-origin-no-curve-change"
WORKDIR /data
CMD ["sfincs"]
'''
 (context/'Dockerfile').write_text(dockerfile)
 print(context)
if __name__=='__main__':main()
