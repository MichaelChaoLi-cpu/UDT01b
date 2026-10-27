"""Version a conservative donor flux limit on the frozen closure-corrected CPU model."""
from pathlib import Path
import sys,json,shutil,difflib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump,COMMIT
OUT='data/processed/three_ward_conservative_positivity_v2'
REPORT='data/exp/data-preprocessing/three_ward_conservative_positivity_v2'
PRIOR='data/processed/three_ward_boundary_closure_fix_v1'
PRIOR_REPORT='data/exp/data-preprocessing/three_ward_boundary_closure_fix'
INTERFACE='data/processed/three_ward_negative_trace_interface_v1'
CASES=['dry_raised','rain60_raised']
TAG='udt01b/sfincs-cpu:galibier-conservative-positivity-v2'
def main():
 out=ROOT/OUT;report=ROOT/REPORT;context=out/'context'
 if(out/'source_binding.json').exists():raise FileExistsError('Retain prepared version')
 context.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
 prior=ROOT/PRIOR;old=json.loads((prior/'source_binding.json').read_text());solver=json.loads((prior/'solver_binding.json').read_text())
 for name in ['sfincs_boundaries.f90','udt_clock_budget.f90','udt_rain_clock.f90']:shutil.copyfile(prior/'context'/name,context/name)
 source=ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src/sfincs_continuity.f90';before=source.read_text();routine=(ROOT/'src/preprocessing/sfincs_conservative_positivity.f90.inc').read_text()
 declarations=' real*8,allocatable,private :: positivity_factor(:)\n real,allocatable,private :: positivity_face_ratio(:)\n integer,private :: positivity_unit\n logical,private :: positivity_started=.false.\n'
 after=before.replace('contains\n',declarations+'contains\n'+routine,1);assert after.count('   call udt_budget_begin()')==1
 after=after.replace('   call udt_budget_begin()','   call udt_limit_fluxes(dt,t)\n   call udt_budget_begin()')
 (context/source.name).write_text(after);delta=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/source/src/'+source.name,tofile='b/source/src/'+source.name));(out/'positivity.patch').write_text(delta);(out/'full_solver.patch').write_bytes((prior/'full_solver.patch').read_bytes()+delta.encode())
 fixture=ROOT/'src/preprocessing/positivity_fixture.f90';shutil.copyfile(fixture,context/fixture.name)
 info={'source_commit':COMMIT,'base_builder_image_id':old['base_builder_image_id'],'prior_runtime_image_id':solver['image_id'],'prior_patch_sha256':solver['patch_sha256'],'full_patch_sha256':sha(out/'full_solver.patch'),'positivity_patch_sha256':sha(out/'positivity.patch'),'continuity_sha256':sha(context/source.name),'source_include_sha256':sha(ROOT/'src/preprocessing/sfincs_conservative_positivity.f90.inc'),'fixture_sha256':sha(fixture),'roundoff_guard':'64 times float32 epsilon and directed-toward-zero flux rounding including subnormals; retained storage, never added water','available_volume':'prior signed native volume plus native rounded nonnegative rainfall; simultaneous inflow is not pre-spent','scope':'regular projected CPU subgrid; shared native face scaled once by interior donor; external inflow unrestricted; no signed-storage clipping','human_record':'MILI-D-20261015-014'}
 dump(out/'source_binding.json',info);dump(context/'POSITIVITY_BINDING.json',info)
 docker='''FROM udt01b/sfincs-cpu:builder-v1 AS positivity_builder
COPY sfincs_boundaries.f90 sfincs_continuity.f90 udt_clock_budget.f90 udt_rain_clock.f90 /usr/src/sfincs/src/
COPY POSITIVITY_BINDING.json /opt/udt-build/POSITIVITY_BINDING.json
COPY positivity_fixture.f90 /opt/udt-build/positivity_fixture.f90
RUN cd /usr/src/sfincs/src && make -j1 > /opt/udt-build/receipts/positivity-make.log 2>&1 && make install > /opt/udt-build/receipts/positivity-install.log 2>&1 && ldconfig && gfortran -fopenmp -I. /opt/udt-build/positivity_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/positivity_fixture
FROM udt01b/sfincs-cpu:galibier-closed-boundary-v1
COPY --from=positivity_builder /usr/local /usr/local
COPY --from=positivity_builder /opt/udt-build /opt/udt-build
RUN ldconfig && /opt/udt-build/netcdf_link_probe && /opt/udt-build/loader_probe
LABEL org.opencontainers.image.version="conservative-donor-limit-v2"
WORKDIR /data
CMD ["sfincs"]
'''
 (context/'Dockerfile').write_text(docker)
 protected={}
 for p in [prior/'freeze.json',prior/'run_manifest.json',prior/'qualification.json',ROOT/PRIOR_REPORT/'summary.json',ROOT/PRIOR_REPORT/'reference_admission_assessment.json',ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json',ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json',ROOT/'src/preprocessing/reference_label_policy.py']:
  protected[str(p.relative_to(ROOT))]=sha(p)
 dump(report/'provenance.json',{'protected_sha256':protected,'human_record':'MILI-D-20261015-014','original_scientific_inputs_unchanged':True});print(json.dumps(info))
if __name__=='__main__':main()
