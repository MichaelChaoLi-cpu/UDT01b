"""Prepare a one-condition boundary closure correction and matched two-control build."""
from pathlib import Path
import sys,json,shutil,difflib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump,COMMIT
OUT='data/processed/three_ward_boundary_closure_fix_v1'
REPORT='data/exp/data-preprocessing/three_ward_boundary_closure_fix'
PRIOR='data/processed/three_ward_negative_trace_controls_v1'
PRIOR_REPORT='data/exp/data-preprocessing/three_ward_negative_trace_controls'
INTERFACE='data/processed/three_ward_negative_trace_interface_v1'
CASES=['dry_raised','rain60_raised']
TAG='udt01b/sfincs-cpu:galibier-closed-boundary-v1'
def main():
 out=ROOT/OUT;report=ROOT/REPORT;context=out/'context'
 if (out/'source_binding.json').exists():raise FileExistsError('Retain prepared source')
 context.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
 old=ROOT/INTERFACE;binding=json.loads((old/'solver_binding.json').read_text());base=json.loads((old/'source_binding.json').read_text())
 source=ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src/sfincs_boundaries.f90'
 before=source.read_text();anchor='if (hnmb < huthresh + 1.0e-6 .or. kcuv(ip) == 3) then';assert before.count(anchor)==1
 after=before.replace(anchor,'if (hnmb < huthresh + 1.0e-6 .or. kcuv(ip) == 3 .or. kcuv(ip) == 0) then')
 (context/source.name).write_text(after)
 delta=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/source/src/'+source.name,tofile='b/source/src/'+source.name))
 (out/'boundary_closure.patch').write_text(delta)
 (out/'full_solver.patch').write_bytes((old/'negative_trace_full.patch').read_bytes()+delta.encode())
 for name in ['udt_clock_budget.f90','udt_rain_clock.f90']:shutil.copyfile(old/'context'/name,context/name)
 fixture=ROOT/'src/preprocessing/closed_boundary_fixture.f90';shutil.copyfile(fixture,context/fixture.name)
 info={'source_commit':COMMIT,'base_builder_image_id':base['base_builder_image_id'],'prior_runtime_image_id':binding['image_id'],'prior_patch_sha256':binding['patch_sha256'],'closure_patch_sha256':sha(out/'boundary_closure.patch'),'full_patch_sha256':sha(out/'full_solver.patch'),'patched_boundary_sha256':sha(context/source.name),'original_boundary_sha256':sha(source),'budget_module_sha256':sha(context/'udt_clock_budget.f90'),'rain_module_sha256':sha(context/'udt_rain_clock.f90'),'fixture_sha256':sha(fixture),'scope':'Only add closed kcuv=0 to the existing shallow/structure zero-exchange branch; no positivity, forcing, geometry or other numerical changes','authorized_control_cases':CASES,'human_record':'MILI-D-20261015-013'}
 dump(out/'source_binding.json',info);dump(context/'CLOSURE_BINDING.json',info)
 docker=f'''FROM udt01b/sfincs-cpu:builder-v1 AS closure_builder
COPY sfincs_boundaries.f90 udt_clock_budget.f90 udt_rain_clock.f90 /usr/src/sfincs/src/
COPY CLOSURE_BINDING.json /opt/udt-build/CLOSURE_BINDING.json
COPY closed_boundary_fixture.f90 /opt/udt-build/closed_boundary_fixture.f90
RUN cd /usr/src/sfincs/src && make -j1 > /opt/udt-build/receipts/closure-make.log 2>&1 && make install > /opt/udt-build/receipts/closure-install.log 2>&1 && ldconfig && gfortran -fopenmp -I. /opt/udt-build/closed_boundary_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/closed_boundary_fixture
FROM udt01b/sfincs-cpu:galibier-negative-trace-v1
COPY --from=closure_builder /usr/local /usr/local
COPY --from=closure_builder /opt/udt-build /opt/udt-build
RUN ldconfig && /opt/udt-build/netcdf_link_probe && /opt/udt-build/loader_probe && /opt/udt-build/closed_boundary_fixture
LABEL org.opencontainers.image.version="closed-boundary-v1"
WORKDIR /data
CMD ["sfincs"]
'''
 (context/'Dockerfile').write_text(docker)
 protected={}
 for p in [source,old/'solver_binding.json',old/'freeze.json',ROOT/PRIOR/'run_manifest.json',ROOT/PRIOR/'qualification.json',ROOT/PRIOR_REPORT/'summary.json',ROOT/PRIOR_REPORT/'reference_admission_assessment.json',ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json',ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json',ROOT/'src/preprocessing/reference_label_policy.py']:
  protected[str(p.relative_to(ROOT))]=sha(p)
 dump(report/'provenance.json',{'protected_sha256':protected,'human_record':'MILI-D-20261015-013','original_scientific_inputs_unchanged':True})
 print(json.dumps(info))
if __name__=='__main__':main()
