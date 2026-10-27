"""Version a coarse-fine extension; frozen regular-grid source branch is retained verbatim."""
from pathlib import Path
import sys,json,shutil,difflib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
OUT='data/processed/three_ward_adaptive_positivity_v2'
REPORT='data/exp/data-preprocessing/three_ward_adaptive_positivity_v2'
PRIOR='data/processed/three_ward_conservative_positivity_v2'
TAG='udt01b/sfincs-cpu:galibier-adaptive-positivity-v2'
CASES=['dry_raised','rain60_raised']
def main():
 out=ROOT/OUT;report=ROOT/REPORT;c=out/'context';c.mkdir(parents=True);report.mkdir(parents=True,exist_ok=True)
 old=ROOT/PRIOR;patches=[]
 for name in ['sfincs_boundaries.f90','sfincs_continuity.f90','udt_clock_budget.f90','udt_rain_clock.f90']:
  before=(old/'context'/name).read_text();after=before
  if name=='sfincs_continuity.f90':
   declarations=' logical,private::ad_started=.false.\n integer,private::ad_unit\n integer,allocatable,private::ad_degree(:),ad_faces(:,:)\n real,allocatable,private::ad_width(:),ad_ratio(:)\n real*8,allocatable,private::ad_factor(:)\n'
   after=after.replace('contains\n',declarations+'contains\n'+(ROOT/'src/preprocessing/sfincs_adaptive_positivity.f90.inc').read_text(),1)
   needle='  if(.not.positivity_started)then'
   after=after.replace(needle,'  if(use_quadtree)then\n   call udt_limit_adaptive(dt,actual_end)\n   return\n  endif\n'+needle,1)
  if name=='udt_clock_budget.f90':
   decl=' real(real64),allocatable::ad_min(:),ad_time(:),ad_previous(:)\n integer(int64),allocatable::ad_min_step(:),ad_counts(:)\n integer,allocatable::ad_groups(:)\n integer::ad_exchange_unit,ad_events_unit\n'
   after=after.replace('contains\n',decl+'contains\n'+(ROOT/'src/preprocessing/sfincs_adaptive_diagnostics.f90.inc').read_text(),1)
   after=after.replace('  call negative_trace_initialize()','  call adaptive_trace_initialize()\n  call negative_trace_initialize()',1)
   after=after.replace('  call negative_trace_finish(actual_time,dt)','  call negative_trace_finish(actual_time,dt)\n  call adaptive_trace_finish()',1)
   after=after.replace('  call negative_trace_close()','  call negative_trace_close()\n  call adaptive_trace_close()',1)
  (c/name).write_text(after)
  if after!=before:patches.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/source/src/'+name,tofile='b/source/src/'+name)))
 for name in ['positivity_fixture.f90','adaptive_positivity_fixture.f90','adaptive_missing_child_fixture.f90']:shutil.copyfile(ROOT/'src/preprocessing'/name,c/name)
 binding={'prior_image_id':json.loads((old/'solver_binding.json').read_text())['image_id'],'source_commit':json.loads((old/'source_binding.json').read_text())['source_commit'],'regular_branch_unchanged':True,'human_record':'MILI-D-20261015-015','source_sha256':{p.name:sha(p)for p in c.iterdir()},'scope':'two existing control companions; not58batch; no training'}
 (out/'adaptive.patch').write_text(''.join(patches));(out/'full_solver.patch').write_bytes((old/'full_solver.patch').read_bytes()+''.join(patches).encode());binding['full_patch_sha256']=sha(out/'full_solver.patch');dump(out/'source_binding.json',binding);dump(c/'ADAPTIVE_BINDING.json',binding)
 docker=(old/'context/Dockerfile').read_text().replace('COPY positivity_fixture.f90 /opt/udt-build/positivity_fixture.f90','COPY positivity_fixture.f90 adaptive_positivity_fixture.f90 adaptive_missing_child_fixture.f90 /opt/udt-build/').replace('COPY POSITIVITY_BINDING.json /opt/udt-build/POSITIVITY_BINDING.json','COPY ADAPTIVE_BINDING.json /opt/udt-build/ADAPTIVE_BINDING.json').replace('\nFROM udt01b/sfincs-cpu:galibier-closed-boundary-v1',' && gfortran -fopenmp -I. /opt/udt-build/adaptive_positivity_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/adaptive_positivity_fixture && gfortran -fopenmp -I. /opt/udt-build/adaptive_missing_child_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/adaptive_missing_child_fixture\nFROM udt01b/sfincs-cpu:galibier-conservative-positivity-v2').replace('conservative-donor-limit-v2','adaptive-conservative-donor-limit-v2')
 (c/'Dockerfile').write_text(docker)
 dump(report/'provenance.json',{'prior_freeze_sha256':sha(old/'freeze.json'),'pending58_manifest_sha256':sha(ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json'),'registry_sha256':sha(ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json'),'human_record':'MILI-D-20261015-015'})
 print(json.dumps(binding))
if __name__=='__main__':main()
