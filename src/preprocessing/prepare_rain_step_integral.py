"""Bind an exact original-curve step integral to the qualified positive CPU solver."""
from pathlib import Path
import sys,json,shutil,difflib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
OUT='data/processed/three_ward_rain_step_integral_v1'
REPORT='data/exp/data-preprocessing/three_ward_rain_step_integral_v1'
PRIOR='data/processed/three_ward_adaptive_positivity_v2'
TAG='udt01b/sfincs-cpu:galibier-rain-step-integral-v1'
CASES=['dry_raised','rain60_raised']
def main():
 out=ROOT/OUT;c=out/'context';c.mkdir(parents=True);report=ROOT/REPORT;report.mkdir(parents=True,exist_ok=True)
 old=ROOT/PRIOR
 for p in (old/'context').iterdir():
  if p.is_file():shutil.copyfile(p,c/p.name)
 source=ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src/sfincs_meteo.f90';before=source.read_text();after=before
 decl='\n logical,private::udt_rain_started=.false.\n integer,private::udt_rain_unit\n real,allocatable,private::udt_rain_rates(:,:),udt_rain_mean(:)\n real*8,allocatable,private::udt_rain_weights(:),udt_rain_volume_rates(:)\n'
 after=after.replace('module sfincs_meteo\n','module sfincs_meteo\n'+decl,1)
 after=after.replace('contains\n','contains\n'+(ROOT/'src/preprocessing/sfincs_rain_step_integral.f90.inc').read_text(),1)
 after=after.replace('   if (meteo3d) then\n','   if (meteo3d.and.precip.and.netamprfile(1:4)/=\'none\') call udt_prepare_rain_integral(t,dt)\n   if (meteo3d) then\n',1)
 after=after.replace('            netprcp(nm) = prcp(nm)','            netprcp(nm) = prcp(nm)\n            if (netamprfile(1:4)/=\'none\') netprcp(nm)=udt_rain_mean(nm)',1)
 after=after.replace('cumprcp(nm) = cumprcp(nm) + prcp(nm) * dt','cumprcp(nm) = cumprcp(nm) + netprcp(nm) * dt',1)
 assert after!=before and after.count('call udt_prepare_rain_integral(t,dt)')==1
 (c/'sfincs_meteo.f90').write_text(after)
 patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/source/src/sfincs_meteo.f90',tofile='b/source/src/sfincs_meteo.f90'))
 (out/'rain_step_integral.patch').write_text(patch);(out/'full_solver.patch').write_bytes((old/'full_solver.patch').read_bytes()+patch.encode())
 shutil.copyfile(ROOT/'src/preprocessing/rain_integral_fixture.f90',c/'rain_integral_fixture.f90')
 binding={'source_commit':json.loads((old/'source_binding.json').read_text())['source_commit'],'prior_image_id':json.loads((old/'solver_binding.json').read_text())['image_id'],'full_patch_sha256':sha(out/'full_solver.patch'),'source_sha256':{p.name:sha(p)for p in c.iterdir()if p.suffix=='.f90'},'prior_freeze_sha256':sha(old/'freeze.json'),'human_record':'MILI-D-20261015-016','scope':'only two existing adaptive controls; no training or58batch'}
 dump(out/'source_binding.json',binding);dump(c/'RAIN_INTEGRAL_BINDING.json',binding)
 docker=(old/'context/Dockerfile').read_text().replace('udt_rain_clock.f90 /usr/src','udt_rain_clock.f90 sfincs_meteo.f90 /usr/src').replace('COPY ADAPTIVE_BINDING.json /opt/udt-build/ADAPTIVE_BINDING.json','COPY RAIN_INTEGRAL_BINDING.json /opt/udt-build/RAIN_INTEGRAL_BINDING.json').replace('adaptive_missing_child_fixture.f90 /opt/udt-build/','adaptive_missing_child_fixture.f90 rain_integral_fixture.f90 /opt/udt-build/')
 docker=docker.replace('\nFROM udt01b/sfincs-cpu:galibier-conservative-positivity-v2',' && gfortran -fopenmp -I. /opt/udt-build/rain_integral_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /opt/udt-build/rain_integral_fixture\nFROM udt01b/sfincs-cpu:galibier-adaptive-positivity-v2').replace('adaptive-conservative-donor-limit-v2','rain-step-integral-v1')
 (c/'Dockerfile').write_text(docker)
 dump(report/'provenance.json',{'prior_freeze_sha256':sha(old/'freeze.json'),'pending58_manifest_sha256':sha(ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json'),'registry_sha256':sha(ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json'),'policy_sha256':sha(ROOT/'src/preprocessing/reference_label_policy.py'),'human_record':'MILI-D-20261015-016'})
 print(json.dumps(binding))
if __name__=='__main__':main()
