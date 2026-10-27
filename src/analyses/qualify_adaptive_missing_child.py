"""Qualify missing-child coarse aggregates against the exact running production library."""
from pathlib import Path
import sys,subprocess,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_adaptive_positivity import OUT,REPORT,TAG
from prepare_sfincs_cpu_build import dump,sha

def main():
 out=ROOT/OUT;report=ROOT/REPORT;dest=out/'missing_child_tests';dest.mkdir();builder_tag=TAG+'-fixture-builder';subprocess.run(['docker','build','--target','positivity_builder','--platform','linux/amd64','--network','none','-t',builder_tag,str(out/'context')],check=True);source=ROOT/'src/preprocessing/adaptive_missing_child_fixture.f90';shutil.copyfile(source,dest/source.name);image=json.loads((out/'solver_binding.json').read_text())['image_id'];hashcmd=['sha256sum','/usr/local/lib/libsfincs.so.0.0.0'];base=['docker','run','--rm','--platform','linux/amd64','--network','none'];rh=subprocess.check_output(base+['--entrypoint',hashcmd[0],image,*hashcmd[1:]],text=True).split()[0]
 script='set -eu\nsha256sum /usr/local/lib/libsfincs.so.0.0.0 > /audit/library.sha256\ngfortran -fopenmp -I/usr/src/sfincs/src /audit/adaptive_missing_child_fixture.f90 -L/usr/local/lib -lsfincs -Wl,-rpath,/usr/local/lib -o /audit/fixture\n'
 for name,threads in [('serial',1),('parallel',4)]:
  (dest/name).mkdir();script+=f'(cd /audit/{name} && OMP_NUM_THREADS={threads} /audit/fixture > stdout.txt 2>stderr.txt)\n'
 (dest/'run.sh').write_text(script);res=subprocess.run(base+['--mount',f'type=bind,source={dest},target=/audit','--entrypoint','/bin/sh',builder_tag,'/audit/run.sh'],text=True,capture_output=True);(dest/'launcher.stdout').write_text(res.stdout);(dest/'launcher.stderr').write_text(res.stderr)
 checks=[{'check':'builder_and_running_runtime_library_byte_identical','pass':rh==(dest/'library.sha256').read_text().split()[0]},{'check':'505_missing_child_updates_pass','pass':res.returncode==0 and'PASS 505 missing-child'in(dest/'serial/stdout.txt').read_text()},{'check':'missing_child_serial_parallel_identical','pass':(dest/'serial/stdout.txt').read_bytes()==(dest/'parallel/stdout.txt').read_bytes()}]
 result={'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'fixture_sha256':sha(source),'production_library_sha256':rh,'production_image_id':image,'new_hydrodynamic_runs':0,'reason':'Existing adaptive geometry contains coarse aggregates with one inactive child represented by the native zero dummy.'};dump(report/'missing_child_validation.json',result);print(json.dumps(result,indent=2));assert result['all_checks_passed']
if __name__=='__main__':main()
