"""Finish the already-started bounded pair: audit dry, run/audit rain, compare.
Does not restart the dry control, retry failures or expand the run set.
"""
from pathlib import Path
import time,json,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_geometry_pair_controls_v1';R=ROOT/'data/exp/estimation/three_ward_urban_geometry_pair_controls_v1';PY='/private/tmp/udt01b-sfincs-builder/bin/python'
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 scripts=['src/analyses/run_urban_geometry_pair_controls_v1.py','src/analyses/audit_urban_geometry_terrain_control_v1.py','src/analyses/compare_urban_geometry_pair_controls_v1.py','src/preprocessing/reference_label_policy.py'];binding={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in scripts};dump(P/'execution_code_binding.json',binding)
 for case in ['dry_raised','rain60_raised']:
  if case=='rain60_raised':
   assert json.loads((R/'dry_raised/T/audit.json').read_text())['all_checks_passed']
   with (R/'rain_runner.log').open('w') as f:subprocess.run([PY,str(ROOT/scripts[0]),case],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  else:
   start=time.monotonic();folder=P/'runs/dry_raised__T_10m';assert (folder/'launch.json').exists()
   while not (folder/'execution.json').exists():
    if time.monotonic()-start>7200:raise TimeoutError('Retain dry attempt and inspect; no restart')
    dump(R/'progress.json',{'phase':'dry_control_running','training':False});time.sleep(10)
  folder=P/'runs'/(case+'__T_10m');ex=json.loads((folder/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker']
  assert all(hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h for s,h in binding.items())
  report=R/case/'T';report.mkdir(parents=True,exist_ok=True)
  with (report/'audit.log').open('w') as f:subprocess.run([PY,str(ROOT/scripts[1]),'execution',case],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert json.loads((report/'audit.json').read_text())['all_checks_passed'];print('CONTROL QUALIFIED',case,flush=True);dump(R/'progress.json',{'phase':case+'_qualified','training':False})
 with (R/'comparison.log').open('w') as f:subprocess.run([PY,str(ROOT/scripts[2])],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 dump(R/'progress.json',{'phase':'paired_comparison_complete','new_runs':2,'training':False});print('ALL DONE',flush=True)
if __name__=='__main__':main()
