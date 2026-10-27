"""No changed baseline static inputs;actual image and geometry frozen before launch."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from qualify_stage_barrier_drainage_v1 import baselines,P,R
from observed_load_common_v1 import sha,dump

def main():
 manifest=json.loads((P/'run_manifest.json').read_text());base=baselines();assert json.loads((R/'solver_qualification.json').read_text())['passed'];assert len(manifest['new_execution_order'])==12;ledger=[]
 for run in manifest['runs']:
  source=base[run['baseline_case']];folder=ROOT/run['folder']
  for n,h in source['input_sha256'].items():
   if n in ['sfincs.inp','udt_run_identity.txt']:continue
   assert sha(folder/n)==h,(run['case_id'],n,'changed static/rain/stage input')
  if not run['reused']:
   text=(folder/'sfincs.inp').read_text();src=(ROOT/source['folder']/'sfincs.inp').read_text();expected=src+('weirfile = barriers.weir\n'if run['measure']in['barrier','combined']else'');assert text==expected
   if run['measure']in['barrier','combined']:assert sha(folder/'barriers.weir')==sha(P/'measures/barriers.weir')
   if run['measure']in['drainage','combined']:assert sha(folder/'udt_pumps.dat')==sha(P/'measures/udt_pumps.dat')
  ledger.append(dict(case=run['case_id'],passed=True,baseline=run['baseline_case'],rain_stage_geometry_unchanged=True))
 for name in ['positivity_fixture','adaptive_positivity_fixture','adaptive_missing_child_fixture','rain_integral_fixture']:
  old=ROOT/'data/processed/three_ward_historical_radar_experiment_v2/fixtures_qualified'/(name+'_1')/'stdout.txt';new=P/'qualification/interfaces'/(name+'_1')/'stdout.txt';assert old.exists()and new.read_bytes()==old.read_bytes(),name+' frozen regression'
 bound=json.loads((P/'protected_sources.json').read_text());files=[P/'protocol.json',P/'run_manifest.json',P/'solver_binding.json',P/'source_binding.json',P/'measure_solver.patch',R/'solver_qualification.json',R/'input_validation.json']+[f for f in(P/'measures').iterdir()if f.is_file()]
 for f in files:bound[str(f.relative_to(ROOT))]=sha(f)
 for r in base.values():
  for n in ['sfincs_map.nc','execution.json']:bound[str((ROOT/r['folder']/n).relative_to(ROOT))]=sha(ROOT/r['folder']/n)
 dump(P/'protected_sources.json',bound);dump(R/'preflight_validation.json',dict(passed=True,static_replays=ledger,original_fixture_regressions_exact=4,all_measure_inputs_bound=True,no_new_hydraulics_in_preflight=True));print('ALL16 PHYSICAL INPUTS AND4 ORIGINAL REGRESSIONS PASSED',flush=True)
if __name__=='__main__':main()
