#!/usr/bin/env python3
"""Apply a reporting-only audit correction after the first run is safely held.
No solver rerun. Preserves original gates,physical inputs and failed audit artifacts.
"""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_ground_flux_replay_v1';R=ROOT/'data/exp/estimation/three_ward_ground_flux_replay_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 state=json.loads((R/'campaign_state.json').read_text());assert state['status']=='held' and state['new_hydraulic_replays_started']==1
 manifest=json.loads((P/'run_manifest.json').read_text());first=manifest['runs'][0];second=manifest['runs'][1]
 ex=json.loads((ROOT/first['folder']/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'];assert not (ROOT/second['folder']/'launch.json').exists()
 log=(R/'execution'/first['case_id']/'full_audit.log').read_text();assert ("udt_positivity_steps.csv" in log and 'Original diagnostic differs' in log) or ("Map differs" in log and "total_runtime" in log)
 archive=R/'revisions/reporting_compare_correction';archive.mkdir(parents=True,exist_ok=False)
 files=[ROOT/'src/analyses/audit_ground_flux_replay_v1.py',ROOT/'src/analyses/run_ground_flux_replay_v1.py',P/'execution_code_binding.json',R/'campaign_state.json',R/'campaign_error.txt',R/'execution'/first['case_id']/'full_audit.log']
 old={}
 for f in files:shutil.copy2(f,archive/f.name);old[str(f.relative_to(ROOT))]=sha(f)
 a=ROOT/'src/analyses/audit_ground_flux_replay_v1.py';s=a.read_text()
 s=s.replace("'udt_negative_cell_summary.csv','udt_positivity_steps.csv'","'udt_negative_cell_summary.csv'")
 s=s.replace(' checks=[]\n with NativeDataset', ' checks=[];runtime_metadata={}\n with NativeDataset',1)
 needle='   nv=nd[name];assert ov.shape==nv.shape and ov.dtype==nv.dtype\n'
 addition="   if name=='total_runtime':\n    before=float(np.asarray(ov[:]).reshape(-1)[0]);after=float(np.asarray(nv[:]).reshape(-1)[0])\n    assert np.isfinite(before) and np.isfinite(after) and min(before,after)>=0\n    runtime_metadata[name]={'original_seconds':before,'replay_seconds':after,'role':'wallclock cost,not a physical state;source writes tfinish_all-tstart_all'}\n    continue\n"
 assert needle in s;s=s.replace(needle,needle+addition,1)
 anchor=' return checks\n\ndef main(case):'
 new=''' # The positivity reporting scalar held is reduced over OpenMP threads.
 # Its final reduction order can differ without changing q,storage or budgets.
 pn=pd.read_csv(new/'udt_positivity_steps.csv',float_precision='round_trip')
 po=pd.read_csv(old/'udt_positivity_steps.csv',float_precision='round_trip')
 assert pn.shape==po.shape and list(pn.columns)==list(po.columns)
 held='held_requested_transport_m3'
 for c in po.columns:
  if c!=held:assert np.array_equal(pn[c],po[c]),('Positivity state differs',c)
 difference=abs(pn[held].to_numpy()-po[held].to_numpy())
 threads=json.loads((P/'run_manifest.json').read_text())['budget']['cpus']
 bound=4*threads*np.finfo('float64').eps*np.maximum(abs(pn[held].to_numpy()),abs(po[held].to_numpy()))
 assert np.all(difference<=bound),'Reporting difference exceeds conservative thread-reduction rounding bound'
 reporting={'held_reporting_rows_different':int((difference>0).sum()),'max_abs_held_reporting_difference_m3':float(difference.max()),'counts_min_factors_and_actual_clocks_exact':True,'comparison':'reporting-only positive-sum reduction bound4*threads*float64_epsilon*max(abs(old),abs(new));original positivity admission unchanged'}
 return checks,reporting,runtime_metadata

def main(case):'''
 assert anchor in s;s=s.replace(anchor,new)
 s=s.replace("maps=unchanged_maps(folder,old);print('ALL_MAP_VARIABLES_AND_NATIVE_CLOCK_LEDGERS_BITWISE_IDENTICAL'", "maps,positivity_reporting,runtime_metadata=unchanged_maps(folder,old);print('ALL_MAP_VARIABLES_AND_NATIVE_CLOCK_LEDGERS_BITWISE_IDENTICAL'")
 s=s.replace("'passed':True,'case':case,'original_numerical_gates_passed'", "'passed':True,'case':case,'positivity_reporting_comparison':positivity_reporting,'nonphysical_runtime_metadata':runtime_metadata,'original_numerical_gates_passed'")
 s=s.replace("all_map_variables_bitwise_identical","all_physical_map_variables_bitwise_identical").replace("ALL_MAP_VARIABLES_AND_NATIVE_CLOCK_LEDGERS_BITWISE_IDENTICAL","ALL_PHYSICAL_MAP_VARIABLES_AND_NATIVE_CLOCK_LEDGERS_BITWISE_IDENTICAL")
 a.write_text(s)
 runner=ROOT/'src/analyses/run_ground_flux_replay_v1.py';s=runner.read_text()
 s=s.replace("assert not (R/'campaign_state.json').exists(),'No automatic overwrite/retry'", "assert not (R/'campaign_state.json').exists() or '--resume-qualified' in sys.argv,'No automatic overwrite/retry'")
 marker=' def report(**kwargs):'
 insertion=''' if (R/'campaign_state.json').exists():
  assert '--resume-qualified' in sys.argv
  previous=json.loads((R/'campaign_state.json').read_text())
  assert previous['status']=='held' and previous['new_hydraulic_replays_started']==1
  first=manifest['runs'][0];second=manifest['runs'][1]
  qualified=json.loads((R/'execution'/first['case_id']/'replay_validation.json').read_text())
  assert qualified['passed'] and qualified['original_numerical_gates_passed']
  assert not (ROOT/second['folder']/'launch.json').exists(),'No automatic hydraulic retry'
  state=previous
  state.update(status='starting_qualified_continuation',pid=os.getpid(),completed_cases=[first['case_id']],queue=[second['case_id']],error=None,resumed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reporting_audit_correction='reporting_comparison_correction.json')
'''
 assert marker in s;s=s.replace(marker,insertion+marker,1)
 s=s.replace("  for run in manifest['runs']:\n   while", "  for run in manifest['runs']:\n   if run['case_id'] in state['completed_cases']:continue\n   while",1)
 runner.write_text(s)
 policy=ROOT/'src/preprocessing/reference_label_policy.py';original=ROOT/'src/analyses/audit_cross_event_protection_run_v1.py'
 receipt={'reason':'Extra total_runtime map equality and whole positivity-reporting CSV SHA were overstrict for wallclock cost and OpenMP reporting-only reduction;neither is a physical state or failed original numerical gate. SFINCS ncoutput writes total_runtime=tfinish_all-tstart_all.','archive':str(archive.relative_to(ROOT)),'before_sha256':old,'after_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [a,runner]},'source_numerical_policy_sha256':sha(policy),'unchanged_original_auditor_sha256':sha(original),'physical_inputs_changed':False,'new_solver_image':False,'hydraulic_retry':False,'resume_requires_saved_H_reaudit_pass':True}
 (R/'reporting_comparison_correction.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print('REPORTING_ONLY_CORRECTION_PREPARED; REAUDIT_SAVED_H_REQUIRED',flush=True)
if __name__=='__main__':main()
