"""Resume only never-launched cases after import failure;original solver/gates unchanged.
Preserves held attempt,all completed outputs,original immutable code binding and budgets.
"""
import os,sys,json,time,inspect,hashlib,datetime as dt,traceback,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
import run_barrier_drainage_v1 as parent
from observed_load_common_v1 import sha,dump
P=ROOT/'data/processed/three_ward_barrier_drainage_v1';R=ROOT/'data/exp/estimation'/P.name;REPAIR=R/'runtime_recovery_20261019';PYTHON=str(ROOT/'.frame-control/runtimes/physical-measures-py312/bin/python')

def prepare():
 assert str(Path(sys.prefix).resolve())==str((ROOT/'.frame-control/runtimes/physical-measures-py312').resolve()),'Use persistent restored runtime'
 assert not(P/'STOP').exists()and not(P/'PAUSE').exists(),'Honor pending user control'
 state=json.loads((R/'campaign_state.json').read_text());assert state['status']=='held';assert 'ModuleNotFoundError: No module named' in(REPAIR/'failed_audit_import.log').read_text();assert 'numpy'in(REPAIR/'failed_audit_import.log').read_text();manifest=json.loads((P/'run_manifest.json').read_text());original_binding=json.loads((P/'execution_code_binding.json').read_text());assert all(sha(ROOT/n)==h for n,h in original_binding.items()),'Original code changed';sources=json.loads((P/'protected_sources.json').read_text());assert all(sha(ROOT/n)==h for n,h in sources.items()),'Scientific input changed';certified=[];already_launched=[]
 for case in manifest['new_execution_order']:
  run=next(r for r in manifest['runs']if r['case_id']==case);folder=ROOT/run['folder'];audit=R/'execution'/case/'audit.json'
  if(folder/'launch.json').exists():
   already_launched.append(case);execution=json.loads((folder/'execution.json').read_text());assert execution['returncode']==0 and execution['completion_marker'];assert audit.exists(),'Existing completed run needs audit,never rerun it';a=json.loads(audit.read_text());assert a['all_checks_passed']and a['admission']['technical_reference'];certified.append(case)
  else:assert not(folder/'sfincs_map.nc').exists()and not(folder/'execution.json').exists(),'Preserve any previous hydraulic attempt'
 assert len(certified)==6 and certified==already_launched;assert set(state['completed_new_cases'])<=set(certified);state['completed_new_cases']=certified;state.pop('reason',None);state.pop('automatic_retry',None);state.update(pid=os.getpid(),status='runtime_repaired_resume',runtime_recovery_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runtime_python=PYTHON,additional_hydraulic_retries=0);old_active=state['active_wall_seconds']
 source=inspect.getsource(parent.main);anchor="assert not(R/'campaign_state.json').exists(),'No automatic restart';";assert source.count(anchor)==1;source=source.replace(anchor,'')
 anchor="bindings={n:sha(ROOT/n)for n in names};dump(P/'execution_code_binding.json',bindings)";assert source.count(anchor)==1;source=source.replace(anchor,"bindings=ORIGINAL_BINDING;assert all(sha(ROOT/n)==h for n,h in bindings.items())")
 i=source.index('state=dict(');j=source.index(';container=None;child=None',i);source=source[:i]+"state=RECOVERED_STATE.copy()"+source[j:]
 anchor="  for case in manifest['new_execution_order']:\n";assert source.count(anchor)==1;source=source.replace(anchor,anchor+"   if case in state['completed_new_cases']:continue\n",1)
 anchor="def unchanged():\n";assert source.count(anchor)==1;source=source.replace(anchor,anchor+"  assert sha(RESUME_SOURCE)==RESUME_SOURCE_SHA,'Resume adapter changed'\n")
 env=parent.__dict__.copy();env.update(PYTHON=PYTHON,RECOVERED_STATE=state,ORIGINAL_BINDING=original_binding,RESUME_SOURCE=Path(__file__),RESUME_SOURCE_SHA=sha(Path(__file__)));exec(compile(source,__file__,'exec'),env)
 oldnames={p.name for p in Path('/private/tmp/udt01b-sfincs-builder/lib/python3.12/site-packages').glob('*.dist-info')};newnames={p.name for p in(Path(sys.prefix)/'lib/python3.12/site-packages').glob('*.dist-info')};assert newnames<=oldnames
 receipt=dict(python=PYTHON,python_version=sys.version,installed_versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},all_distribution_versions_match_prior=True,adapter_sha256=sha(Path(__file__)),derived_loop_sha256=hashlib.sha256(source.encode()).hexdigest(),original_code_binding_sha256=sha(P/'execution_code_binding.json'),original_inputs_unchanged=True,qualified_cases_skipped=certified,remaining_cases=[c for c in manifest['new_execution_order']if c not in certified],original_active_seconds_retained=old_active,no_hydraulic_reruns=True,only_runtime_and_resume_control_changed=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat());dump(REPAIR/'resume_binding.json',receipt);return env['main'],receipt

def main():
 run,receipt=prepare()
 if '--check'in sys.argv:print('RESUME INTERFACE QUALIFIED',json.dumps(receipt),flush=True);return
 assert not(REPAIR/'resume_start.json').exists(),'No automatic adapter restart';dump(REPAIR/'resume_start.json',dict(pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));dump(R/'pipeline_status.json',dict(status='running_physical_campaign',runtime_repaired=True,runtime_python=PYTHON))
 try:
  run();assert json.loads((R/'delivery_validation.json').read_text())['passed'];dump(R/'pipeline_status.json',dict(status='qualified_complete',runtime_repaired=True))
 except BaseException as exc:
  dump(R/'pipeline_status.json',dict(status='held',reason=str(exc),runtime_repaired=True,automatic_retry=False));(REPAIR/'resume_error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
