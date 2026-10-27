"""Open only the authorized eight-run stage after primary-grid qualification; retain a live overlay."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_reference_setup_pilot import OUT,REPORT,PILOT,prepare_pilot
from prepare_sfincs_cpu_build import sha,dump

def update_overlay():
 out=ROOT/OUT;report=ROOT/REPORT;base=ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json';old=json.loads(base.read_text());pilot=json.loads((out/'pilot_manifest.json').read_text());lookup={(r['case_id'],'regular_10m'):r for r in pilot['runs']};rows=[]
 for legacy in old['pending_runs']:
  row=dict(legacy);run=lookup.get((row['case_id'],row['grid']));row['authorized_in_current_pilot']=run is not None;row['training_release']=False
  if run:
   folder=ROOT/run['folder'];audit=report/'pilot'/run['case_id']/'audit.json';ex=folder/'execution.json'
   if ex.exists():
    e=json.loads(ex.read_text());row['status']='completed_pending_audit'if e['returncode']==0 and e['completion_marker']else'failed_retained';row['response_path']=run['folder']
    if audit.exists():
     a=json.loads(audit.read_text());row['numerical_admission']=a['admission'];row['status']='numerically_qualified_partial'if a['all_checks_passed']and a['admission']['technical_reference']else'hold_event'
   elif(folder/'launch.json').exists():row['status']='running';row['response_path']=run['folder']
   else:row['status']='authorized_queued'
  rows.append(row)
 completed=sum(r['status']in ['completed_pending_audit','numerically_qualified_partial','hold_event']for r in rows)
 dump(out/'execution_overlay.json',{'source_snapshot_sha256':sha(base),'source_snapshot_is_historical':True,'current_authoritative_execution_index':True,'runs':rows,'completed_subset_of58':completed,'not_authorized_in_current_pilot':sum(not r['authorized_in_current_pilot']for r in rows),'sealed_test_and_challenge_responses_generated':False,'training':False,'default_registry_unchanged':True})

def main():
 out=ROOT/OUT;report=ROOT/REPORT;gate=report/'primary_grid_qualification.json';print('WAIT PRIMARY GRID QUALIFICATION',flush=True)
 while not gate.exists():time.sleep(15)
 assert json.loads(gate.read_text())['all_checks_passed'];prepare_pilot();update_overlay()
 children=[]
 for script in ['run_reference_setup_pilot.py','process_reference_setup_pilot.py']:
  log=(report/(script.removesuffix('.py')+'_pilot.log')).open('w');proc=subprocess.Popen([sys.executable,'src/analyses/'+script,'pilot'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);children.append((proc,log));print('START',script,'pilot',flush=True)
 while any(p.poll()is None for p,l in children):update_overlay();time.sleep(15)
 for p,l in children:l.close()
 update_overlay();codes=[p.returncode for p,l in children];dump(report/'pilot_stage_execution.json',{'runner_returncode':codes[0],'auditor_returncode':codes[1],'training':False,'sealed':False,'authorized_new_runs':8});assert codes==[0,0],'Preserve findings; no automatic retry'
 print('EIGHT-INPUT PILOT EXECUTION AND AUDITS COMPLETE',flush=True)
if __name__=='__main__':main()
