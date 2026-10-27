"""Execute ten authorized missing responses in five audited batches, preserving prior results."""
from pathlib import Path
import sys,json,subprocess,time,concurrent.futures,argparse,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_expansion_12_4 import OUT,REPORT,PRIOR,PRIOR_REPORT,LIB,HUMAN,prepare
from prepare_sfincs_cpu_build import sha,dump
from run_reference_setup_pilot import memory_used

def update_index():
 out=ROOT/OUT;rep=ROOT/REPORT;plan=json.loads((out/'selection.json').read_text());m=json.loads((out/'expansion_manifest.json').read_text());rows=json.loads((ROOT/PRIOR/'execution_overlay.json').read_text())['runs'];lookup={r['case_id']:r for r in m['runs']}
 for r in rows:
  r['authorized_in_current_expansion']=r['case_id']in lookup and r['grid']=='regular_10m'
  if r['authorized_in_current_expansion']:
   x=lookup[r['case_id']];p=ROOT/x['folder'];a=rep/'expansion'/r['case_id']/'audit.json';r['training_release']=False;r['response_path']=None;r['status']='authorized_queued'
   if(p/'launch.json').exists():r['status']='running';r['response_path']=x['folder']
   if(p/'execution.json').exists():
    e=json.loads((p/'execution.json').read_text());r['status']='completed_pending_audit'if e['returncode']==0 and e['completion_marker']else'failed_retained'
   if a.exists():
    d=json.loads(a.read_text());r['status']='numerically_qualified_partial'if d['all_checks_passed']and d['admission']['technical_reference']else'hold_event';r['numerical_admission']=d['admission']
 dump(out/'execution_overlay.json',{'runs':rows,'prior_snapshot_sha256':sha(ROOT/PRIOR/'execution_overlay.json'),'original58_snapshot_sha256':sha(ROOT/LIB/'run_manifest.json'),'current_authoritative_execution_index':True,'cumulative_completed_subset_of58':sum(r['status']in ['completed_pending_audit','numerically_qualified_partial','hold_event']for r in rows),'remaining_unrun':sum(r['status']=='unrun'for r in rows),'current_expansion_queued_or_running':sum(r['status']in ['authorized_queued','running']for r in rows),'training':False,'sealed_responses_generated':False,'human_record':HUMAN})

def execute(r):
 p=ROOT/r['folder'];e=p/'execution.json'
 if e.exists():
  d=json.loads(e.read_text());assert d['returncode']==0 and d['completion_marker'];return
 assert not(p/'sfincs.log').exists()and not(p/'launch.json').exists(),'Retain incomplete attempt; no automatic rerun'
 cmd=['docker','run','--rm','--name','udt01b-expand124-'+r['case_id'],'--platform','linux/amd64','--network','none','--cpus','4','--memory','7g','--env','OMP_NUM_THREADS=4','--mount',f'type=bind,source={p},target=/data','--mount','type=bind,source='+r['large_readonly_source']+',target=/data/sfincs_subgrid.nc,readonly','--entrypoint','/usr/local/bin/sfincs',r['image_id']]
 assert subprocess.run(['docker','image','inspect',r['image_id'],'--format','{{.Id}}'],capture_output=True,text=True,check=True).stdout.strip()==r['image_id']
 dump(p/'launch.json',{'command':cmd,'authorization':HUMAN,'batch':r['batch'],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'training':False});print('START',r['case_id'],'batch',r['batch'],flush=True);start=time.monotonic()
 with(p/'solver_stdout.txt').open('w')as so,(p/'solver_stderr.txt').open('w')as se:result=subprocess.run(cmd,stdout=so,stderr=se)
 log=(p/'sfincs.log').read_text(errors='replace')if(p/'sfincs.log').exists()else'';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS']);runtime=time.monotonic()-start
 dump(e,{'returncode':result.returncode,'completion_marker':done,'runtime_seconds':runtime,'command':cmd,'input_sha256':r['input_sha256'],'solver_image_id':r['image_id'],'solver_image_digest':r['image_digest'],'phase':'expansion','new_parent_event':True,'library_descendant':True,'training':False,'sealed_response':False});print('FINISH',r['case_id'],result.returncode,done,round(runtime,1),flush=True);assert result.returncode==0 and done

def audit(case):
 import audit_reference_setup_pilot as a
 a.OUT=OUT;a.REPORT=REPORT;a.main('expansion',case)

def main():
 m=prepare();out=ROOT/OUT;rep=ROOT/REPORT;plan=json.loads((out/'selection.json').read_text());assert json.loads((rep/'preflight.json').read_text())['all_checks_passed'];binding=json.loads((out/'pre_execution_binding.json').read_text());assert binding['selection_sha256']==sha(out/'selection.json')and binding['manifest_sha256']==sha(out/'expansion_manifest.json')and binding['runner_source_sha256']==sha(Path(__file__))and binding['preparation_source_sha256']==sha(ROOT/'src/preprocessing/prepare_reference_expansion_12_4.py')and binding['original_auditor_sha256']==sha(ROOT/'src/analyses/audit_reference_setup_pilot.py');assert len(m['runs'])==10 and plan['new_hydraulic_run_limit']==10;update_index();telemetry=[]
 for batch,cases in enumerate(plan['batches'],1):
  runs=[next(r for r in m['runs']if r['case_id']==c)for c in cases];queue=runs.copy();active={};print('BATCH',batch,cases,flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
   while queue or active:
    for future,r in list(active.items()):
     if future.done():future.result();del active[future]
    if queue and len(active)<2:
     ready=not active
     if active and all((ROOT/r['folder']/'udt_step_budget.csv').exists()for r in active.values()):
      used=memory_used()
      if used is not None:
       projected=sum(used)+max(3*2**30,max(used,default=0)+.3*2**30);ready=projected<6.4*2**30;telemetry.append({'batch':batch,'used_bytes':used,'projected_bytes':projected,'parallel_allowed':ready})
     if ready:
      r=queue.pop(0);active[pool.submit(execute,r)]=r
    dump(rep/'execution_progress.json',{'batch':batch,'active':[r['case_id']for r in active.values()],'queued_current_batch':[r['case_id']for r in queue],'completed':[r['case_id']for r in m['runs']if(ROOT/r['folder']/'execution.json').exists()],'max_parallel':2,'training':False});update_index()
    if queue or active:time.sleep(15)
  for r in runs:
   c=r['case_id'];a=rep/'expansion'/c/'audit.json'
   if not a.exists():
    (rep/'expansion'/c).mkdir(parents=True,exist_ok=True)
    with(rep/'expansion'/c/'audit.log').open('w')as f:
     ret=subprocess.run([sys.executable,__file__,'--audit',c],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode:print('AUDIT FAILURE',c,flush=True)
    assert ret.returncode==0,'Preserve finding; stop before next batch'
   d=json.loads(a.read_text());assert d['all_checks_passed']and d['admission']['technical_reference'];print('AUDIT PASS',c,d['admission']['state'],flush=True)
  update_index();dump(rep/f'batch_{batch}_receipt.json',{'batch':batch,'cases':cases,'all_original_gates_passed':True,'training':False});subprocess.run([sys.executable,'src/analyses/review_reference_expansion_12_4.py','--partial'],cwd=ROOT,check=True);print('BATCH QUALIFIED',batch,flush=True)
 dump(rep/'resource_decisions.json',telemetry);dump(rep/'execution_complete.json',{'new_runs':10,'reused_parents':6,'training_parents':12,'validation_parents':4,'all_batches_passed':True,'training':False,'sealed_responses':False});update_index();print('EXPANSION COMPLETE: cumulative 12 train +4 validation; no training.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit');args=p.parse_args()
 if args.audit:audit(args.audit)
 else:main()
