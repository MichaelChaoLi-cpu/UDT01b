"""Continue authorized A2 once A1 has genuinely passed;persist all failures."""
import time,subprocess,sys,datetime as dt
from temporal_transport_common_v1 import *
def main():
 try:
  while True:
   gate();f=R/'A1_execution.json';v=json.loads(f.read_text())if f.exists()else {}
   if v.get('status')=='stopped':raise RuntimeError('A1 stopped:'+v.get('reason',''))
   if v.get('status')=='qualified_complete':break
   time.sleep(5)
  sources();assert json.loads((R/'A1_validation.json').read_text())['passed'];assert json.loads((R/'measure_interface_selfcheck.json').read_text())['passed'];f=R/'A2_execution.json'
  assert not f.exists(),'Preserve previous attempt;no automatic restart'
  with(R/'A2_execution.log').open('ab')as log:
   p=subprocess.Popen([sys.executable,'-u','src/analyses/execute_temporal_stage_v1.py','A2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);dump(R/'A2_process.json',dict(pid=p.pid,started_utc=dt.datetime.now(dt.timezone.utc).isoformat()));code=p.wait()
  # Always expose stopped/incomplete status;never call it completed if failed.
  subprocess.run([sys.executable,'src/analyses/report_temporal_transport_v1.py'],cwd=ROOT,check=True)
  assert code==0,'A2 returned error;see A2_execution.json/log'
  assert json.loads((R/'A2_validation.json').read_text())['passed'];dump(R/'pipeline_status.json',dict(status='qualified_complete',finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
 except BaseException as exc:
  dump(R/'pipeline_status.json',dict(status='stopped',reason=str(exc),utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
if __name__=='__main__':main()
