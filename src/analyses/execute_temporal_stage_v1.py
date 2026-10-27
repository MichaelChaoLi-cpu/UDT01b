"""One bounded temporal stage with persistent failure status;no auto restart."""
import sys,time,datetime as dt,traceback
from temporal_transport_common_v1 import *
def main():
 stage=sys.argv[1];assert stage in ['A1','A2'];sources();gate();begun=time.monotonic();dump(R/(stage+'_execution.json'),dict(status='running',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage=stage))
 try:
  if stage=='A1':from run_temporal_road_population_v1 import main as run
  else:from run_temporal_measures_v1 import main as run
  run();sources();size=sum(f.stat().st_size for root in [P,R]for f in root.rglob('*')if f.is_file());assert size<15*1024**3,'New product budget15GiB'
 except BaseException as exc:
  dump(R/(stage+'_execution.json'),dict(status='stopped',stage=stage,reason=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-begun));raise
 dump(R/(stage+'_execution.json'),dict(status='qualified_complete',stage=stage,seconds=time.monotonic()-begun,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),products_bytes=size))
if __name__=='__main__':main()
