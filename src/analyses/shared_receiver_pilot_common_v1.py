"""Bounded two-case shared-receiver pilot constants and filesystem helpers."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_shared_receiver_pilot_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
PY=ROOT/'.frame-control/runtimes/physical-measures-py312/bin/python';VENV=ROOT/'.venv/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def tail(path):
 if not path.exists():return None
 with path.open('rb') as f:
  names=f.readline().decode().strip().split(',');f.seek(0,2);n=f.tell();f.seek(max(0,n-65536));text=f.read().decode(errors='replace')
 if not text.endswith('\n'):text=text.rsplit('\n',1)[0]
 for line in reversed(text.splitlines()):
  values=line.split(',')
  if len(values)==len(names):
   try:return {k:float(v) for k,v in zip(names,values)}
   except ValueError:pass
 return None
def command(run,container):
 f=ROOT/run['folder'];return ['docker','run','--rm','--name',container,'--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={f},target=/data','--mount',f'type=bind,source={run["large_readonly_source"]},target=/data/sfincs_subgrid.nc,readonly','--mount',f'type=bind,source={f}/historical_rain.nc,target=/data/historical_rain.nc,readonly','--entrypoint','/usr/local/bin/sfincs',run['image_id']]
