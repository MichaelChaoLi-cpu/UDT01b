"""Independent treatment identity and capped allocation expectations for O1/S1."""
from pathlib import Path
import json,hashlib
import numpy as np
from shared_receiver_identity_v2 import CORE,EXTRA,validate_identity_bytes

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def treatment_hash(folder,files):
 if files==['udt_pumps.dat']:return sha(Path(folder)/files[0])
 assert files==['udt_pumps.dat','udt_pool_rules.dat']
 return hashlib.sha256(json.dumps({n:sha(Path(folder)/n) for n in files},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def identity_receipt_check(folder,run):
 try:
  files=run['treatment_files'];assert files==(['udt_pumps.dat','udt_pool_rules.dat'] if run['contrast']=='O1' else ['udt_pumps.dat'])
  assert (folder/'udt_pool_rules.dat').exists()==(run['contrast']=='O1')
  h=treatment_hash(folder,files)
  return validate_identity_bytes((folder/'udt_run_identity.txt').read_bytes(),(folder/'udt_diagnostic_identity.txt').read_bytes(),run['identity'],h) and json.loads((folder/'udt_run_identity.json').read_text())==run['identity'] and all(sha(folder/n)==run['input_sha256'][n] for n in files)
 except (KeyError,OSError,ValueError,AssertionError):return False

def pool_rules(folder,n):
 lines=(Path(folder)/'udt_pool_rules.dat').read_text().splitlines();assert lines[0]=='UDT_WEIGHTED_POOL_V1';ng=int(lines[1]);caps={}
 for i,line in enumerate(lines[2:2+ng],1):
  k,v=line.split();assert int(k)==i;caps[i]=float(v)
 assert int(lines[ng+2])==n and len(lines)==ng+n+3
 groups=[];units=[]
 for i,line in enumerate(lines[ng+3:],1):
  k,g,v=line.split();assert int(k)==i and int(g) in caps;groups.append(int(g));units.append(float(v))
 return np.array(groups),np.array(units),caps

def independent_fill(weights,limits,capacity):
 """Vectorized bisection, independent of the Fortran active-set implementation."""
 limits=np.asarray(limits,float);capacity=np.broadcast_to(np.asarray(capacity,float),limits.shape[:-1]);weights=np.asarray(weights,float)
 assert (weights>0).all() and (limits>=0).all() and (capacity>=0).all()
 lo=np.zeros_like(capacity);hi=capacity/weights.min()
 for _ in range(60):
  mid=(lo+hi)/2;flow=np.minimum(limits,mid[...,None]*weights).sum(axis=-1)
  lo=np.where(flow<=capacity,mid,lo);hi=np.where(flow>capacity,mid,hi)
 result=np.minimum(limits,lo[...,None]*weights)
 return np.where((limits.sum(axis=-1)<=capacity)[...,None],limits,result)

def expected_requested_and_transfer(run,pump,sites,matrix,dt,folder,frame):
 reserve=1+64*np.finfo(np.float32).eps
 prior=matrix['source_prior_volume_m3'];eligible=(matrix['depth_m']>=pump[None,:,3])&(matrix['head_m']<=pump[None,:,4])
 if run['contrast']=='S1':
  req=dt[:,None]*pump[None,:,2];v=np.where(eligible,np.minimum(req,prior/reserve),0.)
 else:
  groups,units,caps=pool_rules(folder,len(pump));req=dt[:,None]*units[None,:];limit=np.where(eligible,np.minimum(req,prior/reserve),0.);v=np.zeros_like(req)
  for g,cap in caps.items():
   m=groups==g;v[:,m]=independent_fill(pump[m,2],limit[:,m],cap*dt)
  for key,expected in [('nominal_reserved_m3',dt[:,None]*pump[None,:,2]),('standalone_limit_m3',req),('receiving_pool_id',np.broadcast_to(groups,req.shape)),('pool_limit_m3',dt[:,None]*np.array([caps[g] for g in groups])[None,:])]:
   assert np.max(np.abs(frame[key].to_numpy().reshape(req.shape)-expected),initial=0)<1e-12,key
 blocked=~eligible
 assert np.array_equal(frame.limited_by_storage.to_numpy().reshape(req.shape)=='T',(prior/reserve<req)&eligible)
 assert np.array_equal(frame.blocked_by_trigger_or_head.to_numpy().reshape(req.shape)=='T',blocked)
 return req,v
