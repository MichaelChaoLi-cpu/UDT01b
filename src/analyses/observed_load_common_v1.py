"""Shared frozen turn-aware network for count-constrained load scenarios."""
from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,hstack,vstack
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
P=ROOT/'data/processed/three_ward_observed_load_pilot_v1';R=ROOT/'data/exp/estimation/three_ward_observed_load_pilot_v1';REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1';C=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1';B=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';F=ROOT/'data/processed/three_ward_cross_event_protection_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def save(n,d):pd.DataFrame(d).to_csv(P/n,index=False,encoding='utf-8-sig')
def network(rr,ep,pr,cost,closed=None):
 S=len(rr['state_physical_arc']);K=len(ep['components']);comp=rr['node_component'];startcomp=comp[pr['from_index']];endcomp=comp[rr['state_arc_end_node']];ac=cost.copy()
 if closed is not None:ac[closed[pr['arc_to_road']]]=np.inf
 g0=csr_matrix((ac[rr['state_physical_arc'][rr['indices']]],rr['indices'],rr['indptr']),shape=tuple(rr['shape']));su=[];sv=[];sc=[];eu=[];ev=[]
 for j,c in enumerate(ep['components']):
  arcs=np.flatnonzero(startcomp==c);su.extend([j]*len(arcs));sv.extend(arcs.tolist());sc.extend(ac[arcs].tolist());states=np.flatnonzero(endcomp==c);eu.extend(states.tolist());ev.extend([j]*len(states))
 starts=csr_matrix((sc,(su,sv)),shape=(K,S));ends=csr_matrix((np.zeros(len(eu)),(eu,ev)),shape=(S,K));return vstack([hstack([g0,csr_matrix((S,K)),ends]),hstack([starts,csr_matrix((K,K*2))]),csr_matrix((K,S+K*2))],format='csr')
