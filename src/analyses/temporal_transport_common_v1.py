"""Shared immutable temporal analysis inputs, exact turn-state reachability and gates."""
from pathlib import Path
import os,sys,json,time,hashlib
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','2');os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/udt01b-temporal-numba');os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-temporal-mpl')
import numpy as np,pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from observed_load_common_v1 import load,dump,sha
D=ROOT/'data/processed';PLAN=ROOT/'data/exp/research-design/three_ward_five_event_temporal_plan_v1';TAG='three_ward_five_event_temporal_analysis_v1';P=D/TAG;R=ROOT/'data/exp/estimation'/TAG;PREV=D/'three_ward_five_event_population_validation_v1';BASE=D/'three_ward_night_population_transport_v1';REP=D/'three_ward_normal_vehicle_repair_v1';M=D/'three_ward_motor_flood_access_v1';C=D/'three_ward_normal_vehicle_choice_v1';MEASURE=D/'three_ward_demand_protection_feedback_v1';N=D/'three_ward_normal_transport_network_v1';WARD=['13101','13102','13103']
def gate():
 if(P/'STOP').exists():raise RuntimeError('User STOP')
 while(P/'PAUSE').exists():
  if(P/'STOP').exists():raise RuntimeError('User STOP while paused')
  time.sleep(5)
def sources():
 b=json.loads((P/'source_bindings.json').read_text())['sha256'];assert all(sha(ROOT/n)==h for n,h in b.items()),'Bound input changed'
 for f in P.glob('*_code_binding.json'):
  binding=json.loads(f.read_text())['sha256'];assert all(sha(ROOT/n)==h for n,h in binding.items()),'Bound execution code changed: '+f.name
def save(p,x):pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def init():
 for q in [P,R,P/'exposure',P/'population',P/'reachability_cache',R/'traffic',P/'traffic',P/'traffic/paths',P/'traffic/fits']:q.mkdir(parents=True,exist_ok=True)
def counts(x,ptr):c=np.r_[0,np.cumsum(x,dtype='i8')];return c[ptr[1:]]-c[ptr[:-1]]
def population_inputs():
 from postprocess_five_event_population_v1 import readblocks
 blocks,support=readblocks();normal=load(BASE/'normal_node_access.npz')['bidirectional'];pop=blocks.residents_2020.to_numpy(dtype='i8');info={}
 for b in [0,10,25]:
  ptr=support[f'offsets_{b}m'];ix=support[f'node_indices_{b}m'];lens=np.diff(ptr);complete=(lens>0)&(counts(normal[ix],ptr)==lens);info[b]=dict(ptr=ptr,ix=ix,lens=lens,complete=complete)
 return blocks,pop,normal,info
class Reachability:
 def __init__(self):
  self.rr=load(REP/'turn_router.npz');self.pr=load(M/'road_projection.npz');r=self.rr;p=self.pr;self.S=len(r['state_physical_arc']);self.comp=r['node_component'];self.nc=int(self.comp.max())+1;self.row=np.repeat(np.arange(self.S,dtype='i4'),np.diff(r['indptr']));self.col=r['indices'];self.road=p['arc_to_road'][r['state_physical_arc']];self.start=self.comp[p['from_index'][r['state_physical_arc']]];self.end=self.comp[r['state_arc_end_node']];ep=pd.read_csv(REP/'multi_network_endpoints.csv');self.gates=ep.loc[ep.kind.eq('conditional_outer_boundary_portal'),'component'].to_numpy();self.sg=np.isin(self.start,self.gates);self.eg=np.isin(self.end,self.gates);self.memory={};self.calls=0;self.hits=0
 def key(self,closed):return hashlib.sha256(np.packbits(closed).tobytes()).hexdigest()
 def direct(self,closed):
  opened=~closed[self.road];keep=opened[self.row]&opened[self.col];u=self.row[keep];v=self.col[keep];eg=np.flatnonzero(self.eg&opened);sg=np.flatnonzero(self.sg&opened);go=coo_matrix((np.ones(len(u)+len(eg),dtype='u1'),(np.r_[v,np.full(len(eg),self.S)],np.r_[u,eg])),shape=(self.S+1,self.S+1)).tocsr();gi=coo_matrix((np.ones(len(u)+len(sg),dtype='u1'),(np.r_[u,np.full(len(sg),self.S)],np.r_[v,sg])),shape=(self.S+1,self.S+1)).tocsr();out=np.zeros(self.S+1,bool);inn=out.copy();out[breadth_first_order(go,self.S,directed=True,return_predecessors=False)]=True;inn[breadth_first_order(gi,self.S,directed=True,return_predecessors=False)]=True;oc=np.bincount(self.start[out[:self.S]&opened],minlength=self.nc)>0;ic=np.bincount(self.end[inn[:self.S]&opened],minlength=self.nc)>0;oc[self.gates]=True;ic[self.gates]=True;return (oc&ic)[self.comp]
 def get(self,closed):
  self.calls+=1;k=self.key(closed)
  if k in self.memory:self.hits+=1;return self.memory[k]
  f=P/'reachability_cache'/(k+'.npz')
  if f.exists():
   z=load(f);assert np.array_equal(z['closed_packed'],np.packbits(closed));a=np.unpackbits(z['node_ok_packed'],count=len(self.comp)).astype(bool);self.hits+=1
  else:
   gate();a=self.direct(closed);np.savez_compressed(f,closed_packed=np.packbits(closed),node_ok_packed=np.packbits(a),node_count=np.array(len(a)))
  self.memory[k]=a
  # Packed files retain all states;limit uncompressed in-memory masks to~64MiB.
  if len(self.memory)>160:self.memory.pop(next(iter(self.memory)))
  return a
