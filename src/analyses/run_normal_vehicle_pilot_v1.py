"""Two conditional shortest-distance flow pilots on the existing turn-memory graph.
No capacities,travel-time costs,hydraulic calls,OD re-estimation ornewturnrules.
"""
from pathlib import Path
import os,json,time,hashlib,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix,hstack,vstack
from scipy.sparse.csgraph import dijkstra
from numba import njit
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_baseline_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
@njit(cache=True)
def propagate(order,pred,load):
 for z in range(len(order)-1,-1,-1):
  node=order[z];parent=pred[node]
  if parent>=0 and parent<load.shape[0]:
   for b in range(load.shape[1]):
    for t in range(load.shape[2]):load[parent,b,t]+=load[node,b,t]
 return load

def main():
 started=time.monotonic();ep=load(P/'endpoint_components.npz');dem=load(P/'routing_demands.npz');rr=load(M/'turn_router.npz');pr=load(M/'road_projection.npz');shape0=tuple(rr['shape']);S=len(rr['state_physical_arc']);E=len(pr['length_m']);K=len(ep['components']);original=csr_matrix((rr['cost'],rr['indices'],rr['indptr']),shape=shape0);base=original[:S,:S].tocsr();assert np.all(base.data>0),'Zero-stateedges require topological propagation,not distorder'
 comp=rr['node_component'];start_comp=comp[pr['from_index']];end_comp=comp[rr['state_arc_end_node']];su=[];sv=[];sc=[];eu=[];ev=[]
 for j,c in enumerate(ep['components']):
  arcs=np.flatnonzero(start_comp==c);su.extend([j]*len(arcs));sv.extend(arcs.tolist());sc.extend(pr['length_m'][arcs].tolist());states=np.flatnonzero(end_comp==c);eu.extend(states.tolist());ev.extend([j]*len(states))
 starts=csr_matrix((sc,(su,sv)),shape=(K,S));ends=csr_matrix((np.zeros(len(eu)),(eu,ev)),shape=(S,K));g=vstack([hstack([base,csr_matrix((S,K)),ends]),hstack([starts,csr_matrix((K,K*2))]),csr_matrix((K,S+2*K))],format='csr');n=S+2*K
 eligible=[dem['base'],dem['through_plus']];origins=sorted(set(map(int,dem['origin_endpoint'][dem['through_plus']])));protocol=json.loads((P/'protocol.json').read_text());assert len(origins)<=protocol['budget']['maximum_routing_origin_representatives']
 flux=np.zeros((2,5,E));routed=np.zeros((2,5));unrouted=np.zeros((2,5));lengths=np.full((K,K),np.inf);complete=[];failures=[]
 for i in origins:
  if (P/'PAUSE').exists():
   np.savez_compressed(P/'pilot_partial_flow.npz',arc_flow=flux,routed=routed,unrouted=unrouted,distance=lengths,completed_origins=np.array(complete));dump(R/'routing_state.json',{'status':'paused_at_origin_boundary','completed_origins':complete,'total_origins':len(origins)});return
  if time.monotonic()-started>protocol['budget']['maximum_route_wall_seconds']:raise RuntimeError('Registeredroutingwallbudgetexceeded')
  dist,pred=dijkstra(g,indices=S+i,directed=True,return_predecessors=True);lengths[i]=dist[S+K:];loads=np.zeros((S,2,5));source_rows=np.flatnonzero(dem['origin_endpoint']==i)
  for b in range(2):
   chosen=source_rows[eligible[b][source_rows]]
   for q in chosen:
    target=int(dem['destination_endpoint'][q]);values=dem['period_weights'][q];terminal=S+K+target
    if np.isfinite(dist[terminal]):
     parent=int(pred[terminal]);assert 0<=parent<S;loads[parent,b]+=values;routed[b]+=values
    else:unrouted[b]+=values;failures.append((i,target,b,float(values[0])))
  order=np.argsort(dist[:S],kind='stable');order=order[np.isfinite(dist[order])];loads=propagate(order,pred,loads)
  for b in range(2):
   for period in range(5):flux[b,period]+=np.bincount(rr['state_physical_arc'],weights=loads[:,b,period],minlength=E)
  complete.append(i)
  if len(complete)%5==0 or len(complete)==len(origins):
   dump(R/'routing_state.json',{'status':'running','completed_origin_count':len(complete),'total_origins':len(origins),'elapsed_s':time.monotonic()-started,'routed_daily_by_branch':routed[:,0].tolist()});print('NORMAL FLOW',len(complete),'/',len(origins),'seconds',round(time.monotonic()-started),flush=True)
 demand=np.array([dem['period_weights'][mask].sum(axis=0)for mask in eligible]);netchecks=[]
 # Independently check node-component divergence against routed OD generation/attraction.
 for b in range(2):
  expected=np.zeros((int(comp.max())+1,5));mask=eligible[b]
  for q in np.flatnonzero(mask):
   i=int(dem['origin_endpoint'][q]);j=int(dem['destination_endpoint'][q])
   if np.isfinite(lengths[i,j]):expected[ep['components'][i]]+=dem['period_weights'][q];expected[ep['components'][j]]-=dem['period_weights'][q]
  for period in range(5):
   balance=np.bincount(comp[pr['from_index']],weights=flux[b,period],minlength=len(expected))-np.bincount(comp[pr['to_index']],weights=flux[b,period],minlength=len(expected));error=float(np.max(np.abs(balance-expected[:,period])));netchecks.append({'branch':b,'period':period,'maximum_node_component_flow_residual':error})
 checks={'all_registered_origins_completed':len(complete)==len(origins),'demand_accounting':bool(np.max(np.abs(demand-routed-unrouted))<1e-6),'nonnegative_arc_flows':bool((flux>=0).all()),'component_flow_conservation':max(x['maximum_node_component_flow_residual']for x in netchecks)<1e-6,'unchanged_turn_state_count':S==717063,'no_capacity_or_speed_costs':True}
 np.savez_compressed(P/'pilot_arc_flows.npz',arc_flow=flux,routed=routed,unrouted=unrouted,input_demand=demand,endpoint_distance_m=lengths);dump(R/'pilot_validation.json',{'passed':all(checks.values()),'checks':checks,'node_balance':netchecks});dump(R/'routing_state.json',{'status':'complete','completed_origin_count':len(complete),'total_origins':len(origins),'elapsed_s':time.monotonic()-started,'branches':['base_no_remote_remote','plus_conditional_chord_through'],'routed_by_branch_period':routed.tolist(),'unrouted_by_branch_period':unrouted.tolist(),'periods':['daily','hour8','hour10','hour17','daytime12h'],'source_turn_memory_states':S,'new_turn_rules':0,'new_hydraulics':False});print('NORMAL FLOW FINISHED',checks,flush=True)
 if not all(checks.values()):raise SystemExit('Pilotflowchecksfailed')
if __name__=='__main__':main()
