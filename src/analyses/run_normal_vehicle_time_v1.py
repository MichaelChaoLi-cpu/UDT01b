"""Time-weighted turn-aware assignment with the prior repaired network and fixed OD."""
from pathlib import Path
import os,sys,time,json,hashlib,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from scipy.sparse.csgraph import dijkstra
from numba import njit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from run_normal_vehicle_pilot_v1 import propagate
P=ROOT/'data/processed/three_ward_normal_vehicle_time_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_time_v1';REPAIR=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
@njit(cache=True)
def path_flags(order,pred,arcmap,motorway,core):
 n=len(arcmap);motor=np.zeros(n,np.bool_);touch=np.zeros(n,np.bool_)
 for i in order:
  a=arcmap[i];p=pred[i];motor[i]=motorway[a];touch[i]=core[a]
  if 0<=p<n:motor[i]=motor[i]or motor[p];touch[i]=touch[i]or touch[p]
 return motor,touch

def run(arm,arc_cost,ep,dem,rr,pr,attrs):
 started=time.monotonic();S=len(rr['state_physical_arc']);E=len(pr['length_m']);K=len(ep['components']);dest_arc=rr['state_physical_arc'][rr['indices']];cost=arc_cost[dest_arc];g0=csr_matrix((cost,rr['indices'],rr['indptr']),shape=tuple(rr['shape']));assert np.all(g0.data>0);comp=rr['node_component'];startcomp=comp[pr['from_index']];endcomp=comp[rr['state_arc_end_node']];su=[];sv=[];sc=[];eu=[];ev=[]
 for j,c in enumerate(ep['components']):
  arcs=np.flatnonzero(startcomp==c);su.extend([j]*len(arcs));sv.extend(arcs.tolist());sc.extend(arc_cost[arcs].tolist());states=np.flatnonzero(endcomp==c);eu.extend(states.tolist());ev.extend([j]*len(states))
 starts=csr_matrix((sc,(su,sv)),shape=(K,S));ends=csr_matrix((np.zeros(len(eu)),(eu,ev)),shape=(S,K));g=vstack([hstack([g0,csr_matrix((S,K)),ends]),hstack([starts,csr_matrix((K,K*2))]),csr_matrix((K,S+K*2))],format='csr');eligible=[dem['base'],dem['through_plus']];origins=sorted(set(map(int,dem['origin_endpoint'][dem['through_plus']])));protocol=json.loads((P/'protocol.json').read_text());assert len(origins)<=protocol['budget']['maximum_origins'];flux=np.zeros((2,5,E));routed=np.zeros((2,5));unrouted=np.zeros((2,5));routecost=np.zeros((2,5));motorway_od=np.zeros((2,5));core_od=np.zeros((2,5));lengths=np.full((K,K),np.inf);expected=np.zeros((int(comp.max())+1,2,5));reached=np.zeros(E,bool);byorigin={i:np.flatnonzero(dem['origin_endpoint']==i)for i in origins}
 for done,i in enumerate(origins,1):
  if(P/'PAUSE').exists():raise RuntimeError('Paused before next origin;no complete-arm release')
  if time.monotonic()-started>protocol['budget']['maximum_route_seconds_per_arm']:raise RuntimeError('Registered time budget reached')
  dist,pred=dijkstra(g,indices=S+i,directed=True,return_predecessors=True);lengths[i]=dist[S+K:];finite=np.isfinite(dist[:S]);order=np.argsort(dist[:S],kind='stable');order=order[finite[order]];reached[rr['state_physical_arc'][finite]]=True;motorflags,coreflags=path_flags(order,pred,rr['state_physical_arc'],attrs['arc_is_motorway'],attrs['arc_is_core']);loads=np.zeros((S,2,5));source_rows=byorigin[i]
  for b in range(2):
   chosen=source_rows[eligible[b][source_rows]]
   for q in chosen:
    target=int(dem['destination_endpoint'][q]);values=dem['period_weights'][q];terminal=S+K+target
    if np.isfinite(dist[terminal]):
     parent=int(pred[terminal]);assert 0<=parent<S;loads[parent,b]+=values;routed[b]+=values;routecost[b]+=values*dist[terminal];motorway_od[b]+=values*motorflags[parent];core_od[b]+=values*coreflags[parent];expected[ep['components'][i],b]+=values;expected[ep['components'][target],b]-=values
    else:unrouted[b]+=values
  loads=propagate(order,pred,loads)
  for b in range(2):
   for t in range(5):flux[b,t]+=np.bincount(rr['state_physical_arc'],weights=loads[:,b,t],minlength=E)
  if done%50==0 or done==len(origins):
   state={'status':'running','arm':arm,'completed_origins':done,'total_origins':len(origins),'elapsed_s':time.monotonic()-started};dump(R/'execution_state.json',state);print(arm,done,'/',len(origins),'seconds',round(state['elapsed_s'],1),flush=True)
 demand=np.array([dem['period_weights'][mask].sum(axis=0)for mask in eligible]);maxbalance=0.
 for b in range(2):
  for t in range(5):
   balance=np.bincount(comp[pr['from_index']],weights=flux[b,t],minlength=len(expected))-np.bincount(comp[pr['to_index']],weights=flux[b,t],minlength=len(expected));maxbalance=max(maxbalance,float(abs(balance-expected[:,b,t]).max()))
 arc_cost_sum=np.einsum('bte,e->bt',flux,arc_cost);cost_resid=float(np.max(abs(arc_cost_sum-routecost)/np.maximum(1,abs(routecost))));old=load(REPAIR/'C_topology_multi_flows.npz');checks={'input_equals_routed_plus_unrouted':bool(abs(demand-routed-unrouted).max()<1e-6),'nonnegative_finite_flow':bool(np.isfinite(flux).all()and(flux>=0).all()),'component_flow_balance':maxbalance<1e-6,'route_cost_equals_arc_loading_cost':cost_resid<1e-9,'all_origins_completed':done==len(origins),'unchanged_eligible_demand':bool(abs(demand-old['input_demand']).max()<1e-6),'original_feasible_endpoint_pairs_not_lost':bool(np.isfinite(lengths[np.isfinite(old['endpoint_distance_m'])]).all()),'motorway_indicator_not_passage_sum':bool((motorway_od<=routed+1e-7).all()and(core_od<=routed+1e-7).all())}
 np.savez_compressed(P/(arm+'_flows.npz'),arc_flow=flux,routed=routed,unrouted=unrouted,input_demand=demand,endpoint_cost_seconds=lengths,od_any_motorway=motorway_od,od_any_core=core_od,reachable_physical_arcs=reached)
 result={'arm':arm,'arc_cost_sha256':hashlib.sha256(arc_cost.tobytes()).hexdigest(),'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':all(checks.values()),'checks':checks,'origins':len(origins),'elapsed_s':time.monotonic()-started,'maximum_component_flow_residual':maxbalance,'route_cost_relative_residual':cost_resid,'routed':routed.tolist(),'unrouted':unrouted.tolist(),'od_any_motorway':motorway_od.tolist(),'od_any_core':core_od.tolist(),'arc_candidate_km':(np.einsum('bte,e->bt',flux,pr['length_m'])/1000).tolist(),'assumed_route_candidate_hours':(routecost/3600).tolist()};dump(R/(arm+'_validation.json'),result);print('FINISHED',arm,result['checks'],flush=True)
 if not result['passed']:raise RuntimeError('Time assignment integrity failure')
 return result

def main():
 assert all(json.loads((R/'preparation_summary.json').read_text())['checks'].values());rr=load(REPAIR/'turn_router.npz');pr=load(M/'road_projection.npz');ep=load(REPAIR/'multi_endpoint_components.npz');dem=load(REPAIR/'multi_routing_demands.npz');attrs=load(P/'time_costs.npz');out=[]
 for i,arm in enumerate(['T1_class_time','T2_survey_time']):
  marker=R/(arm+'_validation.json')
  if marker.exists()and json.loads(marker.read_text())['passed']and json.loads(marker.read_text()).get('arc_cost_sha256')==hashlib.sha256(attrs['arc_cost_seconds'][i].tobytes()).hexdigest():out.append(json.loads(marker.read_text()));continue
  out.append(run(arm,attrs['arc_cost_seconds'][i],ep,dem,rr,pr,attrs))
 dump(R/'execution_state.json',{'status':'complete','completed_arms':[x['arm']for x in out],'route_wall_s':sum(x['elapsed_s']for x in out),'new_hydraulics':False});print('TIME ASSIGNMENT COMPLETE',flush=True)
if __name__=='__main__':main()
