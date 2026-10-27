"""Two bounded distance-assignment arms with frozen OD and external-demand branches."""
from pathlib import Path
import os,sys,time,json,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from run_normal_vehicle_pilot_v1 import propagate
P=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_repair_v1';OLD=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def run(arm,ep,dem,rr,pr):
 started=time.monotonic();S=len(rr['state_physical_arc']);E=len(pr['length_m']);K=len(ep['components']);base=csr_matrix((rr['cost'],rr['indices'],rr['indptr']),shape=tuple(rr['shape']));assert np.all(base.data>0);comp=rr['node_component'];startcomp=comp[pr['from_index']];endcomp=comp[rr['state_arc_end_node']];su=[];sv=[];sc=[];eu=[];ev=[]
 for j,c in enumerate(ep['components']):
  arcs=np.flatnonzero(startcomp==c);su.extend([j]*len(arcs));sv.extend(arcs.tolist());sc.extend(pr['length_m'][arcs].tolist());states=np.flatnonzero(endcomp==c);eu.extend(states.tolist());ev.extend([j]*len(states))
 starts=csr_matrix((sc,(su,sv)),shape=(K,S));ends=csr_matrix((np.zeros(len(eu)),(eu,ev)),shape=(S,K));g=vstack([hstack([base,csr_matrix((S,K)),ends]),hstack([starts,csr_matrix((K,K*2))]),csr_matrix((K,S+K*2))],format='csr');eligible=[dem['base'],dem['through_plus']];origins=sorted(set(map(int,dem['origin_endpoint'][dem['through_plus']])));protocol=json.loads((P/'protocol.json').read_text());assert len(origins)<=protocol['budget']['maximum_origins'];flux=np.zeros((2,5,E));routed=np.zeros((2,5));unrouted=np.zeros((2,5));lengths=np.full((K,K),np.inf);expected=np.zeros((int(comp.max())+1,2,5))
 byorigin={i:np.flatnonzero(dem['origin_endpoint']==i)for i in origins}
 for done,i in enumerate(origins,1):
  if(P/'PAUSE').exists():raise RuntimeError('Paused before next origin;no full-arm result released')
  if time.monotonic()-started>protocol['budget']['maximum_route_seconds_per_arm']:raise RuntimeError('Registered wall budget reached')
  dist,pred=dijkstra(g,indices=S+i,directed=True,return_predecessors=True);lengths[i]=dist[S+K:];loads=np.zeros((S,2,5));source_rows=byorigin[i]
  for b in range(2):
   chosen=source_rows[eligible[b][source_rows]]
   for q in chosen:
    target=int(dem['destination_endpoint'][q]);values=dem['period_weights'][q];terminal=S+K+target
    if np.isfinite(dist[terminal]):
     parent=int(pred[terminal]);assert 0<=parent<S;loads[parent,b]+=values;routed[b]+=values;expected[ep['components'][i],b]+=values;expected[ep['components'][target],b]-=values
    else:unrouted[b]+=values
  order=np.argsort(dist[:S],kind='stable');order=order[np.isfinite(dist[order])];loads=propagate(order,pred,loads)
  for b in range(2):
   for t in range(5):flux[b,t]+=np.bincount(rr['state_physical_arc'],weights=loads[:,b,t],minlength=E)
  if done%25==0 or done==len(origins):
   state={'status':'running','arm':arm,'completed_origins':done,'total_origins':len(origins),'elapsed_s':time.monotonic()-started};dump(R/'execution_state.json',state);print(arm,done,'/',len(origins),'seconds',round(state['elapsed_s'],1),flush=True)
 demand=np.array([dem['period_weights'][mask].sum(axis=0)for mask in eligible]);maxbalance=0.
 for b in range(2):
  for t in range(5):
   balance=np.bincount(comp[pr['from_index']],weights=flux[b,t],minlength=len(expected))-np.bincount(comp[pr['to_index']],weights=flux[b,t],minlength=len(expected));maxbalance=max(maxbalance,float(abs(balance-expected[:,b,t]).max()))
 checks={'input_equals_routed_plus_unrouted':bool(abs(demand-routed-unrouted).max()<1e-6),'nonnegative_finite_flow':bool(np.isfinite(flux).all()and(flux>=0).all()),'component_flow_balance':maxbalance<1e-6,'all_origins_completed':done==len(origins)}
 if arm=='B_topology_single':
  old=load(OLD/'pilot_arc_flows.npz');mask=np.isfinite(old['endpoint_distance_m']);checks['old_feasible_pairs_not_lost']=bool(np.isfinite(lengths[mask]).all());checks['same_endpoints_distance_not_increased']=bool(np.all(lengths[mask]<=old['endpoint_distance_m'][mask]+1e-6));checks['unchanged_eligible_demand']=bool(abs(demand-old['input_demand']).max()<1e-6)
 else:
  old=load(OLD/'pilot_arc_flows.npz');checks['unchanged_eligible_demand']=bool(abs(demand-old['input_demand']).max()<1e-6)
 np.savez_compressed(P/(arm+'_flows.npz'),arc_flow=flux,routed=routed,unrouted=unrouted,input_demand=demand,endpoint_distance_m=lengths)
 result={'arm':arm,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':all(checks.values()),'checks':checks,'origins':len(origins),'elapsed_s':time.monotonic()-started,'maximum_component_flow_residual':maxbalance,'routed':routed.tolist(),'unrouted':unrouted.tolist()};dump(R/(arm+'_validation.json'),result);print('FINISHED',arm,result['checks'],flush=True)
 if not result['passed']:raise RuntimeError('Arm integrity checks failed')
 return result

def main():
 assert all(json.loads((R/'preparation_summary.json').read_text())['checks'].values());rr=load(P/'turn_router.npz');pr=load(M/'road_projection.npz');results=[]
 for arm,epfile,demfile in [('B_topology_single',P/'single_endpoint_components.npz',OLD/'routing_demands.npz'),('C_topology_multi',P/'multi_endpoint_components.npz',P/'multi_routing_demands.npz')]:
  marker=R/(arm+'_validation.json')
  if marker.exists()and json.loads(marker.read_text())['passed']:results.append(json.loads(marker.read_text()));continue
  results.append(run(arm,load(epfile),load(demfile),rr,pr))
 dump(R/'execution_state.json',{'status':'complete','completed_arms':[r['arm']for r in results],'new_hydraulics':False,'total_route_wall_s':sum(r['elapsed_s']for r in results)});print('REPAIR ROUTING COMPLETE',flush=True)
if __name__=='__main__':main()
