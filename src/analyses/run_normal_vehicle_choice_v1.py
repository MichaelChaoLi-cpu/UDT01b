"""Six fixed cost-realization routes; 2x2 toll/multipath development experiment."""
from pathlib import Path
import os,sys,time,json,hashlib,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from scipy.sparse.csgraph import dijkstra
from numba import njit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from run_normal_vehicle_pilot_v1 import propagate
from evaluate_normal_vehicle_baseline_v1 import sha
P=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_choice_v1';REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
@njit(cache=True)
def path_diagnostics(order,pred,arcmap,basecost,motorway,core):
 n=len(arcmap);nominal=np.zeros(n);finger=np.zeros(n,np.uint64);motor=np.zeros(n,np.bool_);touch=np.zeros(n,np.bool_);seed=np.uint64(1469598103934665603);mult=np.uint64(1099511628211)
 for i in order:
  a=arcmap[i];p=pred[i];parenthash=seed;nominal[i]=basecost[a];motor[i]=motorway[a];touch[i]=core[a]
  if 0<=p<n:nominal[i]+=nominal[p];parenthash=finger[p];motor[i]=motor[i]or motor[p];touch[i]=touch[i]or touch[p]
  finger[i]=(parenthash^np.uint64(a+1))*mult
 return nominal,finger,motor,touch

def weighted_quantile(a,w,q):
 order=np.argsort(a,kind='stable');a=a[order];w=w[order];return float(a[np.searchsorted(np.cumsum(w),q*w.sum())])

def run(fee,k,ep,dem,rr,pr,attrs,signature,globalstart,budget):
 name=f'F{fee}_R{k}';started=time.monotonic();S=len(rr['state_physical_arc']);E=len(pr['length_m']);K=len(ep['components']);arc_cost=attrs['arc_cost_seconds'][fee,k];basecost=attrs['unperturbed_generalized_seconds'][fee];edgecost=arc_cost[rr['state_physical_arc'][rr['indices']]];g0=csr_matrix((edgecost,rr['indices'],rr['indptr']),shape=tuple(rr['shape']));assert np.all(g0.data>0);comp=rr['node_component'];startcomp=comp[pr['from_index']];endcomp=comp[rr['state_arc_end_node']];su=[];sv=[];sc=[];eu=[];ev=[]
 for j,c in enumerate(ep['components']):
  arcs=np.flatnonzero(startcomp==c);su.extend([j]*len(arcs));sv.extend(arcs.tolist());sc.extend(arc_cost[arcs].tolist());states=np.flatnonzero(endcomp==c);eu.extend(states.tolist());ev.extend([j]*len(states))
 starts=csr_matrix((sc,(su,sv)),shape=(K,S));ends=csr_matrix((np.zeros(len(eu)),(eu,ev)),shape=(S,K));g=vstack([hstack([g0,csr_matrix((S,K)),ends]),hstack([starts,csr_matrix((K,K*2))]),csr_matrix((K,S+K*2))],format='csr');eligible=[dem['base'],dem['through_plus']];origins=sorted(set(map(int,dem['origin_endpoint'][dem['through_plus']])));assert len(origins)<=budget['maximum_origins'];flux=np.zeros((2,5,E));routed=np.zeros((2,5));unrouted=np.zeros((2,5));routecost=np.zeros((2,5));nominalsum=np.zeros((2,5));motorod=np.zeros((2,5));coreod=np.zeros((2,5));lengths=np.full((K,K),np.inf);nominalmatrix=np.full((K,K),np.inf);fingerprint=np.zeros((K,K),np.uint64);expected=np.zeros((int(comp.max())+1,2,5));byorigin={i:np.flatnonzero(dem['origin_endpoint']==i)for i in origins}
 for done,i in enumerate(origins,1):
  if(P/'PAUSE').exists():raise RuntimeError('Pause requested before next origin;completed trees preserved')
  if time.monotonic()-started>budget['maximum_route_seconds_per_tree']or time.monotonic()-globalstart>budget['maximum_total_route_seconds']:raise RuntimeError('Registered route wall budget reached')
  dist,pred=dijkstra(g,indices=S+i,directed=True,return_predecessors=True);lengths[i]=dist[S+K:];finite=np.isfinite(dist[:S]);order=np.argsort(dist[:S],kind='stable');order=order[finite[order]];nc,fh,mf,cf=path_diagnostics(order,pred,rr['state_physical_arc'],basecost,attrs['arc_is_motorway'],attrs['arc_is_core']);terminalpred=pred[S+K:];good=(terminalpred>=0)&(terminalpred<S)&np.isfinite(lengths[i]);nominalmatrix[i,good]=nc[terminalpred[good]];fingerprint[i,good]=fh[terminalpred[good]];loads=np.zeros((S,2,5));source_rows=byorigin[i]
  for b in range(2):
   for q in source_rows[eligible[b][source_rows]]:
    target=int(dem['destination_endpoint'][q]);values=dem['period_weights'][q];terminal=S+K+target
    if np.isfinite(dist[terminal]):
     parent=int(pred[terminal]);assert 0<=parent<S;loads[parent,b]+=values;routed[b]+=values;routecost[b]+=values*dist[terminal];nominalsum[b]+=values*nc[parent];motorod[b]+=values*mf[parent];coreod[b]+=values*cf[parent];expected[ep['components'][i],b]+=values;expected[ep['components'][target],b]-=values
    else:unrouted[b]+=values
  loads=propagate(order,pred,loads)
  for b in range(2):
   for t in range(5):flux[b,t]+=np.bincount(rr['state_physical_arc'],weights=loads[:,b,t],minlength=E)
  if done%100==0 or done==len(origins):
   st={'status':'running','tree':name,'completed_origins':done,'total_origins':len(origins),'tree_elapsed_s':time.monotonic()-started,'total_elapsed_s':time.monotonic()-globalstart};dump(R/'execution_state.json',st);print(name,done,'/',len(origins),'seconds',round(st['tree_elapsed_s'],1),flush=True)
 demand=np.array([dem['period_weights'][mask].sum(axis=0)for mask in eligible]);maxbalance=0.
 for b in range(2):
  for t in range(5):
   balance=np.bincount(comp[pr['from_index']],weights=flux[b,t],minlength=len(expected))-np.bincount(comp[pr['to_index']],weights=flux[b,t],minlength=len(expected));maxbalance=max(maxbalance,float(abs(balance-expected[:,b,t]).max()))
 costres=float(np.max(abs(np.einsum('bte,e->bt',flux,arc_cost)-routecost)/np.maximum(1,abs(routecost))));nomres=float(np.max(abs(np.einsum('bte,e->bt',flux,basecost)-nominalsum)/np.maximum(1,abs(nominalsum))));original=load(REP/'C_topology_multi_flows.npz');used=dem['through_plus'];oo=dem['origin_endpoint'][used];dd=dem['destination_endpoint'][used];w=dem['period_weights'][used,0];regret=None
 if k:
  optimum=load(P/f'F{fee}_R0_flows.npz')['endpoint_cost_seconds'][oo,dd];ratio=nominalmatrix[oo,dd]/optimum;regret={'max_ratio':float(ratio.max()),'weighted_p95_ratio':weighted_quantile(ratio,w,.95),'bound':1.1/.9}
 checks={'demand_equals_routed_plus_unrouted':bool(abs(demand-routed-unrouted).max()<1e-6),'nonnegative_finite_flow':bool(np.isfinite(flux).all()and(flux>=0).all()),'component_flow_conservation':maxbalance<1e-6,'perturbed_cost_loading_agrees':costres<1e-9,'unperturbed_cost_loading_agrees':nomres<1e-9,'unchanged_eligible_demand':bool(abs(demand-original['input_demand']).max()<1e-6),'original_endpoint_feasibility_retained':bool(np.isfinite(lengths[np.isfinite(original['endpoint_distance_m'])]).all()),'positive_hash_for_used_OD':bool((fingerprint[oo,dd]!=0).all()),'bounded_route_regret':True if regret is None else regret['max_ratio']<=1.1/.9+1e-8,'base_realization_nominal_cost_exact':True if k else bool(np.allclose(nominalmatrix[oo,dd],lengths[oo,dd],rtol=1e-10,atol=1e-7))}
 np.savez_compressed(P/(name+'_flows.npz'),arc_flow=flux,routed=routed,unrouted=unrouted,input_demand=demand,endpoint_cost_seconds=lengths,endpoint_unperturbed_seconds=nominalmatrix,endpoint_path_hash=fingerprint,od_any_motorway=motorod,od_any_core=coreod)
 result={'tree':name,'fee_on':fee,'realization':k,'assignment_signature':signature,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':all(checks.values()),'checks':checks,'elapsed_s':time.monotonic()-started,'maximum_component_flow_residual':maxbalance,'perturbed_cost_relative_residual':costres,'unperturbed_cost_relative_residual':nomres,'regret':regret,'routed':routed.tolist(),'od_any_motorway':motorod.tolist()};dump(R/(name+'_validation.json'),result);print('FINISHED',name,result['checks'],flush=True)
 if not result['passed']:raise RuntimeError('Tree numerical check failed')
 return result

def main():
 assert all(json.loads((R/'preparation_summary.json').read_text())['checks'].values());binding=json.loads((P/'input_bindings.json').read_text())['sha256'];assert all(sha(ROOT/p)==h for p,h in binding.items()),'Prepared input binding changed';globalstart=time.monotonic();budget=json.loads((P/'protocol.json').read_text())['budget'];rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');attrs=load(P/'choice_costs.npz');signature=hashlib.sha256((json.dumps(binding,sort_keys=True)+sha(Path(__file__))).encode()).hexdigest();out=[]
 try:
  for fee in [0,1]:
   for k in range(3):
    marker=R/f'F{fee}_R{k}_validation.json'
    if marker.exists():
     previous=json.loads(marker.read_text())
     if previous['passed']and previous.get('assignment_signature')==signature:out.append(previous);continue
    out.append(run(fee,k,ep,dem,rr,pr,attrs,signature,globalstart,budget))
 except Exception as exc:
  dump(R/'execution_state.json',{'status':'stopped','reason':str(exc),'completed_trees':[x['tree']for x in out],'elapsed_s':time.monotonic()-globalstart});raise
 dump(R/'execution_state.json',{'status':'complete','completed_trees':[x['tree']for x in out],'new_hydraulics':False,'active_route_wall_s':sum(x['elapsed_s']for x in out),'elapsed_s':time.monotonic()-globalstart});print('CHOICE TREES COMPLETE',flush=True)
if __name__=='__main__':main()
