"""One shared fixed-U shortest-path sweep for two separate linear demand increments.
The original U baseline is reused; increments do not interact or alter paths/costs.
"""
from pathlib import Path
import os,sys,json,time,hashlib,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from run_normal_vehicle_pilot_v1 import propagate
from run_normal_vehicle_choice_v1 import path_diagnostics
from scipy.sparse.csgraph import dijkstra
P=ROOT/'data/processed/three_ward_vehicle_demand_recovery_v1';R=ROOT/'data/exp/estimation/three_ward_vehicle_demand_recovery_v1';REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1';C=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1'
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else str(x))+'\n')
def main():
 assert all(json.loads((R/'preparation_summary.json').read_text())['checks'].values());bindings=json.loads((P/'input_bindings.json').read_text())['sha256'];assert all(sha(ROOT/n)==h for n,h in bindings.items());protocol=json.loads((P/'protocol.json').read_text());started=time.monotonic();rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(P/'increment_demands.npz');pr=load(M/'road_projection.npz');attrs=load(C/'choice_costs.npz');baseline=load(C/'F0_R0_flows.npz');arc_cost=attrs['arc_cost_seconds'][0,0];S=len(rr['state_physical_arc']);E=len(pr['length_m']);K=len(ep['components']);comp=rr['node_component'];startcomp=comp[pr['from_index']];endcomp=comp[rr['state_arc_end_node']];g0=csr_matrix((arc_cost[rr['state_physical_arc'][rr['indices']]],rr['indices'],rr['indptr']),shape=tuple(rr['shape']));su=[];sv=[];sc=[];eu=[];ev=[]
 for j,c in enumerate(ep['components']):
  arcs=np.flatnonzero(startcomp==c);su.extend([j]*len(arcs));sv.extend(arcs.tolist());sc.extend(arc_cost[arcs].tolist());states=np.flatnonzero(endcomp==c);eu.extend(states.tolist());ev.extend([j]*len(states))
 starts=csr_matrix((sc,(su,sv)),shape=(K,S));ends=csr_matrix((np.zeros(len(eu)),(eu,ev)),shape=(S,K));g=vstack([hstack([g0,csr_matrix((S,K)),ends]),hstack([starts,csr_matrix((K,K*2))]),csr_matrix((K,S+K*2))],format='csr');origins=sorted(set(map(int,dem['origin_endpoint'])));assert len(origins)<=protocol['budget']['maximum_origins'];flux=np.zeros((2,5,E));routed=np.zeros((2,5));unrouted=np.zeros((2,5));routecost=np.zeros((2,5));motor=np.zeros((2,5));core=np.zeros((2,5));ncpt=int(comp.max())+1;expected=np.zeros((ncpt,2,5));N=len(dem['origin_endpoint']);tripcost=np.full(N,np.nan);triplength=np.full(N,np.nan);tripmotor=np.zeros(N,bool);tripcore=np.zeros(N,bool);replaymax=0.;replayrows=0
 try:
  for done,i in enumerate(origins,1):
   if(P/'PAUSE').exists():raise RuntimeError('Pause requested before next origin')
   if time.monotonic()-started>protocol['budget']['maximum_route_seconds']:raise RuntimeError('Registered routing time budget exceeded')
   dist,pred=dijkstra(g,indices=S+i,directed=True,return_predecessors=True);order=np.argsort(dist[:S],kind='stable');order=order[np.isfinite(dist[:S][order])];nom,_,mf,cf=path_diagnostics(order,pred,rr['state_physical_arc'],arc_cost,attrs['arc_is_motorway'],attrs['arc_is_core']);length,_,_,_=path_diagnostics(order,pred,rr['state_physical_arc'],pr['length_m'],attrs['arc_is_motorway'],attrs['arc_is_core']);old=baseline['endpoint_cost_seconds'][i];valid=np.isfinite(old)
   if valid.any():replaymax=max(replaymax,float(abs(dist[S+K:][valid]-old[valid]).max()));replayrows+=1
   loads=np.zeros((S,2,5))
   for q in np.flatnonzero(dem['origin_endpoint']==i):
    b=int(dem['increment_index'][q]);j=int(dem['destination_endpoint'][q]);values=dem['period_weights'][q];terminal=S+K+j
    if not np.isfinite(dist[terminal]):unrouted[b]+=values;continue
    parent=int(pred[terminal]);assert 0<=parent<S;assert abs(nom[parent]-dist[terminal])<1e-6;loads[parent,b]+=values;routed[b]+=values;routecost[b]+=values*dist[terminal];motor[b]+=values*mf[parent];core[b]+=values*cf[parent];expected[ep['components'][i],b]+=values;expected[ep['components'][j],b]-=values;tripcost[q]=dist[terminal];triplength[q]=length[parent];tripmotor[q]=mf[parent];tripcore[q]=cf[parent]
   loads=propagate(order,pred,loads)
   for b in range(2):
    for t in range(5):flux[b,t]+=np.bincount(rr['state_physical_arc'],weights=loads[:,b,t],minlength=E)
   if done%20==0 or done==len(origins):
    state={'status':'running','origins_completed':done,'origins_total':len(origins),'elapsed_s':time.monotonic()-started};dump(R/'execution_state.json',state);print(state,flush=True)
 except Exception as exc:dump(R/'execution_state.json',{'status':'stopped','reason':str(exc),'elapsed_s':time.monotonic()-started});raise
 demand=np.array([dem['period_weights'][dem['increment_index']==b].sum(axis=0)for b in range(2)]);balance=0.
 for b in range(2):
  for t in range(5):
   actual=np.bincount(comp[pr['from_index']],weights=flux[b,t],minlength=ncpt)-np.bincount(comp[pr['to_index']],weights=flux[b,t],minlength=ncpt);balance=max(balance,float(abs(actual-expected[:,b,t]).max()))
 costres=float(np.max(abs(np.einsum('bte,e->bt',flux,arc_cost)-routecost)/np.maximum(1,routecost)));checks={'source_demand_routed_plus_unrouted':bool(abs(demand-routed-unrouted).max()<1e-6),'all_added_demand_reachable':bool(unrouted.max()<1e-9),'nonnegative_finite_flow':bool(np.isfinite(flux).all()and(flux>=0).all()),'native_component_flow_conservation':balance<1e-6,'same_cost_flow_loading':costres<1e-9,'replayed_existing_U_endpoint_costs':replayrows>0 and replaymax<1e-7,'positive_finite_added_route_length':bool(np.isfinite(triplength).all()and(triplength>0).all()),'frozen_model_inputs_unchanged':all(sha(ROOT/n)==h for n,h in bindings.items())};np.savez_compressed(P/'increment_flows.npz',arc_flow=flux,input_demand=demand,routed=routed,unrouted=unrouted,routecost=routecost,od_any_motorway=motor,od_any_core=core,trip_cost_seconds=tripcost,trip_length_m=triplength,trip_any_motorway=tripmotor,trip_any_core=tripcore);result={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':all(checks.values()),'checks':checks,'elapsed_s':time.monotonic()-started,'origins':len(origins),'existing_U_cost_rows_replayed':replayrows,'maximum_replayed_U_cost_difference_seconds':replaymax,'maximum_component_flow_residual':balance,'cost_loading_relative_residual':costres,'demand':demand,'routed':routed,'unrouted':unrouted,'od_any_motorway':motor,'od_any_core':core,'new_hydraulics':False,'affected_people':False};dump(R/'routing_validation.json',result);dump(R/'execution_state.json',{'status':'complete'if result['passed']else'failed_integrity','elapsed_s':result['elapsed_s'],'origins':len(origins)});print(json.dumps(result,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))
 if not result['passed']:raise SystemExit('Route integrity failure; outputs retained')
if __name__=='__main__':main()
