"""Trace fixed U routes once, preserving physical-arc multiplicities and turn memory."""
import sys,time,datetime as dt
from observed_load_common_v1 import *
from scipy.sparse import csc_matrix,save_npz,load_npz
from scipy.sparse.csgraph import dijkstra
from numba import njit
@njit(cache=True)
def trace(parent,S,pred,arcmap,cutptr,cutidx,ncuts):
 n=len(parent);offset=np.zeros(n+1,np.int64)
 for j in range(n):
  s=parent[j];k=0
  while 0<=s<S:k+=1;s=pred[s]
  offset[j+1]=offset[j]+k
 arcs=np.empty(offset[-1],np.int32);counts=np.zeros((ncuts,n),np.uint16)
 for j in range(n):
  s=parent[j];k=offset[j]
  while 0<=s<S:
   a=arcmap[s];arcs[k]=a;k+=1
   for t in range(cutptr[a],cutptr[a+1]):counts[cutidx[t],j]+=1
   s=pred[s]
 return offset,arcs,counts

def main():
 started=time.monotonic();binding=json.loads((P/'input_bindings.json').read_text())['sha256'];assert all(sha(ROOT/p)==v for p,v in binding.items());rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');cost=load(C/'choice_costs.npz')['arc_cost_seconds'][0,0];old=load(C/'F0_R0_flows.npz');o=pd.read_csv(P/'screenline_inventory.csv');S=len(rr['state_physical_arc']);K=len(ep['components']);E=len(cost);g=network(rr,ep,pr,cost);cutbyroad={}
 for r in o.itertuples():
  for road in str(r.matched_road_indices).split('|'):cutbyroad.setdefault(int(road),[]).append(r.screen_index)
 cptr=np.zeros(E+1,np.int64);cidx=[]
 for a,road in enumerate(pr['arc_to_road']):
  ii=cutbyroad.get(int(road),[]);cidx.extend(ii);cptr[a+1]=len(cidx)
 cidx=np.array(cidx,np.int16);origins=sorted(set(map(int,dem['origin_endpoint'][dem['through_plus']])));allq=[];lens=[];flats=[];cuts=[];distance=[];maxcost=0.
 for done,i in enumerate(origins,1):
  if(P/'PAUSE').exists():raise RuntimeError('Pause before next routing origin')
  if time.monotonic()-started>json.loads((P/'protocol.json').read_text())['budget']['seconds_per_sweep']:raise RuntimeError('Registered basis route budget')
  d,pred=dijkstra(g,indices=S+i,directed=True,return_predecessors=True);q=np.flatnonzero((dem['origin_endpoint']==i)&dem['through_plus']);ends=S+K+dem['destination_endpoint'][q];assert np.isfinite(d[ends]).all();pa=pred[ends];assert ((pa>=0)&(pa<S)).all();offset,arcs,cc=trace(pa,S,pred,rr['state_physical_arc'],cptr,cidx,len(o));allq.append(q);lens.append(np.diff(offset));flats.append(arcs);cuts.append(cc);distance.append(d[ends]);valid=np.isfinite(old['endpoint_cost_seconds'][i]);maxcost=max(maxcost,float(abs(d[S+K:][valid]-old['endpoint_cost_seconds'][i,valid]).max()))
  if done%100==0 or done==len(origins):
   state={'status':'basis_routing','origins_completed':done,'origins_total':len(origins),'elapsed_s':time.monotonic()-started};dump(R/'execution_state.json',state);print(state,flush=True)
 order=np.concatenate(allq);colptr=np.r_[0,np.cumsum(np.concatenate(lens))];ix=np.concatenate(flats);H=csc_matrix((np.ones(len(ix),np.uint8),ix,colptr),shape=(E,len(order)));A=csc_matrix(np.hstack(cuts));save_npz(P/'normal_path_arcs.npz',H);save_npz(P/'screenline_incidence.npz',A);np.savez_compressed(P/'basis_order.npz',source_expanded_od_index=order,route_cost_seconds=np.concatenate(distance));w=dem['period_weights'][order]*dem['base'][order,None];replay=np.asarray(H@w).T;res=float(abs(replay-old['arc_flow'][0]).max());support=np.array(A@w[:,4]).ravel();o['prior_small12h']=support;o['route_support']=np.asarray(A.sum(axis=1)).ravel()>0;o['support_status']=np.where(o.route_support,'supported','structural_no_prior_route');save('observation_route_support.csv',o)
 checks={'all_original_U_path_costs_replayed':maxcost<1e-7,'all_prior_base_arc_flows_reproduced':res<1e-6,'unique_source_OD_permutation':len(order)==int(dem['through_plus'].sum())and len(set(order))==len(order),'positive_path_lengths':bool((np.diff(colptr)>0).all()),'frozen_source_binding':all(sha(ROOT/p)==v for p,v in binding.items())};summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':all(checks.values()),'checks':checks,'seconds':time.monotonic()-started,'route_columns':len(order),'path_arc_entries':len(ix),'supported_screenlines':int(o.route_support.sum()),'unsupported_screenlines':o.loc[~o.route_support,['observation_section_id','road_class','observed_small_daytime12h']].to_dict('records'),'prior_arc_replay_max':res,'prior_cost_replay_max_seconds':maxcost};dump(R/'basis_validation.json',summary);print(json.dumps(summary,indent=2))
 if not summary['passed']:raise SystemExit('Route basis integrity failure')
if __name__=='__main__':main()
