"""Demand-targeted road protection and two-path static capacity feedback.

All demands, capacities and passability rules are scenario assumptions. No network
equilibrium, dynamic queue, real engineering benefit or person impact is claimed.
Phases save bindings and diagnostics; original analyses are never rewritten.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import argparse, collections, datetime as dt, gzip, hashlib, json, re, sys, time
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix,load_npz,save_npz
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize
from numba import njit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import load,dump,sha,network
NAME='three_ward_demand_protection_feedback_v1'
P=ROOT/'data/processed'/NAME;R=ROOT/'data/exp/estimation'/NAME
OLD=ROOT/'data/processed/three_ward_assumed_transport_chain_v1'
REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1'
M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
C=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1'
O=ROOT/'data/processed/three_ward_observed_load_pilot_v1'
F=ROOT/'data/processed/three_ward_cross_event_protection_v1'
N=ROOT/'data/processed/three_ward_normal_transport_network_v1'
ARMS=['normal','flood','old','targeted']
LABEL={'normal':'正常','flood':'积水无措施','old':'原通道方案','targeted':'需求加权方案'}
CAPACITY={'motorway':1800.,'trunk':1500.,'primary':1200.,'secondary':1000.,'tertiary':900.,'unclassified':600.,'residential':600.,'service':300.,'living_street':150.,'other':300.}
def save(path,data):pd.DataFrame(data).to_csv(path,index=False,encoding='utf-8-sig')
def check_sources():
 b=json.loads((P/'input_bindings.json').read_text())['sha256'];assert all(sha(ROOT/n)==h for n,h in b.items()),'Frozen source changed'
def numeric_lane(x):
 s=str(x or '').strip()
 return float(s) if re.fullmatch(r'[1-9][0-9]?',s)and int(s)<=12 else None

def prepare():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 deps=[REP/'turn_router.npz',REP/'multi_endpoint_components.npz',REP/'multi_routing_demands.npz',M/'road_projection.npz',M/'road_inventory.csv',C/'choice_costs.npz',O/'normal_path_arcs.npz',O/'basis_order.npz',OLD/'branch0_normal.npz',OLD/'branch0_flood.npz',OLD/'branch0_protected.npz',OLD/'scenario_masks.npz',F/'motor_common_support.npz',F/'motor_exposures/E20250911__H__U.npz',N/'surface_physical_edges.jsonl.gz',ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson']
 binding={str(f.relative_to(ROOT)):sha(f)for f in deps}
 if (P/'protocol.json').exists():check_sources();return
 protocol={
 'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'human_decision':'MILI-D-20261018-010',
 'question':'Can demand-weighted protection restore more requested trips under the same length budget,and does recovered traffic shift static capacity pressure?',
 'selection_snapshot':'ExistingE20250911__H__U,elapsed14400s,2025-09-11about17:15JST; radar rain+assumed1.5m stage rise,U geometry',
 'transfer_snapshot':'Same event elapsed18000s,about18:15JST; frozen17h demand for controlled transfer,not actual18h reconstruction',
 'protection_budget_m':4766.386718020637,
 'selection':'On lost base-demand OD with a core endpoint,group closed known core-intersecting source roads along existing normal U paths. Greedy choose among32highest own-weight/new-length bundles using exact incidental covered-bundle gain/new-length,within whole-source-road length budget. No optimality claim;in-sample selection. Only closed known roads selected;unknown heights remain unknown.',
 'selection_limits':{'shortlist':32,'maximum_bundle_additions':300},
 'comparison':'No intervention,original fixed corridor and selected plan;normal is shared reference. Ideal protection restores dry reference time and capacity on known selected roads,no engineering rerun.',
 'demand':'Original PT-derived generic small-vehicle proxy. Base/no remote-remote plus original conditional through shape. Core-endpoint proxy and full10km context reported separately. Unknown-clock/coarse/intrazonal/class gaps retained in previous ledger. Not real vehicle totals or people.',
 'capacity_lane_rule':'Explicit numeric directional lanes first;otherwise (numeric total minus shared both_ways lanes)/permitted directions;otherwise1effective lane per direction. These are capacity-model assumptions,not surveyed usable lanes. No capacity inferred from12h census volume or congestion degree.',
 'capacity_veh_per_hour_per_effective_lane':CAPACITY,
 'capacity_uncertainty':'Factors0.5,1,1.5 of assumed class capacity;slow roads0.5factor,closed0. Unknown roads retained at assumed normal condition;previous all-unknown-withheld branch remains evidence stress only.',
 'cost_function':'reference U cost * (1+0.15*(directed arc flow / assumed hourly capacity)^4). Reference U already contains daytime survey/proxy conditions and is not measured free-flow time;output is hypothetical capacity-response cost,not true queue delay.',
 'path_pool':'Two feasible turn-aware paths per reachable OD:reference-time shortest path;and shortest path at central base AON feedback costs clipped at50xreference for candidate generation only. Final objective uncapped. Repeated paths allowed. Restricted two-path convex assignment,not full-network user equilibrium.',
 'fit':'Minimize integrated separable cost over0<=routeB_flow<=ODweight;fixed requested OD;unreachable stays in ledger,not queues. L-BFGS-B,up to2000iterations/120sperfit;restricted relative gap<=0.001required. Objective scaled by1000seconds. Stop subsequent sensitivity fits if central fails gap;retain incomplete status.',
 'schedule':[{'name':'central','demand_factor':1.,'capacity_factor':1.,'shape':'base'},{'name':'low_demand','demand_factor':.5,'capacity_factor':1.,'shape':'base'},{'name':'high_demand','demand_factor':2.,'capacity_factor':1.,'shape':'base'},{'name':'low_capacity','demand_factor':1.,'capacity_factor':.5,'shape':'base'},{'name':'high_capacity','demand_factor':1.,'capacity_factor':1.5,'shape':'base'},{'name':'through_plus','demand_factor':1.,'capacity_factor':1.,'shape':'plus'}],
 'homogeneity':'Low_capacity has same demand/capacity ratio as high_demand;derive flow/time by exact scaling after independent formula/gap checks. No extra optimizer fit.',
 'gates':'Source hashes,nonnegative finite flow,OD demand accounting,closed-arc0,component flow balance<1e-6,cost-load equality<1e-9,restricted gap<=0.001. No observed-calibration claim.',
 'budget':{'maximum_new_path_sweeps':10,'seconds_per_sweep':180,'total_path_seconds':1800,'seconds_per_fit':120,'total_fit_seconds':2400,'new_hydraulics':0,'new_training':0,'people_counted':0},
 'unknown_limit':'Main assumptions retain unverified bridge/tunnel/outside roads;do not present previous normal1.4percent coverage stress as actual loss bound.',
 }
 dump(P/'protocol.json',protocol);dump(P/'input_bindings.json',{'sha256':binding})
 dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');inv=pd.read_csv(M/'road_inventory.csv');state=load(OLD/'branch0_flood.npz');core=load(OLD/'scenario_masks.npz')['core_endpoint_od'];qualified=load(F/'motor_common_support.npz')['qualified_roads'];ridx=pr['arc_to_road']
 order=load(O/'basis_order.npz')['source_expanded_od_index'];H=load_npz(O/'normal_path_arcs.npz').tocsc()[:,np.argsort(order)].astype(np.float64);assert H.shape[1]==len(dem['base'])
 replay=np.asarray(H@(dem['period_weights'][:,3]*dem['base']));assert abs(replay-load(OLD/'branch0_normal.npz')['arc_flow'][0,0]).max()<1e-6
 save_npz(P/'normal_A.npz',H)
 target=dem['base']&core&~np.isfinite(state['trip_cost_seconds']);eligible_road=state['closed']&qualified&inv.inside_core.to_numpy();groups={};excluded_weight=0.
 for q in np.flatnonzero(target):
  roads=np.unique(ridx[H.indices[H.indptr[q]:H.indptr[q+1]]]);need=roads[state['closed'][roads]];assert len(need)>0
  if not eligible_road[need].all():excluded_weight+=float(dem['period_weights'][q,3]);continue
  key=tuple(map(int,need));groups[key]=groups.get(key,0.)+float(dem['period_weights'][q,3])
 bundles=list(sorted(groups));weights=np.array([groups[k]for k in bundles]);rr=[];cc=[]
 for j,k in enumerate(bundles):rr.extend([j]*len(k));cc.extend(k)
 B=csc_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(bundles),len(inv))).tocsr();sizes=np.diff(B.indptr);selected=np.zeros(len(inv),bool);length=inv.length_m.to_numpy();budget=protocol['protection_budget_m'];log=[]
 for step in range(300):
  nleft=sizes-np.asarray(B@selected.astype(float)).ravel();uncovered=nleft>.5;left=np.asarray(B@(length*~selected)).ravel();spent=float(length[selected].sum());candidates=np.flatnonzero(uncovered&(left<=budget-spent+1e-8)&(left>0))
  if not len(candidates):break
  ranking=candidates[np.lexsort((candidates,-weights[candidates]/left[candidates]))[:32]];best=None
  for j in ranking:
   add=np.array(bundles[j],int);add=add[~selected[add]];covered=np.asarray(B[:,add].sum(axis=1)).ravel();gain=float(weights[uncovered&(covered>=nleft-.5)].sum());ratio=gain/length[add].sum();key=(ratio,gain,-int(j))
   if best is None or key>best[0]:best=(key,j,add,gain)
  _,j,add,gain=best;selected[add]=True
  log.append(dict(step=step+1,seed_bundle=int(j),new_roads=len(add),new_length_m=float(length[add].sum()),marginal_normal_path_restored_proxy=gain,cumulative_length_m=float(length[selected].sum()),normal_path_restored_proxy=float(weights[np.asarray(B@selected.astype(float)).ravel()>=sizes-.5].sum())))
 assert float(length[selected].sum())<=budget+1e-6 and not(selected&~eligible_road).any()
 old=load(OLD/'scenario_masks.npz')['protected'];np.savez_compressed(P/'protection_masks.npz',targeted=selected,old=old,core_endpoint_od=core)
 save(P/'selection_steps.csv',log);save(P/'selected_roads.csv',inv[selected]);dump(R/'selection_summary.json',dict(target_lost_hour17_proxy=float(dem['period_weights'][target,3].sum()),excluded_outside_core_closed_bundle_proxy=excluded_weight,candidate_bundles=len(bundles),selected_roads=int(selected.sum()),selected_whole_length_m=float(length[selected].sum()),old_length_m=float(length[old].sum()),normal_path_guaranteed_restored_proxy=log[-1]['normal_path_restored_proxy']if log else 0,selection_is_in_sample=True))
 # Derive capacity assumptions once and retain tag provenance.
 lookup={str(k):j for j,k in enumerate(inv.source_edge_id)};tags={}
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as h:
  for line in h:
   e=json.loads(line);j=lookup.get(e['edge_id'])
   if j is not None:tags[j]=e['tags']
 graph=load(M/'source_motor_graph.npz');forward=graph['node_ids'][pr['from_index']]==inv.from_node.to_numpy()[ridx];directions=np.bincount(ridx,minlength=len(inv));lanes=np.ones(len(ridx));provenance=np.empty(len(ridx),object);cap=np.zeros(len(ridx));roadtypes=inv.highway.to_numpy();rows=[]
 for a,j in enumerate(ridx):
  t=tags[int(j)];suffix='forward'if forward[a]else'backward';directional=numeric_lane(t.get('lanes:'+suffix));total=numeric_lane(t.get('lanes'));shared=numeric_lane(t.get('lanes:both_ways'))or 0
  if directional is not None:lane=directional;why='numeric_directional'
  elif total is not None and total>shared:lane=(total-shared)/directions[j];why='numeric_total_split_assumption'
  else:lane=1.;why='one_lane_per_direction_assumption'
  cls=str(roadtypes[j]).removesuffix('_link');per=CAPACITY.get(cls,CAPACITY['other']);lanes[a]=lane;provenance[a]=why;cap[a]=per*lane
 for k in np.unique(provenance):rows.append(dict(provenance=k,arcs=int((provenance==k).sum()),directed_length_km=float(pr['length_m'][provenance==k].sum()/1000)))
 np.savez_compressed(P/'capacity_assumptions.npz',hourly_capacity=cap,effective_lanes=lanes,lane_provenance=provenance.astype(str));save(P/'capacity_provenance.csv',rows)
 dump(R/'preparation_validation.json',dict(passed=True,normal_prior_flow_replay_max=float(abs(replay-load(OLD/'branch0_normal.npz')['arc_flow'][0,0]).max()),selection_budget_passed=True,selected_known_closed_only=True,positive_capacities=bool((cap>0).all()),declared_not_observed_capacities=True))
 print(json.dumps(json.loads((R/'selection_summary.json').read_text()),indent=2),flush=True)

def road_state(arm,frame=5):
 q=load(F/'motor_common_support.npz')['qualified_roads'];ex=load(F/'motor_exposures/E20250911__H__U.npz');depth=ex['lower'][frame];closed=q&(depth>=.3);slow=q&(depth>=.1)&~closed
 if arm=='normal':closed[:]=False;slow[:]=False
 elif arm in ['old','targeted']:
  protected=load(P/'protection_masks.npz')[arm];closed[protected]=False;slow[protected]=False
 pr=load(M/'road_projection.npz');ridx=pr['arc_to_road'];cost=load(C/'choice_costs.npz')['travel_seconds'].copy();cost[slow[ridx]]*=2;cap=load(P/'capacity_assumptions.npz')['hourly_capacity'].copy();cap[slow[ridx]]*=.5
 return closed,slow,cost,cap

@njit(cache=True)
def trace_paths(parents,S,pred,arcmap):
 ptr=np.zeros(len(parents)+1,np.int64)
 for j in range(len(parents)):
  s=parents[j];k=0
  while 0<=s<S:k+=1;s=pred[s]
  ptr[j+1]=ptr[j]+k
 ix=np.empty(ptr[-1],np.int32)
 for j in range(len(parents)):
  s=parents[j];pos=ptr[j]
  while 0<=s<S:ix[pos]=arcmap[s];pos+=1;s=pred[s]
 return ptr,ix

def path_sweep(arm,kind,frame=5,globalstart=None):
 prefix=f'{arm}_{kind}'if frame==5 else f'transfer_{arm}_A';dest=P/(prefix+'.npz');marker=R/(prefix+'_path_validation.json')
 if marker.exists()and json.loads(marker.read_text()).get('passed'):return
 start=time.monotonic();rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');closed,slow,cost,cap=road_state(arm,frame);S=len(rr['state_physical_arc']);K=len(ep['components']);E=len(cost);Nod=len(dem['base'])
 if kind=='B':
  HA=load_npz(P/f'{arm}_A.npz');fl=HA@(dem['period_weights'][:,3]*dem['base']);mult=1+.15*(fl/cap)**4;cost*=np.minimum(mult,50.)
 g=network(rr,ep,pr,cost,closed);lens=np.zeros(Nod,np.int64);parts=[];indices=[];costs=np.full(Nod,np.inf);origins=np.unique(dem['origin_endpoint'][dem['through_plus']])
 for done,i in enumerate(origins,1):
  if(P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause requested before routing origin')
  if time.monotonic()-start>180 or(globalstart is not None and time.monotonic()-globalstart>1800):raise RuntimeError('Registered path budget reached')
  dist,pred=dijkstra(g,indices=S+int(i),directed=True,return_predecessors=True);q=np.flatnonzero((dem['origin_endpoint']==i)&dem['through_plus']);ends=S+K+dem['destination_endpoint'][q];ok=np.isfinite(dist[ends]);costs[q[ok]]=dist[ends[ok]];chosen=q[ok];parents=pred[ends[ok]];assert ((parents>=0)&(parents<S)).all();ptr,ix=trace_paths(parents,S,pred,rr['state_physical_arc']);lens[chosen]=np.diff(ptr);parts.append(ix);indices.append(chosen)
  if done%150==0 or done==len(origins):print(json.dumps(dict(status='paths',arm=arm,kind=kind,frame=frame,origins=done,total=len(origins),seconds=time.monotonic()-start)),flush=True)
 colorder=np.concatenate(indices);sizes=lens[colorder];H=csc_matrix((np.ones(sum(sizes),np.float64),np.concatenate(parts),np.r_[0,np.cumsum(sizes)]),shape=(E,len(colorder)));allcols=np.full(Nod,-1,np.int64);allcols[colorder]=np.arange(len(colorder));co=H.tocoo();H=csc_matrix((co.data,(co.row,colorder[co.col])),shape=(E,Nod));H.sum_duplicates();reachable=lens>0;w=dem['period_weights'][:,3]*dem['base'];flow=np.asarray(H@w);pathcost=np.asarray(H.T@cost).ravel();known=reachable;err=float(abs(pathcost[known]-costs[known]).max(initial=0));closedflow=float(flow[closed[pr['arc_to_road']]].max(initial=0))
 # Independent incidence divergence for loaded paths.
 nc=int(rr['node_component'].max())+1;comp=rr['node_component'];ec=ep['components'];weights=w*reachable;expected=np.bincount(ec[dem['origin_endpoint']],weights=weights,minlength=nc)-np.bincount(ec[dem['destination_endpoint']],weights=weights,minlength=nc);actual=np.bincount(comp[pr['from_index']],weights=flow,minlength=nc)-np.bincount(comp[pr['to_index']],weights=flow,minlength=nc);balance=float(abs(actual-expected).max())
 checks=dict(path_cost_agreement=err<1e-6,closed_arc_zero=closedflow<1e-9,component_conservation=balance<1e-6,positive_paths=bool((lens[reachable]>0).all()))
 save_npz(dest,H);np.savez_compressed(P/(prefix+'_costs.npz'),trip_cost=costs,reachable=reachable)
 dump(marker,dict(passed=all(checks.values()),checks=checks,seconds=time.monotonic()-start,max_path_cost_error=err,max_component_balance=balance,code_sha256=sha(Path(__file__))));assert all(checks.values()),checks

def paths():
 check_sources();start=time.monotonic()
 for arm in ARMS:
  if arm!='normal':path_sweep(arm,'A',globalstart=start)
  path_sweep(arm,'B',globalstart=start)
 for arm in ['flood','old','targeted']:path_sweep(arm,'A',frame=7,globalstart=start)
 check_sources();dump(R/'path_campaign.json',dict(status='complete',new_sweeps=10,seconds=time.monotonic()-start))

def optimize_pool(H0,H1,q,tref,cap,initial=None,seconds=120):
 """Convex two-route loading. Relative gap is over this fixed path pool only."""
 D=(H1-H0).tocsc();f0=np.asarray(H0@q).ravel();counter=0;started=time.monotonic();last={}
 def evaluate(x):
  nonlocal counter,last
  if time.monotonic()-started>seconds:raise RuntimeError('Registered per-fit budget reached')
  f=f0+np.asarray(D@x).ravel();assert f.min(initial=0)>-1e-6;f=np.maximum(f,0.);ratio=f/cap;tt=tref*(1+.15*ratio**4);obj=float(tref@(f*(1+.03*ratio**4)))/1000;grad=np.asarray(D.T@tt).ravel()/1000;counter+=1;last=dict(flow=f,cost=tt,objective=obj)
  return obj,grad
 x0=np.zeros(len(q))if initial is None else np.clip(initial,0,q)
 out=minimize(evaluate,x0,jac=True,bounds=list(zip(np.zeros(len(q)),q)),method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-13,'gtol':1e-7,'maxls':30,'maxcor':10})
 evaluate(out.x);f=last['flow'];tt=last['cost'];ca=np.asarray(H0.T@tt).ravel();cb=np.asarray(H1.T@tt).ravel();actual=float(f@tt);minimum=float(q@np.minimum(ca,cb));gap=max(0.,(actual-minimum)/max(actual,1));route_time=np.divide((q-out.x)*ca+out.x*cb,q,out=np.zeros_like(q),where=q>0)
 return dict(flow=f,cost=tt,routeB=out.x,mean_od_cost=route_time,gap=gap,iterations=int(out.nit),function_calls=counter,optimizer_success=bool(out.success),message=str(out.message),seconds=time.monotonic()-started)

def test_optimizer():
 h0=csc_matrix(np.array([[1.],[0.]]));h1=csc_matrix(np.array([[0.],[1.]]));z=optimize_pool(h0,h1,np.array([3.]),np.array([10.,10.]),np.array([1.,2.]),seconds=5)
 assert np.allclose(z['flow'],[1,2],atol=1e-4)and z['gap']<1e-6,z
 dump(R/'optimizer_selfcheck.json',dict(passed=True,analytic_two_parallel_routes_flow=z['flow'],expected_flow=[1,2],gap=z['gap'],equal_cost=z['cost']))

def fit():
 check_sources();test_optimizer();start=time.monotonic();protocol=json.loads((P/'protocol.json').read_text());dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');comp=rr['node_component'];ec=ep['components'];nc=int(comp.max())+1;summary=[]
 for arm in ARMS:
  H0=load_npz(P/f'{arm}_A.npz');H1=load_npz(P/f'{arm}_B.npz');assert np.array_equal(np.diff(H0.indptr)>0,np.diff(H1.indptr)>0);reachable=np.diff(H0.indptr)>0;closed,slow,tref,cap=road_state(arm);previous_fraction=None
  for scenario in protocol['schedule']:
   name=arm+'__'+scenario['name'];marker=R/(name+'_fit_validation.json')
   if marker.exists()and json.loads(marker.read_text()).get('passed'):summary.append(json.loads(marker.read_text()));continue
   if time.monotonic()-start>2400:raise RuntimeError('Registered total fit budget reached')
   if(P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause before fit')
   eligible=dem['base']if scenario['shape']=='base'else dem['through_plus'];qq=dem['period_weights'][:,3]*scenario['demand_factor']*eligible;active=reachable&(qq>0);q=qq[active];a=H0[:,active];b=H1[:,active];t=cap*scenario['capacity_factor'];derived=False
   if scenario['name']=='low_capacity':
    high=load(P/(arm+'__high_demand.npz'));flow=high['arc_flow']*.5;cost=tref*(1+.15*(flow/t)**4);routes=high['routeB_flow']*.5;ca=np.asarray(a.T@cost).ravel();cb=np.asarray(b.T@cost).ravel();actual=float(flow@cost);gap=max(0.,(actual-float(q@np.minimum(ca,cb)))/max(actual,1));z=dict(flow=flow,cost=cost,routeB=routes,mean_od_cost=((q-routes)*ca+routes*cb)/q,gap=gap,iterations=0,function_calls=0,optimizer_success=True,message='Exact demand/capacity homogeneity from high_demand',seconds=0.);derived=True
   else:
    warm=previous_fraction*q if previous_fraction is not None and len(previous_fraction)==len(q)else None
    z=optimize_pool(a,b,q,tref,t,warm)
   if scenario['name']=='central':previous_fraction=z['routeB']/q
   flow=z['flow'];weights=qq*reachable;expected=np.bincount(ec[dem['origin_endpoint']],weights=weights,minlength=nc)-np.bincount(ec[dem['destination_endpoint']],weights=weights,minlength=nc);actual=np.bincount(comp[pr['from_index']],weights=flow,minlength=nc)-np.bincount(comp[pr['to_index']],weights=flow,minlength=nc);balance=float(abs(actual-expected).max());closedmax=float(flow[closed[pr['arc_to_road']]].max(initial=0));odcost=np.full(len(qq),np.inf);odcost[active]=z['mean_od_cost'];inactive=reachable&~active;odcost[inactive]=np.asarray(H0[:,inactive].T@z['cost']).ravel();costerr=abs(float(q@z['mean_od_cost'])-float(flow@z['cost']))/max(1,float(flow@z['cost']))
   checks=dict(restricted_gap=z['gap']<=.001,nonnegative_finite_flow=bool(np.isfinite(flow).all()and flow.min()>=0),OD_conservation=bool(((z['routeB']>=-1e-8)&(z['routeB']<=q+1e-8)).all()),component_conservation=balance<1e-6,closed_zero=closedmax<1e-9,cost_loading=costerr<1e-9,request_ledger=float(abs(qq.sum()-weights.sum()-qq[~reachable].sum()))<1e-6)
   np.savez_compressed(P/(name+'.npz'),arc_flow=flow,arc_cost_seconds=z['cost'],od_mean_cost_seconds=odcost,routeB_flow=z['routeB'],active_od_index=np.flatnonzero(active),requested=qq,reachable=reachable,capacity=t)
   val=dict(name=name,arm=arm,scenario=scenario,passed=all(checks.values()),checks=checks,gap=z['gap'],iterations=z['iterations'],optimizer_success=z['optimizer_success'],optimizer_message=z['message'],seconds=z['seconds'],balance_residual=balance,cost_loading_relative_error=costerr,derived_by_homogeneity=derived,code_sha256=sha(Path(__file__)));dump(marker,val);summary.append(val);print(json.dumps({k:val[k]for k in ['name','passed','gap','iterations','seconds']}),flush=True)
   if not val['passed']:dump(R/'fit_campaign.json',dict(status='stopped_numerical_or_gap',failed=name,completed=[x['name']for x in summary]));raise RuntimeError('Predeclared fit checks failed: '+name)
 dump(R/'fit_campaign.json',dict(status='complete',cases=len(summary),seconds=time.monotonic()-start));check_sources()

def report():
 check_sources();assert json.loads((R/'fit_campaign.json').read_text())['status']=='complete';dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');core=load(P/'protection_masks.npz')['core_endpoint_od'];inv=pd.read_csv(M/'road_inventory.csv');protocol=json.loads((P/'protocol.json').read_text());rows=[];savings=[];stress=[];base={};known=load(F/'motor_common_support.npz')['qualified_roads'];ridx=pr['arc_to_road'];own=inv.inside_core.to_numpy()[ridx];fluxout=[]
 for s in protocol['schedule']:
  scenario=s['name'];zs={a:load(P/(a+'__'+scenario+'.npz'))for a in ARMS};normal=zs['normal'];flood=zs['flood'];req=normal['requested'];eligible=dem['base']if s['shape']=='base'else dem['through_plus'];mask=core&eligible;nr=normal['reachable'];fr=flood['reachable'];normal_ok=mask&nr;common=normal_ok&fr
  for arm,z in zs.items():
   good=normal_ok&z['reachable'];lost=normal_ok&~z['reachable'];extra=z['od_mean_cost_seconds'][good]-normal['od_mean_cost_seconds'][good];uncongested=load_npz(P/f'{arm}_A.npz');minimum=np.asarray(uncongested.T@road_state(arm)[2]).ravel();growth=z['od_mean_cost_seconds'][good]-minimum[good]
   rows.append(dict(arm=arm,scenario=scenario,requested_core_proxy=float(req[mask].sum()),normal_reachable_core_proxy=float(req[normal_ok].sum()),new_unreachable_proxy=float(req[lost].sum()),new_unreachable_share=float(req[lost].sum()/req[normal_ok].sum()),restored_vs_flood_proxy=float(req[normal_ok&~fr&z['reachable']].sum()),mean_capacity_response_cost_min=float(req[good]@z['od_mean_cost_seconds'][good]/req[good].sum()/60),mean_extra_vs_normal_survivors_min=float(req[good]@extra/req[good].sum()/60),mean_above_minimum_reference_path_min=float(req[good]@growth/req[good].sum()/60)))
   vc=z['arc_flow']/z['capacity'];closed=road_state(arm)[0][ridx];qual=own&~closed;stress.append(dict(arm=arm,scenario=scenario,directed_core_model_arcs=int(qual.sum()),vc_gt1_directed_core_km=float(pr['length_m'][qual&(vc>1)].sum()/1000),vc_gt1_known_directed_core_km=float(pr['length_m'][qual&known[ridx]&(vc>1)].sum()/1000),vc_gt1_unknown_directed_core_km=float(pr['length_m'][qual&~known[ridx]&(vc>1)].sum()/1000),capacity_pressure_is_assumed=True))
   if scenario=='central':fluxout.append(pd.DataFrame(dict(physical_arc=np.arange(len(ridx)),road_index=ridx,arm=arm,flow=z['arc_flow'],capacity=z['capacity'],cost_seconds=z['arc_cost_seconds'])))
  for arm in ['old','targeted']:
   z=zs[arm];paired=common&z['reachable'];saving=flood['od_mean_cost_seconds'][paired]-z['od_mean_cost_seconds'][paired];restored=normal_ok&~fr&z['reachable'];savings.append(dict(arm=arm,scenario=scenario,flood_lost_proxy=float(req[normal_ok&~fr].sum()),restored_proxy=float(req[restored].sum()),restored_fraction_of_flood_lost=float(req[restored].sum()/req[normal_ok&~fr].sum()),paired_already_reachable_mean_saving_min=float(req[paired]@saving/req[paired].sum()/60),paired_already_reachable_worse_proxy=float(req[paired&(z['od_mean_cost_seconds']>flood['od_mean_cost_seconds']+1e-6)].sum())))
  base[scenario]=zs
 transfer=[]
 for arm in ['flood','old','targeted']:
  h=load_npz(P/f'transfer_{arm}_A.npz');re=np.diff(h.indptr)>0;normal=np.diff(load_npz(P/'normal_A.npz').indptr)>0;mask=core&dem['base']&normal;w=dem['period_weights'][:,3];transfer.append(dict(arm=arm,elapsed_s=18000,frozen_demand_period='17h proxy',normal_core_requested=float(w[mask].sum()),unreachable_proxy=float(w[mask&~re].sum()),unreachable_share=float(w[mask&~re].sum()/w[mask].sum())))
 save(P/'paired_capacity_response_metrics.csv',rows);save(P/'protection_benefits.csv',savings);save(P/'assumed_capacity_pressure.csv',stress);save(P/'transfer_access_check.csv',transfer);pd.concat(fluxout).to_csv(P/'central_arc_feedback.csv',index=False,encoding='utf-8-sig')
 vals=[json.loads(f.read_text())for f in R.glob('*_fit_validation.json')];assert len(vals)==24 and all(v['passed']for v in vals)
 dump(R/'validation.json',dict(passed=True,fit_cases=24,path_sweeps=10,numerical_checks_all_passed=True,maximum_restricted_gap=max(v['gap']for v in vals),maximum_balance_residual=max(v['balance_residual']for v in vals),no_full_network_equilibrium_claim=True,not_observed_traffic_or_people=True))
 summary=dict(selection=json.loads((R/'selection_summary.json').read_text()),central=[x for x in rows if x['scenario']=='central'],central_benefits=[x for x in savings if x['scenario']=='central'],transfer=transfer,new_hydraulics=False,people_counted=False)
 dump(R/'summary.json',summary);plot_and_html(pd.DataFrame(rows),pd.DataFrame(savings),pd.DataFrame(stress),pd.DataFrame(transfer),summary)
 print(json.dumps(summary,indent=2),flush=True)

def plot_and_html(metrics,benefits,pressure,transfer,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import LineCollection
 import geopandas as gpd
 from shapely.geometry import shape
 from shapely.ops import unary_union
 bnd=unary_union(gpd.read_file(ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').to_crs(6677).geometry.to_numpy());inv=pd.read_csv(M/'road_inventory.csv');masks=load(P/'protection_masks.npz');lookup={str(k):i for i,k in enumerate(inv.source_edge_id)};geoms={}
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as h:
  for line in h:
   e=json.loads(line);j=lookup.get(e['edge_id'])
   if j is not None and inv.inside_core.iloc[j]:geoms[j]=shape(e['geometry']).intersection(bnd)
 def lines(g):
  if g.is_empty:return []
  if g.geom_type=='LineString':return[np.asarray(g.coords)/1000]
  return[np.asarray(x.coords)/1000 for x in getattr(g,'geoms',[])if x.geom_type=='LineString']
 polys=[]
 for j in np.flatnonzero(masks['targeted']):
  if j in geoms:polys.append({'type':'Feature','geometry':__import__('shapely').geometry.mapping(geoms[j]),'properties':{'road_index':int(j),'source_edge_id':str(inv.source_edge_id.iloc[j]),'assumption':'ideal passability/capacity protection'}})
 dump(P/'selected_plan_6677.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':polys})
 fig,axs=plt.subplots(1,2,figsize=(10,8),layout='constrained');bounds=np.array(bnd.bounds)/1000
 for ax,arm in zip(axs,['old','targeted']):
  ax.add_collection(LineCollection([l for g in geoms.values()for l in lines(g)],colors='#dce3e5',linewidths=.35));ax.add_collection(LineCollection([l for j,g in geoms.items()if masks[arm][j]for l in lines(g)],colors='#176cad',linewidths=1.8));ax.set(xlim=[bounds[0]-.1,bounds[2]+.1],ylim=[bounds[1]-.1,bounds[3]+.1],aspect='equal',title=arm,xlabel='Easting(km)',ylabel='Northing(km)')
 fig.suptitle('Same maximum length budget | original vs demand-targeted ideal protection');fig.savefig(R/'protection_plans.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');cent=metrics[metrics.scenario.eq('central')].set_index('arm').loc[ARMS];axs[0].bar(ARMS,cent.new_unreachable_share*100,color=['#7696a5','#c25f51','#74947d','#2679a2']);axs[0].set(ylabel='Newly unreachable proxy (%)',title='Central assumed demand')
 for arm in ['flood','old','targeted']:
  rr=metrics[metrics.arm.eq(arm)].set_index('scenario').loc[['low_demand','central','high_demand']];axs[1].plot([.5,1,2],rr.mean_extra_vs_normal_survivors_min,marker='o',label=arm)
 axs[1].set(xlabel='Demand factor',ylabel='Paired survivor added cost (min)',title='Capacity-response cost,not observed queues');axs[1].legend();fig.savefig(R/'capacity_feedback.png',dpi=150);plt.close(fig)
 def tab(df):return'<div class="scroll">'+df.to_html(index=False,border=0,float_format=lambda x:f'{x:,.3f}')+'</div>'
 s=summary['selection'];body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>需求加权通道保障与假设容量反馈</title><style>body{{margin:0;background:#edf2f5;color:#233e4b;font:16px/1.8 system-ui}}main{{max-width:1160px;margin:28px auto;background:white;padding:30px;border-radius:14px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:28px}}.lead{{padding:18px;background:#e7f0ed;border-left:5px solid #428775}}.note{{padding:16px;background:#fff0dd}}table{{border-collapse:collapse;font-size:13px;min-width:750px;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}a{{color:#146587}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main><p>2026-10-18 JST · 都心三区及必要道路上下文 · MILI-D-20261018-010</p><h1>需求加权通道比较与静态容量反馈</h1><div class="lead">已完成同长度预算的需求加权方案、原方案和无措施对照，24组容量／需求组合以及另一个既有时刻的可达性迁移检查。没有新增水动力、机器学习或人口影响计算。</div>
<h2>选路目标与公平比较</h2><p>原方案总长{s['old_length_m']/1000:.3f}km。新方案使用不超过同一长度预算，选中{s['selected_roads']}段、{s['selected_whole_length_m']/1000:.3f}km；仅保障有可靠地表对应、在选定快照关闭的道路。按正常U路径构造需一起恢复的道路束，启发式选择可恢复更多三区起终点17时出行代理的组合。源道路整段长度用于预算，不按裁剪或双向弧重复计费。</p><p>这是在2025-09-11约17:15（起点后4h）、雷达雨场＋假设河道抬升情景上选路并评价，属于样内方案设计，没有全局最优或独立验证声明。路径候选与贪心限制写入合同。原方案保持不变；新旧方案均假设保护道路维持正常通行、参考成本和容量，不模拟工程、水位转移、成本或失效。</p><img src="protection_plans.png" alt="Original and demand-targeted protection plans under the same maximum length budget">
<h2>明确的容量假设</h2><p>有效车道优先使用数值型方向标签，其次将总车道扣除共享车道后按允许方向分配，缺失时每方向假设1条。每有效车道小时容量按道路等级设定：高速1800、trunk1500、primary1200、secondary1000、tertiary900、普通居住/未分类600、service300、living street150，其余300。不是实测容量，不能拿这些假设宣称补齐了观测。</p><p>对需求0.5／1／2倍、容量0.5／1／1.5倍的预定单因素对照，以及原外部穿越敏感性进行比较。缓行道路参考成本×2、容量×0.5。成本函数为 U参考时耗×[1+0.15(v/c)^4]；U含已有日间观测／代理条件，不是实测自由流时耗，因此“增时”应解读为假设容量响应成本，不是真实排队延误。</p><p>每个OD最多两条遵守原单行与转向的路径，在这套候选内求解凸分配。第二条由中央流量压力引导生成，可能与第一条重复。受限相对间隙≤0.001；这证明候选内数值收敛，不证明全路网均衡或真实路径选择。低容量与高需求按齐次关系复用，并独立检查公式与间隙，不能当成独立实验。</p>
<h2>中央条件下的出行后果</h2>{tab(metrics[metrics.scenario.eq('central')])}<p>新增不可达只在正常可达需求中统计；新恢复的需求不被归一化丢弃，也没有计作人口。无法出发／完成的出行留账，但静态模型不生成它们的排队。所有成本均须结合请求完成量理解。</p>{tab(benefits[benefits.scenario.eq('central')])}<p>措施时间收益仅比较无措施下已经可达的同一组出行，恢复可达另列。恢复车辆可能改变其他道路的压力，使部分原有出行变慢；不能只报总收益而隐藏这些变化。</p><img src="capacity_feedback.png" alt="Restored accessibility and capacity response cost across assumed demand factors">
<h2>容量压力转移</h2>{tab(pressure[pressure.scenario.eq('central')])}<p>这里的长度是有方向弧的模型压力长度，可能包含同一源道路两个方向，不能与关闭道路源长度直接相加。分别列出有可靠地表对应和未知道路上的压力。阈值v/c&gt;1仅为假设容量压力，未包含交叉口、回堵传播、信号配时与车辆类别。</p>
<h2>冻结方案后，另一个时刻是否有效</h2>{tab(transfer)}<p>迁移时刻为同场起点后5h（约18:15），方案不重选；为隔离路况变化沿用17时需求，而非重现18时交通。此处仅检查可达性，不追加该时刻的容量反馈或水动力。它是相邻时刻迁移，不是独立暴雨事件验证。</p>
<h2>证据边界与数值核验</h2><p class="note">未知道路在主要情景中假设正常可用，不能宣称高架和隧道已证实安全。上一轮全部扣除未知后正常覆盖仅约1.4%的分支保留为证据诊断，不当成真实损失上界。PT需求、容量和通行阈值都是明确情景；本轮不是实测交通校准、历史拥堵再现或工程评估。</p><p>检查了输入哈希、道路束预算、正常原流量复现、解析双路径算例、全部路径的格弧与OD成本一致、组件节点车辆守恒、封闭道路零流量、请求需求账本和候选内优化间隙。原校准失败证据和此前结果保持不变。</p><ul>'''
 for fn in ['protocol.json','input_bindings.json','selection_steps.csv','selected_roads.csv','capacity_provenance.csv','paired_capacity_response_metrics.csv','protection_benefits.csv','assumed_capacity_pressure.csv','transfer_access_check.csv','selected_plan_6677.geojson']:
  body+=f'<li><a href="../../../processed/{NAME}/{fn}">{fn}</a></li>'
 body+='<li><a href="validation.json">validation.json</a></li><li><a href="summary.json">summary.json</a></li></ul><p>底层道路 © OpenStreetMap contributors / ODbL。原PLATEAU、DEM和统计来源沿用源审计。复现脚本 demand_protection_feedback_v1.py 按 prepare → paths → fit → report 执行。</p></main></html>'
 (R/'review.html').write_text(body)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','paths','fit','report']);a=ap.parse_args()
 try:globals()[a.phase]()
 except Exception as exc:
  if R.exists():dump(R/'execution_state.json',dict(status='stopped',phase=a.phase,reason=str(exc)))
  raise
 else:dump(R/'execution_state.json',dict(status='phase_complete',phase=a.phase,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
if __name__=='__main__':main()
