"""Frozen-demand normal/flood/ideal corridor protection pilot, without hydraulic calls.

PT-derived weights are declared vehicle-trip proxies, not reconstructed observations.
All source files remain read-only. Independent node balance and path cost checks gate
each tree. Existing count-fit failure is retained, not relabelled as validation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']: os.environ.pop(k,None)
from pathlib import Path
import argparse, datetime as dt, gzip, html, json, sys, time
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from numba import njit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import load,sha,dump,network
from run_normal_vehicle_pilot_v1 import propagate
P=ROOT/'data/processed/three_ward_assumed_transport_chain_v1'
R=ROOT/'data/exp/estimation/three_ward_assumed_transport_chain_v1'
BASE=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1'
REP=ROOT/'data/processed/three_ward_normal_vehicle_repair_v1'
M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
C=ROOT/'data/processed/three_ward_normal_vehicle_choice_v1'
O=ROOT/'data/processed/three_ward_observed_load_pilot_v1'
F=ROOT/'data/processed/three_ward_cross_event_protection_v1'
LABEL={'normal':'正常道路','flood':'受淹道路','protected':'通道保障'}
FACTORS={'low':.5,'central':1.,'high':2.}
PERIODS=['hour17','daytime12h']

def save(path,data): pd.DataFrame(data).to_csv(path,index=False,encoding='utf-8-sig')

@njit(cache=True)
def path_attributes(order,pred,arcmap,length,unknown,core):
 n=len(arcmap);ll=np.zeros(n);uu=np.zeros(n,np.bool_);cc=np.zeros(n,np.bool_)
 for s in order:
  a=arcmap[s];q=pred[s];ll[s]=length[a];uu[s]=unknown[a];cc[s]=core[a]
  if 0<=q<n: ll[s]+=ll[q];uu[s]=uu[s]or uu[q];cc[s]=cc[s]or cc[q]
 return ll,uu,cc

def prepare():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 assert json.loads((ROOT/'data/exp/estimation/three_ward_cross_event_protection_v1/motor_validation.json').read_text())['all_passed']
 deps=[REP/'turn_router.npz',REP/'multi_endpoint_components.npz',REP/'multi_routing_demands.npz',REP/'multi_network_endpoints.csv',M/'road_projection.npz',M/'road_inventory.csv',M/'source_motor_graph.npz',C/'choice_costs.npz',C/'F0_R0_flows.npz',O/'capacity_reference_12h.csv',O/'screenline_inventory.csv',O/'protocol.json',O/'branch0_road_state.npz',O/'branch1_road_state.npz',F/'motor_common_support.npz',F/'motor_exposures/E20250911__H__U.npz',BASE/'clock_proxy_ledger.csv',ROOT/'data/processed/three_ward_normal_transport_network_v1/surface_physical_edges.jsonl.gz',ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson',ROOT/'data/processed/three_ward_vehicle_demand_coverage_audit_v1/actionable_gap_inventory.csv',Path(__file__)]
 hashes={str(p.relative_to(ROOT)):sha(p)for p in deps}
 protocol={
 'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'human_direction':'MILI-D-20261018-009',
 'scope':'First complete road-traffic scenario chain for three wards plus frozen10km road context; snapshot only',
 'premise':'Observed baseline fit failed. Do not use failed inverse load_scenarios. Use original PT-derived spatial/clock prior as explicitly assumed generic small-vehicle demand. Passing numerical gates does not calibrate real traffic.',
 'demand_factors':FACTORS,'demand_factor_interpretation':'Sensitivity multipliers of original prior,not a confidence interval or empirically estimated range. Uniform scaling does not repair OD spatial coverage.',
 'external_shapes':['base: no remote-remote movement','plus: original conditional chord-screened remote through pool'],
 'demand_invariance':'Each same-factor/same-shape comparison uses identical requested OD in normal/flood/protected. Do not renormalize lost trips. Keep unknown-clock,coarse,intrazonal,unlocated,goods/taxi gaps in ledger.',
 'snapshot':json.loads((O/'protocol.json').read_text())['flood'],
 'road_uncertainty':'Keep lower-depth/unknown-available and upper-depth/unknown-withheld as joint evidence branches; their contrast cannot isolate depth versus road-height uncertainty. Known source-road class/grade masks retained, no bridges inferred safe.',
 'measure':'Reuse frozen corridor index1 effective_known_protection(102 source roads,about4.766km). Ideal passability and dry travel cost on protected known roads only; no new engineering simulation or water-depth reduction, no permission changes. Not optimized to this pilot outcome.',
 'routing':'Fixed U shortest-time all-or-nothing on existing turn-memory network. Slowed known roads have2x cost; closed roads removed. No congestion feedback, queue model or equilibrium. Demand scaling changes counts and pressure,not route choice.',
 'capacity':'At85existing screenlines only: small-vehicle proxy12h passages / official all-vehicle12h reference capacity. Add an explicitly hypothetical fixed observed-large12h background branch; do not route that background or claim it respects closures. Any closed mapped part marks capacity not comparable; mixed partly slowed cut=>capacity unknown; all mapped parts slowed=>0.5reference. No transfer to other roads; no12h-to-hour capacity conversion.',
 'outputs':['source demand ledger','road-state inventory with clipped ward lengths','normal/flood/protected paired OD reachability and survivor detours','newly restored trips separated from survivor savings','link flow transfers','known-screenline frozen-network12h stress','regional maps and machine-readable validation'],
 'budget':{'maximum_origins':600,'maximum_new_sweeps':6,'maximum_seconds_per_sweep':240,'maximum_total_seconds':1500,'new_hydraulics':0,'new_training':0,'affected_people':0},
 'stop':'Stop on source hash change,negative/nonfinite flow,component balance>1e-6,cost/length residual>1e-9,closed-arc flow>1e-9 or wall budget. Preserve completed verified sweeps and mark incomplete; do not relax gates after results.',
 }
 if (P/'protocol.json').exists():
  prev=json.loads((P/'input_bindings.json').read_text())['sha256'];assert prev==hashes,'Source changed: use a new version'
 else:
  dump(P/'protocol.json',protocol);dump(P/'input_bindings.json',{'sha256':hashes})
 common=load(F/'motor_common_support.npz');selected=common['effective_known_protection'][1];assert selected.sum()==102
 inv=pd.read_csv(M/'road_inventory.csv');assert abs(inv.loc[selected,'length_m'].sum()-4766.386718020637)<1e-6
 src=load(M/'source_motor_graph.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(REP/'multi_routing_demands.npz')
 core_ep=src['inside_core'][ep['native_node_index']];core_od=core_ep[dem['origin_endpoint']]|core_ep[dem['destination_endpoint']]
 np.savez_compressed(P/'scenario_masks.npz',protected=selected,core_endpoint_od=core_od)
 led=[]
 for sh,eligible in [('base',dem['base']),('plus',dem['through_plus'])]:
  for factor,m in FACTORS.items():
   for scope,sm in [('road_halo',np.ones(len(core_od),bool)),('core_endpoint_proxy',core_od)]:
    for t,period in [(0,'daily'),(3,'hour17'),(4,'daytime12h')]:
     led.append(dict(shape=sh,load=factor,factor=m,scope=scope,period=period,requested_vehicle_trip_proxy=float(dem['period_weights'][eligible&sm,t].sum()*m),interpretation='Assumed vehicle-trip proxy,not measured vehicles or people'))
 save(P/'demand_ledger.csv',led)
 gaps=pd.read_csv(deps[-2]);gaps['current_treatment']='Retained unresolved; not silently added to core demand'
 save(P/'unresolved_demand_ledger.csv',gaps)
 dump(R/'execution_state.json',{'status':'prepared','completed_sweeps':[],'planned_sweeps':6,'new_hydraulics':False,'people_counted':False})
 return protocol

def road_state(branch,arm):
 z=load(O/f'branch{branch}_road_state.npz');selected=load(P/'scenario_masks.npz')['protected']
 closed=z['normal_closed'].copy() if arm=='normal' else z['flood_closed'].copy()
 slow=np.zeros_like(closed) if arm=='normal' else z['slow'].copy()
 if arm=='protected': closed[selected]=False;slow[selected]=False
 assert not(closed&slow).any()
 return closed,slow,z['qualified']

def run_sweep(branch,arm,rr,ep,dem,pr,attrs,inv,protocol,campaign_start):
 name=f'branch{branch}_{arm}';marker=R/f'{name}_validation.json';signature=sha(P/'input_bindings.json')+sha(P/'protocol.json')
 if marker.exists():
  old=json.loads(marker.read_text())
  if old.get('passed')and old.get('signature')==signature: return old
 start=time.monotonic();closed,slow,qualified=road_state(branch,arm);ridx=pr['arc_to_road']
 cost=attrs['travel_seconds'].copy();cost[slow[ridx]]*=2
 g=network(rr,ep,pr,cost,closed);S=len(rr['state_physical_arc']);K=len(ep['components']);E=len(cost);nc=int(rr['node_component'].max())+1
 eligible=[dem['base'],dem['through_plus']];w=dem['period_weights'][:,[3,4]];origins=np.unique(dem['origin_endpoint'][dem['through_plus']]);assert len(origins)<=600
 flux=np.zeros((2,2,E));N=len(w);trip_cost=np.full(N,np.inf);trip_length=np.full(N,np.inf);trip_unknown=np.zeros(N,bool);trip_core=np.zeros(N,bool)
 replay_cost=0.;replay_flow=None
 if branch==0 and arm=='normal': replay_flow=load(C/'F0_R0_flows.npz')
 for done,i in enumerate(origins,1):
  if (P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause/stop requested at routing origin boundary')
  if time.monotonic()-start>protocol['budget']['maximum_seconds_per_sweep']or time.monotonic()-campaign_start>protocol['budget']['maximum_total_seconds']:raise RuntimeError('Registered routing budget reached')
  dist,pred=dijkstra(g,indices=S+int(i),directed=True,return_predecessors=True)
  order=np.argsort(dist[:S],kind='stable');order=order[np.isfinite(dist[order])]
  ll,uu,cc=path_attributes(order,pred,rr['state_physical_arc'],pr['length_m'],~qualified[ridx],attrs['arc_is_core'])
  q=np.flatnonzero((dem['origin_endpoint']==i)&dem['through_plus']);terminal=S+K+dem['destination_endpoint'][q];good=np.isfinite(dist[terminal]);q=q[good];parent=pred[terminal[good]];assert ((parent>=0)&(parent<S)).all()
  trip_cost[q]=dist[terminal[good]];trip_length[q]=ll[parent];trip_unknown[q]=uu[parent];trip_core[q]=cc[parent]
  loads=np.zeros((S,2,2))
  for b,mask in enumerate(eligible):
   ii=mask[q];np.add.at(loads[:,b,:],parent[ii],w[q[ii]])
  loads=propagate(order,pred,loads)
  for b in range(2):
   for t in range(2):flux[b,t]+=np.bincount(rr['state_physical_arc'],weights=loads[:,b,t],minlength=E)
  if replay_flow is not None:
   old=replay_flow['endpoint_cost_seconds'][i];good=np.isfinite(old);replay_cost=max(replay_cost,float(abs(dist[S+K:][good]-old[good]).max()))
  if done%100==0 or done==len(origins):
   state=dict(status='routing',sweep=name,origins_done=done,origins_total=len(origins),sweep_elapsed_s=time.monotonic()-start,campaign_elapsed_s=time.monotonic()-campaign_start)
   dump(R/'execution_state.json',state);print(json.dumps(state),flush=True)
 demand=np.array([w[m].sum(0)for m in eligible]);reachable=np.isfinite(trip_cost);routed=np.array([w[m&reachable].sum(0)for m in eligible]);unrouted=np.array([w[m&~reachable].sum(0)for m in eligible]);balance=0.;costres=0.;lenres=0.
 comp=rr['node_component'];ec=ep['components'];oo=dem['origin_endpoint'];dd=dem['destination_endpoint']
 for b,mask in enumerate(eligible):
  chosen=mask&reachable
  for t in range(2):
   expected=np.bincount(ec[oo[chosen]],weights=w[chosen,t],minlength=nc)-np.bincount(ec[dd[chosen]],weights=w[chosen,t],minlength=nc)
   actual=np.bincount(comp[pr['from_index']],weights=flux[b,t],minlength=nc)-np.bincount(comp[pr['to_index']],weights=flux[b,t],minlength=nc)
   balance=max(balance,float(abs(actual-expected).max()))
   rc=float(w[chosen,t]@trip_cost[chosen]);rl=float(w[chosen,t]@trip_length[chosen])
   costres=max(costres,abs(float(flux[b,t]@cost)-rc)/max(rc,1))
   lenres=max(lenres,abs(float(flux[b,t]@pr['length_m'])-rl)/max(rl,1))
 closedmax=float(flux[:,:,closed[ridx]].max(initial=0))
 checks=dict(demand_routed_plus_unrouted=bool(abs(demand-routed-unrouted).max()<1e-6),nonnegative_finite_flow=bool(np.isfinite(flux).all()and(flux>=0).all()),component_flow_conservation=balance<1e-6,path_cost_flow_loading=costres<1e-9,path_length_flow_loading=lenres<1e-9,closed_arc_zero_flow=closedmax<1e-9,all_known_closed_roads_masked=bool((g.data>=0).all()),unknown_withheld_never_used=bool(not trip_unknown[reachable].any())if branch==1 else True)
 replay={}
 if replay_flow is not None:
  err=float(abs(flux-replay_flow['arc_flow'][:,[3,4]]).max());checks['normal_U_cost_replayed']=replay_cost<1e-7;checks['normal_U_flow_replayed']=err<1e-6;replay=dict(max_cost_difference_s=replay_cost,max_arc_flow_difference=err)
 np.savez_compressed(P/f'{name}.npz',arc_flow=flux,trip_cost_seconds=trip_cost,trip_length_m=trip_length,trip_uses_unknown=trip_unknown,trip_touches_core=trip_core,input_demand=demand,routed=routed,unrouted=unrouted,closed=closed,slow=slow)
 result=dict(name=name,branch=branch,arm=arm,passed=all(checks.values()),checks=checks,signature=signature,seconds=time.monotonic()-start,maximum_component_balance=balance,cost_relative_residual=costres,length_relative_residual=lenres,closed_arc_max_flow=closedmax,replay=replay)
 dump(marker,result);assert result['passed'],result
 return result

def weighted_quantile(v,w,q):
 good=np.isfinite(v)&np.isfinite(w)&(w>0)
 if not good.any():return None
 v=v[good];w=w[good];ii=np.argsort(v,kind='stable');return float(v[ii][np.searchsorted(np.cumsum(w[ii]),q*w.sum())])

def report(protocol,validations):
 dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');inv=pd.read_csv(M/'road_inventory.csv');masks=load(P/'scenario_masks.npz');core=masks['core_endpoint_od'];ridx=pr['arc_to_road'];screen=pd.read_csv(O/'screenline_inventory.csv',dtype={'observation_section_id':str});cap=pd.read_csv(O/'capacity_reference_12h.csv',dtype={'observation_section_id':str})
 results={};roadflows={};metrics=[];effects=[];states=[];pressure=[];flowchanges=[];pairchecks=[]
 scopes=[('road_halo',np.ones(len(core),bool)),('core_endpoint_proxy',core)]
 for branch in range(2):
  for arm in LABEL:
   z=load(P/f'branch{branch}_{arm}.npz');results[branch,arm]=z
   roadflows[branch,arm]=np.array([[np.bincount(ridx,weights=z['arc_flow'][b,t],minlength=len(inv))for t in range(2)]for b in range(2)])
  n=results[branch,'normal'];f=results[branch,'flood'];p=results[branch,'protected'];nr=np.isfinite(n['trip_cost_seconds']);fr=np.isfinite(f['trip_cost_seconds']);pp=np.isfinite(p['trip_cost_seconds']);pair=nr&fr;pf=fr&pp
  checks=dict(flood_reachability_subset_normal=bool(not(fr&~nr).any()),flood_subset_protected=bool(not(fr&~pp).any()),protected_subset_normal=bool(not(pp&~nr).any()),flood_survivor_cost_not_below_normal=bool((f['trip_cost_seconds'][pair]>=n['trip_cost_seconds'][pair]-1e-6).all()),protected_survivor_cost_not_above_flood=bool((p['trip_cost_seconds'][pf]<=f['trip_cost_seconds'][pf]+1e-6).all()),protected_cost_not_below_normal=bool((p['trip_cost_seconds'][pp]>=n['trip_cost_seconds'][pp]-1e-6).all()))
  pairchecks.append(dict(branch=branch,checks=checks));assert all(checks.values()),checks
  for b,(shape,eligible)in enumerate([('base',dem['base']),('plus',dem['through_plus'])]):
   for loadlabel,factor in FACTORS.items():
    for ti,period in enumerate(PERIODS):
     w=dem['period_weights'][:,[3,4][ti]]*factor
     for scope,sm in scopes:
      choose=eligible&sm;normal=choose&nr;lost=normal&~fr;restored=lost&pp;survivor=normal&fr
      for arm in LABEL:
       z=results[branch,arm];ar=np.isfinite(z['trip_cost_seconds']);ok=normal&ar
       extra=z['trip_cost_seconds'][ok]-n['trip_cost_seconds'][ok];dr=z['trip_length_m'][ok]-n['trip_length_m'][ok]
       metrics.append(dict(branch=branch,shape=shape,load=loadlabel,factor=factor,period=period,scope=scope,arm=arm,requested=float(w[choose].sum()),normal_unreachable=float(w[choose&~nr].sum()),normal_reachable=float(w[normal].sum()),reachable=float(w[ok].sum()),newly_unreachable=float(w[normal&~ar].sum()),newly_unreachable_share=float(w[normal&~ar].sum()/max(w[normal].sum(),1e-30)),survivor_mean_extra_minutes=float(w[ok]@extra/w[ok].sum()/60)if w[ok].sum()>0 else None,survivor_p90_extra_minutes=weighted_quantile(extra/60,w[ok],.9),survivor_mean_extra_km=float(w[ok]@dr/w[ok].sum()/1000)if w[ok].sum()>0 else None,reachable_unknown_path_weight=float(w[ok&z['trip_uses_unknown']].sum()),sum_extra_vehicle_proxy_hours=float(w[ok]@extra/3600)))
      benefit=f['trip_cost_seconds'][survivor]-p['trip_cost_seconds'][survivor]
      effects.append(dict(branch=branch,shape=shape,load=loadlabel,period=period,scope=scope,flood_new_lost=float(w[lost].sum()),restored_vehicle_trip_proxy=float(w[restored].sum()),remaining_new_lost=float(w[lost&~pp].sum()),restoration_fraction_of_flood_lost=float(w[restored].sum()/w[lost].sum())if w[lost].sum()>0 else None,already_reachable_time_saving_vehicle_proxy_hours=float(w[survivor]@benefit/3600),paired_saving_mean_minutes=float(w[survivor]@benefit/w[survivor].sum()/60)if w[survivor].sum()>0 else None))
     for arm in LABEL:
      rf=roadflows[branch,arm][b,ti]*factor;closed=results[branch,arm]['closed'];slow=results[branch,arm]['slow']
      for i,s in screen.iterrows():
       ids=np.array([int(x)for x in str(s.matched_road_indices).split('|')]);cr=cap.iloc[i];cutclosed=bool(closed[ids].any());mixed=bool(slow[ids].any()and not slow[ids].all());known=bool(cr.capacity_known)and not cutclosed and not mixed
       cf=.5 if slow[ids].all() else 1.;capacity=float(cr.reference_capacity12h_all)*cf if known else None
       if period=='daytime12h':
        small=float(rf[ids].sum());background=float(cr.observed_large12h)
        pressure.append(dict(branch=branch,shape=shape,load=loadlabel,arm=arm,screen_index=i,observation_section_id=s.observation_section_id,road_class=s.road_class,small_proxy12h_passages=small,reference_capacity12h=capacity,capacity_comparable=known,any_mapped_road_closed=cutclosed,mixed_slow_capacity_unknown=mixed,small_component_vc=small/capacity if known else None,with_fixed_observed_large_background_vc=(small+background)/capacity if known else None,background_is_assumed_not_routed=True,capacity_unit='all-vehicle12h; frozen snapshot,not actual17h capacity'))
 for branch in range(2):
  for arm in LABEL:
   closed,slow,q=road_state(branch,arm)
   for state,mask in [('known_closed',closed&q),('known_slow',slow&q),('known_open',q&~closed&~slow),('unknown',~q)]:
    states.append(dict(branch=branch,arm=arm,state=state,source_roads=int(mask.sum()),core_intersecting_source_roads=int((mask&inv.inside_core.to_numpy()).sum()),whole_length_of_core_intersecting_roads_km=float(inv.loc[mask&inv.inside_core,'length_m'].sum()/1000)))
  base=roadflows[branch,'normal'][0,0];flood=roadflows[branch,'flood'][0,0];prot=roadflows[branch,'protected'][0,0]
  part=inv[['road_index','source_edge_id','name','highway','inside_core','reason','length_m']].copy();part['branch']=branch;part['normal_hour17_proxy']=base;part['flood_hour17_proxy']=flood;part['protected_hour17_proxy']=prot;part['flood_minus_normal']=flood-base;part['protected_minus_flood']=prot-flood;part['flood_closed']=results[branch,'flood']['closed'];part['protected']=masks['protected'];flowchanges.append(part)
 save(P/'paired_access_metrics.csv',metrics);save(P/'measure_effects.csv',effects);save(P/'road_state_inventory.csv',states);save(P/'screenline_stress12h.csv',pressure);pd.concat(flowchanges).to_csv(P/'road_flow_changes.csv',index=False,encoding='utf-8-sig')
 # Precise core clipping is independent of the demand and outcomes.
 import geopandas as gpd
 from shapely.geometry import shape
 from shapely.ops import unary_union
 wards=gpd.read_file(ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').to_crs(6677);boundary=unary_union(wards.geometry.to_numpy());lookup={str(x):i for i,x in enumerate(inv.source_edge_id)};geoms={};corelens=np.zeros(len(inv))
 source=ROOT/'data/processed/three_ward_normal_transport_network_v1/surface_physical_edges.jsonl.gz'
 with gzip.open(source,'rt')as h:
  for line in h:
   e=json.loads(line);j=lookup.get(e['edge_id'])
   if j is None or not inv.inside_core.iloc[j]:continue
   clipped=shape(e['geometry']).intersection(boundary)
   if not clipped.is_empty:geoms[j]=clipped;corelens[j]=clipped.length
 clippedstates=[]
 for s in states:
  closed,slow,q=road_state(s['branch'],s['arm']);mask={'known_closed':closed&q,'known_slow':slow&q,'known_open':q&~closed&~slow,'unknown':~q}[s['state']]
  clippedstates.append(dict(**s,ward_clipped_length_km=float(corelens[mask].sum()/1000)))
 save(P/'road_state_clipped_lengths.csv',clippedstates)
 plot(regions=geoms,inv=inv,results=results,roadflows=roadflows,boundary=boundary,metrics=pd.DataFrame(metrics),effects=pd.DataFrame(effects))
 df=pd.DataFrame(metrics);eff=pd.DataFrame(effects);st=pd.DataFrame(clippedstates);stress=pd.DataFrame(pressure);summary=[]
 for branch in range(2):
  for shape in ['base','plus']:
   selection=df[(df.branch==branch)&df['shape'].eq(shape)&df.load.eq('central')&df.period.eq('hour17')&df.scope.eq('core_endpoint_proxy')]
   e=eff[(eff.branch==branch)&eff['shape'].eq(shape)&eff.load.eq('central')&eff.period.eq('hour17')&eff.scope.eq('core_endpoint_proxy')].iloc[0]
   summary.append(dict(branch=branch,shape=shape,access=selection.to_dict('records'),measure=e.to_dict()))
 allchecks={'six_sweeps_passed':len(validations)==6 and all(v['passed']for v in validations),'pair_monotonicity':all(all(x['checks'].values())for x in pairchecks),'source_hashes_unchanged':all(sha(ROOT/n)==v for n,v in json.loads((P/'input_bindings.json').read_text())['sha256'].items()),'core_clipped_lengths_not_greater_than_whole':bool((corelens<=inv.length_m.to_numpy()+.01).all()),'selected_measure_known_only':bool((load(P/'scenario_masks.npz')['protected']&~load(F/'motor_common_support.npz')['qualified_roads']).sum()==0)}
 assert all(allchecks.values()),allchecks
 dump(R/'validation.json',dict(all_passed=True,checks=allchecks,pair_checks=pairchecks,meaning='Numerical consistency of an assumed-demand scenario; not traffic calibration,observed loss or engineering effectiveness validation'))
 dump(R/'summary.json',dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),central_hour17_core_endpoint_results=summary,road_inventory=clippedstates,route_sweep_seconds=sum(v['seconds']for v in validations),new_hydraulics=False,people_counted=False,congestion_feedback=False,count_fit_gate_still_failed=True))
 review(protocol,df,eff,st,stress,validations)
 dump(R/'execution_state.json',dict(status='complete',sweeps=6,assumption_combinations=36,periods=2,new_hydraulics=False,people_counted=False,actual_congestion_or_queues_simulated=False))

def plot(regions,inv,results,roadflows,boundary,metrics,effects):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.collections import LineCollection
 from matplotlib.lines import Line2D
 def parts(g):return [np.asarray(g.coords)/1000]if g.geom_type=='LineString'else[np.asarray(x.coords)/1000 for x in g.geoms if x.geom_type=='LineString']if hasattr(g,'geoms')else[]
 lines={i:parts(g)for i,g in regions.items()};bounds=np.array(boundary.bounds)/1000
 fig,axs=plt.subplots(2,2,figsize=(11,12),layout='constrained')
 for b in range(2):
  for c,arm in enumerate(['flood','protected']):
   ax=axs[b,c];z=results[b,arm];closed=z['closed'];slow=z['slow'];qualified=load(F/'motor_common_support.npz')['qualified_roads']
   for name,mask,color in [('Unknown',~qualified,'#b9c0c4'),('Open',qualified&~closed&~slow,'#d6e6df'),('Slowed',qualified&slow,'#dc9d32'),('Closed',qualified&closed,'#bd4040')]:
    ax.add_collection(LineCollection([a for i,x in lines.items()if mask[i]for a in x],colors=color,linewidths=.7 if name in ('Closed','Slowed')else .35))
   if arm=='protected':
    selected=load(P/'scenario_masks.npz')['protected'];ax.add_collection(LineCollection([a for i,x in lines.items()if selected[i]for a in x],colors='#1767ab',linewidths=1.3))
   ax.set(xlim=[bounds[0]-.2,bounds[2]+.2],ylim=[bounds[1]-.2,bounds[3]+.2],aspect='equal',title=f'Branch{b} | {arm}',xlabel='Easting(km)',ylabel='Northing(km)')
 fig.legend(handles=[Line2D([0],[0],color=c,lw=3,label=n)for c,n in [('#bd4040','Closed hypothesis'),('#dc9d32','Slowed hypothesis'),('#b9c0c4','Unknown'),('#1767ab','Ideal corridor protection')]],loc='outside lower center',ncol=2)
 fig.suptitle('Three wards | existing2025-09-11 radar + hypothetical raised stage | elapsed4h')
 fig.savefig(R/'road_states.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');sub=metrics[metrics['shape'].eq('base')&metrics.load.eq('central')&metrics.period.eq('hour17')&metrics.scope.eq('core_endpoint_proxy')];x=np.arange(2)
 for j,arm in enumerate(['flood','protected']):
  rows=sub[sub.arm.eq(arm)].sort_values('branch');axs[0].bar(x+(j-.5)*.3,rows.newly_unreachable_share*100,width=.3,label=arm);axs[1].bar(x+(j-.5)*.3,rows.survivor_mean_extra_minutes,width=.3,label=arm)
 axs[0].set(ylabel='Newly unreachable proxy share (%)');axs[1].set(ylabel='Mean added route minutes (survivors only)')
 for ax in axs:ax.set_xticks(x,['Unknown available/lower','Unknown withheld/upper']);ax.tick_params(axis='x',labelsize=8);ax.legend()
 fig.suptitle('Central prior | core-endpoint demand | route-time costs exclude congestion')
 fig.savefig(R/'paired_outcomes.png',dpi=150);plt.close(fig)

def review(protocol,df,eff,st,stress,validations):
 def table(frame):return '<div class="scroll">'+frame.to_html(index=False,escape=True,float_format=lambda x:f'{x:,.3f}',border=0)+'</div>'
 core=df[df['shape'].eq('base')&df.load.eq('central')&df.period.eq('hour17')&df.scope.eq('core_endpoint_proxy')][['branch','arm','normal_reachable','newly_unreachable','newly_unreachable_share','survivor_mean_extra_minutes','survivor_mean_extra_km','reachable_unknown_path_weight']].copy()
 core.columns=['分支','状态','正常可达需求代理','新增不可达需求代理','新增不可达比例','仍可达平均增时min','仍可达平均增距km','路径含未知需求代理']
 measures=eff[eff['shape'].eq('base')&eff.load.eq('central')&eff.period.eq('hour17')&eff.scope.eq('core_endpoint_proxy')][['branch','restored_vehicle_trip_proxy','remaining_new_lost','paired_saving_mean_minutes']]
 ss=[]
 for (b,sh,ld,arm),g in stress.groupby(['branch','shape','load','arm']):
  yes=g.capacity_comparable;ss.append(dict(branch=b,shape=sh,load=ld,arm=arm,comparable_screens=int(yes.sum()),closed_or_mixed_unknown=int((~yes).sum()),small_only_vc_gt1=int((g.loc[yes,'small_component_vc']>1).sum()),with_assumed_large_background_gt1=int((g.loc[yes,'with_fixed_observed_large_background_vc']>1).sum())))
 save(P/'screenline_stress_summary.csv',ss)
 links=['protocol.json','input_bindings.json','demand_ledger.csv','unresolved_demand_ledger.csv','paired_access_metrics.csv','measure_effects.csv','road_state_clipped_lengths.csv','road_flow_changes.csv','screenline_stress12h.csv','screenline_stress_summary.csv']
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>假设需求：正常—积水—通道保障首轮对照</title><style>body{{margin:0;background:#edf2f5;color:#233e4b;font:16px/1.8 system-ui,sans-serif}}main{{max-width:1160px;margin:28px auto;background:white;padding:30px;border-radius:14px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:30px}}.lead{{padding:18px;background:#e8f2ec;border-left:5px solid #43806e}}.note{{padding:16px;background:#fff0dd}}table{{border-collapse:collapse;font-size:13px;min-width:700px;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}a{{color:#146587}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main>
<p>2026-10-18 JST · 都心三区与10km道路上下文 · MILI-D-20261018-009</p><h1>先用明确假设跑通正常—积水—措施链条</h1>
<div class="lead">已完成6次路由、3档需求×2档外部穿越×2档道路未知处理×3种道路状态，共36种组合；分别报告17时需求代理和固定路况12h压力。沿用现有水动力，不追加模拟、不计算人数。上轮观测拟合未通过的结果仍保留，本轮不采用其失败反演负荷。</div>
<h2>这一轮的假设</h2><p>原2018 PT推导的驾驶者候选量仅作为一般小型车辆出行的空间／时钟先验，按0.5、1、2倍设低／中／高需求。这些倍数是敏感性设置，不是统计置信区间，也不保证覆盖真实需求。保留不含外部—外部和加入既有弦线筛选穿越池两种口径；车型、货运、粗区与区内短途、未知时钟等缺口继续单列。完整道路上下文需求不能称为三区本地需求，三区重点表仅包含起点或终点模型代表位于三区的OD代理。</p>
<p>选择已冻结的E20250911__H__U，窗口起点后4h，2025-09-11约17:15 JST；降雨来自历史雷达重建，河道+1.5m是情景假设，不是当日实测。积水≥0.30m关闭，0.10–0.30m行程成本加倍；阈值为研究假设。未知道路可用／低水深和未知道路扣除／高水深是两个联合分支，不能用其差异单独识别桥隧或水深误差。</p>
<p>保障措施沿用既有102段、约4.766km通道清单，只假设这些有可靠地表对应的道路可以维持正常通行及成本，不改变道路权限、单行或转向。没有降低DEM水深或求解排水工程，因此表示理想通道保障的条件效果，不能当作真实工程效益。没有基于这轮交通结果重新选路。</p>
<h2>三区道路关闭和缓行</h2><p>以下按官方三区边界裁剪几何后计长度，按源道路一次计数；不把双向弧计两次。高架／隧道等未知部分保留未知；蓝色为冻结通道方案。仅为同一时刻，不能据此声称关闭持续时长。</p>{table(st[st.arm.isin(['flood','protected'])][['branch','arm','state','source_roads','ward_clipped_length_km']])}<img src="road_states.png" alt="Clipped three-ward road closure,slowdown and unknown states with fixed ideal corridor protection">
<h2>中需求：三区起终点代理的配对结果</h2><p>先扣除同一分支正常状态已不可达的需求，再计算暴雨新增不可达；不能把正常路网缺失归因于暴雨。下面单位为17时条件车辆出行代理，含分数权重，不是人数、实测车辆或真实历史交通损失。</p>{table(core)}<img src="paired_outcomes.png" alt="Paired reachability and survivor route time for central assumed demand">
<p>“仍可达平均增时”比较各状态自己的幸存OD集合，集合可能变化，不能直接相减作为措施收益。措施表另外固定为受淹状态已经可达的同一批OD，单列恢复可达的需求：</p>{table(measures)}
<h2>流量转移与拥堵风险能说到哪里</h2><p>路由复用正常U成本、现有转向网络；受淹改变封路与缓行，车辆重新选择固定成本最短路。已计算道路流量变化和车辆守恒，但没有流量—速度反馈、排队或均衡。因此额外时间来自绕行和设定缓行，不包含拥堵排队。当前不能输出真实拥堵时长。</p>
<p>对85个有参考断面，报告小型车辆代理／全车型12h容量，以及加固定观测大型车背景的假设分支。大型车背景没有路由，不能声称它遵守本次封路；有任何映射道路关闭的断面不计算比值，部分缓行导致容量不可统一时保留未知。只有全断面映射道路都缓行时才采用0.5容量假设；不把12h容量除以12。把单个时刻的封路冻结12h仅为静态压力实验，不是当天累计拥堵。未观测道路容量保持未知。</p>
{table(pd.DataFrame(ss).query("shape == 'base' and load == 'central'"))}
<p class="note">0.5／1／2倍均匀缩放不会改变当前固定成本路径，所以不可达比例和平均绕行时间相同是模型结构使然，不能据此声称对需求不确定性稳健。数量和容量压力会变化；外部穿越和道路未知分支的比较才改变本轮空间支持条件。未来如研究实际拥堵，需引入并检验容量和需求响应情景。</p>
<h2>核验与接下来的完整实验</h2><p>6个路由均检查请求需求=可达+不可达、组件节点流量守恒、按路径与格弧分别计算的成本和长度一致、封闭道路零流量。正常开放分支还精确复现原U成本和流量。另检查封路不新增可达、保障不使原有可达丢失和配对成本单调性；源数据哈希保持不变。数值通过不代表通过观测校准或证实真实损失。</p>
<p>本轮是第一个快照下的完整链条，尚未完成多暴雨、多时刻、全部措施、拥堵反馈和受影响人数实验。后续可以沿用同一合同扩展现有事件和时刻，再加入透明容量假设的交通反馈；人口白天／夜间与楼宇分配区间继续保留。不能把无法完成的车辆出行直接当成受影响人数。</p>
<p>活动路由约{sum(v['seconds']for v in validations)/60:.1f}分钟。预先预算每次≤240s、全轮≤1500s、最多6次，任一守恒／成本／源版本检查失败即停止；没有修改原观测准入规则或删除不合观测。</p><ul>'''
 for n in links:body+=f'<li><a href="../../../processed/{P.name}/{n}">{n}</a></li>'
 body+='<li><a href="validation.json">validation.json</a></li><li><a href="summary.json">summary.json</a></li></ul><p>底层道路 © OpenStreetMap contributors / ODbL；PLATEAU城市几何、现有地形及官方统计来源沿用源审计。原始资料和既有实验保持不变。</p></main></html>'
 (R/'review.html').write_text(body)

def main():
 started=time.monotonic();protocol=prepare();rr=load(REP/'turn_router.npz');ep=load(REP/'multi_endpoint_components.npz');dem=load(REP/'multi_routing_demands.npz');pr=load(M/'road_projection.npz');attrs=load(C/'choice_costs.npz');inv=pd.read_csv(M/'road_inventory.csv');completed=[]
 try:
  for b in range(2):
   for arm in LABEL:completed.append(run_sweep(b,arm,rr,ep,dem,pr,attrs,inv,protocol,started))
  report(protocol,completed)
 except Exception as exc:
  dump(R/'execution_state.json',dict(status='stopped',reason=str(exc),completed_sweeps=[v['name']for v in completed],elapsed_s=time.monotonic()-started));raise
 print('COMPLETE',str(R/'review.html'),flush=True)
if __name__=='__main__':main()
