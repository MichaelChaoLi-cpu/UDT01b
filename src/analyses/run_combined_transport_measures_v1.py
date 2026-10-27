"""Frozen protection x explicit deferral x ideal guidance in two existing storms.

Original demand = deferred + retained/served + retained/unreachable. Objective
and optimality checks distinguish private-cost and system-cost path allocation.
No new hydraulics, observed-traffic claims, dynamic queues or person estimates.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,time,datetime as dt,argparse
import numpy as np,pandas as pd
from scipy.sparse import csc_matrix,load_npz,save_npz
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import minimize,brentq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import demand_protection_feedback_v1 as prev
from observed_load_common_v1 import load,dump,sha,network
P=ROOT/'data/processed/three_ward_combined_transport_measures_v1'
R=ROOT/'data/exp/estimation/three_ward_combined_transport_measures_v1'
CASES=['E20250911__H__U','E20240830__H__U'];REDUCTIONS=[0.,.1,.2,.3]
def save(p,data):pd.DataFrame(data).to_csv(p,index=False,encoding='utf-8-sig')
def check():
 b=json.loads((P/'input_bindings.json').read_text())['sha256'];assert all(sha(ROOT/n)==v for n,v in b.items()),'Frozen source changed'
def original_request(dem):return dem['period_weights'][:,3]*dem['base']*2.
def key(case,arm,r,g):return f'{case}__{arm}__defer{round(100*r):02d}__{g}'
def paths_for(case,arm,kind):
 if case=='normal' or arm=='normal':return prev.P/f'normal_{kind}.npz'
 if case==CASES[0]:return prev.P/f'{arm}_{kind}.npz'
 return P/'paths'/f'{case}__{arm}_{kind}.npz'

def prepare():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'paths').mkdir(exist_ok=True);(P/'fits').mkdir(exist_ok=True)
 assert json.loads((prev.R/'delivery_validation.json').read_text())['passed']
 files=[prev.REP/'turn_router.npz',prev.REP/'multi_endpoint_components.npz',prev.REP/'multi_routing_demands.npz',prev.M/'road_projection.npz',prev.M/'road_inventory.csv',prev.C/'choice_costs.npz',prev.P/'protection_masks.npz',prev.P/'capacity_assumptions.npz',prev.P/'protocol.json',prev.F/'motor_common_support.npz',prev.F/'run_manifest.json']
 for case in CASES:files.append(prev.F/'motor_exposures'/(case+'.npz'))
 for arm in ['normal','flood','targeted']:
  for kind in ['A','B']:files.append(prev.P/f'{arm}_{kind}.npz')
  files.extend([prev.P/(arm+'__high_demand.npz'),prev.R/(arm+'__high_demand_fit_validation.json')])
 bindings={str(f.relative_to(ROOT)):sha(f)for f in files}
 if(P/'protocol.json').exists():check();return
 manifest=json.loads((prev.F/'run_manifest.json').read_text())['runs'];clock={x['case_id']:x['window_start_utc']for x in manifest if x['case_id']in CASES}
 protocol={
 'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'decision_id':'MILI-D-20261018-011',
 'question':'Can frozen road protection with deferred demand and ideal guidance reduce the accessibility/congestion-cost tradeoff,and does it transfer to another existing rainfall event?',
 'events':CASES,'event_parent_count':2,'selection_event':CASES[0],'transfer_event':CASES[1],
 'snapshot_elapsed_s':14400,'frame_index':5,'event_window_start_utc':clock,
 'clock_interpretation':'Fixed17h PT proxy demand on both elapsed4h storm snapshots:2025eventabout17:15JST,2024eventabout06:00JST. Second is a controlled commute-load weather-shape transfer,not observed06h traffic replay.',
 'hydraulics':'Reuse existing qualified U urban geometry,radar rain and same hypothetical1.5m raised-stage scenarios. No new hydraulics;two storms only,not four independent low/high stage events.',
 'factorial':{'protection':['flood','targeted'],'deferred_fraction':REDUCTIONS,'guidance':['UE','SO']},
 'scenario_counts':{'storm_intervention_cells':32,'normal_controls':8,'total_cases':40,'inherited_verified_no_deferral_UE_cases':3,'maximum_new_optimizer_fits':37},
 'demand':'High reference=2xoriginal PT-derived base/no remote-remote17h generic small-vehicle proxy. Defer0/10/20/30percent uniformly in every OD across10km context. Deferred means removed from current hour only;not assumed completed later or benefiting. Through-traffic,class,unknown-clock and location gaps retained in prior ledgers;not completed by multiplier.',
 'ledger':'Original = deferred + retained served + retained unreachable,globally and core-endpoint subset. Deferring disconnected trips is not counted as restoring access. Current-hour noncompletion includes both deferred and retained unreachable. No people counts or welfare cost assigned to deferral.',
 'protection':'Freeze118roads,4743.231745120262m from prior event;no new selection in transfer storm. Ideal dry passability,time and capacity,not engineering effectiveness.',
 'capacity_and_reference_cost':'Exactly prior lane/class capacities and Ureference cost. Slow:time2x,capacity0.5x;closed:path excluded. Unknown roads remain assumed available,not verified safe. Do not use previous all-unknown-withheld1.4percent normal coverage as a physical upper bound.',
 'cost':'c(f)=tref*(1+0.15*(f/C)^4). tref includes existing daytime/proxy conditions,not measured free-flow time. Model cost,not observed queue delay.',
 'guidance':'UE:minimize integrated cost(sum tref*f*(1+0.03*(f/C)^4)). SO:ideal fully complied coordinated allocation minimizes actual total modeled vehicle cost(sum tref*f*(1+0.15*(f/C)^4)) over identical two paths. SO derivative=tref*(1+0.75*(f/C)^4). No traffic-signal,intersection,compliance,information or dynamic queue simulation;not full-network equilibrium or social optimum.',
 'path_pool':'Same two paths for all management/guidance cells of one road state. For transfer,reference shortest path plus shortest at1xprior AON capacity cost capped50xreference for generation only;uncapped final objective. Same established turn-memory graph.',
 'comparisons':'Guidance versus UE at SAME retained demand and physical protection. Protection versus no protection at SAME deferral. Additional comparison against original no-action full-hour request lists deferred separately. Fixed cohort=original no-action flood-reachable core OD,original weights;avoid survivorship/deferral being called time savings.',
 'fit':{'maximum_iterations':2000,'maximum_seconds_per_fit':120,'ftol':1e-13,'gtol':1e-7,'restricted_gap_max':.001,'scale_seconds':1000,'warm_start':'UEuses previous lower-deferral solution,SOstarts at same-cellUE;raw no-deferralUE inherited if exact source checks pass.'},
 'gates':'Finite nonnegative flow;ODroute bounds;component flow balance<1e-6;closed-arc0;path cost loading relative error<1e-9;original demand ledger<1e-6;gap<=.001(using marginal cost forSO). SO total cost<=same-cellUE*(1+1e-6). No observed calibration claim.',
 'budget':{'maximum_new_path_sweeps':4,'maximum_seconds_per_sweep':180,'total_path_seconds':900,'maximum_new_optimizer_fits':37,'maximum_fit_active_seconds':5400,'new_hydraulics':0,'new_training':0,'people_counted':0},
 'stop':'Stop failed numerical/gap/source/budget cases;retain completed artifacts and explicit incomplete status. Never relax gates to improve a measure. PAUSE/STOP honored before each path origin/fit and during optimization.',
 'interpretation':'Scenario comparisons,not actualhistorical congestion or independent forecasting validation. Two events do not establish broad generality. System-cost gains may redistribute costs to core travelers;deferral has an unmodeled inconvenience cost.',
 }
 dump(P/'protocol.json',protocol);dump(P/'input_bindings.json',{'sha256':bindings})
 mask=load(prev.P/'protection_masks.npz')['targeted'];inv=pd.read_csv(prev.M/'road_inventory.csv');assert mask.sum()==118 and abs(inv.loc[mask,'length_m'].sum()-4743.231745120262)<1e-6
 dem=load(prev.REP/'multi_routing_demands.npz');q0=original_request(dem);core=load(prev.P/'protection_masks.npz')['core_endpoint_od'];ledger=[]
 for r in REDUCTIONS:
  for scope,choose in [('all_road_context',np.ones(len(q0),bool)),('core_endpoint_proxy',core)]:ledger.append(dict(deferred_fraction=r,scope=scope,original_requested_proxy=float(q0[choose].sum()),deferred_proxy=float((q0[choose]*r).sum()),retained_proxy=float((q0[choose]*(1-r)).sum()),deferred_is_not_restored_or_completed=True))
 save(P/'management_input_ledger.csv',ledger);dump(R/'preparation_validation.json',dict(passed=True,plan_frozen=True,events_predeclared=True,original_requests_conserved=True,new_hydraulics=False));print(json.dumps(protocol['scenario_counts']),flush=True)

def state(case,arm):
 qualified=load(prev.F/'motor_common_support.npz')['qualified_roads']
 if case=='normal':closed=np.zeros(len(qualified),bool);slow=closed.copy()
 else:
  ex=load(prev.F/'motor_exposures'/(case+'.npz'));assert abs(ex['actual_times'][5]-14400)<.001;depth=ex['lower'][5];closed=qualified&(depth>=.3);slow=qualified&(depth>=.1)&~closed
  if arm=='targeted':protection=load(prev.P/'protection_masks.npz')['targeted'];closed[protection]=False;slow[protection]=False
 pr=load(prev.M/'road_projection.npz');ridx=pr['arc_to_road'];tref=load(prev.C/'choice_costs.npz')['travel_seconds'].copy();capacity=load(prev.P/'capacity_assumptions.npz')['hourly_capacity'].copy();tref[slow[ridx]]*=2;capacity[slow[ridx]]*=.5
 return closed,slow,tref,capacity

def generate_paths():
 check();start=time.monotonic();case=CASES[1];rr=load(prev.REP/'turn_router.npz');ep=load(prev.REP/'multi_endpoint_components.npz');dem=load(prev.REP/'multi_routing_demands.npz');pr=load(prev.M/'road_projection.npz');S=len(rr['state_physical_arc']);K=len(ep['components']);E=len(pr['length_m']);N=len(dem['base']);nc=int(rr['node_component'].max())+1;comp=rr['node_component'];ec=ep['components'];origins=np.unique(dem['origin_endpoint'][dem['through_plus']])
 for arm in ['flood','targeted']:
  closed,slow,tref,capacity=state(case,arm)
  for kind in ['A','B']:
   name=f'{case}__{arm}_{kind}';marker=R/(name+'_path_validation.json');out=paths_for(case,arm,kind)
   if marker.exists()and json.loads(marker.read_text())['passed']:continue
   begun=time.monotonic();cost=tref.copy()
   if kind=='B':a=load_npz(paths_for(case,arm,'A'));flow=a@(dem['period_weights'][:,3]*dem['base']);cost*=np.minimum(1+.15*(flow/capacity)**4,50.)
   graph=network(rr,ep,pr,cost,closed);lengths=np.zeros(N,np.int64);chunks=[];columns=[];costs=np.full(N,np.inf)
   for done,i in enumerate(origins,1):
    if(P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause before next origin')
    if time.monotonic()-begun>180 or time.monotonic()-start>900:raise RuntimeError('Path budget reached')
    dist,pred=dijkstra(graph,indices=S+int(i),directed=True,return_predecessors=True);q=np.flatnonzero((dem['origin_endpoint']==i)&dem['through_plus']);ends=S+K+dem['destination_endpoint'][q];ok=np.isfinite(dist[ends]);chosen=q[ok];parents=pred[ends[ok]];assert ((parents>=0)&(parents<S)).all();ptr,arcs=prev.trace_paths(parents,S,pred,rr['state_physical_arc']);lengths[chosen]=np.diff(ptr);costs[chosen]=dist[ends[ok]];chunks.append(arcs);columns.append(chosen)
    if done%150==0 or done==len(origins):print(json.dumps(dict(status='paths',case=case,arm=arm,kind=kind,origins=done,total=len(origins),seconds=time.monotonic()-begun)),flush=True)
   order=np.concatenate(columns);sizes=lengths[order];H=csc_matrix((np.ones(sizes.sum()),np.concatenate(chunks),np.r_[0,np.cumsum(sizes)]),shape=(E,len(order))).tocoo();H=csc_matrix((H.data,(H.row,order[H.col])),shape=(E,N));H.sum_duplicates();reachable=lengths>0;w=dem['period_weights'][:,3]*dem['base'];f=np.asarray(H@w);closedmax=float(f[closed[pr['arc_to_road']]].max(initial=0));error=float(abs(np.asarray(H.T@cost).ravel()[reachable]-costs[reachable]).max(initial=0));weights=w*reachable;expected=np.bincount(ec[dem['origin_endpoint']],weights=weights,minlength=nc)-np.bincount(ec[dem['destination_endpoint']],weights=weights,minlength=nc);actual=np.bincount(comp[pr['from_index']],weights=f,minlength=nc)-np.bincount(comp[pr['to_index']],weights=f,minlength=nc);balance=float(abs(actual-expected).max());checks=dict(cost_match=error<1e-6,closed_zero=closedmax<1e-9,component_conservation=balance<1e-6)
   save_npz(out,H);dump(marker,dict(passed=all(checks.values()),checks=checks,seconds=time.monotonic()-begun,code_sha256=sha(Path(__file__))));assert all(checks.values()),checks
 check();dump(R/'path_campaign.json',dict(status='complete',sweeps=4,seconds=time.monotonic()-start))

def optimize(A,B,q,tref,capacity,guidance,initial,gradient_tolerance=1e-7):
 D=(B-A).tocsc();baseline=np.asarray(A@q).ravel();started=time.monotonic();calls=0;coef=.03 if guidance=='UE'else .15;gradcoef=.15 if guidance=='UE'else .75
 def calc(x):
  nonlocal calls
  if time.monotonic()-started>120:raise RuntimeError('Fit wall budget reached')
  if(P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause in optimization')
  f=baseline+np.asarray(D@x).ravel();assert f.min(initial=0)>-1e-6;f=np.maximum(f,0.);r4=(f/capacity)**4;objective=float(tref@(f*(1+coef*r4)))/1000;marginal=tref*(1+gradcoef*r4);gradient=np.asarray(D.T@marginal).ravel()/1000;calls+=1;return objective,gradient
 x0=np.clip(initial,0,q);res=minimize(calc,x0,jac=True,bounds=list(zip(np.zeros(len(q)),q)),method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-13,'gtol':gradient_tolerance,'maxls':30,'maxcor':10});calc(res.x);f=np.maximum(baseline+np.asarray(D@res.x).ravel(),0.);r4=(f/capacity)**4;tt=tref*(1+.15*r4);mc=tref*(1+gradcoef*r4);ma=np.asarray(A.T@mc).ravel();mb=np.asarray(B.T@mc).ravel();gap=max(0.,(float(f@mc)-float(q@np.minimum(ma,mb)))/max(float(f@mc),1));ca=np.asarray(A.T@tt).ravel();cb=np.asarray(B.T@tt).ravel();mean=((q-res.x)*ca+res.x*cb)/q
 return dict(flow=f,cost=tt,routeB=res.x,mean_cost=mean,gap=gap,iterations=int(res.nit),optimizer_success=bool(res.success),message=str(res.message),seconds=time.monotonic()-started,function_calls=calls)

def optimizer_test():
 A=csc_matrix(np.array([[1.],[0.]]));B=csc_matrix(np.array([[0.],[1.]]));out={}
 for mode,mult in [('UE',.15),('SO',.75)]:
  root=brentq(lambda x:10*(1+mult*x**4)-20*(1+mult*(3-x)**4),0,3);z=optimize(A,B,np.array([3.]),np.array([10.,20.]),np.ones(2),mode,np.zeros(1),gradient_tolerance=1e-10);assert abs(z['flow'][0]-root)<1e-4 and z['gap']<1e-6;out[mode]=dict(flow=z['flow'],analytic_first_arc=root,gap=z['gap'],total_cost=float(z['flow']@z['cost']))
 assert out['SO']['total_cost']<=out['UE']['total_cost']+1e-8;dump(R/'optimizer_selfcheck.json',dict(passed=True,results=out))

def inherit(case,arm,r,g,active,requested):
 if r!=0 or g!='UE' or(case not in [CASES[0],'normal']):return None
 src=prev.P/(arm+'__high_demand.npz');validation=prev.R/(arm+'__high_demand_fit_validation.json');v=json.loads(validation.read_text());assert v['passed'];z=load(src);assert np.array_equal(active,z['active_od_index'])and np.allclose(z['requested'],requested,rtol=0,atol=1e-10)
 return dict(flow=z['arc_flow'],cost=z['arc_cost_seconds'],routeB=z['routeB_flow'],mean_cost=z['od_mean_cost_seconds'][active],gap=v['gap'],iterations=0,optimizer_success=True,message='Inherited exact verified high-demand UE cell',seconds=0.,function_calls=0,inherited_source=str(src.relative_to(ROOT)))

def fit(case):
 check();optimizer_test();started=time.monotonic();dem=load(prev.REP/'multi_routing_demands.npz');pr=load(prev.M/'road_projection.npz');rr=load(prev.REP/'turn_router.npz');ep=load(prev.REP/'multi_endpoint_components.npz');comp=rr['node_component'];ec=ep['components'];nc=int(comp.max())+1;q0=original_request(dem);arms=['normal']if case=='normal'else['flood','targeted'];records=[]
 for arm in arms:
  H0=load_npz(paths_for(case,arm,'A'));H1=load_npz(paths_for(case,arm,'B'));reachable=np.diff(H0.indptr)>0;assert np.array_equal(reachable,np.diff(H1.indptr)>0);closed,slow,tref,capacity=state(case,arm);warm=None
  for r in REDUCTIONS:
   qfull=q0*(1-r);idx=np.flatnonzero(reachable&(qfull>0));q=qfull[idx];A=H0[:,idx];B=H1[:,idx]
   for mode in ['UE','SO']:
    name=key(case,arm,r,mode);marker=R/(name+'_fit_validation.json');outfile=P/'fits'/(name+'.npz')
    if marker.exists()and json.loads(marker.read_text())['passed']:
     old=load(outfile)
     if mode=='UE':warm=old['routeB']/q
     records.append(json.loads(marker.read_text()));continue
    if time.monotonic()-started>2700:raise RuntimeError('Per-scope campaign budget reached')
    z=inherit(case,arm,r,mode,idx,qfull)
    if z is None:
     initial=np.zeros(len(q))if warm is None else warm*q;z=optimize(A,B,q,tref,capacity,mode,initial)
    if mode=='UE':warm=z['routeB']/q
    f=z['flow'];cost=z['cost'];weights=qfull*reachable;expected=np.bincount(ec[dem['origin_endpoint']],weights=weights,minlength=nc)-np.bincount(ec[dem['destination_endpoint']],weights=weights,minlength=nc);actual=np.bincount(comp[pr['from_index']],weights=f,minlength=nc)-np.bincount(comp[pr['to_index']],weights=f,minlength=nc);balance=float(abs(actual-expected).max());closedmax=float(f[closed[pr['arc_to_road']]].max(initial=0));costres=abs(float(q@z['mean_cost'])-float(f@cost))/max(1,float(f@cost));original=float(q0.sum());deferred=float((q0-qfull).sum());served=float(qfull[reachable].sum());failed=float(qfull[~reachable].sum());total=float(f@cost);ue_total=total if mode=='UE'else float(load(P/'fits'/(key(case,arm,r,'UE')+'.npz'))['total_model_cost'])
    route_loading_error=float(abs(np.asarray(A@(q-z['routeB'])+B@z['routeB']).ravel()-f).max(initial=0))
    checks=dict(gap=z['gap']<=.001,finite_nonnegative=bool(np.isfinite(f).all()and f.min()>=0),route_bounds=bool(((z['routeB']>=-1e-8)&(z['routeB']<=q+1e-8)).all()),two_path_flow_loading=route_loading_error<1e-6,component_conservation=balance<1e-6,closed_zero=closedmax<1e-9,cost_loading=costres<1e-9,original_demand_ledger=abs(original-deferred-served-failed)<1e-6,SO_network_cost_not_worse_than_UE=total<=ue_total*(1+1e-6))
    mean=np.full(len(q0),np.nan);mean[idx]=z['mean_cost'];np.savez_compressed(outfile,arc_flow=f,arc_cost=cost,mean_od_cost=mean,routeB=z['routeB'],active_od_index=idx,original_requested=q0,retained_requested=qfull,reachable=reachable,total_model_cost=np.array(total))
    val=dict(case_id=case,arm=arm,deferred_fraction=r,guidance=mode,passed=all(checks.values()),checks=checks,gap=z['gap'],iterations=z['iterations'],seconds=z['seconds'],optimizer_success=z['optimizer_success'],message=z['message'],inherited_source=z.get('inherited_source'),balance_residual=balance,cost_loading_residual=costres,original_requested=original,deferred=deferred,served=served,unreachable=failed,total_model_cost=total,paired_UE_total_model_cost=ue_total,code_sha256=sha(Path(__file__)));dump(marker,val);records.append(val);print(json.dumps({k:val[k]for k in ['case_id','arm','deferred_fraction','guidance','passed','gap','seconds']}),flush=True)
    if not val['passed']:raise RuntimeError('Numerical/gap gate failed:'+name)
 check();dump(R/(case+'_campaign.json'),dict(status='complete',cases=len(records),seconds=time.monotonic()-started))

def report():
 check();dem=load(prev.REP/'multi_routing_demands.npz');core=load(prev.P/'protection_masks.npz')['core_endpoint_od'];q0=original_request(dem);mask=core&dem['base'];pr=load(prev.M/'road_projection.npz');inv=pd.read_csv(prev.M/'road_inventory.csv');known=load(prev.F/'motor_common_support.npz')['qualified_roads'];ridx=pr['arc_to_road'];corearc=inv.inside_core.to_numpy()[ridx];rows=[];effects=[]
 for case in CASES:
  baseline=load(P/'fits'/(key(case,'flood',0.,'UE')+'.npz'));cohort=mask&baseline['reachable']&(q0>0);normal=load(P/'fits'/(key('normal','normal',0.,'UE')+'.npz'));normalcohort=mask&normal['reachable']&(q0>0)
  for arm in ['flood','targeted']:
   closed,slow,tref,capacity=state(case,arm)
   for r in REDUCTIONS:
    unmanaged=load(P/'fits'/(key(case,'flood',r,'UE')+'.npz'));sameue=load(P/'fits'/(key(case,arm,r,'UE')+'.npz'))
    for mode in ['UE','SO']:
     name=key(case,arm,r,mode);z=load(P/'fits'/(name+'.npz'));request=z['retained_requested'];served=mask&z['reachable']&(request>0);failed=mask&~z['reachable'];original=float(q0[mask].sum());deferred=float((q0[mask]-request[mask]).sum());servedw=float(request[served].sum());unserved=float(request[failed].sum());assert abs(original-deferred-servedw-unserved)<1e-6
     cohort_ok=cohort&z['reachable'];reference_cost=baseline['mean_od_cost'][cohort_ok];change=z['mean_od_cost'][cohort_ok]-reference_cost;weights=q0[cohort_ok];sameuechange=z['mean_od_cost'][served]-sameue['mean_od_cost'][served];normalpaired=served&normalcohort;vsnormal=z['mean_od_cost'][normalpaired]-normal['mean_od_cost'][normalpaired];vc=z['arc_flow']/capacity;eligiblearc=corearc&~closed[ridx];restored=mask&~unmanaged['reachable']&z['reachable'];restoredw=float(request[restored].sum());den=float(request[mask&~unmanaged['reachable']].sum());totaloriginal=float(q0.sum());retained=float(request.sum())
     rows.append(dict(case_id=case,protection=arm,deferred_fraction=r,guidance=mode,original_core_request=original,deferred_core_proxy=deferred,retained_served_core_proxy=servedw,retained_unreachable_core_proxy=unserved,completion_fraction_original=servedw/original,total_noncompletion_fraction_original=(deferred+unserved)/original,unreachable_fraction_retained=unserved/max(original-deferred,1e-30),restored_same_deferral_proxy=restoredw,restored_same_deferral_fraction=restoredw/den if den else None,fixed_original_cohort_mean_cost_min=float(weights@z['mean_od_cost'][cohort_ok]/weights.sum()/60),fixed_original_cohort_cost_change_vs_full_no_action_min=float(weights@change/weights.sum()/60),fixed_original_cohort_over1min_worse_share=float(weights[change>60].sum()/weights.sum()),same_retained_guidance_mean_saving_min=float(-request[served]@sameuechange/request[served].sum()/60),served_mean_cost_min=float(request[served]@z['mean_od_cost'][served]/request[served].sum()/60),retained_survivor_added_vs_original_normal_min=float(request[normalpaired]@vsnormal/request[normalpaired].sum()/60),total_network_cost=float(z['total_model_cost']),network_cost_per_retained_request_min=float(z['total_model_cost'])/retained/60,original_all_context_request=totaloriginal,deferred_all_context_proxy=float((q0-request).sum()),vc_gt1_known_directed_core_intersect_km=float(pr['length_m'][eligiblearc&known[ridx]&(vc>1)].sum()/1000),vc_gt1_unknown_directed_core_intersect_km=float(pr['length_m'][eligiblearc&~known[ridx]&(vc>1)].sum()/1000)))
     if mode=='SO':effects.append(dict(case_id=case,protection=arm,deferred_fraction=r,same_retained_network_total_cost_change_fraction=float(z['total_model_cost']/sameue['total_model_cost']-1),core_served_guidance_saving_min=float(-request[served]@sameuechange/request[served].sum()/60),guidance_can_redistribute_cost=True))
 df=pd.DataFrame(rows);save(P/'combined_outcomes.csv',df);save(P/'same_demand_guidance_effects.csv',effects)
 fronts=[]
 for case,g in df.groupby('case_id'):
  completion=g.completion_fraction_original.to_numpy();cost=g.fixed_original_cohort_mean_cost_min.to_numpy()
  for i,(_,z)in enumerate(g.iterrows()):
   dominated=((completion>=completion[i]-1e-12)&(cost<=cost[i]+1e-8)&((completion>completion[i]+1e-12)|(cost<cost[i]-1e-8))).any();fronts.append(dict(case_id=case,protection=z.protection,deferred_fraction=z.deferred_fraction,guidance=z.guidance,nondominated_completion_vs_fixed_cohort_cost=not bool(dominated)))
 save(P/'descriptive_tradeoff_front.csv',fronts)
 fits=[json.loads(f.read_text())for f in R.glob('*_fit_validation.json')];pathvals=[json.loads(f.read_text())for f in R.glob('*_path_validation.json')];assert len(fits)==40 and all(v['passed']for v in fits);assert len(pathvals)==4 and all(v['passed']for v in pathvals);assert sum(v['seconds']for v in fits)<=5400
 validation=dict(passed=True,combination_cells=32,normal_controls=8,total_fits=40,new_optimization_cases=sum(v['inherited_source']is None for v in fits),inherited_verified_cases=sum(v['inherited_source']is not None for v in fits),maximum_gap=max(v['gap']for v in fits),maximum_component_balance=max(v['balance_residual']for v in fits),fit_active_seconds=sum(v['seconds']for v in fits),path_active_seconds=sum(v['seconds']for v in pathvals),all_original_requests_accounted=True,SO_comparison_same_retained_demand=True,plan_frozen_across_two_events=True,new_hydraulics=0,people_counted=0)
 dump(R/'validation.json',validation);dump(R/'summary.json',dict(validation=validation,outcomes=df.astype(object).where(pd.notna(df),None).to_dict('records')));render(df,pd.DataFrame(effects));print(json.dumps(validation,indent=2),flush=True)

def render(df,effects):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,case in zip(axs,CASES):
  g=df[df.case_id.eq(case)]
  for arm,color in [('flood','#b35c50'),('targeted','#277ca5')]:
   for mode,style in [('UE','--'),('SO','-')]:
    h=g[g.protection.eq(arm)&g.guidance.eq(mode)].sort_values('deferred_fraction');ax.plot(h.deferred_fraction*100,h.fixed_original_cohort_cost_change_vs_full_no_action_min,color=color,linestyle=style,marker='o',label=arm+'/'+mode)
  ax.axhline(0,color='#777',lw=.8);ax.set(title=case,xlabel='Deferred from current hour (%)',ylabel='Same-cohort modeled cost change (min)');ax.legend(fontsize=8)
 fig.suptitle('Cost versus full no-action flood baseline | deferred requests are not completed trips');fig.savefig(R/'combined_cost_response.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,case in zip(axs,CASES):
  for arm,color in [('flood','#b35c50'),('targeted','#277ca5')]:
   h=df[df.case_id.eq(case)&df.protection.eq(arm)&df.guidance.eq('UE')].sort_values('deferred_fraction');ax.plot(h.deferred_fraction*100,h.completion_fraction_original*100,color=color,marker='o',label=arm)
  ax.set(title=case,xlabel='Deferred from current hour (%)',ylabel='Completed share of original request (%)',ylim=[0,100]);ax.legend()
 fig.suptitle('Accessibility is not demand deferral | guidance does not create new physical links');fig.savefig(R/'original_request_completion.png',dpi=150);plt.close(fig)
 # Network system objective need not improve the core subset in every cell.
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
 for case,marker in zip(CASES,['o','s']):
  h=effects[effects.case_id.eq(case)&effects.protection.eq('targeted')].sort_values('deferred_fraction');ax.plot(h.deferred_fraction*100,h.core_served_guidance_saving_min,marker=marker,label=case)
 ax.axhline(0,color='#777',lw=.8);ax.set(xlabel='Deferred from current hour (%)',ylabel='Same-demand core mean saving from guidance (min)',title='Ideal system-cost guidance | negative means core users lose');ax.legend();fig.savefig(R/'guidance_core_tradeoff.png',dpi=150);plt.close(fig)
 def tab(z):return'<div class="scroll">'+z.to_html(index=False,border=0,float_format=lambda v:f'{v:,.3f}')+'</div>'
 compact=df[['case_id','protection','deferred_fraction','guidance','completion_fraction_original','fixed_original_cohort_cost_change_vs_full_no_action_min','fixed_original_cohort_over1min_worse_share','same_retained_guidance_mean_saving_min']]
 protocol=json.loads((P/'protocol.json').read_text());body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>通道保障、需求管理与理想疏导的两场暴雨比较</title><style>body{{margin:0;background:#edf2f5;color:#233e4b;font:16px/1.8 system-ui}}main{{max-width:1160px;margin:28px auto;background:white;padding:30px;border-radius:14px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:28px}}.lead{{padding:18px;background:#e7f1ec;border-left:5px solid #478b73}}.note{{padding:16px;background:#fff0dd}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;min-width:800px;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%;height:auto}}a{{color:#146587}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main><p>2026-10-18 JST · 都心三区与道路上下文 · MILI-D-20261018-011</p><h1>通道保障＋需求管理／理想协同疏导</h1><div class="lead">已完成2场已有雷达暴雨×2档物理保障×4档需求移出×2种路径分配，共32组组合及8组正常控制。新通道方案冻结118段、4.743km；不重新选路、不追加水动力、不计算人数。</div>
<h2>本轮的问题和固定条件</h2><p>上一轮高需求下，通道保障能恢复可达性，却使部分原本可达出行变慢。本轮比较无保障／冻结新方案、当前小时移出0／10／20／30%需求，以及私人参考成本分配UE／理想协同分配SO。高需求基准沿用原PT推导17时车辆出行代理×2，容量、两条候选路径、道路权限和数值门槛保持同一体系。</p><p>SO在相同路径、相同保留需求下最小化整个10km道路上下文的总模型车辆成本，假设完全执行；没有信号配时、交叉口、真实信息和服从率模型。它可能将成本转移到三区旅客，不能称为所有人的社会最优。UE与SO均仅为有限候选内的静态解，不是全路网或动态排队模型。</p>
<h2>先把需求账本说清楚</h2><p class="note">原请求＝主动移出当前小时＋保留且完成＋保留但因断连未完成。移出的需求没有计作已完成或恢复通行，也不假设以后必然完成。每个OD按同一比例移出，因此这里只测试均匀减量；尚未模拟谁能居家办公、货运错峰或出行必要性。未为错峰／取消赋予福利成本。</p>{tab(pd.read_csv(P/'management_input_ledger.csv'))}<img src="original_request_completion.png" alt="Original-demand completion across deferral fractions,with physical road protection distinguished">
<h2>两场雨与时钟的含义</h2><p>采用2025-09-11与2024-08-30两批已通过数值规则的U城市几何结果，均为历史雷达重建雨场＋假设河道抬升1.5m、窗口起点后4h。2025时刻约17:15JST；2024时刻约06:00JST。为控制交通负荷，两者都施加同一17时需求代理，因此第二场是天气形态迁移实验，不是早晨6时历史交通再现。</p><p>方案在2025事件选出，此处仍为样内评估；2024事件不重选方案，是另一天的迁移检查。两场并不能证明一般有效。均保留桥隧高程／地表对应未知，以及未知道路按正常可用的情景假设；不能把缺证据道路当成已核验安全。</p>
<h2>固定原有可达出行群体的成本响应</h2><p>下图与表中固定群体是各场无措施、未减量时本来可达的三区起终点OD，并使用原请求权重。每档保留需求在该群体中按相同比例采样，避免把取消某些出行或新增幸存者混合当作成本改善。新增恢复的出行与移出的出行仍分别列账。</p><img src="combined_cost_response.png" alt="Same-cohort modeled cost response to deferred demand and ideal guidance">{tab(compact)}
<h2>疏导的总成本与局部代价</h2>{tab(effects)}<img src="guidance_core_tradeoff.png" alt="Core users can gain or lose under ideal system objective guidance"><p>SO检查的总成本不高于同一保留需求UE，保证比较没有靠取消更多出行来获益。这个约束针对整个道路上下文，不约束每个OD、三区平均或每条路都改善。报告同时给出核心出行的成本变化和未知对应道路的容量压力。</p>
<h2>如何使用这些结果</h2><p>应在同一移出比例下比较保障与疏导；跨比例比较必须同时看原请求完成率、固定群体成本和被移出量。描述性非支配表只涵盖这些模型指标，不代表最佳政策，因为移出出行的社会成本与保障工程费用尚未计入。</p><p>tref沿用既有U日间观测／代理时耗，并非实测自由流时耗；c=tref[1+0.15(v/C)^4]给出容量响应成本，不是真实排队延误。容量基于道路等级和有效车道假设，大量车道缺失仍用每方向1条；车型和外部穿越未知未补齐。未完成需求留在账本，静态模型不会让它在道路上真实排队。</p>
<h2>核验和复现</h2><p>所有组检查原请求账本、车辆守恒、封闭道路零流量、路径成本加载、候选内优化间隙≤0.001、相同保留需求下SO总成本不高于UE。解析双路径算例分别核验UE实际成本与SO边际成本条件。3组无移出UE精确继承前次已验证结果，其余37组新分配，不把继承或两个时刻当作独立暴雨事件。</p><ul>'''
 for f in ['protocol.json','input_bindings.json','management_input_ledger.csv','combined_outcomes.csv','same_demand_guidance_effects.csv','descriptive_tradeoff_front.csv']:
  body+=f'<li><a href="../../../processed/{P.name}/{f}">{f}</a></li>'
 body+='<li><a href="validation.json">validation.json</a></li><li><a href="summary.json">summary.json</a></li></ul><p>道路 © OpenStreetMap contributors / ODbL；PLATEAU、DEM、雷达与统计来源沿用已有审计。复现 run_combined_transport_measures_v1.py：prepare → paths → fit --case normal／各事件 → report。</p></main></html>'
 (R/'review.html').write_text(body)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','paths','fit','report']);ap.add_argument('--case',choices=['normal']+CASES);a=ap.parse_args()
 try:
  if a.phase=='prepare':prepare()
  elif a.phase=='paths':generate_paths()
  elif a.phase=='fit':
   if a.case is None:raise ValueError('fit requires --case')
   fit(a.case)
  else:report()
 except Exception as exc:
  if R.exists():dump(R/('execution_'+(a.case or a.phase)+'.json'),dict(status='stopped',phase=a.phase,case=a.case,reason=str(exc)))
  raise
 else:dump(R/('execution_'+(a.case or a.phase)+'.json'),dict(status='complete',phase=a.phase,case=a.case,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
if __name__=='__main__':main()
