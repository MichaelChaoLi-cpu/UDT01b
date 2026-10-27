"""A2: registered H anchors;identical two-path measures and numerical admission.
Reuse is per exact state/input/path/objective fingerprint,then independently audited.
"""
import sys,time,json,shutil,hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz
from temporal_transport_common_v1 import *
import run_combined_transport_measures_v1 as original
TP=P/'traffic';TR=R/'traffic';OLD=D/'three_ward_combined_transport_measures_v1';OLDR=ROOT/'data/exp/estimation'/OLD.name
CONTEXT={};START=0.;REUSE=[];AUDITS=[];REFERENCES={}

def check():
 gate()
 if START:assert time.monotonic()-START<21600,'A2 stage budget reached'
 sources()

def paths_for(case,arm,kind):return MEASURE/f'normal_{kind}.npz'if case=='normal'else TP/'paths'/f'{case}__{arm}_{kind}.npz'
@lru_cache(maxsize=12)
def state_codes(case):
 clip=load(P/'clipped_road_lengths.npz');q=clip['qualified'];c=np.zeros(len(q),bool);s=c.copy()
 if case!='normal':
  row=REFERENCES[case];z=load(P/'population'/(row['case_id']+'.npz'));code=z['road_state_codes'][int(row['frame']),0];qi=z['road_indices'];c[qi]=code==2;s[qi]=code==1
 return c,s

def state(case,arm):
 c,s=state_codes(case);c=c.copy();s=s.copy()
 if arm=='targeted':c[CONTEXT['protection']]=False;s[CONTEXT['protection']]=False
 t=CONTEXT['tref'].copy();cap=CONTEXT['capacity'].copy();ridx=CONTEXT['pr']['arc_to_road'];t[s[ridx]]*=2;cap[s[ridx]]*=.5
 return c,s,t,cap

def configure():
 global REFERENCES
 init();REFERENCES=pd.read_csv(P/'reference_states.csv').set_index('reference_id').to_dict('index');CONTEXT.update(pr=load(M/'road_projection.npz'),rr=load(REP/'turn_router.npz'),ep=load(REP/'multi_endpoint_components.npz'),dem=load(REP/'multi_routing_demands.npz'),tref=load(C/'choice_costs.npz')['travel_seconds'],capacity=load(MEASURE/'capacity_assumptions.npz')['hourly_capacity'],protection=load(P/'clipped_road_lengths.npz')['protected']);CONTEXT['q0']=original.original_request(CONTEXT['dem']);CONTEXT['core']=load(MEASURE/'protection_masks.npz')['core_endpoint_od']&CONTEXT['dem']['base']
 original.P=TP;original.R=TR;original.check=check;original.paths_for=paths_for;original.state=state;original.inherit=lambda *args:None
 raw_dijkstra=original.dijkstra;raw_minimize=original.minimize
 def tick():
  gate()
  if START:assert time.monotonic()-START<21600,'A2 stage budget reached'
 def guarded_dijkstra(*args,**kwargs):tick();return raw_dijkstra(*args,**kwargs)
 def guarded_minimize(fun,*args,**kwargs):
  def guarded_fun(x):tick();return fun(x)
  return raw_minimize(guarded_fun,*args,**kwargs)
 original.dijkstra=guarded_dijkstra;original.minimize=guarded_minimize
 static=[REP/'turn_router.npz',REP/'multi_endpoint_components.npz',REP/'multi_routing_demands.npz',M/'road_projection.npz',C/'choice_costs.npz',MEASURE/'capacity_assumptions.npz',MEASURE/'protection_masks.npz']
 hashes={str(f.relative_to(ROOT)):sha(f)for f in static};oldbind=json.loads((OLD/'input_bindings.json').read_text())['sha256'];assert all(oldbind.get(n)==h for n,h in hashes.items()),'Prior normal/static controls differ'
 CONTEXT['static_hashes']=hashes;CONTEXT['optimizer_sha256']=sha(ROOT/'src/analyses/run_combined_transport_measures_v1.py');CONTEXT['contract']=dict(static=hashes,optimizer=CONTEXT['optimizer_sha256'],path_rule='A:turn-aware reference shortest;B:base17h AON cost1+.15vC^4,capped50 for candidate generation only',fit_rule=dict(UE_integral=.03,UE_marginal=.15,SO_objective=.15,SO_marginal=.75,maximum_iterations=2000,maximum_seconds=120,gap=.001),request='frozen17hproxy*2',slow_time=2,slow_capacity=.5)
 dump(TP/'static_traffic_contract.json',CONTEXT['contract'])

def state_hash(c,s):return hashlib.sha256(np.packbits(np.stack([c,s])).tobytes()).hexdigest()
def audit_fit(case,arm,r,mode,rawpath,v):
 z=load(rawpath);c,s,t,cap=state(case,arm);dem=CONTEXT['dem'];pr=CONTEXT['pr'];rr=CONTEXT['rr'];ep=CONTEXT['ep'];q0=CONTEXT['q0'];qfull=q0*(1-r);H=load_npz(paths_for(case,arm,'A'));J=load_npz(paths_for(case,arm,'B'));reachable=np.diff(H.indptr)>0;assert np.array_equal(reachable,np.diff(J.indptr)>0);idx=np.flatnonzero(reachable&(qfull>0));assert np.array_equal(z['original_requested'],q0)and np.array_equal(z['retained_requested'],qfull)and np.array_equal(z['reachable'],reachable)and np.array_equal(z['active_od_index'],idx)
 q=qfull[idx];A=H[:,idx];B=J[:,idx];x=z['routeB'];f=z['arc_flow'];cost=z['arc_cost'];loaderr=float(abs(np.asarray(A@(q-x)+B@x).ravel()-f).max(initial=0));costerr=float(abs(t*(1+.15*(f/cap)**4)-cost).max(initial=0));ttA=np.asarray(A.T@cost).ravel();ttB=np.asarray(B.T@cost).ravel();expectedmean=((q-x)*ttA+x*ttB)/q;meanerr=float(abs(expectedmean-z['mean_od_cost'][idx]).max(initial=0));loadcost=abs(float(q@expectedmean)-float(f@cost))/max(1,float(f@cost));marginal=t*(1+(.15 if mode=='UE'else .75)*(f/cap)**4);ma=np.asarray(A.T@marginal).ravel();mb=np.asarray(B.T@marginal).ravel();gap=max(0.,(float(f@marginal)-float(q@np.minimum(ma,mb)))/max(float(f@marginal),1));comp=rr['node_component'];ec=ep['components'];nc=int(comp.max())+1;w=qfull*reachable;expected=np.bincount(ec[dem['origin_endpoint']],weights=w,minlength=nc)-np.bincount(ec[dem['destination_endpoint']],weights=w,minlength=nc);actual=np.bincount(comp[pr['from_index']],weights=f,minlength=nc)-np.bincount(comp[pr['to_index']],weights=f,minlength=nc);balance=float(abs(expected-actual).max());closedmax=float(f[c[pr['arc_to_road']]].max(initial=0));ledger=float(abs((q0-qfull)+qfull*reachable+qfull*~reachable-q0).max(initial=0));total=float(f@cost);ue=total if mode=='UE'else float(load(TP/'fits'/(original.key(case,arm,r,'UE')+'.npz'))['total_model_cost']);checks=dict(prior_report_passed=bool(v['passed']),gap=gap<=.001,gap_replayed=abs(gap-v['gap'])<1e-8,nonnegative_finite=bool(np.isfinite(f).all()and f.min(initial=0)>=0),route_bounds=bool((x>=-1e-8).all()and(x<=q+1e-8).all()),route_loading=loaderr<1e-6,cost_formula=costerr<1e-6,per_OD_mean_cost=meanerr<1e-6,cost_loading=loadcost<1e-9,component_conservation=balance<1e-6,closed_zero=closedmax<1e-9,per_OD_ledger=ledger<1e-6,total_cost=abs(total-float(z['total_model_cost']))/max(1,total)<1e-12,SO_not_worse=total<=ue*(1+1e-6));assert all(checks.values()),(case,arm,r,mode,checks)
 return dict(case_id=case,arm=arm,deferred_fraction=r,guidance=mode,passed=True,checks=checks,gap=gap,balance_residual=balance,cost_loading_residual=loadcost,max_cost_error=costerr,max_route_loading_error=loaderr,max_per_OD_ledger_error=ledger,original_requested=float(q0.sum()),deferred=float((q0-qfull).sum()),served=float(qfull[reachable].sum()),unreachable=float(qfull[~reachable].sum()),total_model_cost=total,paired_UE_total_model_cost=ue)

def library():
 """Legacy snapshots are candidates only;state counts never authorize reuse."""
 q=load(PREV/'motor_common_support.npz')['qualified_roads'];prot=CONTEXT['protection'];out=[dict(case='normal',arm='normal',c=np.zeros(len(q),bool),s=np.zeros(len(q),bool),data=OLD,report=OLDR)]
 for case in ['E20250911__H__U','E20240830__H__U','E20240618__H__U','E20240731__H__U','E20250710__H__U']:
  ex=load(PREV/'motor_exposures'/(case+'.npz'));assert abs(ex['actual_times'][5]-14400)<.001;c=q&(ex['lower'][5]>=.3);s=q&(ex['lower'][5]>=.1)&~c
  for arm in ['flood','targeted']:
   cc=c.copy();ss=s.copy()
   if arm=='targeted':cc[prot]=False;ss[prot]=False
   old=case in ['E20250911__H__U','E20240830__H__U'];out.append(dict(case=case,arm=arm,c=cc,s=ss,data=OLD if old else PREV/'traffic',report=OLDR if old else ROOT/'data/exp/estimation'/PREV.name/'traffic'))
 return out

def library_paths(item,kind):
 if item['case']=='normal':return MEASURE/f'normal_{kind}.npz'
 if item['case']=='E20250911__H__U':return MEASURE/f"{item['arm']}_{kind}.npz"
 return item['data']/'paths'/f"{item['case']}__{item['arm']}_{kind}.npz"

def reuse_arm(case,arm,item):
 c,s,t,cap=state(case,arm);assert np.array_equal(c,item['c'])and np.array_equal(s,item['s']);sh=state_hash(c,s);fingerprint=dict(contract=CONTEXT['contract'],state_sha256=sh,path_A_sha256=sha(library_paths(item,'A')),path_B_sha256=sha(library_paths(item,'B')));fp=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
 for kind in ['A','B']:
  src=library_paths(item,kind);dst=paths_for(case,arm,kind)
  if src!=dst:
   assert not dst.exists();shutil.copy2(src,dst);assert sha(src)==sha(dst)
  dump(TR/(f'{case}__{arm}_{kind}_path_validation.json'),dict(passed=True,reused=True,source=str(src.relative_to(ROOT)),source_sha256=sha(src),traffic_fingerprint_sha256=fp,state_sha256=sh,seconds=0.,note='Prior validated turn-aware path pool at exact same open/slow/cost/capacity/OD state;every fit independently re-audited.'))
 for r in original.REDUCTIONS:
  for mode in ['UE','SO']:
   name=original.key(case,arm,r,mode);oldname=original.key(item['case'],item['arm'],r,mode);src=item['data']/'fits'/(oldname+'.npz');srcval=item['report']/(oldname+'_fit_validation.json');v=json.loads(srcval.read_text());assert v['code_sha256']==CONTEXT['optimizer_sha256']and v['passed'],'Prior objective/code differs';dest=TP/'fits'/(name+'.npz');assert not dest.exists();shutil.copy2(src,dest);a=audit_fit(case,arm,r,mode,dest,v);a.update(seconds=0.,iterations=0,optimizer_success=v['optimizer_success'],message='Exact traffic fingerprint reuse;all numerical rules independently replayed',inherited_source=str(src.relative_to(ROOT)),source_sha256=sha(src),source_validation_sha256=sha(srcval),traffic_fingerprint_sha256=fp,code_sha256=CONTEXT['optimizer_sha256']);dump(TR/(name+'_fit_validation.json'),a);AUDITS.append(a);REUSE.append(dict(case_id=case,arm=arm,deferred_fraction=r,guidance=mode,action='verified_exact_reuse',source=str(src.relative_to(ROOT)),traffic_fingerprint_sha256=fp))
 print('EXACT TRAFFIC REUSE',case,arm,'from',item['case'],item['arm'],flush=True)

def unknown_diagnostic(refs):
 engine=Reachability();blocks,pop,normal,infos=population_inputs();q=load(P/'clipped_road_lengths.npz')['qualified'];normalstrict=engine.get(~q);assert not(normalstrict&~normal).any();rows=[]
 for case in refs:
  for arm in ['flood','targeted']:
   closed,_,_,_=state(case,arm);ok=engine.get((~q)|closed);assert not(ok&~normalstrict).any();lost=normalstrict&~ok
   for buf,info in infos.items():
    complete=(info['lens']>0)&(counts(normalstrict[info['ix']],info['ptr'])==info['lens']);n=counts(lost[info['ix']],info['ptr']);lo=complete&(n==info['lens']);up=complete&(n>0)
    rows.append(dict(reference_id=case,arm=arm,buffer_m=buf,total_stock=int(pop.sum()),strict_normal_supported_stock=int(pop[complete].sum()),strict_normal_unresolved_stock=int(pop[~complete].sum()),main_normal_supported_stock=int(pop[info['complete']].sum()),conditional_loss_lower=int(pop[lo].sum()),conditional_loss_upper=int(pop[up].sum()),physical_flood_upper_bound=False))
 save(P/'unknown_withheld_reference_diagnostic.csv',rows)

def summarize(refs):
 q0=CONTEXT['q0'];core=CONTEXT['core'];full=float(q0[core].sum());rows=[];effects=[]
 for case in refs:
  baseline=load(TP/'fits'/(original.key(case,'flood',0.,'UE')+'.npz'));cohort=core&baseline['reachable']&(q0>0);weights=q0[cohort];assert weights.sum()>0;basecost=float(weights@baseline['mean_od_cost'][cohort]/weights.sum());meta=REFERENCES[case]
  for arm in ['flood','targeted']:
   for r in original.REDUCTIONS:
    ue=load(TP/'fits'/(original.key(case,arm,r,'UE')+'.npz'));noaction=load(TP/'fits'/(original.key(case,'flood',r,'UE')+'.npz'))
    for mode in ['UE','SO']:
     z=load(TP/'fits'/(original.key(case,arm,r,mode)+'.npz'));ret=z['retained_requested'];serv=core&z['reachable'];fail=core&~z['reachable'];deferred=float((q0[core]-ret[core]).sum());served=float(ret[serv].sum());unreachable=float(ret[fail].sum());assert abs(full-deferred-served-unreachable)<1e-6;assert z['reachable'][cohort].all();cost=float(weights@z['mean_od_cost'][cohort]/weights.sum());rest=core&~noaction['reachable']&z['reachable'];row=dict(reference_id=case,event_id=meta['event_id'],frame=meta['frame'],elapsed_s=meta['analysis_elapsed_s'],selection_reasons=meta['selection_reasons'],protection=arm,deferred_fraction=r,guidance=mode,original_core_request_proxy=full,deferred_core_proxy=deferred,retained_served_core_proxy=served,retained_unreachable_core_proxy=unreachable,completion_fraction_original=served/full,restored_same_deferral_proxy=float(ret[rest].sum()),fixed_original_cohort_cost_index=100*cost/basecost,fixed_original_cohort_mean_cost_min=cost/60,total_network_model_cost=float(z['total_model_cost']),same_retained_guidance_network_cost_index=100*float(z['total_model_cost'])/float(ue['total_model_cost']));rows.append(row)
     if mode=='SO':effects.append(dict(reference_id=case,event_id=meta['event_id'],protection=arm,deferred_fraction=r,network_cost_reduction_fraction=1-float(z['total_model_cost'])/float(ue['total_model_cost']),same_retained_demand=True))
 save(TP/'combined_outcomes.csv',rows);save(TP/'same_demand_guidance_effects.csv',effects);save(TP/'fit_reuse_ledger.csv',REUSE)

def main():
 global START
 START=time.monotonic();assert json.loads((R/'A1_validation.json').read_text())['passed'];configure();check();original.optimizer_test();refs=list(REFERENCES);assert 5<=len(refs)<=10 and len({REFERENCES[k]['event_id']for k in refs})==5;lib=library();reuse_arm('normal','normal',lib[0]);unknown_diagnostic(refs);complete=[];exclusions=[]
 for number,case in enumerate(refs,1):
  check();dump(R/'A2_progress.json',dict(status='running',current_reference=case,references_completed=len(complete),references_total=len(refs),completed_measure_cells=16*len(complete),normal_controls_reused=8,elapsed_s=time.monotonic()-START))
  for arm in ['flood','targeted']:
   c,s,_,_=state(case,arm);match=next((item for item in lib if np.array_equal(item['c'],c)and np.array_equal(item['s'],s)),None)
   if match:reuse_arm(case,arm,match)
   else:exclusions.append(dict(reference_id=case,arm=arm,reason='No COMPLETE open/slow state match in validated library;counts/time alone do not permit reuse',action='new_paths_and_fits'))
  original.CASES=['unused_no_inheritance',case];original.generate_paths();shutil.copy2(TR/'path_campaign.json',TR/(case+'_path_campaign.json'));original.fit(case)
  for arm in ['flood','targeted']:
   c,s,_,_=state(case,arm)
   fingerprint=dict(contract=CONTEXT['contract'],state_sha256=state_hash(c,s),path_A_sha256=sha(paths_for(case,arm,'A')),path_B_sha256=sha(paths_for(case,arm,'B')));fp=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest();dump(TP/'paths'/(case+'__'+arm+'_fingerprint.json'),dict(sha256=fp,inputs=fingerprint))
   for r in original.REDUCTIONS:
    for mode in ['UE','SO']:
     name=original.key(case,arm,r,mode);v=json.loads((TR/(name+'_fit_validation.json')).read_text());a=audit_fit(case,arm,r,mode,TP/'fits'/(name+'.npz'),v);AUDITS.append(a)
     if not v.get('inherited_source'):REUSE.append(dict(case_id=case,arm=arm,deferred_fraction=r,guidance=mode,action='new_bounded_fit',source='',traffic_fingerprint_sha256=fp))
   lib.append(dict(case=case,arm=arm,c=c,s=s,data=TP,report=TR))
  complete.append(case);summarize(complete);save(TP/'reuse_exclusion_ledger.csv',exclusions);save(TR/'independent_fit_audit.csv',AUDITS);dump(R/'A2_progress.json',dict(status='running',references_completed=len(complete),references_total=len(refs),completed_measure_cells=16*len(complete),normal_controls_reused=8,elapsed_s=time.monotonic()-START));print('A2 REFERENCE QUALIFIED',case,len(complete),'/',len(refs),flush=True)
  from report_temporal_transport_v1 import main as report
  report()
 vals=[json.loads(f.read_text())for f in TR.glob('*_fit_validation.json')];paths=[json.loads(f.read_text())for f in TR.glob('*_path_validation.json')];assert len(vals)==len(refs)*16+8 and all(v['passed']for v in vals+paths);assert len(paths)==len(refs)*4+2;assert time.monotonic()-START<21600;check();new=sum(not v.get('inherited_source')for v in vals);v=dict(passed=True,reference_states=len(refs),unique_rainfall_parents=5,measure_cells=len(refs)*16,normal_controls_reused=8,new_fits=new,verified_reused_fits=len(vals)-new,new_path_sweeps=sum(not v.get('reused',False)for v in paths),maximum_gap=max(v['gap']for v in vals),maximum_component_balance=max(v['balance_residual']for v in vals),all_fits_independently_audited=True,all_original_requests_accounted=True,seconds=time.monotonic()-START,new_hydraulics=0,training=False,sealed_tests=False);dump(R/'A2_validation.json',v);dump(R/'A2_progress.json',dict(status='qualified_complete',references_completed=len(refs),references_total=len(refs),completed_measure_cells=len(refs)*16,elapsed_s=v['seconds']));print('A2 ALL MEASURE CHECKS PASSED',json.dumps(v),flush=True)
if __name__=='__main__':main()
