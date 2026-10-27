"""A1: all1090actual snapshots,unique road accounting and resident-stock intervals."""
from pathlib import Path
import sys,json,time,csv,datetime as dt
import numpy as np,pandas as pd
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from temporal_transport_common_v1 import *
from prepare_historical_surface_coverage_v2 import edge_maximum

def envelope(nodes,info,pop):
 n=counts(nodes[info['ix']],info['ptr']);return info['complete']&(n==info['lens']),info['complete']&(n>0)
def signed_envelope(delta,info,pop):
 pos=counts((delta[info['ix']]>0),info['ptr']);neg=counts((delta[info['ix']]<0),info['ptr']);lo=np.where(neg>0,-1,np.where(pos==info['lens'],1,0));up=np.where(pos>0,1,np.where(neg==info['lens'],-1,0));return lo*info['complete']*pop,up*info['complete']*pop
def episodes(case,bi,ai,closed,ts,qi,length,inventory,writer):
 duration=np.trapezoid(closed.astype(float),ts,axis=0)/60.;left=np.sum(closed[:-1]*np.diff(ts)[:,None],axis=0)/60;right=np.sum(closed[1:]*np.diff(ts)[:,None],axis=0)/60;longlo=np.zeros(len(qi));longup=longlo.copy();ever=closed.any(0);last=closed[-1];firstlo=np.full(len(qi),np.nan);firsthi=firstlo.copy()
 for j in np.flatnonzero(ever&(length[qi]>0)):
  c=closed[:,j];changes=np.flatnonzero(np.diff(np.r_[False,c,False].astype('i1')));assert len(changes)%2==0
  for number,(a,b)in enumerate(zip(changes[::2],changes[1::2]),1):
   lo=ts[a-1]if a>0 else 0.;hi=ts[a];rlo=ts[b-1];rup=ts[b]if b<len(ts)else None;dl=max(0.,rlo-hi)/60;du=((ts[-1]if rup is None else rup)-lo)/60;longlo[j]=max(longlo[j],dl);longup[j]=max(longup[j],du)
   if number==1:firstlo[j]=lo;firsthi[j]=hi
   writer.writerow(dict(case=case,bound=bi,arm=ai,road_index=int(qi[j]),source_edge_id=inventory.source_edge_id.iloc[qi[j]],episode=number,onset_lower_s=lo,onset_upper_s=hi,recovery_lower_s=rlo,recovery_upper_s=rup,left_censored=bool(a==0),right_censored=bool(b==len(ts)),observed_span_min=dl,conditional_span_upper_min=du,three_ward_clipped_length_m=float(length[qi[j]])))
 return dict(sampled_closed_min=duration,left_hold_min=left,right_hold_min=right,longest_observed_span_min=longlo,longest_conditional_span_upper_min=longup,ever_closed=ever,closed_at_horizon=last,first_onset_lower_s=firstlo,first_onset_upper_s=firsthi)
def readkey(key,n):return np.unpackbits(load(P/'reachability_cache'/(str(key)+'.npz'))['node_ok_packed'],count=n).astype(bool)

def main():
 started=time.monotonic();init();assert json.loads((R/'A0_validation.json').read_text())['passed'];sources();inv=pd.read_csv(P/'input_inventory.csv');roadinv=pd.read_csv(M/'road_inventory.csv');clip=load(P/'clipped_road_lengths.npz');q=clip['qualified'];qi=np.flatnonzero(q);protected=clip['protected'];effective_q=protected[qi];length=clip['union_length_m'];wlength=clip['ward_length_m'];weights=np.vstack([length,wlength]);scopes=['Three wards']+WARD;engine=Reachability();blocks,pop,normal,infos=population_inputs();wardarr=blocks.ward.to_numpy(dtype=str);select=[np.ones(len(pop),bool)]+[wardarr==w for w in WARD];pr=engine.pr;roadrows=[];prows=[];restrows=[];everrows=[];peakrows=[];ref=[];codecount=0;reusechecks=[];priorpop=pd.read_csv(PREV/'population/population_effect_summary.csv');priorroads=load(PREV/'motor_common_support.npz');assert np.array_equal(q,priorroads['qualified_roads']);timing=pd.read_csv(PLAN/'rain_timing_context.csv').set_index('event_id');episodefile=P/'road_closure_episodes.csv';columns=['case','bound','arm','road_index','source_edge_id','episode','onset_lower_s','onset_upper_s','recovery_lower_s','recovery_upper_s','left_censored','right_censored','observed_span_min','conditional_span_upper_min','three_ward_clipped_length_m'];ef=episodefile.open('w',newline='',encoding='utf-8-sig');writer=csv.DictWriter(ef,fieldnames=columns);writer.writeheader()
 try:
  for done,(_,run)in enumerate(inv.iterrows(),1):
   gate();assert time.monotonic()-started<14400,'A1activebudget';case=run.case_id;clock=load(ROOT/run.clock_path)['Actual Map Time'];assert len(clock)==109 and np.max(abs(clock-np.arange(109)*300))<=.001;ts=clock.copy();ts[0]=0.;ts[-1]=32400.;lower=[];upper=[]
   with Dataset(ROOT/run.map_path)as ds:
    bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel()
    for t in range(109):
     gate();eta=np.flipud(np.ma.asarray(ds['zs'][t],dtype='f8').filled(np.nan)).ravel();a,b=edge_maximum(pr['point_z'],pr['point_coarse'],pr['point_refs'],pr['offsets'],q,eta,bed);lower.append(a[qi]);upper.append(b[qi])
   lower=np.array(lower);upper=np.array(upper);assert np.isfinite(lower).all()and np.isfinite(upper).all()and(lower<=upper).all();np.savez_compressed(P/'exposure'/(case+'.npz'),qualified_road_indices=qi,lower=lower,upper=upper,actual_times=clock,analysis_times=ts);states=np.stack([np.where(lower>=.3,2,np.where(lower>=.1,1,0)),np.where(upper>=.3,2,np.where(upper>=.1,1,0))],axis=1).astype('u1');alllost=np.zeros((109,2,2,3,len(pop)),bool);anylost=alllost.copy();restoreall=np.zeros((109,2,3,len(pop)),bool);restoreany=restoreall.copy();ever=np.zeros((2,2,len(normal)),bool);keys=np.empty((109,2,2),dtype='U64');durations={}
   for bi,bound in enumerate(['lower','upper']):
    for ai,arm in enumerate(['flood','targeted']):
     code=states[:,bi].copy()
     if ai:code[:,effective_q]=0
     c=code==2;s=code==1
     for ti in range(109):
      for si,scope in enumerate(scopes):
       closed_m=float(weights[si,qi]@c[ti]);slow_m=float(weights[si,qi]@s[ti]);open_m=float(weights[si,qi]@(code[ti]==0));known_m=float(weights[si,qi].sum());assert abs(open_m+slow_m+closed_m-known_m)<1e-6
       roadrows.append(dict(case=case,event_id=run.event_id,stage=run.stage,frame=ti,actual_elapsed_s=float(clock[ti]),analysis_elapsed_s=float(ts[ti]),bound=bound,arm=arm,scope=scope,qualified_open_km=open_m/1000,qualified_slow_km=slow_m/1000,qualified_closed_km=closed_m/1000,qualified_total_km=known_m/1000,unknown_exposure_km=float(weights[si,~q].sum()/1000)))
     durations[bound+'__'+arm]=episodes(case,bound,arm,c,ts,qi,length,roadinv,writer)
   ef.flush()
   for ti in range(109):
    gate()
    for bi,bound in enumerate(['lower','upper']):
     oks=[]
     for ai,arm in enumerate(['flood','targeted']):
      closed=np.zeros(len(q),bool);closed[qi]=states[ti,bi]==2
      if ai:closed[protected]=False
      ok=engine.get(closed);keys[ti,bi,ai]=engine.key(closed);assert not(ok&~normal).any();lost=normal&~ok;ever[bi,ai]|=lost;oks.append(ok);codecount+=1
      for ui,(buf,info)in enumerate(infos.items()):
       lo,up=envelope(lost,info,pop);assert not(lo&~up).any();alllost[ti,bi,ai,ui]=lo;anylost[ti,bi,ai,ui]=up
       for sel,scope in zip(select,scopes):
        prows.append(dict(case=case,event_id=run.event_id,stage=run.stage,frame=ti,actual_elapsed_s=float(clock[ti]),analysis_elapsed_s=float(ts[ti]),bound=bound,arm=arm,buffer_m=buf,scope=scope,total_population=int(pop[sel].sum()),conditional_loss_lower=int(pop[sel&lo].sum()),conditional_loss_upper=int(pop[sel&up].sum()),baseline_unresolved_population=int(pop[sel&~info['complete']].sum())))
     assert not(oks[0]&~oks[1]).any(),'Protection removed access';recover=normal&~oks[0]&oks[1]
     for ui,(buf,info)in enumerate(infos.items()):
      a,b=envelope(recover,info,pop);restoreall[ti,bi,ui]=a;restoreany[ti,bi,ui]=b
      for sel,scope in zip(select,scopes):restrows.append(dict(case=case,frame=ti,elapsed_s=float(ts[ti]),bound=bound,buffer_m=buf,scope=scope,paired_restored_lower=int(pop[sel&a].sum()),paired_restored_upper=int(pop[sel&b].sum())))
    if ti%24==0:print('TEMPORAL',case,'frame',ti,'/108','unique_cached',len(list((P/'reachability_cache').glob('*.npz'))),flush=True)
   for bi,bound in enumerate(['lower','upper']):
    for ai,arm in enumerate(['flood','targeted']):
     for ui,(buf,info)in enumerate(infos.items()):
      a,b=envelope(ever[bi,ai],info,pop)
      for sel,scope in zip(select,scopes):
       ll=alllost[:,bi,ai,ui][:,sel]@pop[sel];uu=anylost[:,bi,ai,ui][:,sel]@pop[sel];il=int(np.argmax(ll));iu=int(np.argmax(uu));peakrows.append(dict(case=case,event_id=run.event_id,stage=run.stage,bound=bound,arm=arm,buffer_m=buf,scope=scope,peak_loss_lower=int(ll[il]),peak_loss_upper=int(uu[iu]),lower_peak_frame=il,upper_peak_frame=iu,lower_peak_elapsed_s=float(ts[il]),upper_peak_elapsed_s=float(ts[iu]),lower_peak_at_horizon=bool(il==108),upper_peak_at_horizon=bool(iu==108)));everrows.append(dict(case=case,bound=bound,arm=arm,buffer_m=buf,scope=scope,ever_stock_lower=int(pop[sel&a].sum()),ever_stock_upper=int(pop[sel&b].sum()),unresolved_population=int(pop[sel&~info['complete']].sum())))
   # Source snapshot replay uses original whole-source and population definitions.
   for bi,bound in enumerate(['lower','upper']):
    for ai,arm in enumerate(['flood','targeted']):
     for ui,buf in enumerate(infos):
      old=priorpop[(priorpop.case==case)&(priorpop.bound==bound)&(priorpop.arm==arm)&(priorpop.buffer_m==buf)]
      if len(old):
       row=old.iloc[0];assert int(alllost[48,bi,ai,ui]@pop)==row.conditional_loss_lower and int(anylost[48,bi,ai,ui]@pop)==row.conditional_loss_upper;reusechecks.append(dict(case=case,bound=bound,arm=arm,buffer_m=buf,original4h_population_reproduced=True))
   payload=dict(actual_times=clock,analysis_times=ts,road_indices=qi,road_state_codes=states,node_cache_keys=keys,block_ids=blocks.block_id.to_numpy(dtype=str),block_all_lost=alllost,block_any_lost=anylost,block_all_restored=restoreall,block_any_restored=restoreany)
   for prefix,values in durations.items():
    for k,v in values.items():payload[prefix+'__'+k]=v
   np.savez_compressed(P/'population'/(case+'.npz'),**payload)
   if run.stage=='H':
    clo=(states[:,0]==2)@length[qi];impact=int(np.argmax(clo));rainsec=int(timing.loc[run.event_id,'rolling60_peak_end_elapsed_s']);rain=int(np.argmin(abs(clock-rainsec)));assert abs(clock[rain]-rainsec)<=.001;by={}
    for reason,frame in [('rain_rolling60_peak_end',rain),('H_noaction_closed_length_peak',impact)]:by.setdefault(frame,[]).append(reason)
    for frame,reasons in by.items():ref.append(dict(reference_id=case+'__f'+str(frame).zfill(3),case_id=case,event_id=run.event_id,stage='H',frame=frame,actual_elapsed_s=float(clock[frame]),analysis_elapsed_s=float(ts[frame]),selection_reasons='|'.join(reasons),known_closed_clipped_km=float(clo[frame]/1000),maximum_known_closed_km=float(clo.max()/1000),peak_at_horizon=bool(frame==108),no_known_closure=bool(clo.max()==0),selected_using_intervention_benefit=False))
   save(P/'road_trajectories.csv',roadrows);save(P/'population_trajectories.csv',prows);save(P/'paired_population_restoration.csv',restrows);save(P/'population_peak_intervals.csv',peakrows);save(P/'population_ever_intervals.csv',everrows);save(P/'reference_states.csv',ref);dump(R/'A1_progress.json',dict(status='running',cases_completed=done,cases_total=10,current_case=case,frames_completed=done*109,unique_reachability_states=len(list((P/'reachability_cache').glob('*.npz'))),graph_calls=engine.calls,cache_hits=engine.hits,elapsed_s=time.monotonic()-started));print('CASE TEMPORAL QUALIFIED',case,done,'/10',flush=True)
 finally:ef.close()
 # Same-time paired stage effects from candidate differences,not marginal endpoint subtraction.
 stagepop=[];stageRoad=[]
 for event in sorted(inv.event_id.unique()):
  h=load(P/'population'/(event+'__H__U.npz'));l=load(P/'population'/(event+'__L__U.npz'))
  for ti in range(109):
   for bi,bound in enumerate(['lower','upper']):
    for ai,arm in enumerate(['flood','targeted']):
     delta=readkey(l['node_cache_keys'][ti,bi,ai],len(normal)).astype('i1')-readkey(h['node_cache_keys'][ti,bi,ai],len(normal)).astype('i1');delta[~normal]=0
     for buf,info in infos.items():
      lo,up=signed_envelope(delta,info,pop)
      for sel,scope in zip(select,scopes):stagepop.append(dict(event_id=event,frame=ti,elapsed_s=float(h['analysis_times'][ti]),bound=bound,arm=arm,buffer_m=buf,scope=scope,paired_stage_loss_lower=int(lo[sel].sum()),paired_stage_loss_upper=int(up[sel].sum())))
 save(P/'paired_stage_population_effects.csv',stagepop);road=pd.DataFrame(roadrows);ph=pd.DataFrame(prows);k=['event_id','frame','bound','arm','scope'];hr=road[road.stage=='H'];lr=road[road.stage=='L'];sr=hr.merge(lr,on=k,suffixes=('_H','_L'));sr['stage_closed_km_difference']=sr.qualified_closed_km_H-sr.qualified_closed_km_L;save(P/'paired_stage_road_effects.csv',sr);tot=ph[ph.scope=='Three wards'];assert tot.total_population.eq(496345).all();assert (ph.conditional_loss_lower<=ph.conditional_loss_upper).all();assert len(ref)<=10 and len({r['event_id']for r in ref})==5;assert len(road)==10*109*2*2*4;assert len(ph)==10*109*2*2*3*4;assert len(reusechecks)==96;save(R/'original_4h_replay.csv',reusechecks);sources();validation=dict(passed=True,hydraulic_cases=10,unique_rain_parents=5,actual_snapshots=1090,primary_graph_states=codecount,unique_cached_graphs=len(list((P/'reachability_cache').glob('*.npz'))),cache_hits=engine.hits,population_total=496345,reference_states=len(ref),maximum_measure_cells=len(ref)*16,original4h_replays=len(reusechecks),ward_length_accounting=True,paired_population_differences=True,no_time_slice_population_sum=True,seconds=time.monotonic()-started,new_hydraulics=0,training=False,sealed_tests=False);assert validation['seconds']<=14400;dump(R/'A1_validation.json',validation);dump(R/'A1_progress.json',dict(status='qualified_complete',cases_completed=10,cases_total=10,frames_completed=1090,reference_states=len(ref),elapsed_s=validation['seconds']));print('A1 ALL TEMPORAL CHECKS PASSED',json.dumps(validation),flush=True)
if __name__=='__main__':main()
