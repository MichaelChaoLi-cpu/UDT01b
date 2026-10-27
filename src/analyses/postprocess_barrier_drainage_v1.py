"""Matched physical flood states -> roads and uncertain resident vehicle access."""
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from temporal_transport_common_v1 import Reachability,population_inputs,counts,load,dump,sha
from run_temporal_road_population_v1 import envelope,signed_envelope
from prepare_historical_surface_coverage_v2 import edge_maximum
P=ROOT/'data/processed/three_ward_barrier_drainage_v1';R=ROOT/'data/exp/estimation'/P.name;D=ROOT/'data/processed';TEMP=D/'three_ward_five_event_temporal_analysis_v1'
def save(p,x):pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def gate():
 if(P/'STOP').exists():raise RuntimeError('User STOP')
 while(P/'PAUSE').exists():time.sleep(5)
def main():
 start=time.monotonic();runs=json.loads((P/'run_manifest.json').read_text())['runs'];clip=load(TEMP/'clipped_road_lengths.npz');q=clip['qualified'].copy();common=load(D/'three_ward_five_event_population_validation_v1/motor_common_support.npz')['common_admitted'].copy();pr=load(D/'three_ward_motor_flood_access_v1/road_projection.npz');prot=clip['protected'];weights=np.vstack([clip['union_length_m'],clip['ward_length_m']]);scopes=['Three wards','13101','13102','13103'];grid=load(P/'measures/spatial_contract.npz');corridor=np.zeros(1012*770,bool);corridor[grid['native_grid_north_index'][grid['corridor_native']]]=True
 for run in runs:
  if run['reused']:continue
  a=json.loads((R/'execution'/run['case_id']/'audit.json').read_text());assert a['all_checks_passed']and a['admission']['technical_reference'];common&=load(P/'execution'/(run['case_id']+'_candidate_reference.npz'))['Admitted Mask']
 for j in np.flatnonzero(q):q[j]=common.ravel()[pr['point_coarse'][pr['point_refs'][pr['offsets'][j]:pr['offsets'][j+1]]]].all()
 assert q.any()and not(q&~clip['qualified']).any();qi=np.flatnonzero(q);np.savez_compressed(P/'motor_common_support.npz',common_admitted=common,qualified_roads=q,protected_source_roads=prot);blocks,pop,normal,infos=population_inputs();ward=blocks.ward.to_numpy(dtype=str);sels=[np.ones(len(pop),bool)]+[ward==s for s in scopes[1:]];engine=Reachability();cache={};roadrows=[];prows=[];paired=[];transfer=[];reference=[]
 for f in ['exposure','population','node_cache']:(P/f).mkdir(exist_ok=True)
 def okmask(closed):
  key=engine.key(closed)
  if key not in cache:
   f=P/'node_cache'/(key+'.npz')
   if f.exists():v=load(f);ok=np.unpackbits(v['ok'],count=len(normal)).astype(bool)
   else:ok=engine.direct(closed);np.savez_compressed(f,ok=np.packbits(ok),closed=np.packbits(closed))
   cache[key]=ok
   if len(cache)>160:cache.pop(next(iter(cache)))
  return cache[key],key
 for number,run in enumerate(runs,1):
  gate();assert time.monotonic()-start<14400;case=run['case_id'];base=run['baseline_case'];folder=ROOT/run['folder'];cp=(folder.parents[1]/'execution'/(base+'_output_clock.npz'))if run['reused']else P/'execution'/(case+'_output_clock.npz');clock=load(cp)['Actual Map Time'];assert len(clock)==109 and np.max(abs(clock-np.arange(109)*300))<=.001;ts=clock.copy();ts[0]=0;ts[-1]=32400;lo=[];up=[];keys=np.empty((109,2),dtype='U64');lowerflags=np.zeros((109,2,3,len(pop)),bool);upperflags=lowerflags.copy();native_depth=[]
  with Dataset(folder/'sfincs_map.nc')as ds:
   bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for i in range(109):
    gate();eta=np.flipud(np.ma.asarray(ds['zs'][i],dtype='f8').filled(np.nan)).ravel();a,b=edge_maximum(pr['point_z'],pr['point_coarse'],pr['point_refs'],pr['offsets'],q,eta,bed);lo.append(a[qi]);up.append(b[qi]);native_depth.append(np.where(np.isfinite(eta),np.maximum(0,eta-bed),np.nan).astype('f4'))
  lo=np.array(lo);up=np.array(up);assert np.isfinite(lo).all()and np.isfinite(up).all()and(lo<=up).all();states=np.stack([np.where(lo>=.3,2,np.where(lo>=.1,1,0)),np.where(up>=.3,2,np.where(up>=.1,1,0))],axis=1).astype('u1');np.savez_compressed(P/'exposure'/(case+'.npz'),lower=lo,upper=up,road_indices=qi,road_state_codes=states,actual_times=clock,analysis_times=ts,native_depth=np.array(native_depth));basekeys=None
  if run['measure']!='none':basekeys=load(P/'population'/(base+'__none.npz'))['node_cache_keys']
  for i in range(109):
   for bi,bound in enumerate(['lower','upper']):
    state=states[i,bi];closed=np.zeros(len(q),bool);closed[qi]=state==2;ok,key=okmask(closed);keys[i,bi]=key;assert not(ok&~normal).any();lost=normal&~ok
    for si,scope in enumerate(scopes):
     for stratum,choose in [('all',np.ones(len(qi),bool)),('protected_corridor',prot[qi]),('other_roads',~prot[qi])]:
      w=weights[si,qi]*choose;openkm=float(w@(state==0)/1000);slowkm=float(w@(state==1)/1000);closedkm=float(w@(state==2)/1000);assert abs(openkm+slowkm+closedkm-w.sum()/1000)<1e-7;roadrows.append(dict(case=case,event_id=run['event_id'],stage=run['stage'],measure=run['measure'],frame=i,elapsed_s=float(ts[i]),bound=bound,scope=scope,stratum=stratum,open_km=openkm,slow_km=slowkm,closed_km=closedkm,qualified_km=float(w.sum()/1000),unknown_km=float(weights[si,~q].sum()/1000)if stratum=='all'else None))
    for ui,(buf,info)in enumerate(infos.items()):
     a,b=envelope(lost,info,pop);lowerflags[i,bi,ui]=a;upperflags[i,bi,ui]=b
     for sel,scope in zip(sels,scopes):prows.append(dict(case=case,event_id=run['event_id'],stage=run['stage'],measure=run['measure'],frame=i,elapsed_s=float(ts[i]),bound=bound,buffer_m=buf,scope=scope,total_stock=int(pop[sel].sum()),loss_lower=int(pop[sel&a].sum()),loss_upper=int(pop[sel&b].sum()),unresolved_stock=int(pop[sel&~info['complete']].sum())))
    if basekeys is not None:
     key0=str(basekeys[i,bi]);v=load(P/'node_cache'/(key0+'.npz'));baselineok=np.unpackbits(v['ok'],count=len(normal)).astype(bool);delta=baselineok.astype('i1')-ok.astype('i1');delta[~normal]=0
     for buf,info in infos.items():
      a,b=signed_envelope(delta,info,pop);ra,rb=envelope(normal&~baselineok&ok,info,pop);wa,wb=envelope(normal&baselineok&~ok,info,pop)
      for sel,scope in zip(sels,scopes):paired.append(dict(case=case,event_id=run['event_id'],stage=run['stage'],measure=run['measure'],frame=i,elapsed_s=float(ts[i]),bound=bound,buffer_m=buf,scope=scope,net_added_loss_lower=int(a[sel].sum()),net_added_loss_upper=int(b[sel].sum()),restored_lower=int(pop[sel&ra].sum()),restored_upper=int(pop[sel&rb].sum()),new_loss_lower=int(pop[sel&wa].sum()),new_loss_upper=int(pop[sel&wb].sum())))
  np.savez_compressed(P/'population'/(case+'.npz'),actual_times=clock,node_cache_keys=keys,block_all_lost=lowerflags,block_any_lost=upperflags,sampled_closed_min=np.trapezoid((states==2).astype(float),ts,axis=0)/60.)
  if run['measure']!='none':
   z=load(P/'exposure'/(base+'__none.npz'));baseline_depth=z['native_depth'];changed=np.array(native_depth);valid=common.ravel();outside=valid&~corridor
   for i in range(109):
    # Shallow NaNs mean censored0..0.01m only under passed parent censor gate.
    d=np.where(np.isfinite(changed[i]),changed[i],0)-np.where(np.isfinite(baseline_depth[i]),baseline_depth[i],.01);e=np.where(np.isfinite(baseline_depth[i]),baseline_depth[i],0)-np.where(np.isfinite(changed[i]),changed[i],.01)
    transfer.append(dict(case=case,event_id=run['event_id'],stage=run['stage'],measure=run['measure'],frame=i,elapsed_s=float(ts[i]),outside_corridor_cell_area_head_increase_ge0_1m_km2=float(np.count_nonzero(outside&(d>=.1))*.0001),outside_corridor_cell_area_head_decrease_ge0_1m_km2=float(np.count_nonzero(outside&(e>=.1))*.0001),native_cell_area_not_measured_flood_area=True))
  save(P/'road_trajectories.csv',roadrows);save(P/'population_trajectories.csv',prows);save(P/'paired_population_effects.csv',paired);save(P/'neighbor_transfer_screen.csv',transfer);dump(R/'postprocessing_progress.json',dict(status='running',cases_completed=number,cases_total=16,qualified_roads=int(q.sum()),original_qualified_roads=int(clip['qualified'].sum()),elapsed_s=time.monotonic()-start));print('PHYSICAL ACCESS QUALIFIED',case,number,'/16',flush=True)
 road=pd.DataFrame(roadrows);summary=[]
 for key,g in road.groupby(['case','event_id','stage','measure','bound','scope','stratum']):
  g=g.sort_values('frame');i=g.closed_km.idxmax();summary.append(dict(zip(['case','event_id','stage','measure','bound','scope','stratum'],key),peak_closed_km=float(g.loc[i,'closed_km']),peak_frame=int(g.loc[i,'frame']),closed_at9h_km=float(g.closed_km.iloc[-1]),closed_km_hours=float(np.trapezoid(g.closed_km,g.elapsed_s)/3600)))
 save(P/'road_summary.csv',summary);pf=pd.DataFrame(prows);ps=pf.groupby(['case','event_id','stage','measure','bound','buffer_m','scope'],as_index=False).agg(peak_loss_lower=('loss_lower','max'),peak_loss_upper=('loss_upper','max'),unresolved_stock=('unresolved_stock','first'));save(P/'population_peak_intervals.csv',ps);assert pf[pf.scope=='Three wards'].total_stock.eq(496345).all();assert(pf.loss_lower<=pf.loss_upper).all();gate();dump(R/'postprocessing_validation.json',dict(passed=True,cases=16,hydraulic_new=12,rain_parents=2,frames=1744,original_qualified_roads=int(clip['qualified'].sum()),common_qualified_roads=int(q.sum()),quality_only_shrinks=True,population_total=496345,paired_candidate_effects=True,neighbor_worsening_retained=True,no_ideal_dry_overlay=True,seconds=time.monotonic()-start,training=False,sealed_test=False));print('PHYSICAL ROAD/POPULATION CHECKS PASSED',flush=True)
if __name__=='__main__':main()
