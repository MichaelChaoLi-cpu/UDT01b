"""After six admitted hydraulics, export common-supported roads and census consequences.
Retains original resident stock, candidate uncertainty and unknown-road semantics.
"""
from pathlib import Path
import os,sys,json,time,argparse
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','2')
import numpy as np,pandas as pd
from scipy.sparse.csgraph import dijkstra
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from prepare_five_event_population_validation_v1 import P,R,PRIOR,load,dump,sha
from integrate_night_population_transport_v1 import distances,selfcheck as routing_selfcheck
from observed_load_common_v1 import network
from prepare_historical_surface_coverage_v2 import edge_maximum
D=ROOT/'data/processed';BASE=D/'three_ward_night_population_transport_v1';REP=D/'three_ward_normal_vehicle_repair_v1';M=D/'three_ward_motor_flood_access_v1';C=D/'three_ward_normal_vehicle_choice_v1';MEASURE=D/'three_ward_demand_protection_feedback_v1';POP=D/'three_ward_population_block_stability_v1'

def gate():
 if(P/'STOP').exists():raise RuntimeError('User STOP during population')
 while(P/'PAUSE').exists():
  if(P/'STOP').exists():raise RuntimeError('User STOP while population paused')
  time.sleep(10)
def counts(values,ptr):
 c=np.r_[0,np.cumsum(values,dtype='i8')];return c[ptr[1:]]-c[ptr[:-1]]
def population_rows(blocks,support,normal,ok,ratio,case,bound,arm,knownnode):
 rows=[];pop=blocks.residents_2020.to_numpy()
 for buf in [0,10,25]:
  ptr=support[f'offsets_{buf}m'];ix=support[f'node_indices_{buf}m'];lens=np.diff(ptr);bn=counts(normal[ix],ptr);cn=counts(ok[ix],ptr);lost=counts(normal[ix]&~ok[ix],ptr);detour=counts(normal[ix]&ok[ix]&(ratio[ix]>=1.5),ptr);complete=(lens>0)&(bn==lens);a=blocks.copy();a['case']=case;a['bound']=bound;a['arm']=arm;a['buffer_m']=buf;a['candidate_nodes']=lens;a['normal_bidirectional_candidates']=bn;a['current_bidirectional_candidates']=cn;a['lost_candidates']=lost;a['detour_candidates']=detour;a['candidates_with_local_qualified_road']=counts(knownnode[ix],ptr);a['conditional_loss_population_lower']=np.where(complete&(lost==lens),pop,0.);a['conditional_loss_population_upper']=np.where(complete&(lost>0),pop,0.);a['baseline_unresolved_population']=np.where(~complete,pop,0.);a['access_class']=np.select([lens==0,bn==0,bn<lens,cn==0,cn<lens,detour==lens,detour>0],['no_candidate_support','normal_access_missing','normal_support_mixed','all_candidates_lost','partial_candidate_loss','connected_detour_all','connected_detour_mixed'],default='connected_below_detour_threshold');rows.append(a)
 return pd.concat(rows,ignore_index=True)
def readblocks():
 census=pd.read_csv(POP/'official_2020_block_count_baseline.csv',dtype={'block_id':str,'ward':str,'town_id':str});census=census.rename(columns={'official_2020_residents':'residents_2020'});support=load(BASE/'block_access_support.npz');census=census.set_index('block_id').loc[support['block_ids']].reset_index();assert census.residents_2020.sum()==496345;return census,support

def selftest():
 assert routing_selfcheck();blocks,support=readblocks();normal=load(BASE/'normal_node_access.npz')['bidirectional'];case='E20250911__H__U';z=load(BASE/(case+'__lower__flood_node_access.npz'));rows=population_rows(blocks,support,normal,z['bidirectional'],z['reference_detour_ratio'],case,'lower','flood',np.ones(len(normal),bool));old=pd.read_csv(BASE/'block_population_access.csv',dtype={'block_id':str});old=old[(old.case==case)&(old.bound=='lower')&(old.arm=='flood')].set_index(['buffer_m','block_id']);new=rows.set_index(['buffer_m','block_id']).loc[old.index]
 for k in ['candidate_nodes','normal_bidirectional_candidates','current_bidirectional_candidates','lost_candidates','detour_candidates','conditional_loss_population_lower','conditional_loss_population_upper','baseline_unresolved_population','access_class']:assert np.array_equal(old[k].to_numpy(),new[k].to_numpy()),k
 (R/'population').mkdir(exist_ok=True);dump(R/'population/interface_selfcheck.json',dict(passed=True,fixture='Exact replay of original2025H lower-depth flood population rows;all4746units and3searchradii',rows=len(new),synthetic_router=True));print('POPULATION INTERFACE SELFCHECK PASSED',flush=True)

def main():
 started=time.monotonic();gate();selftest();runs=json.loads((P/'run_manifest.json').read_text())['runs'];assert all(json.loads((R/'execution'/r['case_id']/'audit.json').read_text())['all_checks_passed']for r in runs);sources=json.loads((P/'protected_sources.json').read_text());assert all(sha(ROOT/n)==h for n,h in sources.items());out=P/'population';rep=R/'population';out.mkdir(exist_ok=True);rep.mkdir(exist_ok=True);(out/'node_access').mkdir(exist_ok=True);exdir=P/'motor_exposures';exdir.mkdir(exist_ok=True);pr=load(M/'road_projection.npz');oldsupport=load(PRIOR/'motor_common_support.npz');common=oldsupport['common_admitted'].copy()
 for r in runs:common &=load(P/'execution'/(r['case_id']+'_candidate_reference.npz'))['Admitted Mask']
 qualified=oldsupport['qualified_roads'].copy();coarse=pr['point_coarse'];refs=pr['point_refs'];offset=pr['offsets']
 for j in np.flatnonzero(qualified):qualified[j]=common.ravel()[coarse[refs[offset[j]:offset[j+1]]]].all()
 assert common.any()and qualified.any(),'No common admitted ground/road support';original=load(MEASURE/'protection_masks.npz')['targeted'];effective=original&qualified;np.savez_compressed(P/'motor_common_support.npz',common_admitted=common,qualified_roads=qualified,targeted_original=original,targeted_effective=effective);assert not(qualified&~oldsupport['qualified_roads']).any();times=json.loads((P/'protocol.json').read_text())['grid']['times_s']
 for run in runs:
  gate();case=run['case_id'];clock=load(P/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];idx=[int(np.argmin(abs(clock-t)))for t in times];assert np.max(abs(clock[idx]-times))<=.001;lo=[];up=[]
  with Dataset(ROOT/run['folder']/'sfincs_map.nc')as ds:
   bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for i in idx+[None]:
    eta=np.flipud(np.ma.asarray(ds['zs'][i]if i is not None else ds['zsmax'][0],dtype='f8').filled(np.nan)).ravel();a,b=edge_maximum(pr['point_z'],coarse,refs,offset,qualified,eta,bed);lo.append(a);up.append(b)
  lo=np.array(lo);up=np.array(up);assert np.isfinite(lo[:,qualified]).all()and np.isfinite(up[:,qualified]).all()and(lo[:,qualified]<=up[:,qualified]).all();np.savez_compressed(exdir/(case+'.npz'),lower=lo,upper=up,actual_times=clock[idx]);print('EXPOSURE QUALIFIED',case,flush=True)
 cases=['E20250911__H__U','E20240830__H__U']+[r['case_id']for r in runs]
 for case in cases[:2]:
  dest=exdir/(case+'.npz')
  if not dest.exists():dest.hardlink_to(PRIOR/'motor_exposures'/(case+'.npz'))
 blocks,support=readblocks();rr=load(REP/'turn_router.npz');epcsv=pd.read_csv(REP/'multi_network_endpoints.csv');gateix=epcsv.loc[epcsv.kind=='conditional_outer_boundary_portal','endpoint_index'].to_numpy();gates=epcsv.loc[epcsv.kind=='conditional_outer_boundary_portal','component'].to_numpy();normal=load(BASE/'normal_node_access.npz');normalok=normal['bidirectional'];tref=load(C/'choice_costs.npz')['travel_seconds'];ep=load(REP/'multi_endpoint_components.npz');S=len(rr['state_physical_arc']);K=len(ep['components']);node=ep['native_node_index'];nonnormalgate=~np.isin(np.arange(K),gateix);knownnode=np.zeros(len(normalok),bool);q=qualified[pr['arc_to_road']];knownnode[pr['from_index'][q]]=True;knownnode[pr['to_index'][q]]=True;rows=[];restored=[];maxerr=0.;checks={};roadrows=[]
 inv=pd.read_csv(M/'road_inventory.csv');core=inv.inside_core.to_numpy();baseprotectionlength=float(inv.loc[original,'length_m'].sum())
 for case in cases:
  ex=load(exdir/(case+'.npz'));assert abs(ex['actual_times'][5]-14400)<=.001
  for bound in ['lower','upper']:
   depth=ex[bound][5];baseclosed=qualified&(depth>=.3);baseslow=qualified&(depth>=.1)&~baseclosed;results={}
   for arm in ['flood','targeted']:
    gate();closed=baseclosed.copy();slow=baseslow.copy()
    if arm=='targeted':closed[effective]=False;slow[effective]=False
    cost=tref.copy();cost[slow[pr['arc_to_road']]]*=2;oo,ii=distances(rr,pr,cost,closed,gates);ok=np.isfinite(oo)&np.isfinite(ii);assert not(ok&~normalok).any();oratio=np.divide(oo,normal['outward_reference_seconds'],out=np.full_like(oo,np.inf),where=np.isfinite(normal['outward_reference_seconds'])&(normal['outward_reference_seconds']>0));iratio=np.divide(ii,normal['inward_reference_seconds'],out=np.full_like(ii,np.inf),where=np.isfinite(normal['inward_reference_seconds'])&(normal['inward_reference_seconds']>0));ratio=np.maximum(oratio,iratio);assert not np.isnan(ratio).any();results[arm]=ok;np.savez_compressed(out/'node_access'/(case+'__'+bound+'__'+arm+'.npz'),bidirectional=ok,reference_detour_ratio=ratio,outward_reference_seconds=oo,inward_reference_seconds=ii);rows.append(population_rows(blocks,support,normalok,ok,ratio,case,bound,arm,knownnode))
    # Independently use the unchanged production endpoint network for every case.
    if bound=='lower'and arm=='flood':
     g=network(rr,ep,pr,cost,closed);inc=dijkstra(g,indices=S+gateix,directed=True,min_only=True)[S+K:S+2*K];outc=dijkstra(g.T.tocsr(),indices=S+K+gateix,directed=True,min_only=True)[S:S+K]
     for expected,actual in [(inc,ii[node]),(outc,oo[node])]:
      assert np.array_equal(np.isfinite(expected[nonnormalgate]),np.isfinite(actual[nonnormalgate]));sel=nonnormalgate&np.isfinite(expected);maxerr=max(maxerr,float(abs(expected[sel]-actual[sel]).max(initial=0)))
     assert maxerr<1e-6
    roadrows.append(dict(case=case,bound=bound,arm=arm,known_closed_source_roads=int(closed.sum()),known_closed_core_intersect_km=float(inv.loc[closed&core,'length_m'].sum()/1000),known_slow_source_roads=int(slow.sum()),effective_protected_roads=int(effective.sum()),effective_protected_length_m=float(inv.loc[effective,'length_m'].sum())))
   assert not(results['flood']&~results['targeted']).any();rec=normalok&~results['flood']&results['targeted']
   for buf in [0,10,25]:
    ptr=support[f'offsets_{buf}m'];ix=support[f'node_indices_{buf}m'];lens=np.diff(ptr);complete=(lens>0)&(counts(normalok[ix],ptr)==lens);nrec=counts(rec[ix],ptr);d=blocks[['block_id','ward','residents_2020']].copy();d['case']=case;d['bound']=bound;d['buffer_m']=buf;d['conditional_restored_lower']=np.where(complete&(nrec==lens),d.residents_2020,0.);d['conditional_restored_upper']=np.where(complete&(nrec>0),d.residents_2020,0.);restored.append(d)
  print('POPULATION QUALIFIED',case,flush=True)
 tab=pd.concat(rows,ignore_index=True);rest=pd.concat(restored,ignore_index=True);tab.to_csv(out/'block_population_access.csv',index=False,encoding='utf-8-sig');rest.to_csv(out/'block_population_restoration.csv',index=False,encoding='utf-8-sig');key=['case','bound','arm','buffer_m'];summary=tab.groupby(key,as_index=False).agg(total_population=('residents_2020','sum'),conditional_loss_lower=('conditional_loss_population_lower','sum'),conditional_loss_upper=('conditional_loss_population_upper','sum'),baseline_unresolved_population=('baseline_unresolved_population','sum'));summary.to_csv(out/'population_effect_summary.csv',index=False,encoding='utf-8-sig');rest.groupby(['case','bound','buffer_m'],as_index=False).agg(conditional_restored_lower=('conditional_restored_lower','sum'),conditional_restored_upper=('conditional_restored_upper','sum')).to_csv(out/'population_restoration_summary.csv',index=False,encoding='utf-8-sig');tab.groupby(key+['ward'],as_index=False).agg(population=('residents_2020','sum'),loss_lower=('conditional_loss_population_lower','sum'),loss_upper=('conditional_loss_population_upper','sum'),unresolved=('baseline_unresolved_population','sum')).to_csv(out/'ward_population_effect_summary.csv',index=False,encoding='utf-8-sig');pd.DataFrame(roadrows).to_csv(out/'road_state_summary.csv',index=False,encoding='utf-8-sig')
 # River-stage effect at the same event-relative time and fixed geometry/protection.
 matched=summary[summary.case.str.startswith(tuple(r['event_id']for r in runs))].copy();matched['event']=matched.case.str.split('__').str[0];matched['stage']=matched.case.str.split('__').str[1];h=matched[matched.stage=='H'];l=matched[matched.stage=='L'];stage=h.merge(l,on=['event','bound','arm','buffer_m'],suffixes=('_H','_L'));stage['stage_loss_lower_difference']=stage.conditional_loss_lower_H-stage.conditional_loss_lower_L;stage['stage_loss_upper_difference']=stage.conditional_loss_upper_H-stage.conditional_loss_upper_L;stage.to_csv(out/'stage_population_effects.csv',index=False,encoding='utf-8-sig')
 checks=dict(all6original_audits=True,population_total_all_states=bool(summary.total_population.eq(496345).all()),unique_block_state=not tab.duplicated(key+['block_id']).any(),all_bounds_ordered=bool((tab.conditional_loss_population_lower<=tab.conditional_loss_population_upper).all()),unresolved_held=bool(tab.loc[tab.baseline_unresolved_population>0,'conditional_loss_population_upper'].eq(0).all()),common_support_only_shrinks=not bool((qualified&~oldsupport['qualified_roads']).any()),unknown_never_protected=not bool((effective&~qualified).any()),production_router_matches=maxerr<1e-6,no_vehicle_to_person_multiplier=True,original_sources_unchanged=all(sha(ROOT/n)==h for n,h in sources.items()));assert all(checks.values()),checks;val=dict(passed=True,checks=checks,cases=cases,unique_rainfall_parents=5,states=len(summary),population_total=496345,original_qualified_roads=int(oldsupport['qualified_roads'].sum()),common_qualified_roads=int(qualified.sum()),original_targeted_roads=int(original.sum()),effective_targeted_roads=int(effective.sum()),original_targeted_length_m=baseprotectionlength,max_independent_router_error_s=maxerr,seconds=time.monotonic()-started);dump(rep/'validation.json',val);dump(rep/'summary.json',dict(validation=val,main_summary=summary[(summary.buffer_m==10)&(summary.bound=='lower')].to_dict('records'),interpretation='Conditional resident-stock candidate-location intervals,not actual stranded persons. Unknown roads assumed available;night/day actual occupancy not inferred;0/10/25m sensitivity unchanged.'))
 print('ALL POPULATION GATES PASSED',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--selfcheck',action='store_true');args=ap.parse_args();selftest()if args.selfcheck else main()
