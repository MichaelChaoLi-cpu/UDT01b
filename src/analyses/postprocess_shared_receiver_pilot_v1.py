#!/usr/bin/env python3
"""Same-time common-support roads and off-target water-level changes;no new people counts."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from netCDF4 import Dataset
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'preprocessing')]
from shared_receiver_pilot_common_v1 import ROOT,P,R,dump
from prepare_historical_surface_coverage_v2 import edge_maximum

def load(p):return dict(np.load(p,allow_pickle=False))
def main():
 runs=json.loads((P/'run_manifest.json').read_text())['runs'];D=ROOT/'data/processed';clip=load(D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');pr=load(D/'three_ward_motor_flood_access_v1/road_projection.npz');support=load(D/'three_ward_five_event_population_validation_v1/motor_common_support.npz');common=support['common_admitted'].copy();qualified=clip['qualified'].copy()
 for run in runs:
  a=json.loads((R/'execution'/run['case_id']/'audit.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'];common &=load(P/'execution'/(run['case_id']+'_candidate_reference.npz'))['Admitted Mask']
 for i in np.flatnonzero(qualified):qualified[i]=common.ravel()[pr['point_coarse'][pr['point_refs'][pr['offsets'][i]:pr['offsets'][i+1]]]].all()
 assert not(qualified&~clip['qualified']).any();qi=np.flatnonzero(qualified);weights=np.vstack([clip['union_length_m'],clip['ward_length_m']]);scopes=['Three wards','Chiyoda','Chuo','Minato'];ward=load(D/'three_ward_surrogate_inputs/static_tensor_10m.npz')['Ward Code'].ravel()
 target=np.zeros(1012*770,bool);target[load(D/'three_ward_intake_receiver_candidates_v1/candidate_index.npz')['source_grid']]=True
 road=[];transfer=[];pumps=[]
 (P/'exposure').mkdir(exist_ok=True)
 for run in runs:
  case=run['case_id'];base=ROOT/run['original_folder'];new=ROOT/run['folder'];clock=load(P/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];analysis=clock.copy();analysis[0]=0.;analysis[-1]=32400.
  previous_clock=pd.read_csv(base/'udt_output_clock.csv');oldtime=previous_clock.loc[previous_clock.write_map.eq('T'),'actual_state_time'].to_numpy();assert np.max(abs(oldtime-clock))<=.001
  paths=[base/'sfincs_map.nc',new/'sfincs_map.nc'];states=[]
  for arm,path in zip(['baseline','drainage'],paths):
   lo=[];up=[]
   with Dataset(path) as ds:
    bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype=float).filled(np.nan)).ravel()
    for frame in range(109):
     eta=np.flipud(np.ma.asarray(ds['zs'][frame],dtype=float).filled(np.nan)).ravel();a,b=edge_maximum(pr['point_z'],pr['point_coarse'],pr['point_refs'],pr['offsets'],qualified,eta,bed);a=a[qi];b=b[qi];assert np.isfinite(a).all() and np.isfinite(b).all() and (a<=b).all();lo.append(a);up.append(b)
   depth=np.stack([np.array(lo),np.array(up)],axis=1);codes=np.where(depth>=.3,2,np.where(depth>=.1,1,0)).astype('u1');states.append(codes)
   np.savez_compressed(P/'exposure'/(case+'__'+arm+'.npz'),lower=depth[:,0],upper=depth[:,1],road_indices=qi,road_state_codes=codes,actual_times=clock,analysis_times=analysis)
   for frame in range(109):
    for bi,bound in enumerate(['lower','upper']):
     state=codes[frame,bi]
     for j,scope in enumerate(scopes):
      w=weights[j,qi]/1000.;closed=float(w[state==2].sum());slow=float(w[state==1].sum());opened=float(w[state==0].sum());assert abs(opened+slow+closed-w.sum())<1e-7
      prior=states[0][frame,bi];restored=float(w[(prior==2)&(state<2)].sum()) if arm=='drainage' else 0.;added=float(w[(prior<2)&(state==2)].sum()) if arm=='drainage' else 0.
      road.append({'Case':case,'Baseline Case':run['baseline_case'],'Stage':run['stage'],'Arm':arm,'Frame':frame,'Analysis Time':float(analysis[frame]),'Actual Time':float(clock[frame]),'Depth Branch':bound,'Scope':scope,'Closed Road Length':closed,'Slow Road Length':slow,'Open Road Length':opened,'Same Time Restored Road Length':restored,'Same Time Newly Closed Road Length':added,'Common Qualified Road Length':float(w.sum()),'Unknown Road Length':float(weights[j,~qualified].sum()/1000.)})
  with Dataset(paths[0]) as before,Dataset(paths[1]) as after:
   bed0=np.flipud(np.ma.asarray(before['zb'][:],dtype=float).filled(np.nan)).ravel();bed1=np.flipud(np.ma.asarray(after['zb'][:],dtype=float).filled(np.nan)).ravel()
   for frame in range(109):
    a=np.flipud(np.ma.asarray(before['zs'][frame],dtype=float).filled(np.nan)).ravel();b=np.flipud(np.ma.asarray(after['zs'][frame],dtype=float).filled(np.nan)).ravel()
    increase=np.where(np.isfinite(b),b,bed1)-np.where(np.isfinite(a),a,bed0+.01);decrease=np.where(np.isfinite(a),a,bed0)-np.where(np.isfinite(b),b,bed1+.01)
    for code,scope in [(0,'Whole supported domain'),(-1,'Three wards'),(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
     use=common.ravel().copy();use &= (ward>0) if code==-1 else ((ward==code) if code>0 else True);outside=use&~target
     transfer.append({'Case':case,'Stage':run['stage'],'Frame':frame,'Analysis Time':float(analysis[frame]),'Scope':scope,'Outside Target Cells Certain Head Rise Area':float(np.count_nonzero(outside&(increase>=.10))*.0001),'Outside Target Cells Certain Head Fall Area':float(np.count_nonzero(outside&(decrease>=.10))*.0001),'Interpretation':'10m cell area km2,conservative dry censor bound;not measured flood area'})
  pumps.append(json.loads((R/'execution'/case/'pump_budget_summary.json').read_text()));print('PILOT_ROAD_TRANSFER_CASE_COMPLETE',case,flush=True)
 roads=pd.DataFrame(road);roads.to_csv(P/'road_trajectories.csv',index=False);pd.DataFrame(transfer).to_csv(P/'off_target_head_changes.csv',index=False);summary=[]
 for key,g in roads.groupby(['Case','Stage','Arm','Depth Branch','Scope']):
  g=g.sort_values('Frame');summary.append(dict(zip(['Case','Stage','Arm','Depth Branch','Scope'],key),**{'Peak Closed Road Length':float(g['Closed Road Length'].max()),'Closed Road Length At End':float(g['Closed Road Length'].iloc[-1]),'Closed Road Length Hours':float(np.trapezoid(g['Closed Road Length'],g['Analysis Time'])/3600),'Maximum Same Time Newly Closed Road Length':float(g['Same Time Newly Closed Road Length'].max()),'Maximum Same Time Restored Road Length':float(g['Same Time Restored Road Length'].max())}))
 pd.DataFrame(summary).to_csv(P/'road_summary.csv',index=False);np.savez_compressed(P/'common_road_support.npz',common_admitted=common,qualified_roads=qualified)
 dump(R/'postprocessing_validation.json',{'passed':True,'paired_cases':2,'common_qualified_roads':int(qualified.sum()),'original_qualified_roads':int(clip['qualified'].sum()),'quality_only_shrinks':True,'same_time_restoration_and_new_closures_separate':True,'off_target_increase_and_decrease_separate':True,'new_population_counts':False,'traffic_and_people_not_yet_computed':True,'pump_actuals':pumps})
 print('SHARED_PILOT_ROAD_TRANSFER_COMPLETE',flush=True)
if __name__=='__main__':main()
