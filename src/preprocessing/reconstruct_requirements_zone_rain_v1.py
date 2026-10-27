#!/usr/bin/env python3
"""Reconstruct fixed-control-volume rain from unchanged encoded roof-adjusted fields.
Existing-result reader only. Exact START block overlaps and float32 rain arithmetic;
no hydraulic integration or inference of missing ground-face discharge.
"""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_zone_rain_flux_readiness_v1';OLD='three_ward_protection_requirements_register_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG;OP=ROOT/'data/processed'/OLD
CTX=ROOT/'data/processed/three_ward_historical_radar_experiment_v2/context'

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 sources={}
 def bind(p):
  k=str(p.relative_to(ROOT));st=p.stat();sources[k]={'sha256':sha(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 invpath=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv';inv=pd.read_csv(invpath);runs={}
 for folder in ['three_ward_five_event_population_validation_v1','three_ward_cross_event_protection_v1']:
  f=ROOT/'data/processed'/folder/'run_manifest.json';bind(f)
  for row in json.loads(f.read_text())['runs']:runs[row['case_id']]=row
 with np.load(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz') as z:ng=z['native_grid_north_index'];kcs=z['native_kcs'];xy=z['native_xy']
 with np.load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz') as z:ward=z['Ward Code'].ravel()[ng]
 rr=1011-ng//770;cc=ng%770;ground=kcs==1;checks=[];alltr=[];allcs=[];intervals=[];snapshots=[]
 for p in [invpath,CTX/'sfincs_meteo.f90',CTX/'udt_clock_budget.f90',CTX/'sfincs_continuity.f90',ROOT/'data/processed/three_ward_historical_radar_experiment_v2/solver_binding.json']:bind(p)
 for number,row in inv.iterrows():
  if (P/'STOP').exists():raise RuntimeError('User stop')
  assert time.monotonic()-start<1800,'Read-only reconstruction budget'
  case=row.case_id;run=runs[case];folder=(ROOT/row.map_path).parent;rainfolder=ROOT/run['rain_source_folder'];assert run['arm']=='U'
  for path in [folder/'historical_rain.nc',folder/'udt_step_budget.csv',folder/'udt_rain_integral.csv',folder/'udt_output_clock.csv',folder/'udt_run_identity.txt',OP/'zones'/f'{case}.npz',rainfolder/'forcing_intervals_10m.nc',rainfolder/'spatial_forcing_qa.json']:bind(path)
  with np.load(OP/'zones'/f'{case}.npz') as z:
   gridzone=np.zeros(1012*770,dtype='int32');gridzone[z['node_grid_index']]=z['zone_index'];zids=z['zone_ids'];actual=z['actual_times']
  zone=gridzone[ng];nz=len(zids);group=zone*4+ward;nb=(nz+1)*4
  ledger=pd.read_csv(folder/'udt_step_budget.csv',float_precision='round_trip');clock=pd.read_csv(folder/'udt_output_clock.csv',float_precision='round_trip');integ=pd.read_csv(folder/'udt_rain_integral.csv',float_precision='round_trip')
  t=ledger.actual_state_time.to_numpy()[1:];dt=ledger.actual_step_dt.to_numpy()[1:];ns=len(t);maps=clock[clock.write_map=='T'];ms=maps.step_index.to_numpy(dtype=int)
  assert np.array_equal(ms,ledger.step_index.to_numpy()[ms]) and np.array_equal(actual,maps.actual_state_time.to_numpy())
  assert np.array_equal(integ.actual_end_time_s.to_numpy(),t) and np.array_equal(integ.dt_s.to_numpy(),dt)
  with Dataset(folder/'historical_rain.nc') as d,Dataset(rainfolder/'forcing_intervals_10m.nc') as f:
   d.set_auto_mask(False);knots=np.rint(np.asarray(d['time'][:])*60);x=np.asarray(d['x'][:]);y=np.asarray(d['y'][:]);assert np.array_equal(knots,np.arange(109)*300)
   assert np.array_equal(x[cc+1]+5,xy[:,0]) and np.array_equal(y[rr+1]+5,xy[:,1]),'Noncenter forcing requires full spatial interpolation replay'
   rates=np.zeros((109,len(ng)),dtype='float32');recipient_match=True
   for j in range(109):
    raw=np.asarray(d['Precipitation'][j],dtype='float32');assert np.isfinite(raw).all() and raw.min()>=0
    if j<108:
     expected=np.flipud(np.ma.asarray(f['urban_effective_rainfall'][j]).filled(np.nan).astype('float64')*run['rainfall_scale']).astype('float32');recipient_match &= bool(np.array_equal(raw[1:-1,1:-1],expected))
    else:assert (raw==0).all()
    # At every native center the source bilinear weights are exactly [1,0,0,0].
    rates[j]=raw[rr+1,cc+1]/np.float32(3600000)
   assert recipient_match
  # START-stamped holds. Actual finish-dt retains the solver's overlap rounding.
  left=t-dt;ki=np.clip(np.searchsorted(knots,left,side='right')-1,0,108);kj=np.clip(np.searchsorted(knots,t,side='left')-1,0,108)
  end=np.where(ki<108,knots[np.minimum(ki+1,108)],np.inf)
  wl=np.maximum(0,np.minimum(t,end)-np.maximum(left,knots[ki]));wr=np.where(ki!=kj,np.maximum(0,t-np.maximum(left,knots[kj])),0.)
  assert np.max(abs(wl+wr-dt))<1e-9
  patterns,index=np.unique(np.column_stack([ki,kj,wl,wr,dt]),axis=0,return_inverse=True)
  apply=np.zeros((len(patterns),nb));quadrature=np.zeros(len(patterns));globalrates=rates[:,ground].sum(axis=1,dtype='float64')*100.
  for i,(ka,kb,wa,wb,step) in enumerate(patterns):
   ra=rates[int(ka)].astype('float64');rb=rates[int(kb)].astype('float64')
   mean=((ra*wa+rb*wb)/step).astype('float32')
   dv=((mean*np.float32(100))*np.float32(step)).astype('float64')
   apply[i]=np.bincount(group[ground],weights=dv[ground],minlength=nb)
   quadrature[i]=wa*globalrates[int(ka)]+wb*globalrates[int(kb)]
  cum=np.vstack([np.zeros(nb),np.cumsum(apply[index],axis=0)]);globalcum=cum.sum(axis=1)
  ledgerdiff=globalcum-ledger.rain_applied_cumulative.to_numpy();scale=max(1.,float(ledger.rain_applied_cumulative.iloc[-1]));applied_error=float(np.max(abs(ledgerdiff)));assert applied_error<=scale*1e-6
  qdiff=np.cumsum(quadrature[index])-np.cumsum(integ.integrated_native_rate_volume_m3.to_numpy());quad_error=float(np.max(abs(qdiff)));assert quad_error<=scale*1e-6
  c=cum[ms].reshape(len(ms),nz+1,4);baseline=c[0].copy();since=c-baseline;assert np.min(since)>=0
  # All cells not assigned to a selected zone are a separate rain partition,not ignored.
  for frame in range(109):
   state=since[frame]
   for zi,zid in enumerate(zids,1):
    per=state[zi];alltr.append({'Case':case,'Zone':str(zid),'Frame':frame,'Actual Time':float(actual[frame]),'Actual Step':int(ms[frame]),'Reconstructed Rain Since First Map':float(per.sum()),'Chiyoda Rain Since First Map':float(per[1]),'Chuo Rain Since First Map':float(per[2]),'Minato Rain Since First Map':float(per[3]),'Buffer Rain Since First Map':float(per[0]),'Independent Local Applied Rain Saved':False})
   snapshots.append({'Case':case,'Frame':frame,'Actual Time':float(actual[frame]),'Selected Zone Rain Since First Map':float(state[1:].sum()),'Unselected Ground Rain Since First Map':float(state[0].sum()),'Whole Model Rain Since First Map':float(state.sum()),'Three Ward Selected Zone Rain Since First Map':float(state[1:,1:].sum()),'Buffer Selected Zone Rain Since First Map':float(state[1:,0].sum())})
  # Interval physical input totals are separate from step-applied rounding.
  for j in range(108):
   v=rates[j].astype('float64')*100.*300.;agg=np.bincount(group[ground],weights=v[ground],minlength=nb).reshape(nz+1,4)
   for zi,zid in enumerate(zids,1):intervals.append({'Case':case,'Zone':str(zid),'Rain Interval':j,'Interval Start':j*300,'Interval End':(j+1)*300,'Encoded Rain Volume':float(agg[zi].sum()),'Three Ward Encoded Rain Volume':float(agg[zi,1:].sum()),'Buffer Encoded Rain Volume':float(agg[zi,0])})
  for zi,zid in enumerate(zids,1):allcs.append({'Case':case,'Zone':str(zid),'Reconstructed Rain Since First Map':float(since[-1,zi].sum()),'Three Ward Rain Since First Map':float(since[-1,zi,1:].sum()),'Buffer Rain Since First Map':float(since[-1,zi,0]),'Rain Meaning':'roof-adjusted applied input reconstructed on fixed cells;not complete contributing catchment rain','Local Applied Rain Independently Saved':False})
  nominal=json.loads((rainfolder/'spatial_forcing_qa.json').read_text())['total_reconstructed_source_rain_volume_m3']*run['rainfall_scale'];encoded=float(np.dot(globalrates[:-1],np.diff(knots)))
  assert abs(encoded-nominal)<=max(1.,nominal)*1e-6
  assert np.max(abs(since.sum(axis=(1,2))-(globalcum[ms]-globalcum[ms[0]])))<1e-5
  checks.append({'case':case,'zones':nz,'actual_steps':ns,'unique_overlap_patterns':len(patterns),'unchanged_roof_recipient_fields_exact':recipient_match,'native_centers_and_source_bilinear_sample_coincide':True,'max_applied_global_prefix_difference_m3':applied_error,'max_native_quadrature_prefix_difference_m3':quad_error,'original_relative_rain_tolerance':1e-6,'encoded_native_rain_m3':encoded,'original_roof_source_rain_m3':nominal,'selected_zone_rain_since_first_map_m3':float(since[-1,1:].sum()),'unselected_ground_rain_since_first_map_m3':float(since[-1,0].sum()),'original_simulation_runtime_seconds':float(json.loads((folder/'execution.json').read_text())['runtime_seconds'])})
  dump(R/'progress.json',{'cases_completed':len(checks),'last_case':case,'elapsed_seconds':time.monotonic()-start});print('RAIN_CASE_COMPLETE',case,'zones',nz,'max_applied_error_m3',applied_error,flush=True)
 for name,rows in [('zone_rain_trajectories.csv',alltr),('zone_rain_summary.csv',allcs),('zone_encoded_rain_intervals.csv',intervals),('case_rain_partition.csv',snapshots)]:pd.DataFrame(rows).to_csv(R/name,index=False)
 assert all((ROOT/k).stat().st_size==v['bytes'] and (ROOT/k).stat().st_mtime_ns==v['mtime_ns'] for k,v in sources.items())
 dump(R/'rain_validation.json',{'passed':True,'cases':checks,'source_bindings':sources,'source_stats_unchanged':True,'original_roof_weights_rain_fields_and_masks_unchanged':True,'method':'exact START block overlap;native float32 mean,area and dt products;partition all active ground before selecting zones','local_applied_rain_independently_saved':False,'new_hydraulic_runs':0,'required_pump_capacity_computed':False,'elapsed_seconds':time.monotonic()-start})
 print('ALL_RAIN_COMPLETE',len(checks),flush=True)
if __name__=='__main__':main()
