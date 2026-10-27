"""Stage eight source-bound10m runs without changing city geometry, solver or measures."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,datetime as dt,shutil,hashlib,itertools,time
import numpy as np
from scipy.sparse import load_npz
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_historical_surface_access_v1 import load,read,write,dump,sha
from acquire_cross_event_radar_v1 import P,R,D,OLD,EVENTS
from prepare_formal_transport_experiments_v1 import IMAGE,BUILDER,TIME,INCREMENT
F=ROOT/'data/processed/three_ward_formal_transport_experiments_v1';PAIR=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1';MEASURE=ROOT/'data/processed/three_ward_motor_road_protection_v1';MOTOR=ROOT/'data/processed/three_ward_motor_flood_access_v1'

def prepare():
 start=time.monotonic();selection=json.loads((P/'event_selection.json').read_text());assert json.loads((R/'acquisition_state.json').read_text())['status']=='radar_acquisition_qualified';assert not(P/'run_manifest.json').exists(),'No silent input overwrite'
 pair=load(PAIR/'paired_support_and_native_mapping.npz');old=json.loads((F/'run_manifest.json').read_text());templates={g:next(r for r in old['runs']if r['case_id']=='R1__L__'+g)for g in ['T','U']};matrices={g:load_npz(OLD/(('terrain'if g=='T'else'urban')+'_source_area_matrix.npz'))for g in ['T','U']};mapping=json.loads((OLD/'source_recipient_map_contract.json').read_text());assert all(sha(OLD/n)==h for n,h in mapping['matrix_sha256'].items());assert all(sha(ROOT/n)==h for n,h in json.loads((MEASURE/'frozen_selection_binding.json').read_text()).items())
 cases=[e['event_id']+'__'+s+'__'+g for e in selection['events']for g in ['U','T']for s in ['L','H']];batches=[dict(batch_id=f'B{i+1}',new_cases=cases[i*2:i*2+2])for i in range(4)]
 budget=dict(concurrent_hydraulic_runs=1,cpus=6,memory_gib=7,maximum_new_runs=8,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=57600,maximum_new_output_gib=120,minimum_free_gib=100,nominal_per_run_wall_minutes=[45,90],nominal_campaign_hours=[6,12],automatic_retries=0,timing_basis='Prior completed10-run campaign8h54min; new wetness may alter runtime.16h active campaign cap,2h per case; estimates not guarantees.')
 protocol=dict(human_decision='MILI-D-20261017-011',purpose='Cross-event conditional motorcar accessibility and fixed road-protection transfer; no rainfall prediction or engineering certification.',events=selection['events'],selection_protocol_sha256=sha(D/'rainfall_selection_protocol.json'),execution_order=cases,batches=batches,new_case_cap=8,budget=budget,solver_image=IMAGE,prior_reference_policy_sha256=sha(ROOT/'src/preprocessing/reference_label_policy.py'),geometry='Exactly reused T/U templates,10m grid,original0.5m terrain support,urban buildings/roads and roof redistribution. T/U is a bundle, not full3D versus2D flow.',rain='Two different-date numerical5min archived representative-intensity parents; select9h maximum source-weighted rainfall window without observing network consequences; hold preceding interval300s, no scaling or calibration.',stage=dict(time_s=TIME,high_increment_m=INCREMENT,low_increment_m=[0]*6,meaning='Same elapsed-time seven-group low/high stage scenarios for both dates; not observed stage or tide and no rain-peak alignment optimization.'),grid=dict(shape=[1012,770],dx_m=10,epsg=6677,horizon_s=32400,map_step_s=300,map_records=109,times_s=[0,3600,7200,10800,12600,14400,16200,18000,21600,25200,28800,32400],separate_non_simultaneous_peak=True),measures=dict(source=str(MEASURE.relative_to(ROOT)),arms=['none','corridor','distributed'],length_m=json.loads((MEASURE/'protocol.json').read_text())['actual_length_m'],fixed=True,unknown_roads_never_reopened=True,meaning='Idealized flood passability only; no water-level feedback, costs, capacity or depth-dependent efficacy model. Conditional network benefits, not realized engineering efficacy.'),motor_network='Reuse exact original turn-aware conservative motorcar router; preserve one-way, access, native levels and turn history. Keep original road thresholds0.1/0.3m and lower/upper plus unknown branches.',support_policy='Intersect original common12case ground admission with all8newcase admission masks. A previously qualified road becomes unknown if any sampled coarse cell is outside common new admission. Recompute matched normal unknown-withheld baseline; retain original93outward/89bidirectional as upper bound and report additional attrition. Measures remain fixed; only known qualified selected roads can be reopened. No mask relaxation.',postprocessing_network_states=8*13*2*2*3,numerical_gates='Unchanged28checks and reference_label_policy,actual clocks,exact step rainfall integral,positivity,conservation,closed boundary flux and native directions. Metadata-only audit derivative bound and checked before launch.',stop_rules=['Any original numerical/identity/clock/geometry/rain/volume gate fails: stop queue and preserve failed attempt; no automatic retry or threshold relaxation.','Stop if actual clock stalls900s after diagnostics initialization, resource cap exceeded, protected input/code changes, or user STOP.','PAUSE flag freezes current container and active-budget clock; no new case while paused.','After each numeric batch retain results and update live report. Two fixed plans only; no adaptive road reselection after outcomes.'],independence='Different historical rainfall dates are additional event parents, not iid random samples; two extra events cannot establish general superiority. Same modern static road/city snapshot applied to historical forcing.',initial_state='Same dry initial surface; antecedent rainfall outside selected9h window not included. Report excluded survey rainfall. Not a complete reconstruction of past flood impacts.',training=False,sealed_tests_opened=False,people_counted=False,traffic_flow_or_queue=False,real_event_accuracy_established=False)
 dump(D/'protocol.json',protocol)
 protected={};hcache={}
 def bind(f):
  st=f.stat();k=(st.st_dev,st.st_ino)
  if k not in hcache:hcache[k]=sha(f)
  protected[str(f.relative_to(ROOT))]=hcache[k];return hcache[k]
 for f in [D/'protocol.json',D/'rainfall_selection_protocol.json',P/'event_selection.json',ROOT/'src/preprocessing/reference_label_policy.py',PAIR/'paired_support_and_native_mapping.npz',PAIR/'terrain_face_topology.npz',OLD/'source_recipient_map_contract.json',MEASURE/'frozen_selection_binding.json',MEASURE/'protected_roads.npz',MEASURE/'protocol.json',MEASURE/'protected_road_inventory.csv',MOTOR/'turn_router.npz',MOTOR/'road_projection.npz',MOTOR/'evidence_support.npz',MOTOR/'probe_inventory.csv',MOTOR/'road_inventory.csv',F/'factor_comparison_v1/common_support.npz',ROOT/'data/processed/three_ward_minimum_scenarios_v1/partition_manifest.json',ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/selection.json']+[OLD/n for n in mapping['matrix_sha256']]:bind(f)
 for g in ['T','U']:
  for n in templates[g]['input_sha256']:bind(ROOT/templates[g]['folder']/n)
 ratesby={};qa=[];knots=np.arange(109,dtype='f8')*300
 for event in selection['events']:
  eid=event['event_id'];folder=P/'rainfall'/eid;rad=load(folder/'radar_intervals.npz');flat=rad['representative_intensity_mm_h'].reshape(108,-1).astype('f8');area=np.asarray(matrices['T'].sum(axis=0)).ravel();used=area>0;assert np.isfinite(flat[:,used]).all();flat[:,~used]=0.;nominal=float((flat@area).sum()/12/1000);bind(folder/'radar_intervals.npz');bind(folder/'acquisition_manifest.json');units='minutes since '+dt.datetime.fromisoformat(event['window_start_utc']).strftime('%Y-%m-%d %H:%M:%S')
  intermediate=folder/'forcing_intervals_10m.nc'
  with Dataset(intermediate,'w')as cf:
   for k,n in [('time',108),('y',1012),('x',770)]:cf.createDimension(k,n)
   tv=cf.createVariable('time','f8',('time',));tv.units='seconds since 1970-01-01 00:00:00 UTC';tv[:]=rad['interval_start_unix_s']+150
   cf.parent_id=event['parent_id'];cf.rain_rule='preceding300s representative intensity hold; intermediate comparison array'
   for g in ['T','U']:
    target=P/'forcing'/(eid+'__'+g);target.mkdir(parents=True,exist_ok=True);inside=pair['Terrain Active Mask']==1 if g=='T'else pair['Urban Active Mask']==1;rates=[];var='terrain_effective_rainfall'if g=='T'else'urban_effective_rainfall';cv=cf.createVariable(var,'f4',('time','y','x'),zlib=True,complevel=2);cv.units='mm h-1'
    with Dataset(target/'historical_rain.nc','w')as nc:
     for k,n in [('time',109),('y',1014),('x',772)]:nc.createDimension(k,n)
     v=nc.createVariable('time','f8',('time',));v.units=units;v[:]=knots/60
     for k,vals in [('x',-11370+np.arange(772)*10),('y',-42450+np.arange(1014)*10)]:v=nc.createVariable(k,'f8',(k,));v.units='m';v[:]=vals
     v=nc.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999.,zlib=True,complevel=2,chunksizes=(1,128,128));v.unit='mm.hr-1';v.units='mm h-1'
     for k in range(108):
      field=(matrices[g]@flat[k]/100).astype('f4').reshape(1012,770);assert np.isfinite(field).all()and(field>=0).all();cv[k]=field;v[k]=np.flipud(np.pad(field,1));rates.append(float(field[inside].sum(dtype='f8')*100/3.6e6))
     v[108]=0;nc.parent_id=event['parent_id'];nc.geometry_arm=g;nc.interval_convention='START-stamped right-held300s representative intensity;finalzero';nc.amprblock=1
    rates=np.r_[rates,0.];encoded=float(rates[:-1]@np.diff(knots));assert abs(encoded-nominal)<=max(1.,nominal)*1e-6
    ratesby[eid,g]=dict(path=target/'historical_rain.nc',nominal=nominal,units=units);np.savez_compressed(target/'forcing_budget.npz',time_s=knots,rate_m3_s=rates,cumulative_m3=np.r_[0,np.cumsum(rates[:-1]*300)]);qa.append(dict(event_id=eid,arm=g,nominal_volume_m3=nominal,encoded_volume_m3=encoded,absolute_error_m3=abs(encoded-nominal),relative_gate=1e-6));print('FORCING',eid,g,encoded,flush=True)
  dump(folder/'spatial_forcing_qa.json',dict(total_reconstructed_source_rain_volume_m3=nominal,source_recipient_matrices_unchanged=True,interval_count=108));bind(intermediate);bind(folder/'spatial_forcing_qa.json')
 write(R/'forcing_volume_preflight.csv',qa)
 runs=[]
 for event in selection['events']:
  for g,s in itertools.product(['U','T'],['L','H']):
   eid=event['event_id'];case=eid+'__'+s+'__'+g;folder=P/'runs'/case;folder.mkdir(parents=True,exist_ok=True);template=templates[g];src=ROOT/template['folder'];names=[n for n in template['input_sha256']if n not in ['sfincs.inp','sfincs.bzs','udt_run_identity.txt','historical_rain.nc','roof_surface_precip.nc','terrain_surface_precip.nc']]
   for n in names:
    if n=='sfincs_subgrid.nc':(folder/n).hardlink_to(src/n)
    else:shutil.copy2(src/n,folder/n)
   (folder/'historical_rain.nc').hardlink_to(ratesby[eid,g]['path']);base_bzs=ROOT/next(r['folder']for r in old['runs']if r['case_id']=='R1__'+s+'__T')/'sfincs.bzs';shutil.copy2(base_bzs,folder/'sfincs.bzs')
   begin=dt.datetime.fromisoformat(event['window_start_utc']);end=dt.datetime.fromisoformat(event['window_end_utc']);changes={'tref':begin.strftime('%Y%m%d %H%M%S'),'tstart':begin.strftime('%Y%m%d %H%M%S'),'tstop':end.strftime('%Y%m%d %H%M%S')};lines=[];seen=set()
   for line in(src/'sfincs.inp').read_text().splitlines():
    key=line.partition('=')[0].strip();seen.add(key);lines.append(f'{key:25s}= {changes[key]}'if key in changes else line)
   assert set(changes)<=seen;(folder/'sfincs.inp').write_text('\n'.join(lines)+'\n')
   ident=dict(template['identity']);ident.update(run_id=case+'__cross_event_10m',parent_id=event['parent_id'],family_id=event['parent_id'],split='external_cross_event_transport',forcing_sha256=hashlib.sha256((sha(folder/'historical_rain.nc')+sha(folder/'sfincs.bzs')).encode()).hexdigest());dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');inputs=names+['sfincs.inp','sfincs.bzs','historical_rain.nc','udt_run_identity.txt'];hashes={n:bind(src/n)if n in names else sha(folder/n)for n in inputs}
   runs.append(dict(case_id=case,run_id=ident['run_id'],event_id=eid,parent_id=event['parent_id'],folder=str(folder.relative_to(ROOT)),rain_source_folder=str((P/'rainfall'/eid).relative_to(ROOT)),rain_time_units=ratesby[eid,g]['units'],rainfall_scale=1.,arm=g,stage=s,reused=False,identity=ident,image_id=IMAGE,input_sha256=hashes,large_readonly_source=str((src/'sfincs_subgrid.nc').resolve()),horizon_s=32400,dtout_s=300,split=ident['split'],stage_times_s=TIME,stage_increment_m=INCREMENT if s=='H'else[0]*6,nominal_rain_volume_m3=ratesby[eid,g]['nominal'],window_start_utc=event['window_start_utc'],training=False,sealed_test=False))
 assert len(runs)==8 and{r['case_id']for r in runs}==set(cases);(P/'execution').mkdir(exist_ok=True);dump(P/'run_manifest.json',dict(runs=runs,new_execution_order=cases,maximum_new_runs=8,protocol_sha256=sha(D/'protocol.json'),budget=budget));dump(P/'protected_sources.json',protected);dump(R/'input_validation.json',dict(all_passed=True,runs=8,events=2,source_fields_exact=True,source_volume_gate=1e-6,geometry_and_measures_unchanged=True,real_utc_epoch_in_inp_and_rain=True,preparation_s=time.monotonic()-start));print('EIGHT RUN INPUTS PREPARED',round(time.monotonic()-start),flush=True)
if __name__=='__main__':prepare()
