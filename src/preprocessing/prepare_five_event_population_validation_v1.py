"""Three additional rain parents, six serial urban-geometry runs; unchanged physics."""
from pathlib import Path
import os,sys,json,datetime as dt,hashlib,shutil,itertools,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.sparse import load_npz
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_historical_surface_access_v1 import load,read,write,dump,sha
from prepare_formal_transport_experiments_v1 import IMAGE,BUILDER,TIME,INCREMENT
TAG='three_ward_five_event_population_validation_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
NEW=ROOT/'data/processed/three_ward_population_additional_radar_v1';OLD=ROOT/'data/processed/tokyo_historical_radar_20240821_v1';PRIOR=ROOT/'data/processed/three_ward_cross_event_protection_v1';PAIR=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1'
EVENT_ORDER=['E20240731','E20250710','E20240618'];UTC=dt.timezone.utc

def prepare():
 started=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 if(P/'run_manifest.json').exists():raise RuntimeError('Existing preparation preserved; run validation or resume explicitly')
 assert json.loads((ROOT/'data/exp/data-preprocessing/three_ward_population_additional_radar_v1/delivery_validation.json').read_text())['passed']
 events=json.loads((NEW/'event_inventory.json').read_text())['events'];by={e['event_id']:e for e in events};events=[by[k]for k in EVENT_ORDER]
 for e in events:
  for s in ['start','end']:e['window_'+s+'_utc']=dt.datetime.fromisoformat(e['window_'+s+'_jst']).astimezone(UTC).isoformat()
 cases=[e['event_id']+'__'+s+'__U'for e in events for s in ['H','L']];budget=dict(concurrent_hydraulic_runs=1,cpus=6,memory_gib=7,maximum_new_runs=6,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=43200,maximum_new_output_gib=120,minimum_free_gib=100,nominal_campaign_hours=[6,8],automatic_retries=0,postprocessing_active_wall_s=10800)
 protocol=dict(decision='MILI-D-20261018-013',events=events,execution_order=cases,batches=[dict(batch_id=f'B{i+1}',new_cases=cases[2*i:2*i+2])for i in range(3)],budget=budget,solver_image=IMAGE,grid=dict(dx_m=10,shape=[1012,770],horizon_s=32400,map_step_s=300,times_s=[0,3600,7200,10800,12600,14400,16200,18000,21600,25200,28800,32400]),geometry='Frozen U urban-geometry10m model, building barriers, road constraints and conservative roof redistribution; no geometry or numerical changes.',rain='Native1km5min JMA representative intensity reconstructed over preceding300s, original rain-only selected9h windows. Not10m radar observations.',stage=dict(times_s=TIME,high_increment_m=INCREMENT,low_increment_m=[0]*6,meaning='Same hypothetical river-stage scenarios as old events,not observed stage.'),initial_state='Same dry initial surface; excluded antecedent rain retained in event metadata.20240618 has21.53mm before selected9h;controlled comparison,not full historical flood reconstruction.',gates='Exactly original numerical admission,actual second clocks,exact step rain integral,positive native volume,closed-boundary zero exchange,water balance and quarantine;no threshold relaxation.',validation_claim='Numerical/interface checks and additional-event transfer only;without independent observed inundation or traffic closures,real-event predictive accuracy remains unverified.',automatic_progression=['Run and audit each of6cases serially;hold the entire queue on any original failed gate.','After all6pass,intersect old/new admitted ground;newly unsupported roads remain unknown;protect only original selected roads retaining qualified support.','Reuse official2020resident totals and frozen0/10/25m candidate access support;analyze old2H plus new3H/3L,retain uncertainty,no vehicle-to-person conversion.','If population identity,route and bounds checks pass,run3newH events with fixed protection off/on,deferral0/10/20/30percent,UE/SO guidance;reuse8normal controls.','Generate joint five-event results and report coverage and incomplete/failed steps honestly.'],measures='Freeze original118roads4.743231745km;no population/outcome-based reselection,ideal passability only.',population='Historical at-residence stock496345;not synchronous actual occupancy or actual stranded persons. Daytime destination population not invented.',traffic='Same17h high-demand2xPT driver proxy and assumed capacity across events;not historical traffic reconstruction;two-path static UE/SO,not measured queue delay.',postprocessing_budget=dict(new_path_sweeps=12,seconds_per_sweep=180,new_optimizer_fits=48,seconds_per_fit=120,maximum_fit_active_seconds=7200,maximum_path_active_seconds=2700),stop='PAUSE pauses containers and active clocks. STOP halts and preserves attempt; no automatic hydraulic retries,solver changes or gate relaxation.',training=False,sealed_tests_opened=False)
 dump(P/'protocol.json',protocol);dump(P/'event_selection.json',dict(events=events,selection_uses_rainfall_only=True,measures_not_reselected=True));protected={}
 def bind(f):protected[str(f.relative_to(ROOT))]=sha(f);return protected[str(f.relative_to(ROOT))]
 deps=[NEW/'protocol.json',NEW/'event_inventory.json',P/'protocol.json',P/'event_selection.json',ROOT/'src/preprocessing/reference_label_policy.py',ROOT/'src/analyses/audit_cross_event_protection_run_v1.py',ROOT/'src/analyses/integrate_night_population_transport_v1.py',ROOT/'src/analyses/run_combined_transport_measures_v1.py',PAIR/'paired_support_and_native_mapping.npz',OLD/'source_recipient_map_contract.json',PRIOR/'motor_common_support.npz',ROOT/'data/processed/three_ward_demand_protection_feedback_v1/protection_masks.npz',ROOT/'data/processed/three_ward_minimum_scenarios_v1/partition_manifest.json',ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/selection.json',ROOT/'data/processed/three_ward_night_population_transport_v1/input_bindings.json']
 mapping=json.loads((OLD/'source_recipient_map_contract.json').read_text());deps +=[OLD/n for n in mapping['matrix_sha256']]
 assert all(sha(OLD/n)==h for n,h in mapping['matrix_sha256'].items())
 for f in deps:bind(f)
 populationbindings=json.loads((ROOT/'data/processed/three_ward_night_population_transport_v1/input_bindings.json').read_text())['sha256']
 for n,h in populationbindings.items():assert sha(ROOT/n)==h;protected[n]=h
 for f in (ROOT/'data/processed/three_ward_night_population_transport_v1').glob('*.npz'):bind(f)
 manifest=json.loads((PRIOR/'run_manifest.json').read_text());template=next(r for r in manifest['runs']if r['case_id']=='E20240830__L__U');src=ROOT/template['folder'];pair=load(PAIR/'paired_support_and_native_mapping.npz');inside=pair['Urban Active Mask']==1;mat=load_npz(OLD/'urban_source_area_matrix.npz');area=np.asarray(mat.sum(0)).ravel();used=area>0;knots=np.arange(109)*300.;qa=[];runs=[]
 for n in template['input_sha256']:bind(src/n)
 for e in events:
  eid=e['event_id'];rain=P/'rainfall'/eid;rain.mkdir(parents=True,exist_ok=True);source=NEW/eid/'radar_intervals.npz';bind(source);shutil.copy2(source,rain/'radar_intervals.npz');rad=load(source);flat=rad['representative_intensity_mm_h'].reshape(108,-1).astype('f8');assert np.isfinite(flat[:,used]).all();flat[:,~used]=0.;nominal=float((flat@area).sum()/12000);units='minutes since '+dt.datetime.fromisoformat(e['window_start_utc']).strftime('%Y-%m-%d %H:%M:%S');target=P/'forcing'/eid;target.mkdir(parents=True,exist_ok=True);rates=[]
  with Dataset(rain/'forcing_intervals_10m.nc','w')as inter,Dataset(target/'historical_rain.nc','w')as nc:
   for k,n in [('time',108),('y',1012),('x',770)]:inter.createDimension(k,n)
   cv=inter.createVariable('urban_effective_rainfall','f4',('time','y','x'),zlib=True,complevel=2);cv.units='mm h-1'
   for k,n in [('time',109),('y',1014),('x',772)]:nc.createDimension(k,n)
   tv=nc.createVariable('time','f8',('time',));tv.units=units;tv[:]=knots/60
   for k,vals in [('x',-11370+np.arange(772)*10),('y',-42450+np.arange(1014)*10)]:v=nc.createVariable(k,'f8',(k,));v.units='m';v[:]=vals
   v=nc.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999.,zlib=True,complevel=2,chunksizes=(1,128,128));v.units='mm h-1';v.unit='mm.hr-1'
   for k in range(108):
    field=(mat@flat[k]/100).astype('f4').reshape(1012,770);assert np.isfinite(field).all()and(field>=0).all();cv[k]=field;v[k]=np.flipud(np.pad(field,1));rates.append(float(field[inside].sum(dtype='f8')*100/3.6e6))
   v[108]=0.;nc.parent_id=e['parent_id'];nc.interval_convention='START-stamped right-held300s representative intensity;finalzero';nc.amprblock=1
  rates=np.r_[rates,0.];encoded=float(rates[:-1]@np.diff(knots));assert abs(encoded-nominal)<=max(1.,nominal)*1e-6;np.savez_compressed(target/'forcing_budget.npz',time_s=knots,rate_m3_s=rates,cumulative_m3=np.r_[0,np.cumsum(rates[:-1]*300)]);dump(rain/'spatial_forcing_qa.json',dict(total_reconstructed_source_rain_volume_m3=nominal,source_recipient_matrices_unchanged=True,interval_count=108));qa.append(dict(event_id=eid,nominal_volume_m3=nominal,encoded_volume_m3=encoded,error_m3=abs(encoded-nominal)))
  for f in [rain/'radar_intervals.npz',rain/'forcing_intervals_10m.nc',rain/'spatial_forcing_qa.json',target/'historical_rain.nc',target/'forcing_budget.npz']:bind(f)
  for stage in ['H','L']:
   case=eid+'__'+stage+'__U';folder=P/'runs'/case;folder.mkdir(parents=True,exist_ok=True);names=[n for n in template['input_sha256']if n not in ['sfincs.inp','sfincs.bzs','udt_run_identity.txt','historical_rain.nc','roof_surface_precip.nc','terrain_surface_precip.nc']]
   for n in names:
    if n=='sfincs_subgrid.nc':(folder/n).hardlink_to(src/n)
    else:shutil.copy2(src/n,folder/n)
   (folder/'historical_rain.nc').hardlink_to(target/'historical_rain.nc');stage_src=ROOT/next(r['folder']for r in manifest['runs']if r['case_id']=='E20240830__'+stage+'__U')/'sfincs.bzs';shutil.copy2(stage_src,folder/'sfincs.bzs');bind(stage_src);begin=dt.datetime.fromisoformat(e['window_start_utc']);end=dt.datetime.fromisoformat(e['window_end_utc']);changes={'tref':begin.strftime('%Y%m%d %H%M%S'),'tstart':begin.strftime('%Y%m%d %H%M%S'),'tstop':end.strftime('%Y%m%d %H%M%S')};lines=[]
   for line in(src/'sfincs.inp').read_text().splitlines():key=line.partition('=')[0].strip();lines.append(f'{key:25s}= {changes[key]}'if key in changes else line)
   (folder/'sfincs.inp').write_text('\n'.join(lines)+'\n');ident=dict(template['identity']);ident.update(run_id=case+'__five_event_10m',parent_id=e['parent_id'],family_id=e['parent_id'],split='external_population_transfer',forcing_sha256=hashlib.sha256((sha(folder/'historical_rain.nc')+sha(folder/'sfincs.bzs')).encode()).hexdigest());dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n');hashes={n:sha(folder/n)for n in names+['sfincs.inp','sfincs.bzs','historical_rain.nc','udt_run_identity.txt']}
   runs.append(dict(case_id=case,run_id=ident['run_id'],event_id=eid,parent_id=e['parent_id'],folder=str(folder.relative_to(ROOT)),rain_source_folder=str(rain.relative_to(ROOT)),rain_time_units=units,rainfall_scale=1.,arm='U',stage=stage,reused=False,identity=ident,image_id=IMAGE,input_sha256=hashes,large_readonly_source=str((src/'sfincs_subgrid.nc').resolve()),horizon_s=32400,dtout_s=300,split=ident['split'],stage_times_s=TIME,stage_increment_m=INCREMENT if stage=='H'else[0]*6,nominal_rain_volume_m3=nominal,window_start_utc=e['window_start_utc'],training=False,sealed_test=False))
  print('FORCING PREPARED',eid,encoded,flush=True)
 (P/'execution').mkdir(exist_ok=True);(R/'execution').mkdir(exist_ok=True);write(R/'forcing_volume_preflight.csv',qa);dump(P/'run_manifest.json',dict(runs=runs,new_execution_order=cases,maximum_new_runs=6,protocol_sha256=sha(P/'protocol.json'),budget=budget));dump(P/'protected_sources.json',protected);dump(R/'input_validation.json',dict(all_passed=True,case_count=6,event_count=3,source_volume_gate=1e-6,geometry_unchanged=True,preparation_s=time.monotonic()-started));print('SIX RUNS PREPARED',flush=True)
if __name__=='__main__':prepare()
