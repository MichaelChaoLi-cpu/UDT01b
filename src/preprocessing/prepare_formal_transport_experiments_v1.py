"""Preregister finite formal scenario matrix and stage bounded serial inputs."""
from pathlib import Path
import os,sys,json,shutil,time,hashlib,itertools
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_historical_surface_access_v1 import load,read,write,dump,sha
P=ROOT/'data/processed/three_ward_formal_transport_experiments_v1';R=ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1';D=ROOT/'data/exp/research-design/three_ward_formal_transport_matrix_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2';HR=ROOT/'data/exp/estimation/three_ward_historical_radar_experiment_v2';RAIN=ROOT/'data/processed/tokyo_historical_radar_20240821_v1';PAIR=ROOT/'data/processed/three_ward_urban_geometry_pair_inputs_v1';COV=ROOT/'data/processed/three_ward_surface_coverage_v2';COVR=ROOT/'data/exp/estimation/three_ward_surface_coverage_v2'
IMAGE='sha256:5415c73fc6ac335bd225336d0b2055c0ea0345c2eced42ba9cef8e1c44ae1ace'
FACTORS={'R0':0.,'R1':1.,'R125':1.25};STAGES=['L','H'];ARMS=['T','U'];TIME=[0,10800,12600,14400,18000,32400];INCREMENT=[0,0,1.5,1.5,0,0];BUILDER='/private/tmp/udt01b-sfincs-builder/bin/python'
def register():
 D.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 batch1=['R1__L__T','R1__H__T'];batch2=['R0__L__U','R0__L__T','R0__H__U','R0__H__T'];batch3=['R125__L__U','R125__L__T','R125__H__U','R125__H__T']
 budget=dict(concurrent_hydraulic_runs=1,cpus=6,memory_gib=7,maximum_new_runs=10,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=50400,maximum_new_output_gib=120,minimum_free_gib=100,nominal_per_run_wall_minutes=[45,90],nominal_first_batch_hours=[2,3],nominal_campaign_hours=[8,14],timing_basis='Existing urban9h runs56.5/57.7min; terrain and stronger inputs can differ. Estimates exclude user pauses; caps do not imply guaranteed completion.',automatic_retries=0)
 plan=dict(human_decision='MILI-D-20261017-004',research_objective='Conditional storm-related transport isolation and alternative access across Chiyoda, Chuo and Minato, not rainfall forecasting or engineering certification.',
 physical_factors={'rainfall':{'R0':'No-rain control on the same9h clock','R1':'Original2024-08-21 archived radar-derived representative intensity,108 five-minute intervals','R125':'Multiply the same complete source rain field by1.25; sensitivity descendant, not another observed storm or return period'},'river_stage':{'L':'Frozen low-stage assumptions','H':'Same low stage plus joint7-group0->1.5m18:00-18:30JST rise,18:30-19:00 plateau,19:00-20:00 recession; not observed river/tide'},'urban_geometry':{'T':'Original bare terrain, restored footprint storage, uniform n0.04, no artificial building barriers or road conveyance; same total source rain on terrain','U':'Frozen urban bundle: building obstacles/barriers, road roughness/connections and conservative instantaneous roof redistribution; not a fully3D water solver or isolated building-height effect'}},
 hydraulic_cases=12,reused_cases=['R1__L__U','R1__H__U'],new_case_cap=10,batches=[dict(batch_id='B1',purpose='Historical-rain matched T/U geometry pairs',new_cases=batch1),dict(batch_id='B2',purpose='No-rain controls separate imposed-stage inundation from rainfall contribution',new_cases=batch2),dict(batch_id='B3',purpose='Same-family rainfall-intensity stress and compound stage/geometry comparisons',new_cases=batch3)],execution_order=batch1+batch2+batch3,
 grid=dict(dx_m=10,shape=[1012,770],epsg=6677,horizon_s=32400,map_step_s=300,map_records=109,synchronized_times_s=sorted(set(range(0,32401,3600))|{12600,16200}),peak_envelope='separate non-simultaneous product'),
 transport_postprocessing={'rail_shutdown':['no_assumed_rail_shutdown','entrance','station_pass_through','service_section'],'road_projection':['strict_classified_road','ground_path_supplement'],'uncertainty':['lower_depth_unknown_available','upper_depth_unknown_withheld'],'road_threshold_m':[.1,.3],'station_identity':['original','source_normalized'],'combinations_per_state_per_hydraulic_case':64,'states_per_case_including_separate_peak':13,'maximum_network_evaluations':9984,'rail_no_shutdown_meaning':'Road flood restrictions remain; no inferred rail closure. Normal comparison separately has no flood closures and uses the matching unknown branch.'},
 comparison_rules=['Compare same rain, stage and solver across T/U; compare stage at fixed rain/geometry; compare rainfall at fixed stage/geometry.','City geometry only changes hydraulics and runoff allocation. The frozen layered transport graph and both projection policies remain the same across T/U.','Roof rainfall allocation differs by geometry, while source rainfall volume is conserved; this is a bundled urban-information comparison, not single-component causal identification.','Use original115 support and added29 as separate cohorts; use the144 maximum qualified cohort and per-case shared-admission masks. Report209 total and65 preexisting evidence-insufficient points, plus additional case-specific attrition.','All rainfall scalings and geometry/stage descendants share HIST-RADAR-TOKYO-20240821. No independent-event inflation; no original12+4 or sealed-test changes.'],
 numerical_stop_rules=['Any existing completion, identity, native geometry, field, actual-clock, rain-volume, conservation, positivity or boundary-exchange gate fails: hold case and stop queue.','Missing diagnostics, unsupported profile, nonfinite state, negative native volume or nonzero closed-boundary exchange: stop; never change masks or thresholds to pass.','Native actual clock stalls for15min after diagnostic initialization, container exits abnormally, resource cap exceeded, or protected inputs change: stop and preserve the attempt.','No automatic hydraulic restart, input repair, replacement case or solver-version switch.','User PAUSE request can be represented by a PAUSE flag; pause container and active-budget clock. User STOP flag stops the campaign and preserves outputs.'],
 budget=budget,solver_image=IMAGE,solver_patch_sha256=json.loads((H/'source_binding.json').read_text())['patch_sha256'],source_commit=json.loads((H/'source_binding.json').read_text())['source_commit'],reference_policy_sha256=sha(ROOT/'src/preprocessing/reference_label_policy.py'),
 scope_extension='Existing qualified9h block-rain implementation is reused on independently bound T topology and0/1.25 forcing descendants. Each new case requires unchanged numerical gates; no automatic training-reference release.',
 outputs=['Actual-clock water states and online peak depth intervals','Whole-domain water budget and7-group directional exchange','Synchronized station-neighborhood exposure separated from assumed shutdown','Native surface exit and rail alternative access with walking-distance burden','Robust, assumption-sensitive, insufficient-evidence and retained-alternative classes with coverage','Matched geometry and stage/rain differences on common cohorts'],
 active_measures='Later paired measure phase after core baseline comparisons. No active-measure capacity or engineering effect invented in this matrix.',people_counted=False,training=False,sealed_tests_opened=False,real_event_accuracy_established=False)
 dest=D/'protocol.json'
 if dest.exists():assert json.loads(dest.read_text())==plan,'Registered plan differs; do not silently amend'
 else:dump(dest,plan)
 return plan

def prepare():
 started=time.monotonic();plan=register()
 if (P/'run_manifest.json').exists():return
 assert not list((P/'runs').glob('*/launch.json')) and not list((P/'runs').glob('*/sfincs_map.nc')),'Never overwrite a hydraulic attempt'
 assert json.loads((COVR/'independent_validation.json').read_text())['all_passed'];assert json.loads((COVR/'batch_readiness.json').read_text())['scoped_formal_scenario_batch_recommended']
 pair=load(PAIR/'paired_support_and_native_mapping.npz');hmanifest=json.loads((H/'run_manifest.json').read_text());uold={x['case_id']:x for x in hmanifest['runs']};tm=json.loads((ROOT/'data/processed/three_ward_urban_geometry_pair_controls_v1/execution_manifest.json').read_text());tsource=next(x for x in tm['runs'] if x['case_id']=='rain60_raised');templates={'T':tsource,'U':uold['radar_low']};tsfolder=ROOT/tsource['folder'];geom={g:templates[g]['identity']['geometry_sha256'] for g in ARMS}
 assert json.loads((HR/'runtime_qualification.json').read_text())['all_passed'];assert all(json.loads((HR/'execution'/c/'audit.json').read_text())['all_checks_passed'] for c in ['radar_low','radar_raised'])
 protected={};hashcache={}
 def bind(p):
  key=str(p.relative_to(ROOT));st=p.stat();hk=(st.st_dev,st.st_ino)
  if hk not in hashcache:hashcache[hk]=sha(p)
  protected[key]=hashcache[hk];return hashcache[hk]
 for p in [D/'protocol.json',ROOT/'src/preprocessing/reference_label_policy.py',PAIR/'paired_support_and_native_mapping.npz',PAIR/'terrain_face_topology.npz',RAIN/'forcing_intervals_10m.nc',RAIN/'spatial_forcing_qa.json',COVR/'delivery_manifest.json',COV/'ground_path_supplement.npz',COV/'local_walking_graph.npz',ROOT/'data/processed/three_ward_minimum_scenarios_v1/partition_manifest.json',ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/selection.json']:
  bind(p)
 for g in ARMS:
  for n in templates[g]['input_sha256']:bind(ROOT/templates[g]['folder']/n)
 for c in ['radar_low','radar_raised']:
  for p in [HR/'execution'/c/'audit.json',H/'runs'/c/'sfincs_map.nc',H/'execution'/(c+'_candidate_reference.npz')]:bind(p)
 knots=np.arange(109,dtype='f8')*300;nominal0=json.loads((RAIN/'spatial_forcing_qa.json').read_text())['total_reconstructed_source_rain_volume_m3'];forcing={};qa=[]
 for rain,factor in FACTORS.items():
  for g in ARMS:
   f=P/'forcing'/(rain+'__'+g);f.mkdir(parents=True,exist_ok=True);dest=f/'historical_rain.nc';rates=[];inside=pair['Terrain Active Mask']==1 if g=='T' else pair['Urban Active Mask']==1
   with Dataset(RAIN/'forcing_intervals_10m.nc') as src,Dataset(dest,'w') as ds:
    for k,n in [('time',109),('y',1014),('x',772)]:ds.createDimension(k,n)
    v=ds.createVariable('time','f8',('time',));v.units='minutes since 2024-08-21 06:00:00';v[:]=knots/60
    for k,vals in [('x',-11370+np.arange(772)*10),('y',-42450+np.arange(1014)*10)]:v=ds.createVariable(k,'f8',(k,));v.units='m';v[:]=vals
    v=ds.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999.,zlib=True,complevel=2,chunksizes=(1,128,128));v.unit='mm.hr-1';v.units='mm h-1';source_var='terrain_effective_rainfall' if g=='T' else 'urban_effective_rainfall'
    for i in range(108):
     field=(np.ma.asarray(src[source_var][i]).filled(np.nan).astype('f8')*factor).astype('f4');assert np.isfinite(field).all() and (field>=0).all();rates.append(float(field[inside].sum(dtype='f8')*100/3.6e6));v[i]=np.flipud(np.pad(field,1))
    v[108]=0.;ds.parent_id='HIST-RADAR-TOKYO-20240821';ds.rainfall_scale=factor;ds.geometry_arm=g;ds.interval_convention='START-stamped right-held300s representative intensity; final zero sentinel';ds.source_sha256=protected[str((RAIN/'forcing_intervals_10m.nc').relative_to(ROOT))];ds.amprblock=1
   rates=np.r_[rates,0.];encoded=float(rates[:-1]@np.diff(knots));nominal=nominal0*factor;assert abs(encoded-nominal)<=max(1.,nominal)*1e-6
   np.savez_compressed(f/'forcing_budget.npz',time_s=knots,rate_m3_s=rates,cumulative_m3=np.r_[0,np.cumsum(rates[:-1]*300)])
   forcing[rain,g]=dict(path=dest,encoded_volume_m3=encoded,nominal_volume_m3=nominal);qa.append(dict(rainfall=rain,geometry=g,scale=factor,encoded_m3=encoded,nominal_m3=nominal,absolute_difference_m3=abs(encoded-nominal),relative_gate=1e-6))
   print('FORCING',rain,g,encoded,flush=True)
 for rain in FACTORS:assert abs(forcing[rain,'T']['encoded_volume_m3']-forcing[rain,'U']['encoded_volume_m3'])<=max(1.,nominal0*FACTORS[rain])*1e-6
 write(R/'forcing_volume_preflight.csv',qa)
 runs=[]
 for rain,stage,g in itertools.product(FACTORS,STAGES,ARMS):
  case='__'.join([rain,stage,g]);reuse=case in plan['reused_cases'];inc=INCREMENT if stage=='H' else [0]*6
  if reuse:
   x=uold['radar_raised' if stage=='H' else 'radar_low'];runs.append(dict(x,case_id=case,original_case_id=x['case_id'],rainfall=rain,rainfall_scale=FACTORS[rain],arm=g,stage=stage,reused=True,original_audit=str((HR/'execution'/x['case_id']/'audit.json').relative_to(ROOT))));continue
  folder=P/'runs'/case;folder.mkdir(parents=True,exist_ok=True);src=ROOT/templates[g]['folder'];names=[n for n in templates[g]['input_sha256'] if n not in ['sfincs.inp','sfincs.bzs','udt_run_identity.txt','roof_surface_precip.nc','terrain_surface_precip.nc','historical_rain.nc']]
  for n in names:
   if n=='sfincs_subgrid.nc':
    if not (folder/n).exists():(folder/n).hardlink_to(src/n)
   else:shutil.copy2(src/n,folder/n)
  
  if not (folder/'historical_rain.nc').exists():(folder/'historical_rain.nc').hardlink_to(forcing[rain,g]['path'])
  shutil.copy2(H/'runs'/('radar_raised' if stage=='H' else 'radar_low')/'sfincs.bzs',folder/'sfincs.bzs')
  changes={'tref':'20240821 060000','tstart':'20240821 060000','tstop':'20240821 150000','dtmaxout':'32400.0','dthisout':'0.0001','netamprfile':'historical_rain.nc','ampr_block':'1','amprblock':'1'};lines=[];seen=set()
  for line in (src/'sfincs.inp').read_text().splitlines():
   key=line.partition('=')[0].strip();seen.add(key);lines.append(f'{key:25s}= {changes[key]}' if key in changes else line)
  assert set(changes)<=seen;(folder/'sfincs.inp').write_text('\n'.join(lines)+'\n')
  ident=dict(templates[g]['identity']);ident.update(run_id=case+'__formal_10m',parent_id='HIST-RADAR-TOKYO-20240821',family_id='HIST-RADAR-TOKYO-20240821',split='external_formal_scenario',forcing_sha256=hashlib.sha256((sha(folder/'historical_rain.nc')+sha(folder/'sfincs.bzs')).encode()).hexdigest(),solver_digest=IMAGE,patch_sha256=plan['solver_patch_sha256']);dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n')
  inputs=names+['sfincs.inp','sfincs.bzs','historical_rain.nc','udt_run_identity.txt'];ih={n:(protected.get(str((src/n).relative_to(ROOT))) if n in names else None) or sha(folder/n) for n in inputs}
  runs.append(dict(case_id=case,run_id=ident['run_id'],folder=str(folder.relative_to(ROOT)),rainfall=rain,rainfall_scale=FACTORS[rain],arm=g,stage=stage,reused=False,identity=ident,image_id=IMAGE,input_sha256=ih,large_readonly_source=str((src/'sfincs_subgrid.nc').resolve()),horizon_s=32400,dtout_s=300,split='external_formal_scenario',stage_times_s=TIME,stage_increment_m=inc,nominal_rain_volume_m3=forcing[rain,g]['nominal_volume_m3'],forcing_budget=str((forcing[rain,g]['path'].parent/'forcing_budget.npz').relative_to(ROOT)),training=False,sealed_test=False))
 assert len(runs)==12 and sum(not x['reused'] for x in runs)==10;assert len({(x['rainfall'],x['stage'],x['arm']) for x in runs})==12
 dump(P/'run_manifest.json',dict(runs=runs,new_execution_order=plan['execution_order'],maximum_new_runs=10,protocol_sha256=sha(D/'protocol.json'),budget=plan['budget']));dump(P/'protected_sources.json',protected);(P/'execution').mkdir(exist_ok=True)
 dump(R/'input_validation.json',dict(all_passed=True,hydraulic_cases=12,reused_cases=2,new_run_cap=10,budget_frozen=True,source_rain_shared_by_geometry=True,all_rain_volumes_pass=True,whole_domain_same_grid=True,source_masks_unchanged=True,no_training_or_sealed_access=True,preparation_seconds=time.monotonic()-started));print('PREPARED',len(runs),time.monotonic()-started,flush=True)
if __name__=='__main__':prepare()
