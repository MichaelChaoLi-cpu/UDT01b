"""Build synthetic forcing inputs only. No solver, training, network, or subprocess calls.

Run with the project geospatial Python runtime. Source arrays are immutable.
Versioned numerical design settings are engineering choices, not observations.
"""
from pathlib import Path
import argparse, csv, hashlib, json, itertools
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT=Path(__file__).resolve().parents[2]
LIB='data/processed/three_ward_minimum_scenarios_v1'
REPORT='data/exp/data-preprocessing/three_ward_minimum_scenarios'
PLAN='data/exp/research-design/ml-surrogate-methods-r2'
OLD='data/processed/three_ward_surrogate_inputs'
CONFIG={'version':'minimum-forcing-v1','master_seed':20261015,'horizon_s':14400,
 'normalized_samples':257,'near_rmse':0.02,'near_joint_rmse':0.04,'near_correlation':0.995,
 'max_attempts':512,'synthetic_settings_not_observations':True,
 'new_simulations_authorized':False,'training_authorized':False,
 'time_interpolation':'piecewise linear, exact knot union; no coarse resampling',
 'rain_spatial':'uniform physical rain; fixed conservative roof redistribution',
 'stage_spatial':'source low point levels plus coherent additive triangular pulse across all seven groups',
 'stage_reference':'inherited working T.P. reference; epoch alignment uncertified; no measured hydrographs',
 'deduplication':'normalize start, duration and peak, then compare rainfall shape irrespective of stage; controls share legacy lineage',
 'initial_state':'dry interior; fixed existing geometry, roughness and no-action setting'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def arrays(path):
 with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def integral(t,r):return float(np.sum(np.diff(t)*(r[:-1]+r[1:])*0.5)/3600)
def semantic(a):
 h=hashlib.sha256()
 for k in sorted(a):
  v=np.ascontiguousarray(a[k]);h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()
def stable_seed(slot,attempt):return int(digest([CONFIG['master_seed'],slot,attempt])[:16],16)

def normalized(t,r,start,duration):
 if np.max(r)==0:return None
 return np.interp(np.linspace(start,start+duration,CONFIG['normalized_samples']),t,r)/np.max(r)
def distance(a,b):
 rmse=float(np.sqrt(np.mean((a-b)**2)))
 corr=float(np.corrcoef(a,b)[0,1])
 near=rmse<=CONFIG['near_rmse'] or (rmse<=CONFIG['near_joint_rmse'] and corr>=CONFIG['near_correlation'])
 return rmse,corr,bool(near)

def candidate(slot,attempt,shape,axis=None,challenge_number=1):
 seed=stable_seed(slot,attempt);g=np.random.default_rng(seed)
 # Parameters are sampled independently of final split labels and response data.
 duration=int(g.integers(30,91))*60
 if axis=='rainfall_duration':duration=int(g.integers(105,121))*60
 if axis=='rain_stage_lag' and challenge_number==1:duration=int(g.integers(75,91))*60;shape='late'
 peak=round(float(g.uniform(20,100)),2)
 amplitude=0.0 if slot%5==0 else round(float(g.uniform(.05,1.5)),3)
 lag=int(g.integers(-30,31))*60
 if axis=='withheld_compound':peak=round(float(g.uniform(90,100)),2);amplitude=round(float(g.uniform(1.2,1.5)),3)
 if axis=='rainfall_magnitude':peak=round(float(g.uniform(110,120)),2)
 if axis=='rain_stage_lag':lag=(-3600 if challenge_number==1 else 3600);amplitude=round(float(g.uniform(.2,1.5)),3)
 if axis!='withheld_compound' and peak>=90 and amplitude>=1.2:return None,'excluded compound corner'
 if shape=='to_allocate_unique_realization':shape=['early','central','late','multiple_peak'][(slot-1)%4]
 ranges={'early':(.15,.30),'central':(.40,.60),'late':(.70,.85)}
 u=np.concatenate(([0.],np.arange(1,11)/11+g.uniform(-.025,.025,10),[1.]))
 if shape=='multiple_peak':
  p1=float(g.uniform(.18,.35));p2=float(g.uniform(.65,.84));main=0 if g.random()<.5 else 1
  up=np.array([0,p1,(p1+p2)*.5,p2,1.]);yp=np.array([0,1 if main==0 else g.uniform(.65,.9),g.uniform(.05,.3),1 if main==1 else g.uniform(.65,.9),0])
  u=np.unique(np.r_[u,up]);y=np.interp(u,up,yp);peak_u=[p1,p2][main]
 else:
  peak_u=float(g.uniform(*ranges[shape]));u=np.unique(np.r_[u,peak_u])
  y=np.maximum(0,np.minimum(u/peak_u,(1-u)/(1-peak_u)))**float(g.uniform(.65,2.3))
 y=np.minimum(y*g.uniform(.65,1.25,len(y)),.96);y[np.argmin(abs(u-peak_u))]=1;y[[0,-1]]=0
 tr=3600+np.rint(u*duration);rr=y*peak
 if len(np.unique(tr))!=len(tr):return None,'collapsed integer-second rain knots'
 rain_peak=float(tr[np.argmax(rr)]);stage_peak=rain_peak+lag
 if stage_peak-1800<0 or stage_peak+1800>14400:return None,'pulse outside horizon'
 ts=np.array([0,stage_peak-1800,stage_peak,stage_peak+1800,14400.]);ss=np.array([0,0,amplitude,0,0.])
 ts,idx=np.unique(ts,return_index=True);ss=ss[idx]
 time=np.unique(np.r_[0,tr,ts,14400])
 rain=np.interp(time,tr,rr,left=0,right=0);delta=np.interp(time,ts,ss)
 data={'Event Time':time.astype('f8'),'Physical Rainfall Rate':rain.astype('f8'),
       'River Stage Increment By Group':np.repeat(delta[:,None],7,axis=1)}
 meta={'seed':seed,'accepted_attempt':attempt,'slot':slot,'shape_class':shape,'challenge_axis':axis,
       'rain_start_s':3600,'rain_duration_s':duration,'rain_peak_mm_h':peak,'rain_peak_time_s':rain_peak,
       'stage_amplitude_m':amplitude,'stage_peak_time_s':stage_peak,'scheduled_peak_lag_s':lag,
       'actual_peak_lag_s':lag if amplitude>0 else None,
       'rain_knot_time_s':tr.tolist(),'rain_knot_rate_mm_h':rr.tolist(),
       'stage_knot_time_s':ts.tolist(),'stage_knot_increment_m':ss.tolist()}
 return (data,meta,normalized(time,rain,3600,duration)),None

def legacy(peak,raised):
 tr=np.array([0,3600,3601,7200,7201,14400.]);rr=np.array([0,0,peak,peak,0,0.])
 ts=np.array([0,3600,5400,7200,10800,14400.]);ss=np.array([0,0,1.5,1.5,0,0.]) if raised else np.zeros(6)
 t=np.union1d(tr,ts);r=np.interp(t,tr,rr);s=np.interp(t,ts,ss)
 return {'Event Time':t,'Physical Rainfall Rate':r,'River Stage Increment By Group':np.repeat(s[:,None],7,axis=1)}

def shared_geometry(root):
 a10=rasterio.open(root/'data/processed/three_ward_fixed_runoff/contributing_area_10m.tif').read(1).astype('f8')
 a5=rasterio.open(root/'data/processed/three_ward_adaptive_connectivity/contributing_area_5m.tif').read(1).astype('f8')
 ground10=arrays(root/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')['Ground Area']
 ground5=np.zeros(a5.shape,'f8')
 with rasterio.open(root/'data/processed/three_ward_sfincs_scenarios/active_mask_20m.tif') as m, rasterio.open(root/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif') as c:
  mask=m.read(1)
  for r in range(0,len(mask),40):
   piece=mask[r:r+40];inside=np.repeat(np.repeat(piece==1,40,0),40,1)
   cls=c.read(1,window=Window(0,r*40,c.width,len(piece)*40))
   op=inside&(cls!=1)
   out=op.reshape(len(piece)*4,10,a5.shape[1],10).sum((1,3))*.25
   ground5[r*4:(r+len(piece))*4]=out
 def aggregate(v):return v.reshape(a10.shape[0],2,a10.shape[1],2).sum((1,3))
 assert np.array_equal(aggregate(a5),a10),'Contributing-area nesting changed'
 assert np.array_equal(aggregate(ground5),ground10),'Open-ground nesting changed'
 assert np.min(a5-ground5)>=0 and np.min(a10-ground10)>=0
 assert float(a10.sum())==38976800 and float(ground10.sum())==25936197
 geo=arrays(root/'data/processed/three_ward_adaptive_connectivity/adaptive_cell_geometry.npz')
 row=geo['row_fine']//10;col=geo['col_fine']//10;fine=geo['size_m']==5
 native=np.where(fine,a5[row,col],a10[row//2,col//2]);native[geo['mask']!=1]=0
 assert np.array_equal(native,geo['contributing_area_m2'])
 graph=arrays(root/OLD/'surface_connectivity_graph.npz')
 return {'Contributing Area 10m':a10,'Contributing Area 5m':a5,'Ground Area 10m':ground10,'Ground Area 5m':ground5,
   'Roof Assigned Area 10m':a10-ground10,'Roof Assigned Area 5m':a5-ground5,
   'Effective Rainfall Weight 10m':a10/100,'Effective Rainfall Weight 5m':a5/25,
   'Adaptive Contributing Area':native,'Adaptive Row 5m':row,'Adaptive Column 5m':col,
   'Adaptive Size':geo['size_m'],'Adaptive Interior':geo['mask']==1,
   'Boundary Low Stage':graph['Boundary Low Stage'],'Boundary Group Index':graph['Boundary Group Index'],
   'Boundary Group Codes':graph['Boundary Group Codes'],'Boundary Group Names':graph['Boundary Group Names'],
   'Boundary Point Position':graph['Boundary Point Position']}

def run_binding(case,grid):
 return {'run_id':case['case_id']+'__'+grid,'case_id':case['case_id'],'parent_id':case['parent_id'],
 'family_id':case['family_id'],'split':case['split'],'grid':grid,'forcing_sha256':case['forcing_sha256'],
 'status':'unrun','response_path':None,'measure':'no_action','initial_state':'dry_interior'}

def build(root=ROOT):
 out=root/LIB;report=root/REPORT;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 freeze=out/'freeze.json'
 if freeze.exists():raise FileExistsError('Frozen library exists; validate it or build a new version, never overwrite')
 source_paths=[PLAN+'/design-contract.json',PLAN+'/reserved-parent-allocations.json',OLD+'/input_schema.json',OLD+'/quality_policy.json',OLD+'/surface_connectivity_graph.npz',
 'data/processed/three_ward_fixed_runoff/contract.json','data/processed/three_ward_fixed_runoff/contributing_area_10m.tif',
 'data/processed/three_ward_adaptive_connectivity/contributing_area_5m.tif','data/processed/three_ward_adaptive_connectivity/adaptive_cell_geometry.npz',
 'data/processed/three_ward_common_depth_labels/common_support_10m.npz','data/processed/three_ward_urban_constraints/constraints_0_50m.tif','data/processed/three_ward_sfincs_scenarios/active_mask_20m.tif']
 source_paths += [str(p.relative_to(root)) for p in sorted((root/OLD).glob('*_inputs.npz'))]
 source_paths += ['data/processed/three_ward_fixed_runoff/sfincs_ncinput.F90','data/processed/three_ward_sfincs_scenarios/solver_provenance/sfincs_meteo.f90',
  'data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison/rain60_low_corrected_10m/sfincs.inp',
  'data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison/rain60_low_corrected_10m/sfincs.bnd']
 sources={p:sha(root/p) for p in source_paths}
 roof_contract=json.loads((root/'data/processed/three_ward_fixed_runoff/contract.json').read_text())
 assert all(sources[p]==h for p,h in roof_contract['source_sha256'].items()),'Frozen roof donor geometry changed'
 # Keep pre-existing input/label products in the provenance protection set.
 protected={str(p.relative_to(root)):sha(p) for folder in [root/OLD,root/'data/processed/three_ward_common_depth_labels'] for p in sorted(folder.iterdir()) if p.is_file()}
 allocations=json.loads((root/PLAN/'reserved-parent-allocations.json').read_text())
 dump(out/'generator_config.json',CONFIG)
 print('Building conservative shared geometry...',flush=True)
 geom=shared_geometry(root);np.savez_compressed(out/'shared_geometry.npz',**geom)
 cases=[];signals={};norms={};rejected=[]
 for slot,allocation in enumerate(allocations['parents'],1):
  for attempt in range(CONFIG['max_attempts']):
   result,why=candidate(slot,attempt,allocation['shape_class'],allocation.get('challenge_axis'),int(allocation['parent_id'][-1]))
   if result is not None:
    data,meta,norm=result
    close=[cid for cid,v in norms.items() if v is not None and distance(norm,v)[2]]
    if close:why='near rainfall shape: '+','.join(close)
   if why:rejected.append({'slot':slot,'attempt':attempt,'reason':why});continue
   break
  else:raise RuntimeError('No distinct feasible realization for '+allocation['parent_id'])
  cid=allocation['parent_id'];parent=f'P{slot:04d}'
  np.savez_compressed(out/(cid+'.npz'),**data)
  record={'case_id':cid,'parent_id':parent,'family_id':parent,'split':allocation['split'],
   'independent_parent_candidate':True,'forcing_file':cid+'.npz','forcing_sha256':semantic(data),
   'normalized_shape_sha256':semantic({'Normalized Rainfall Shape':norm}),
   'adaptive_companion':allocation['adaptive_companion'],'source_allocation':allocation,
   'status':'forcing_generated_response_absent','parameters':meta}
  cases.append(record);signals[cid]=data;norms[cid]=norm
 print('Generated 48 de-duplicated synthetic parents',flush=True)
 # Existing and new controls are all descendants of one shared legacy forcing family.
 controls=[('rain60_low',60,False,True),('rain100_low',100,False,True),('rain60_raised',60,True,True),('dry_raised',0,True,True),
 ('dry_low_current_geometry',0,False,False),('rain100_raised_current_geometry',100,True,False)]
 for cid,peak,raised,existing in controls:
  data=legacy(peak,raised)
  if existing:
   old=arrays(root/OLD/(cid+'_inputs.npz'))
   for k,v in data.items():assert np.allclose(v,old[k],rtol=0,atol=1e-7),(cid,k)
   # Existing group increments were stored in float32; hash their actual bytes.
   data={k:old[k] for k in data}
   forcing_file='../three_ward_surrogate_inputs/'+cid+'_inputs.npz'
  else:np.savez_compressed(out/(cid+'.npz'),**data);forcing_file=cid+'.npz'
  cases.append({'case_id':cid,'parent_id':'CONTROL-LEGACY','family_id':'CONTROL-LEGACY','split':'open_control',
    'independent_parent_candidate':False,'forcing_file':forcing_file,'forcing_sha256':semantic(data),
    'adaptive_companion':True,'existing_completed':existing,'status':'existing_reference_only' if existing else 'forcing_generated_response_absent',
    'parameters':{'rain_peak_mm_h':peak,'stage_amplitude_m':1.5 if raised else 0,'rain_start_s':3600,'rain_duration_s':3601,'shape_class':'legacy_plateau' if peak else 'dry'}})
  signals[cid]=data;norms[cid]=normalized(data['Event Time'],data['Physical Rainfall Rate'],3600,3601)
 catalog={'version':CONFIG['version'],'configuration_sha256':digest(CONFIG),'cases':cases,
   'source_sha256':sources,'protected_existing_sha256':protected,'new_simulations':0,'training_runs':0,
   'label_admission_policy_source':OLD+'/quality_policy.json','response_release':False,
   'independence_limit':'48 unique normalized synthetic realizations under the specified screening rule; not 48 independent observed events or proof of IID sampling.'}
 dump(out/'catalog.json',catalog)
 dump(out/'schema.json',{'grid':{'epsg':6677,'shape_10m':[1012,770],'left':-11360,'top':-32320,'dx_m':10},
  'forcing_keys':{'Event Time':'s since event start; variable-length sorted knot union','Physical Rainfall Rate':'mm/h; uniform physical rain','River Stage Increment By Group':'m; time by seven distinct group codes'},
  'factorization':{'effective_rain':'physical_rate multiplied by Effective Rainfall Weight; nominal cell area, not open-ground area, is the denominator','point_stage':'Boundary Low Stage plus group increment indexed by Boundary Group Index'},
  'roof_rule':'Immediate complete roof transfer to frozen nearest road-connected ground recipients; zero delay or loss; no measured downpipes or underground removal',
  'initial_state':'dry interior inherited from fixed reference; boundary stages supplied separately',
  'variable_length_batch':'exact native knots with Time Valid padding; no temporal coarsening',
  'response_schema':'absent; retain separate reference admission and exact/interval masks when later responses are authorized',
  'normalization':'none fitted; future fit on train parents only','training_release':False})
 runs=[]
 for case in cases:
  if case.get('existing_completed'):continue
  runs.append(run_binding(case,'regular_10m'))
  if case['adaptive_companion']:runs.append(run_binding(case,'adaptive_10_5m'))
 dump(out/'run_manifest.json',{'pending_runs':runs,'completed_new_runs':0,'no_execution_authorization':True})
 dump(out/'partition_manifest.json',{'policy':'All derivatives inherit family split. Test and challenge responses sealed; forcing QC is permitted.',
 'bindings':[{k:c[k] for k in ['case_id','parent_id','family_id','split','forcing_sha256']} for c in cases],
 'release_state':'sealed_unrun','release_token_present':False,'custody_limit':'Application-level guard; local files are not cryptographically isolated.'})
 dump(report/'generation_rejections.json',rejected)
 pairs=[]
 for a,b in itertools.combinations(cases,2):
  na,nb=norms[a['case_id']],norms[b['case_id']]
  if na is None or nb is None:rmse=corr=None;near=False
  else:rmse,corr,near=distance(na,nb)
  pairs.append({'case_a':a['case_id'],'case_b':b['case_id'],'rmse':rmse,'correlation':corr,'near':near,
    'same_family':a['family_id']==b['family_id'],'cross_partition':a['split']!=b['split']})
 write_csv(report/'shape_pair_audit.csv',pairs)
 mass=[]
 for c in cases:
  a=signals[c['case_id']];depth=integral(a['Event Time'],a['Physical Rainfall Rate'])
  ground=depth*geom['Ground Area 10m'].sum()/1000;roof=depth*geom['Roof Assigned Area 10m'].sum()/1000
  expected=depth*geom['Contributing Area 10m'].sum()/1000
  v10=integral(a['Event Time'],a['Physical Rainfall Rate'])*float((geom['Effective Rainfall Weight 10m']*100).sum())/1000
  v5=depth*float((geom['Effective Rainfall Weight 5m']*25).sum())/1000
  vad=depth*float(geom['Adaptive Contributing Area'].sum())/1000
  mass.append({'case_id':c['case_id'],'split':c['split'],'physical_depth_mm':depth,'direct_ground_volume_m3':ground,
   'roof_volume_m3':roof,'expected_volume_m3':expected,'regular_volume_m3':v10,'fine_volume_m3':v5,'adaptive_volume_m3':vad,
   'max_abs_volume_error_m3':max(abs(v10-expected),abs(v5-expected),abs(vad-expected),abs(ground+roof-expected))})
 write_csv(report/'rainfall_volume_audit.csv',mass)
 variables=[{'readable_name':k,'source':'generated physical curve and inherited geometry','is_final_variable':'no','units':u} for k,u in [
 ('Event Time','s'),('Physical Rainfall Rate','mm/h'),('Effective Surface Rainfall Rate','mm/h'),('River Stage Increment By Group','m'),('River Stage','m in inherited working reference'),('Parent Event','1'),('Event Family','1'),('Partition','1')]]
 dump(report/'decisions.json',{'human_decision':'MILI-D-20261015-005','variables':variables,'format_adaptation':'NPZ for geospatial arrays; JSON/CSV for metadata, no Excel/parquet conversion','agent_settings':CONFIG,'is_final_variable':'no'})
 write_csv(report/'variable_list.csv',variables)
 dump(report/'build_summary.json',{'new_parent_realizations':48,'main_train':24,'main_validation':8,'main_sealed_test':8,'challenge':8,'new_control_forcings':2,
 'new_forcings':50,'existing_controls_referenced':4,'pending_regular':50,'pending_adaptive':8,'new_hydrodynamic_runs':0,'training_runs':0,
 'rejected_candidates':len(rejected),'ground_area_m2':float(geom['Ground Area 10m'].sum()),'roof_area_m2':float(geom['Roof Assigned Area 10m'].sum()),
 'total_rained_area_m2':float(geom['Contributing Area 10m'].sum()),'max_algebraic_mass_error_m3':max(r['max_abs_volume_error_m3'] for r in mass),
 'cross_family_near_duplicates':sum(p['near'] and not p['same_family'] for p in pairs),'cross_partition_near_duplicates':sum(p['near'] and p['cross_partition'] for p in pairs)})
 print('Generated factorized inputs and manifests; run independent validation before freezing.',flush=True)

def write_csv(path,rows):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();build(args.root)
