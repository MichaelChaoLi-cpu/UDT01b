"""Independent forcing, lineage and file-reader-equivalent checks; never executes SFINCS."""
from pathlib import Path
import sys, json, argparse, collections, itertools
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
import generate_three_ward_minimum_scenarios as gen
import load_three_ward_minimum_scenarios as api

def validate(root=ROOT,freeze=False):
 root=Path(root);out=root/gen.LIB;report=root/gen.REPORT;cat=api.catalog(root);cases=cat['cases'];g=api.load_geometry(root)
 checks=[];details=[]
 def check(name,ok,evidence=None):
  checks.append({'check':name,'pass':bool(ok),'evidence':evidence})
  print(('PASS ' if ok else 'FAIL ')+name,flush=True)
 data={c['case_id']:api.load_forcing(c['case_id'],root) for c in cases}
 check('counts_and_reserved_partitions',collections.Counter(c['split'] for c in cases)=={'train':24,'validation':8,'sealed_test':8,'challenge':8,'open_control':6})
 check('unique_case_ids_and_48_unique_parent_families',len(data)==54 and len({c['family_id'] for c in cases if c['independent_parent_candidate']})==48)
 plan=json.loads((root/gen.PLAN/'reserved-parent-allocations.json').read_text())
 byid={c['case_id']:c for c in cases}
 check('reserved_allocation_binding',all(byid[p['parent_id']]['split']==p['split'] and byid[p['parent_id']]['adaptive_companion']==p['adaptive_companion'] for p in plan['parents']))
 check('balanced_main_shape_classes',all(collections.Counter(c['parameters']['shape_class'] for c in cases if c['split']==s)=={k:n for k in ['early','central','late','multiple_peak']} for s,n in [('train',6),('validation',2),('sealed_test',2)]))
 check('forcing_semantic_hashes_match_actual_arrays',all(gen.semantic(data[c['case_id']])==c['forcing_sha256'] for c in cases))
 check('full_forcing_semantic_hashes_unique',len({c['forcing_sha256'] for c in cases})==54)
 time_valid=True;integrals=True;envelopes=True;repro=True;norms={};max_dense_error=0
 for c in cases:
  d=data[c['case_id']];t=d['Event Time'];r=d['Physical Rainfall Rate'];s=d['River Stage Increment By Group'];p=c['parameters']
  time_valid &= t[0]==0 and t[-1]==14400 and np.all(np.diff(t)>0) and np.all(np.isfinite(r)) and np.min(r)>=0 and r[0]==r[-1]==0 and s.shape==(len(t),7) and np.all(np.isfinite(s)) and np.min(s)>=0 and np.all(s[[0,-1]]==0)
  dense=np.interp(np.arange(14401),t,r);dense_depth=float((dense[:-1]+dense[1:]).sum()/7200)
  err=abs(dense_depth-gen.integral(t,r));max_dense_error=max(max_dense_error,err);integrals &= err<1e-10
  norms[c['case_id']]=gen.normalized(t,r,p['rain_start_s'],p['rain_duration_s'])
  if not c['independent_parent_candidate']:continue
  peak=float(np.max(r));dur=p['rain_duration_s']/60;amp=float(np.max(s));lag=p['scheduled_peak_lag_s']/60;axis=p['challenge_axis']
  if c['split']!='challenge':
   envelopes &= 20<=peak<=100 and 30<=dur<=90 and 0<=amp<=1.5 and -30<=lag<=30 and not(peak>=90 and amp>=1.2)
  elif axis=='withheld_compound':envelopes &= 90<=peak<=100 and 1.2<=amp<=1.5 and 30<=dur<=90 and -30<=lag<=30
  elif axis=='rainfall_magnitude':envelopes &= 110<=peak<=120 and 30<=dur<=90 and -30<=lag<=30
  elif axis=='rainfall_duration':envelopes &= 105<=dur<=120 and 20<=peak<=100 and -30<=lag<=30
  elif axis=='rain_stage_lag':envelopes &= abs(lag)==60 and 20<=peak<=100 and 30<=dur<=90 and amp>0
  envelopes &= p['stage_peak_time_s']-1800>=0 and p['stage_peak_time_s']+1800<=14400
  envelopes &= p['rain_peak_time_s']==t[np.argmax(r)] and abs(gen.integral(t,s[:,0])*3600-amp*1800)<1e-8
  if amp>0:envelopes &= t[np.argmax(s[:,0])]-t[np.argmax(r)]==p['scheduled_peak_lag_s']
  a=c['source_allocation'];rebuilt,reason=gen.candidate(p['slot'],p['accepted_attempt'],a['shape_class'],a.get('challenge_axis'),int(c['case_id'][-1]))
  repro &= reason is None and gen.semantic(rebuilt[0])==c['forcing_sha256']
 check('time_knots_nonnegative_curves_and_horizon',time_valid)
 check('one_second_independent_rain_integral',integrals,{'maximum_depth_error_mm':max_dense_error})
 check('main_challenge_envelopes_and_peak_lag_semantics',envelopes)
 check('all_48_seed_and_attempt_reproductions',repro)
 allpairs=[]
 for a,b in itertools.combinations(cases,2):
  x=norms[a['case_id']];y=norms[b['case_id']]
  if x is None or y is None:continue
  rm,co,near=gen.distance(x,y)
  allpairs.append((a,b,rm,co,near))
 illegal=[(a['case_id'],b['case_id']) for a,b,rm,co,near in allpairs if near and a['family_id']!=b['family_id']]
 check('no_cross_family_normalized_shape_duplicates',not illegal,{'offenders':illegal,'rule':{k:gen.CONFIG[k] for k in ['near_rmse','near_joint_rmse','near_correlation']}})
 clone=gen.normalized(np.array([0,2,5,10.])*900+500,np.array([0,2,1,0.])*7,500,9000)
 original=gen.normalized(np.array([0,2,5,10.]),np.array([0,2,1,0.]),0,10)
 check('scaled_and_time_stretched_clone_rejected',gen.distance(clone,original)[2])
 check('control_rain_integrals',all(abs(gen.integral(data[cid]['Event Time'],data[cid]['Physical Rainfall Rate'])-v)<1e-10 for cid,v in [('dry_low_current_geometry',0),('rain100_raised_current_geometry',100),('rain60_low',60),('rain100_low',100)]))
 a10=g['Contributing Area 10m'];a5=g['Contributing Area 5m'];op10=g['Ground Area 10m'];op5=g['Ground Area 5m']
 agg=lambda a:a.reshape(1012,2,770,2).sum((1,3))
 check('nested_ground_roof_contributing_areas',np.array_equal(agg(a5),a10) and np.array_equal(agg(op5),op10) and np.array_equal(agg(g['Roof Assigned Area 5m']),g['Roof Assigned Area 10m']) and np.min(g['Roof Assigned Area 5m'])>=0)
 check('fixed_area_contract',a10.sum()==38976800 and op10.sum()==25936197 and g['Roof Assigned Area 10m'].sum()==13040603 and g['Adaptive Contributing Area'].sum()==38976800)
 max_mass=0
 for cid,d in data.items():
  depth=gen.integral(d['Event Time'],d['Physical Rainfall Rate'])
  expected=depth*38976800/1000
  vals=[depth*x/1000 for x in [float(a10.sum()),float(a5.sum()),float(g['Adaptive Contributing Area'].sum()),float(op10.sum()+g['Roof Assigned Area 10m'].sum())]]
  max_mass=max(max_mass,max(abs(v-expected) for v in vals))
 check('all_54_factorized_rain_roof_volume_budgets',max_mass<1e-8,{'maximum_volume_error_m3':max_mass})
 check('seven_boundary_groups_preserved',g['Boundary Low Stage'].shape==(10488,) and len(g['Boundary Group Codes'])==7 and len(set(g['Boundary Group Codes'].tolist()))==7 and set(g['Boundary Group Index'].tolist())==set(range(7)))
 runs=json.loads((out/'run_manifest.json').read_text())['pending_runs']
 check('58_pending_descendants_inherit_family_and_split',len(runs)==58 and sum(r['grid']=='adaptive_10_5m' for r in runs)==8 and all(r['status']=='unrun' and r['response_path'] is None and all(r[k]==byid[r['case_id']][k] for k in ['parent_id','family_id','split','forcing_sha256']) for r in runs))
 check('adaptive_selection_excludes_sealed_responses',all(r['split'] in ['train','validation','open_control'] for r in runs if r['grid']=='adaptive_10_5m'))
 def rejects(fn,kind):
  try:fn()
  except kind:return True
  return False
 check('split_override_guard',rejects(lambda:api.bind_descendant('train-01','adaptive_10_5m',requested_split='sealed_test',root=root),ValueError))
 check('sealed_response_and_challenge_guards',all(rejects(lambda c=c:api.response_access(c,'train',root),PermissionError) for c in ['sealed_test-01','challenge-withheld_compound-1']))
 check('training_and_validation_response_release_closed',all(rejects(lambda c=c,p=p:api.response_access(c,p,root),PermissionError) for c,p in [('train-01','train'),('validation-01','validation')]))
 batch=api.padded_forcing_batch(['train-01','rain60_low','challenge-rainfall_duration-1'],root)
 check('variable_knot_batch_preserves_legacy_one_second_ramps',all(np.array_equal(batch['Event Time'][i,batch['Time Valid'][i]],data[c]['Event Time']) for i,c in enumerate(['train-01','rain60_low','challenge-rainfall_duration-1'])))
 # Reader-equivalent file preflight: files only; explicitly not official engine execution.
 geo=gen.arrays(root/'data/processed/three_ward_adaptive_connectivity/adaptive_cell_geometry.npz')
 sample_specs=[('train-01','regular_10m'),('challenge-rainfall_duration-1','regular_10m'),('challenge-rain_stage_lag-1','regular_10m'),('dry_low_current_geometry','regular_10m'),('rain100_raised_current_geometry','regular_10m'),('rain100_raised_current_geometry','adaptive_10_5m')]
 export_ok=True;max_rel=0;max_cell=0
 for cid,grid in sample_specs:
  dest=report/'forcing_file_preflight'/(cid+'__'+grid)
  if not dest.exists():api.export_galibier(cid,dest,grid,root)
  d=data[cid];t=d['Event Time'];rates=d['Physical Rainfall Rate'];fine=grid=='adaptive_10_5m';dx=5 if fine else 10
  if fine:
   inside=geo['mask']==1;pos=np.column_stack([geo['y'][inside],geo['x'][inside]]);areas=geo['size_m'][inside]**2;contrib=geo['contributing_area_m2'][inside]
  else:
   rr,cc=np.where(a10>0);pos=np.column_stack([-32320-(rr+.5)*10,-11360+(cc+.5)*10]);areas=np.full(len(rr),100.);contrib=a10[rr,cc]
  volumes=[]
  with Dataset(dest/'roof_surface_precip.nc') as nc:
   # Check masked reads too: zero must stay a valid number.
   v=nc.variables['Precipitation'];x=nc['x'][:].data;y=nc['y'][:].data
   export_ok &= np.max(abs(nc['time'][:].data*60-t))<1e-6 and v._FillValue!=0 and v.units=='mm h-1'
   expected_x=-11360-dx+np.arange(len(x))*dx;expected_y=-42440-dx+np.arange(len(y))*dx
   export_ok &= np.array_equal(x,expected_x) and np.array_equal(y,expected_y)
   for i,rate in enumerate(rates):
    frame=v[i];export_ok &= not np.any(np.ma.getmaskarray(frame)) and np.all(np.isfinite(frame)) and np.min(frame)>=0
    raw=np.asarray(frame,dtype='f8')
    # The source reader adds halfcell before bilinear interpolation.
    interp=RegularGridInterpolator((y+dx*.5,x+dx*.5),raw,bounds_error=True)
    native=interp(pos);expected_rate=rate*contrib/areas
    max_cell=max(max_cell,float(np.max(abs(native-expected_rate))))
    volumes.append(float(np.sum(native*areas)/1000))
   computed=float(np.sum(np.diff(t)*(np.array(volumes[:-1])+volumes[1:])*.5)/3600)
  expected=gen.integral(t,rates)*38976800/1000;relative=abs(computed-expected)/max(expected,1)
  max_rel=max(max_rel,relative);export_ok &= relative<=1e-6
  stages=np.loadtxt(dest/'sfincs.bzs');expected_stages=g['Boundary Low Stage'][None]+d['River Stage Increment By Group'][:,g['Boundary Group Index']]
  export_ok &= np.array_equal(stages[:,0],t) and np.max(abs(stages[:,1:]-expected_stages))<1e-9
  src=root/'data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison/rain60_low_corrected_10m/sfincs.bnd'
  export_ok &= gen.sha(src)==gen.sha(dest/'sfincs.bnd')
  details.append({'case_id':cid,'grid':grid,'expected_volume_m3':expected,'readback_volume_m3':computed,'relative_error':relative})
 check('six_Galibier_file_preflights_no_engine_execution',export_ok,{'max_relative_volume_error':max_rel,'max_native_rate_roundoff_mm_h':max_cell,'sample_count':len(sample_specs),'official_engine_executed':False})
 check('source_files_unchanged',all(gen.sha(root/p)==h for p,h in cat['source_sha256'].items()))
 check('existing_inputs_and_labels_unchanged',all(gen.sha(root/p)==h for p,h in cat['protected_existing_sha256'].items()))
 check('no_new_response_products_or_release',not list(out.rglob('sfincs_map.nc')) and not list(out.rglob('sfincs_his.nc')) and not (out/'sealed_test_release.json').exists() and cat['new_simulations']==cat['training_runs']==0)
 result={'all_passed':all(c['pass'] for c in checks),'checks':checks,'file_preflights':details,'official_engine_executed':False,'training_executed':False,'new_hydrodynamic_runs':0}
 gen.dump(report/'validation.json',result)
 if freeze and result['all_passed']:
  hashes={str(p.relative_to(out)):gen.sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='freeze.json'}
  code={str(p.relative_to(root)):gen.sha(p) for p in [root/'src/preprocessing/generate_three_ward_minimum_scenarios.py',root/'src/preprocessing/load_three_ward_minimum_scenarios.py',Path(__file__)]}
  record={'version':gen.CONFIG['version'],'state':'forcing_catalog_frozen_responses_unrun','files_sha256':hashes,'code_sha256':code,
    'validation_sha256':gen.sha(report/'validation.json'),'training_release':False,'test_response_release':False}
  if (out/'freeze.json').exists():
   existing=json.loads((out/'freeze.json').read_text())
   if existing!=record:raise ValueError('Frozen library, code or validation receipt changed')
  else:gen.dump(out/'freeze.json',record)
 print(json.dumps({'all_passed':result['all_passed'],'checks':len(checks),'new_simulations':0,'training':0}))
 return result['all_passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--freeze',action='store_true');a=p.parse_args();sys.exit(0 if validate(a.root,a.freeze) else 1)
