"""Read-only audit of eight existing runs under the reference policy. No solver calls."""
import os
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
from pathlib import Path
import sys,json,csv,hashlib,collections,argparse,time
import numpy as np
import netCDF4,rasterio
from rasterio.transform import Affine
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from reference_label_policy import POLICY,CONTRACT,classify_labels,integrate_rate_at,balance_check,event_admission
OUT='data/processed/three_ward_reference_contract_v1'
REPORT='data/exp/data-preprocessing/three_ward_reference_contract'
BASE='data/processed/three_ward_sfincs_framework'
INPUT='data/processed/three_ward_surrogate_inputs'
LABEL='data/processed/three_ward_common_depth_labels'
CASES=['rain60_low','rain100_low','rain60_raised','dry_raised']

def read_npz(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def arr(v):return np.ma.asarray(v,dtype='f8').filled(np.nan)
def inp(p):return {k.strip():v.strip() for line in Path(p).read_text().splitlines() if '=' in line for k,v in [line.split('=',1)]}
def polyline_sections(p):
 lines=Path(p).read_text().splitlines();out=[];i=0
 while i<len(lines):
  if not lines[i].strip():i+=1;continue
  name=lines[i].strip();n=int(lines[i+1].split()[0]);coords=np.array([[float(x) for x in line.split()[:2]] for line in lines[i+2:i+2+n]])
  if n!=2:raise ValueError('Expected one segment per native boundary face')
  out.append((name,coords));i+=2+n
 return out

def boundary_binding(folder,grid,mask,g,face,graph,old_boundary):
 sections=json.loads((folder/'flux_sections.json').read_text());lines=polyline_sections(folder/'sfincs.crs')
 assert [x['name'] for x in sections]==[name for name,xy in lines]
 points=np.array([xy.mean(0) for name,xy in lines]);vectors=np.array([xy[1]-xy[0] for name,xy in lines]);normal=np.c_[-vectors[:,1],vectors[:,0]]
 lookup={(int(x['row_north']),int(x['col_east'])):int(x['water_group']) for x in old_boundary}
 if grid=='corrected_10m':
  expected=[];outward=[];groups=[]
  for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
   select=(mask==1)&(np.roll(mask,(-dr,-dc),(0,1))==2)
   if dr==1:select[-1]=False
   if dr==-1:select[0]=False
   if dc==1:select[:,-1]=False
   if dc==-1:select[:,0]=False
   rr,cc=np.where(select)
   expected.extend(np.c_[-11360+(cc+.5+.5*dc)*10,-32320-(rr+.5+.5*dr)*10].tolist());outward.extend([[dc,-dr]]*len(rr))
   groups.extend(lookup[(int((r+dr)//2),int((c+dc)//2))] for r,c in zip(rr,cc))
  expected=np.array(expected);outward=np.array(outward);groups=np.array(groups)
  distance,index=cKDTree(expected).query(points)
  ok=len(points)==len(expected) and len(np.unique(index))==len(expected) and float(distance.max())<=.001 and np.all(np.sum(normal*outward[index],axis=1)>0)
  groups=groups[index]
 else:
  a=face['cell_a'];b=face['cell_b'];m=g['mask'];sel=np.flatnonzero(((m[a]==1)&(m[b]==2))|((m[b]==1)&(m[a]==2)))
  ids=np.array([s['face'] for s in sections]);di=face['direction'][ids];length=face['length_pixels'][ids]*.5
  x=-11360+face['col_fine'][ids]*.5;y=-32320-face['row_fine'][ids]*.5
  centres=np.c_[np.where(di==1,x+length/2,x),np.where(di==1,y-.5,y-length/2)]
  bi=np.where(m[a[ids]]==2,a[ids],b[ids]);land=np.where(m[a[ids]]==1,a[ids],b[ids]);outward=np.c_[g['x'][bi]-g['x'][land],g['y'][bi]-g['y'][land]]
  groups=np.array([lookup[(int(g['row_fine'][j]//40),int(g['col_fine'][j]//40))] for j in bi])
  ok=np.array_equal(np.sort(ids),sel) and len(np.unique(ids))==len(sel) and np.max(abs(points-centres))<=.001 and np.all(np.sum(normal*outward,axis=1)>0)
 return bool(ok),[{'section':s['name'],'boundary_group_code':int(code)} for s,code in zip(sections,groups)]

def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 if (out/'freeze.json').exists():raise FileExistsError('Frozen contract audit exists; validate without overwriting or create a new version')
 dump(out/'label_policy.json',POLICY);dump(out/'output_contract.json',CONTRACT)
 support=read_npz(root/LABEL/'common_support_10m.npz');area=support['Ground Area'];valid=support['Eligible Ground Mask'];ward=support['Ward Code']
 graph=read_npz(root/INPUT/'surface_connectivity_graph.npz');g=read_npz(root/'data/processed/three_ward_adaptive_connectivity/adaptive_cell_geometry.npz');face=read_npz(root/'data/processed/three_ward_adaptive_connectivity/adaptive_face_geometry.npz')
 with (root/'data/processed/three_ward_sfincs_scenarios/boundary_points.csv').open(encoding='utf-8-sig') as f:old_boundary=list(csv.DictReader(f))
 protected={str(p.relative_to(root)):sha(p) for folder in [root/INPUT,root/LABEL,root/'data/processed/three_ward_minimum_scenarios_v1'] for p in sorted(folder.iterdir()) if p.is_file()}
 sources={};fingerprints={};cache={};rows=[];admissions={};scoperows=[];wetrows=[];pairrows=[];bindings={};budgetrows=[];union=np.zeros(valid.shape,bool);oldbudgets={x['run']:x for x in json.loads((root/'data/exp/data-briefing/three_ward_adaptive_sensitivity/water_budgets.json').read_text())}
 def bind(p):
  stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  sources[str(p.relative_to(root))]=cache[key];fingerprints[str(p.relative_to(root))]=[stat.st_size,stat.st_mtime_ns]
  return cache[key]
 for p in [root/'data/processed/three_ward_sfincs_scenarios/solver_provenance'/f for f in ['sfincs_ncoutput.F90','sfincs_crosssections.f90','sfincs_continuity.f90','sfincs_lib.f90','manifest.json']]:bind(p)
 scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3),'Cross-ward contributing area':valid&(ward==0)}
 for case in CASES:
  case_diagnostics={}
  for grid in ['corrected_10m','adaptive_10_5m']:
   tag=case+'_'+grid;folder=root/BASE/('adaptive_connectivity_comparison' if case=='rain60_low' else 'adaptive_sensitivity')/tag
   print('READ EXISTING',tag,flush=True);start=time.monotonic()
   ex=json.loads((folder/'execution.json').read_text());cfg=inp(folder/'sfincs.inp');scenario=json.loads((folder/'scenario_contract.json').read_text())
   identity=all(bind(folder/k)==h for k,h in ex['input_sha256'].items())
   for f in ['execution.json','scenario_contract.json','flux_sections.json','sfincs_map.nc','sfincs_his.nc']:bind(folder/f)
   fine=grid=='adaptive_10_5m'
   with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
    ts=arr(d['time'][:]);raw_mask=arr(d['msk'][:]);mask0=np.where(np.isfinite(raw_mask),raw_mask,0).astype('uint8');mask=mask0 if fine else np.flipud(mask0);inside=mask==1
    if fine:
     geometry=bool(np.array_equal(mask,g['mask']) and np.max(abs(arr(d['mesh2d_face_x'][:])-g['x']))<.001 and np.max(abs(arr(d['mesh2d_face_y'][:])-g['y']))<.001)
    else:
     xx=-11360+(np.arange(770)+.5)*10;yy=-32320-(np.arange(1012)+.5)*10
     geometry=bool(np.array_equal(mask==1,valid) and np.max(abs(arr(d['x'][:])-xx[None,:]))<.001 and np.max(abs(np.flipud(arr(d['y'][:]))-yy[:,None]))<.001)
    def native(k,i=None):
     v=arr(d[k][:] if i is None else d[k][i]);return v if fine else np.flipud(v)
    required=CONTRACT['required_peak_profile']['map_variables'];fields=all(k in d.variables for k in required)
    fields &= all(d[k].units==u for k,u in [('zb','m'),('zs','m'),('zsmax','m'),('subgrid_volume','m3'),('h','m'),('u','m s-1'),('v','m s-1'),('cumprcp','m')])
    peak=native('zsmax',0);zmin=native('zb');n=len(ts);storage=[];negative_count=[];negative_total=[];debt=np.zeros(mask.shape,bool);minimum=np.inf;state_finite=True;h_semantic=0.;peak_violation=0.;unproved_censor=0
    for k in range(n):
     vol=native('subgrid_volume',k);lev=native('zs',k);h=native('h',k)
     state_finite &= bool(np.all(np.isfinite(vol[inside])))
     storage.append(float(vol[inside].sum()));negative_count.append(int(np.sum(vol[inside]<-.001)));negative_total.append(float(-np.minimum(vol[inside],0).sum()))
     minimum=min(minimum,float(vol[inside].min()));debt|=inside&(vol<-.001)
     both=inside&np.isfinite(lev)&np.isfinite(h);h_semantic=max(h_semantic,float(np.max(abs(h[both]-(lev[both]-zmin[both])))) if both.any() else 0)
     seen=inside&np.isfinite(lev)&np.isfinite(peak)
     if seen.any():peak_violation=max(peak_violation,float(np.max(lev[seen]-peak[seen])))
     unproved_censor+=int(np.count_nonzero(inside&~np.isfinite(lev)&~np.isfinite(zmin)))
     for key in ['u','v']:
      vel=native(key,k);state_finite &= bool(np.all(np.isfinite(vel[inside])))
    native_area=g['size_m']**2 if fine else np.full(mask.shape,100.)
    reported_rain=float(np.sum(native('cumprcp',-1)[inside]*native_area[inside]))
    tm=arr(d['timemax'][:]);map_status=float(arr(d['status'][:])[0]);storage=np.array(storage)
    actual_state_clock_present='actual_state_time' in d.variables
    peak_window=bool(len(tm)==1 and abs(tm[0]-14400)<=.001)
   boundary,section_binding=boundary_binding(folder,grid,mask,g,face,graph,old_boundary);bindings[tag]=section_binding
   # Integrate actual effective-input bytes, including valid zeros in legacy files.
   with netCDF4.Dataset(folder/'roof_surface_precip.nc') as d:
    v=d['Precipitation'];v.set_auto_mask(False);rt=arr(d['time'][:])*60;unit=getattr(v,'units',getattr(v,'unit',''))
    rates=[]
    for k in range(len(rt)):
     a=np.asarray(v[k],dtype='f8');assert np.all(np.isfinite(a)) and np.min(a)>=0
     rates.append(float(a[1:-1,1:-1].sum())*(25 if fine else 100)/3.6e6)
    rain=integrate_rate_at(rt,np.array(rates),ts)
   forcing_target=float(scenario['original_rain_interior_volume_m3']);rain_rel=abs(rain[-1]-forcing_target)/max(forcing_target,1)
   # Stream complete boundary-face histories: never replace missing flux by zero.
   with netCDF4.Dataset(folder/'sfincs_his.nc') as d:
    qt=arr(d['time'][:]);names=[str(x).strip(' \x00') for x in netCDF4.chartostring(d['crosssection_name'][:])];history_status=float(arr(d['status'][:])[0]);actual_history_clock_present='actual_history_time' in d.variables
    boundary &= names==[s['section'] for s in section_binding] and len(names)==len(set(names)) and d['crosssection_discharge'].units=='m3 s-1'
    qout=np.zeros(len(qt));qin=qout.copy();qvalid=True;group_codes=np.array([s['boundary_group_code'] for s in section_binding]);group_net=np.zeros((len(qt),7));codes=graph['Boundary Group Codes']
    for i in range(0,len(qt),1000):
     q=arr(d['crosssection_discharge'][i:i+1000]);qvalid &= bool(np.all(np.isfinite(q)))
     qout[i:i+len(q)]=np.maximum(q,0).sum(1);qin[i:i+len(q)]=np.maximum(-q,0).sum(1)
     for j,code in enumerate(codes):group_net[i:i+len(q),j]=q[:,group_codes==code].sum(1)
   dt=np.diff(qt);cumulative_out=np.r_[0,np.cumsum((qout[:-1]+qout[1:])*.5*dt)];inc=np.r_[0,np.cumsum((qin[:-1]+qin[1:])*.5*dt)]
   right=np.r_[0,np.cumsum((qout[1:]-qin[1:])*dt)];trap=cumulative_out-inc
   bo=np.interp(ts,qt,cumulative_out);bi=np.interp(ts,qt,inc);bal=balance_check(ts,storage,rain,bo,bi)
   quadrature_delta=float(abs(trap[-1]-right[-1]));quadrature_ok=quadrature_delta<=max(1.,.001*bal['scale'][-1])
   with (report/(tag+'_budget_checkpoints.csv')).open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['time_s','signed_storage_m3','rain_m3','boundary_out_m3','boundary_in_m3','residual_m3','scale_m3','prefix_gate_m3','negative_native_cells','negative_bookkeeping_m3'])
    w.writerows(zip(ts,storage,rain,bo,bi,bal['residual'],bal['scale'],bal['prefix_gate'],negative_count,negative_total))
   time_ok=bool(ts[0]==qt[0]==0 and abs(ts[-1]-14400)<=.001 and abs(qt[-1]-14400)<=.001 and np.all(np.diff(ts)>0) and np.max(np.diff(ts))<=300.001 and np.all(np.diff(qt)>0) and np.max(np.diff(qt))<=.401 and peak_window)
   completion=bool(ex['returncode']==0 and ex['completion_marker'] and map_status==history_status==0)
   censor=bool(float(cfg.get('huthresh','nan'))==.01 and np.all(np.isfinite(zmin[inside])) and h_semantic<=1e-5 and peak_violation<=1e-5 and unproved_censor==0)
   fields=bool(fields and state_finite and qvalid)
   baseline_terms=bool(float(cfg.get('qinf','nan'))==0 and all(cfg.get(k,'none').strip("'\"").lower()=='none' for k in ['disfile','srcfile','infiltrationfile','evapfile','precipfile']))
   budget=bool(bal['endpoint_pass'] and bal['prefix_pass'] and quadrature_ok and baseline_terms)
   clock_alignment=False # Existing Galibier outputs do not store actual integration clocks; presence alone would require value/provenance verification.
   checks={'completion':completion,'clock_alignment':clock_alignment,'identity':bool(identity),'geometry':geometry,'fields':fields,'time':time_ok,'boundary':bool(boundary),'rain':bool(rain_rel<=1e-6 and unit in ['mm.hr-1','mm h-1']),'budget':budget,'censoring':censor}
   # Remap negative diagnostics to the same common10m ground support.
   if fine:
    debt5=np.zeros((2024,1540),bool)
    for size in [5,10]:
     j=np.flatnonzero(g['size_m']==size);rr=g['row_fine'][j]//10;cc=g['col_fine'][j]//10
     for dr in range(size//5):
      for dc in range(size//5):debt5[rr+dr,cc+dc]=debt[j]
    common_debt=debt5.reshape(1012,2,770,2).any((1,3))
   else:common_debt=debt
   labels=read_npz(root/LABEL/(tag+'.npz'));debt_equal=np.array_equal(common_debt,labels['Sampled Negative Volume Flag'])
   checks['identity']=checks['identity'] and bool(debt_equal)
   diagnostics={'checks':checks,'nominal_endpoint_budget_pass':bool(bal['endpoint_pass']),'nominal_prefix_budget_pass':bool(bal['prefix_pass']),'quadrature_pass':bool(quadrature_ok),
    'actual_state_clock_present':actual_state_clock_present,'actual_history_clock_present':actual_history_clock_present,'time_semantics':'Pinned sfincs_lib advances t and state then writes state using nominal tout; exact state integration clock not saved',
    'end_residual_m3':float(bal['residual'][-1]),'end_relative_residual':bal['end_relative_residual'],
    'max_prefix_relative_residual':bal['max_prefix_relative_residual'],'endpoint_gate_m3':float(bal['endpoint_gate']),
    'rain_volume_m3':float(rain[-1]),'rain_relative_error':float(rain_rel),'boundary_gross_out_m3':float(cumulative_out[-1]),'boundary_gross_in_m3':float(inc[-1]),
    'right_vs_trapezoid_difference_m3':quadrature_delta,'cumprcp_roundoff_m3':reported_rain-float(rain[-1]),'min_native_bookkeeping_m3':minimum,
    'sampled_negative_native_cells':int(debt.sum()),'max_sampled_negative_total_m3':float(max(negative_total)),'negative_observability':'300s snapshots only',
    'barrier_overtopped_fine_area_m2_from_prior_audit':oldbudgets[tag]['building_overtopped_fine_area_m2'],
    'h_semantic_max_error_m':h_semantic,'native_peak_below_snapshot_max_error_m':max(peak_violation,0),
    'history_max_gap_s':float(np.max(np.diff(qt))),'group_codes':codes.tolist(),'boundary_group_net_out_m3':{str(code):float(np.sum((group_net[:-1,j]+group_net[1:,j])*.5*dt)) for j,code in enumerate(codes)},
    'continuous_state_nonnegativity_proved':False,'dynamic_face_flux_profile_complete':False,'new_execution':False}
   admissions[tag]=diagnostics;case_diagnostics[grid]=(labels,common_debt,checks)
   rows.append({'run':tag,'role':'adaptive_diagnostic' if fine else 'primary_reference','required_checks_pass':all(checks.values()),'failed_checks':','.join(k for k,v in checks.items() if not v),
    'nominal_endpoint_budget_pass':diagnostics['nominal_endpoint_budget_pass'],'nominal_prefix_budget_pass':diagnostics['nominal_prefix_budget_pass'],'actual_clock_verified':False,'end_relative_residual':diagnostics['end_relative_residual'],'end_residual_m3':diagnostics['end_residual_m3'],'max_prefix_relative_residual':diagnostics['max_prefix_relative_residual'],'quadrature_difference_m3':quadrature_delta,
    'negative_native_cells':int(debt.sum()),'strict_all_step_nonnegative':False,'dynamic_path_labels_admitted':False,'training_release':False})
   print('AUDITED',tag,'failed',[k for k,v in checks.items() if not v], 'end_residual',round(bal['end_relative_residual'],7),'elapsed_s',round(time.monotonic()-start,1),flush=True)
  lab,negative,checks=case_diagnostics['corrected_10m'];old=read_npz(root/INPUT/(case+'_label_admission.npz'));barrier=old['Assumed Barrier Overtopping Flag']
  masks=classify_labels(lab['Peak Mean Depth Lower'],lab['Peak Mean Depth Upper'],valid,negative,barrier,interval_source_known=checks['censoring'])
  state=event_admission(checks,masks);admissions[case+'_corrected_10m']['label_admission']=state
  unchanged=all(np.array_equal(masks[k],old[k]) for k in ['Exact Label Mask','Interval Label Mask','Quarantine Mask','Wide Interval Mask'])
  admissions[case+'_corrected_10m']['previous_masks_preserved']=bool(unchanged)
  np.savez_compressed(out/(case+'_reference_masks.npz'),**masks,**{'Label Lower':lab['Peak Mean Depth Lower'],'Label Upper':lab['Peak Mean Depth Upper'],'Eligible Ground Mask':valid,'Sampled Negative Volume Flag':negative,'Assumed Barrier Overtopping Flag':barrier})
  flags=np.where(~valid,0,np.where(masks['Quarantine Mask'],3,np.where(masks['Wide Interval Mask'],4,np.where(masks['Interval Label Mask'],2,1)))).astype('uint8');union|=masks['Quarantine Mask']
  tf=Affine(*support['Affine Transform'])
  with rasterio.open(out/(case+'_reference_class_10m.tif'),'w',driver='GTiff',height=1012,width=770,count=1,dtype='uint8',crs=6677,transform=tf,nodata=0,compress='deflate') as d:
   d.write(flags,1);d.set_band_description(1,'0 outside;1 exact;2 interval;3 quarantine;4 wide interval');d.update_tags(interpretation='Numerical admission masks; snapshot-screened, not physical validation',policy=POLICY['version'])
  for scope,sel in scopes.items():
   scoperows.append({'case':case,'scope':scope,'eligible_ground_km2':float(area[sel].sum()/1e6),**{k:float(area[sel&masks[v]].sum()/1e6) for k,v in [('exact_km2','Exact Label Mask'),('interval_km2','Interval Label Mask'),('quarantine_km2','Quarantine Mask'),('wide_km2','Wide Interval Mask')]},
    'quarantine_percent':float(area[sel&masks['Quarantine Mask']].sum()/area[sel].sum()*100),'run_state':state['state']})
   for threshold in POLICY['peak_thresholds_m']:
    lo=lab['Peak Mean Depth Lower'];up=lab['Peak Mean Depth Upper'];wet=sel&(lo>threshold);possible=sel&(up>threshold);qwet=wet&masks['Quarantine Mask'];amb=sel&(lo<=threshold)&(up>threshold)
    wetrows.append({'case':case,'scope':scope,'threshold_m':threshold,'certain_wet_km2':float(area[wet].sum()/1e6),'possible_wet_km2':float(area[possible].sum()/1e6),
     'ambiguous_km2':float(area[amb].sum()/1e6),'quarantined_certain_wet_km2':float(area[qwet].sum()/1e6),'quarantined_share_certain_wet_percent':float(area[qwet].sum()/area[wet].sum()*100) if area[wet].sum()>0 else None,
     'admitted_certain_wet_km2':float(area[wet&masks['Admitted Mask']].sum()/1e6)})
  adaptive=case_diagnostics['adaptive_10_5m'][0];width=adaptive['Peak Mean Depth Upper']-adaptive['Peak Mean Depth Lower'];pairq=masks['Quarantine Mask']|adaptive['Sampled Negative Volume Flag']
  # Adaptive barrier contact is not packaged at common-cell resolution; no clean paired regression admission is asserted.
  pairrows.append({'case':case,'adaptive_width_gt_1cm_km2':float(area[valid&(width>.010001)].sum()/1e6),'direct_pair_quarantine_lower_bound_km2':float(area[valid&pairq].sum()/1e6),'paired_adaptive_mask_complete':False,'reason':'Adaptive barrier contact common-cell mask is not yet packaged; adaptive results remain diagnostics, not alternate primary truth.'})
 np.savez_compressed(out/'development_quarantine_union.npz',**{'Quarantine Union':union,'Eligible Ground Mask':valid,'Ground Area':area,'Ward Code':ward})
 csvwrite(report/'run_admission.csv',rows);csvwrite(report/'label_admission_by_scope.csv',scoperows);csvwrite(report/'wet_quarantine_coverage.csv',wetrows);csvwrite(report/'adaptive_diagnostic_status.csv',pairrows)
 dump(report/'run_diagnostics.json',admissions);dump(out/'boundary_section_binding.json',bindings)
 for p in [root/'src/preprocessing/reference_label_policy.py',Path(__file__)]:bind(p)
 before_after=all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==fp for p,fp in fingerprints.items())
 protected_ok=all(sha(root/p)==h for p,h in protected.items())
 summary={'existing_runs_audited':8,'primary_controls':4,'regular_required_checks_passed':sum(all(v['checks'].values()) for k,v in admissions.items() if k.endswith('_corrected_10m')),
 'adaptive_required_checks_passed':sum(all(v['checks'].values()) for k,v in admissions.items() if k.endswith('_adaptive_10_5m')),
 'nominal_endpoint_passes':sum(v['nominal_endpoint_budget_pass'] for v in admissions.values()),'nominal_prefix_passes':sum(v['nominal_prefix_budget_pass'] for v in admissions.values()),
 'event_holds':[k for k,v in admissions.items() if not all(v['checks'].values())],
 'primary_masks_unchanged':all(v['previous_masks_preserved'] for k,v in admissions.items() if k.endswith('_corrected_10m')),
 'development_quarantine_union_km2':float(area[union].sum()/1e6),'quarantine_scope':'Direct flagged cells only; no proof of bounded error propagation',
 'sources_unchanged':before_after,'old_labels_inputs_and_frozen_forcings_unchanged':protected_ok,
 'policy_frozen_for_new_reference_screen':True,'formal_reference_training_release':False,'dynamic_profile_ready':False,'new_simulations':0,'training':0}
 dump(report/'summary.json',summary);dump(report/'provenance.json',{'sources_sha256':sources,'protected_sha256':protected,'source_stat_fingerprints':fingerprints,'new_simulations':0,'training':0})
 dump(report/'decisions.json',{'human_decision':'MILI-D-20261015-006','numerical_settings':'Agent engineering defaults inherited or explicitly versioned; not universal standards','label_policy':POLICY,'output_contract':CONTRACT,'format':'NPZ/GeoTIFF for geospatial arrays; JSON/CSV audits; no package installation','is_final_variable':'no'})
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT);args=a.parse_args();main(args.root)
