#!/usr/bin/env python3
"""Prepare existing common10m inputs, typed surface graph and supervision masks.
No solver, training, new forcing generation, network access or raw/source edits.
Requires the existing numpy/numba/rasterio/netCDF4/scipy environment.
"""
import os
os.environ['NUMBA_NUM_THREADS']='4'
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
from contextlib import ExitStack
from collections import Counter
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
import numpy as np
import rasterio,netCDF4
from rasterio.windows import Window
from numba import njit,prange
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2]
CASES=['rain60_low','rain100_low','rain60_raised','dry_raised']
NR,NC=1012,770

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def array(v):return np.ma.filled(np.ma.asarray(v,dtype='float32'),np.nan)
def integrate(y,t):return np.sum((y[1:]+y[:-1])*.5*np.diff(t).reshape((-1,)+(1,)*(y.ndim-1)),axis=0)

@njit(parallel=True)
def statistics(z,cl,rough,source,hyd,context,interior):
 nr=z.shape[0]//20;nc=z.shape[1]//20
 out=np.full((10,nr,nc),np.nan,np.float32);barrier=np.full((nr,nc),np.inf,np.float32)
 for index in prange(nr*nc):
  r=index//nc;c=index%nc
  if not context[r,c]:continue
  n=0;s=ss=nrgh=nb=nroad=nstructure=nfall=0.;lo=np.inf;hi=-np.inf
  for dr in range(20):
   for dc in range(20):
    rr=r*20+dr;cc=c*20+dc;v=z[rr,cc];cls=cl[rr,cc]
    building=interior[r,c] and cls==1
    if building:
     nb+=1.
     if np.isfinite(hyd[rr,cc]):barrier[r,c]=min(barrier[r,c],hyd[rr,cc])
    if interior[r,c] and cls==2:nroad+=1.
    if interior[r,c] and cls==3:nstructure+=1.
    if not building and np.isfinite(v):
     n+=1;s+=v;ss+=v*v;lo=min(lo,v);hi=max(hi,v);nrgh+=rough[rr,cc]
     if source[rr,cc]==2:nfall+=1.
  if n:
   out[0,r,c]=s/n;out[1,r,c]=lo;out[2,r,c]=hi;out[3,r,c]=np.sqrt(max(0.,ss/n-(s/n)**2));out[8,r,c]=nrgh/n;out[9,r,c]=nfall/n
  out[4,r,c]=n*.25 if interior[r,c] else 0.
  out[5,r,c]=nb/400.;out[6,r,c]=nroad/400.;out[7,r,c]=nstructure/400.
 return out,barrier

def endpoints(rr,cc,di):
 ar=np.where(di==0,rr//20,(rr+1)//20);ac=np.where(di==0,cc//20-1,cc//20)
 return ar,ac,ar-(di==1),ac+(di==0)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT);args=ap.parse_args();root=args.root.resolve();start=time.monotonic()
 dest=root/'data/processed/three_ward_surrogate_inputs';out=root/'data/exp/data-preprocessing/three_ward_surrogate_inputs'
 dest.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
 old=root/'data/processed/three_ward_sfincs_scenarios';con=root/'data/processed/three_ward_urban_constraints';geo=root/'data/processed/three_ward_adaptive_connectivity'
 labeldir=root/'data/processed/three_ward_common_depth_labels';base=root/'data/processed/three_ward_sfincs_framework'
 baseline=base/'adaptive_connectivity_comparison/rain60_low_corrected_10m'
 hashes={};cache={}
 def bind(p):
  st=p.stat();key=(st.st_dev,st.st_ino,st.st_mtime_ns,st.st_size)
  if key not in cache:cache[key]=sha(p)
  hashes[str(p.relative_to(root))]=cache[key];return cache[key]
 label_manifest=json.loads((root/'data/exp/data-preprocessing/three_ward_common_depth_labels/product_manifest.json').read_text())
 assert all(bind(root/q['file'])==q['sha256'] for q in label_manifest)
 ana_before=sha(root/'docs/AnaSOP.md')
 with np.load(labeldir/'common_support_10m.npz') as d:area=d['Ground Area'];roads=d['Road Ground Area'];valid=d['Eligible Ground Mask'];ward=d['Ward Code']
 with rasterio.open(old/'active_mask_20m.tif') as d:m20=d.read(1)
 inside=np.repeat(np.repeat(m20==1,2,0),2,1);context=np.repeat(np.repeat(m20>0,2,0),2,1)
 with netCDF4.Dataset(baseline/'sfincs_map.nc') as d:
  mask=np.flipud(array(d['msk'][:])).astype('uint8');zmin=np.flipud(array(d['zb'][:]))
 assert np.array_equal(valid,mask==1)
 files=[old/'terrain_sampling_0_50m.tif',con/'constraints_0_50m.tif',con/'manning_0_50m.tif',old/'terrain_source_id_0_50m.tif',con/'hydraulic_obstacle_0_50m.tif']
 stat=np.full((10,NR,NC),np.nan,np.float32);barrier=np.full((NR,NC),np.inf,np.float32)
 with ExitStack() as stack:
  readers=[stack.enter_context(rasterio.open(f)) for f in files]
  for f in files:bind(f)
  for r in range(0,NR,32):
   n=min(32,NR-r);win=Window(0,r*20,NC*20,n*20)
   a,b=statistics(*(d.read(1,window=win) for d in readers),context[r:r+n],inside[r:r+n]);stat[:,r:r+n]=a;barrier[r:r+n]=b
  print('STATIC GROUND READY',round(time.monotonic()-start,1),flush=True)
 assert np.array_equal(np.nan_to_num(stat[4],nan=0.),area.astype('float32'))
 assert np.allclose(stat[6][inside]*100,roads[inside],atol=1e-5)
 with rasterio.open(root/'data/processed/three_ward_fixed_runoff/contributing_area_10m.tif') as d:contributing=d.read(1).astype('float64')
 assert contributing.sum()==38976800 and np.all(contributing[~valid]==0)
 bind(root/'data/processed/three_ward_fixed_runoff/contributing_area_10m.tif')
 with rasterio.open(con/'subcell_connectivity_flags_10m.tif') as d:ambiguous=d.read(1)>0
 with rasterio.open(geo/'automatic_refinement_10m.tif') as d:refine=d.read(1)>0
 bind(con/'subcell_connectivity_flags_10m.tif');bind(geo/'automatic_refinement_10m.tif')
 with np.load(geo/'regular_face_geometry.npz') as d:faces={k:d[k] for k in d.files}
 bind(geo/'regular_face_geometry.npz')
 rr,cc,di=faces['row_fine'],faces['col_fine'],faces['direction'];ar,ac,br,bc=endpoints(rr,cc,di)
 ma=mask[ar,ac];mb=mask[br,bc];assert np.all(ma>0)&np.all(mb>0)
 width=faces['native_open_width_m'];roadwidth=faces['native_road_width_m'];touch=(ma==1)|(mb==1)
 blocked=touch&(width==0)
 snap=json.loads((root/'data/exp/data-briefing/three_ward_adaptive_connectivity/building_closure_snap_audit.json').read_text())[0]
 assert snap['passed'] and int(blocked.sum())==snap['expected_closed_faces']==40310
 with netCDF4.Dataset(baseline/'sfincs_his.nc') as d:
  px=array(d['thindam_x'][:]);py=array(d['thindam_y'][:]);assert len(px)==int(blocked.sum())
  expected=np.c_[-11360+(ac[blocked]+bc[blocked]+1)*5,-32320-(ar[blocked]+br[blocked]+1)*5]
  actual=np.c_[px,py]
  # Official snapping outputs neighboring-cell midpoint in source-face order.
  aa=np.lexsort((actual[:,1],actual[:,0]));ee=np.lexsort((expected[:,1],expected[:,0]));assert np.max(abs(actual[aa]-expected[ee]))<.01
 with netCDF4.Dataset(baseline/'sfincs_subgrid.nc') as d:
  crest=array(d['uv_zmin'][:]);friction=array(d['uv_navg'][:])
 assert len(crest)==len(width) and np.isfinite(crest).all() and np.isfinite(friction).all()
 names=['Ground Elevation Mean','Ground Elevation Minimum','Ground Elevation Maximum','Ground Elevation Standard Deviation','Ground Area Fraction','Building Occupancy Fraction','Road Ground Fraction','Structure Uncertainty Fraction','Ground Manning Coefficient','PLATEAU Ground Fraction','Effective Rainfall Weight','Subcell Connectivity Ambiguity','Automatic Refinement Flag','Interior Ground Mask','Prescribed Boundary Mask','Context Mask']
 physical=[*stat[:4],area/100,*stat[5:10],contributing/100,ambiguous.astype(float),refine.astype(float),valid.astype(float),(mask==2).astype(float),context.astype(float)]
 assert len(names)==len(physical)==16
 # Face features are shared with CNN and graph; missing faces have explicit feature-valid flags.
 face_planes=np.full((12,NR,NC),np.nan,np.float32)
 for direction,index_a,index_b in [(0,0,1),(1,2,3)]:
  sel=di==direction
  for j,values in enumerate([width,roadwidth,crest]):
   face_planes[index_a*3+j,ar[sel],ac[sel]]=values[sel];face_planes[index_b*3+j,br[sel],bc[sel]]=values[sel]
 for direction in ['East','West','North','South']:
  names.extend([direction+' Open Face Width',direction+' Road Face Width',direction+' Face Crest'])
 physical=np.asarray([*physical,*face_planes],dtype='float32')
 feature_valid=np.isfinite(physical)&context[None,:,:]
 tensor=np.where(feature_valid,physical,0).astype('float32')
 np.savez_compressed(dest/'static_tensor_10m.npz',**{'Static Tensor':tensor,'Static Feature Valid':feature_valid,'Static Channel Names':np.array(names),'Eligible Ground Mask':valid,'Context Mask':context,'Native Hydraulic Mask':mask,'Ward Code':ward,'Ground Area':area})
 # Surface and receiver indices use distinct namespaces; prescribed water cannot bridge two ground nodes.
 grid_ids=np.flatnonzero(valid);rnodes,cnodes=np.unravel_index(grid_ids,(NR,NC));node_index=np.full((NR,NC),-1,np.int32);node_index[valid]=np.arange(len(grid_ids))
 bnd=np.loadtxt(baseline/'sfincs.bnd');brow=np.rint((-32320-bnd[:,1])/10-.5).astype(int);bcol=np.rint((bnd[:,0]+11360)/10-.5).astype(int)
 assert np.all(mask[brow,bcol]==2) and len(set(zip(brow,bcol)))==len(bnd)==10488
 assert np.max(abs(-11360+(bcol+.5)*10-bnd[:,0]))<.001 and np.max(abs(-32320-(brow+.5)*10-bnd[:,1]))<.001
 bindex=np.full((NR,NC),-1,np.int32);bindex[brow,bcol]=np.arange(len(bnd))
 with (old/'boundary_points.csv').open(encoding='utf-8-sig') as f:boundary=list(csv.DictReader(f))
 lookup={(int(x['row_north']),int(x['col_east'])):x for x in boundary}
 rec=[lookup[(int(r//2),int(c//2))] for r,c in zip(brow,bcol)]
 groups=sorted({int(x['water_group']) for x in rec});gindex=np.array([groups.index(int(x['water_group'])) for x in rec],np.int16)
 group_names=[next(x['water_name'] for x in rec if int(x['water_group'])==g) for g in groups]
 low=np.loadtxt(baseline/'sfincs.bzs')[0,1:];assert len(low)==len(bnd)
 assert np.max(abs(low-np.array([float(x['low_stage_m']) for x in rec])))<1e-6
 core=(ma==1)&(mb==1)&(width>0);receiver=((ma==1)&(mb==2)|(ma==2)&(mb==1))&(width>0)
 ids=np.flatnonzero(core);a=node_index[ar[ids],ac[ids]];b=node_index[br[ids],bc[ids]]
 index=np.stack([np.r_[a,b],np.r_[b,a]]).astype('int32')
 dx=(bc[ids]-ac[ids])*10;dy=(ar[ids]-br[ids])*10
 ea=np.c_[width[ids],roadwidth[ids],crest[ids],friction[ids],dx,dy].astype('float32');edgeattr=np.r_[ea,ea*np.array([1,1,1,1,-1,-1],np.float32)]
 assert np.all(index>=0) and len(set(zip(a,b)))==len(a)
 graph=coo_matrix((np.ones(index.shape[1],np.uint8),(index[0],index[1])),shape=(len(grid_ids),len(grid_ids))).tocsr()
 components,component_id=connected_components(graph,directed=False)
 rid=np.flatnonzero(receiver);surface_a=ma[rid]==1
 sr=np.where(surface_a,ar[rid],br[rid]);sc=np.where(surface_a,ac[rid],bc[rid]);tr=np.where(surface_a,br[rid],ar[rid]);tc=np.where(surface_a,bc[rid],ac[rid])
 links=np.stack([node_index[sr,sc],bindex[tr,tc]]).astype('int32');assert np.all(links>=0)
 linkattr=np.c_[width[rid],roadwidth[rid],crest[rid],friction[rid],(tc-sc)*10,(sr-tr)*10].astype('float32')
 road_sel=edgeattr[:,1]>0
 np.savez_compressed(dest/'surface_connectivity_graph.npz',**{
  'Node Grid Index':grid_ids.astype('int32'),'Node Features':tensor[:,rnodes,cnodes].T,'Node Feature Valid':feature_valid[:,rnodes,cnodes].T,
  'Node Position':np.c_[-11360+(cnodes+.5)*10,-32320-(rnodes+.5)*10].astype('float32'),'Node Ground Area':area[valid],
  'Surface Edge Index':index,'Surface Edge Features':edgeattr,'Surface Edge Native Face Id':np.r_[ids,ids].astype('int32'),
  'Road Edge Index':index[:,road_sel],'Road Edge Native Face Id':np.r_[ids,ids][road_sel].astype('int32'),
  'Edge Feature Names':np.array(['Open Face Width','Road Face Width','Face Crest','Face Manning Coefficient','Displacement East','Displacement North']),
  'Surface Component Id':component_id.astype('int32'),
  'Receiver Link Index':links,'Receiver Link Features':linkattr,'Receiver Link Native Face Id':rid.astype('int32'),
  'Boundary Point Position':bnd,'Boundary Point Row':brow,'Boundary Point Column':bcol,'Boundary Low Stage':low,'Boundary Group Index':gindex,
  'Boundary Group Codes':np.array(groups),'Boundary Group Names':np.array(group_names),
  'Blocked Native Face Id':np.flatnonzero(blocked).astype('int32'),'Blocked Face Grid Endpoints':np.stack([ar[blocked]*NC+ac[blocked],br[blocked]*NC+bc[blocked]]).astype('int32')})
 # Bind actual source inputs once per hardlink, and retain all forcing knots, including one-second rain ramps.
 records=[];clock=set();contracts={};quality_rows=[];checks={};scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3),'Cross-ward contributing area':valid&(ward==0)}
 budgets={x['run']:x for x in json.loads((root/'data/exp/data-briefing/three_ward_adaptive_sensitivity/water_budgets.json').read_text())}
 for case in CASES:
  folder=base/('adaptive_connectivity_comparison' if case=='rain60_low' else 'adaptive_sensitivity')/(case+'_corrected_10m')
  execution=json.loads((folder/'execution.json').read_text());assert execution['completion_marker'] and execution['returncode']==0
  assert all(bind(folder/f)==h for f,h in execution['input_sha256'].items())
  bind(folder/'execution.json');sc=json.loads((folder/'scenario_contract.json').read_text());contracts[case]=sc
  with netCDF4.Dataset(folder/'roof_surface_precip.nc') as d:
   d['Precipitation'].set_auto_mask(False);rain=array(d['Precipitation'][:])[:,1:-1,1:-1][:,::-1,:]
   seconds=np.asarray(d['time'][:])*60;rt=np.rint(seconds);assert np.max(abs(rt-seconds))<1e-6
  stages=np.loadtxt(folder/'sfincs.bzs');clock.update(rt.tolist());clock.update(stages[:,0].tolist())
  with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
   assert float(d['status'][0])==0 and float(d['time'][-1])==14400
   initial=np.flipud(array(d['h'][0]));peak=np.flipud(array(d['zsmax'][0]));assert np.all(initial[valid]==0)
  records.append((case,folder,rt,rain,stages,peak))
 clock=np.array(sorted(clock),dtype='float64');assert np.array_equal(clock,[0,3600,3601,5400,7200,7201,10800,14400])
 for case,folder,rt,rain,stages,peak in records:
  scalar=contracts[case]['rain_mm_h']*np.array([0,0,1,1,0,0],float);assert len(scalar)==len(rt)
  expected=scalar[:,None,None]*(contributing/100)[None,:,:]
  max_input_error=float(np.max(abs(rain-expected)));assert max_input_error<.001
  physical_rain=np.interp(clock,rt,scalar)
  effective=np.empty((len(clock),NR,NC),np.float32)
  for j,t in enumerate(clock):
   k=min(max(np.searchsorted(rt,t,side='right')-1,0),len(rt)-2);w=(t-rt[k])/(rt[k+1]-rt[k]);effective[j]=(1-w)*rain[k]+w*rain[k+1]
  stage=np.stack([np.interp(clock,stages[:,0],stages[:,j+1]) for j in range(len(bnd))],axis=1)
  increments=stage-low[None,:];group_increment=np.stack([increments[:,gindex==i].mean(1) for i in range(len(groups))],axis=1)
  group_error=float(np.max(abs(stage-(low[None,:]+group_increment[:,gindex]))));assert group_error<1e-6
  volume=float(integrate(effective[:,valid].astype('float64'),clock).sum()*100/(3600*1000))
  intended=contracts[case]['original_rain_interior_volume_m3'];relative=abs(volume-intended)/max(intended,1);assert relative<1e-6
  np.savez_compressed(dest/(case+'_inputs.npz'),**{'Event Time':clock,'Physical Rainfall Rate':physical_rain.astype('float32'),
   'Effective Surface Rainfall Rate':effective,'River Stage':stage.astype('float32'),'River Stage Increment By Group':group_increment.astype('float32'),
   'Initial Water State':np.zeros((NR,NC),np.float32),'Initial State Known':valid,'Measure Configuration':np.array('no action')})
  with np.load(labeldir/(case+'_corrected_10m.npz')) as d:
   lower=d['Peak Mean Depth Lower'];upper=d['Peak Mean Depth Upper'];width_interval=d['Peak Interval Width'];debt=d['Sampled Negative Volume Flag']
  # Quarantine masks never modify the original target. Response-derived flags are label-side only.
  overtop=valid&np.isfinite(peak)&np.isfinite(barrier)&(peak>=barrier-.01)
  finite=valid&np.isfinite(lower)&np.isfinite(upper)&(lower>=0)&(upper>=lower)
  inconsistent=valid&~finite;quarantine=valid&(debt|overtop|inconsistent)
  exact=finite&(width_interval<=1e-7)&~quarantine
  interval=finite&(width_interval>1e-7)&(width_interval<=.010001)&~quarantine
  wide=finite&(width_interval>.010001)
  flags=np.zeros((NR,NC),np.uint16)
  for bit,sel in [(1,~valid),(2,valid&(width_interval>1e-7)),(4,valid&debt),(8,overtop),(16,inconsistent),(32,wide),(64,valid&ambiguous),(128,valid&(ward==0)),(256,valid&(stat[9]>0)),(512,valid&(stat[7]>0))]:flags[sel]|=bit
  np.savez_compressed(dest/(case+'_label_admission.npz'),**{'Exact Label Mask':exact,'Interval Label Mask':interval,'Wide Interval Mask':wide,'Quarantine Mask':quarantine,
   'Eligible Ground Mask':valid,'Quality Flags':flags,'Numerical Volume Flag':debt,'Assumed Barrier Overtopping Flag':overtop,'Label Lower':lower,'Label Upper':upper})
  budget=budgets[case+'_corrected_10m'];assert budget['passed']
  for name,scope in scopes.items():
   quality_rows.append(dict(case=case,scope=name,eligible_ground_km2=float(area[scope].sum()/1e6),exact_admitted_km2=float(area[scope&exact].sum()/1e6),interval_admitted_km2=float(area[scope&interval].sum()/1e6),quarantined_km2=float(area[scope&quarantine].sum()/1e6),negative_volume_flag_km2=float(area[scope&debt].sum()/1e6),barrier_overtopping_flag_km2=float(area[scope&overtop].sum()/1e6),wide_interval_km2=float(area[scope&wide].sum()/1e6),static_connectivity_ambiguity_km2=float(area[scope&ambiguous].sum()/1e6)))
  checks[case]=dict(input_hashes_passed=True,rain_volume_m3=volume,intended_volume_m3=intended,rain_volume_relative_error=relative,physical_rain_integral_mm=float(integrate(physical_rain,clock)/3600),effective_rain_max_input_error_mm_h=max_input_error,stage_group_factorization_error_m=group_error,initial_ground_state_verified_dry=True,source_budget_passed=True,source_budget_relative_error=budget['relative_error'],finite_label_support=int(finite.sum()),quarantined_cells=int(quarantine.sum()),exact_admitted_cells=int(exact.sum()),interval_admitted_cells=int(interval.sum()),technical_control_ready=bool(finite.sum()==valid.sum()),formal_training_release=False)
  print('CASE PREPARED',case,'quarantine',int(quarantine.sum()),flush=True)
 csvwrite(out/'label_admission_by_scope.csv',quality_rows)
 bind(root/'src/preprocessing/prepare_three_ward_surrogate_inputs.py');bind(old/'boundary_points.csv');bind(root/'data/exp/data-briefing/three_ward_adaptive_connectivity/building_closure_snap_audit.json')
 policy=dict(version='2026-10-15-v1',scope='Existing controls only; technical supervision policy, not physical validation or train authorization',
  hard_exclusions=['no eligible ground','unknown/nonfinite/inconsistent target bounds','sampled negative bookkeeping-volume cell pending review','any original assumed-building barrier overtopping in the common cell pending review'],
  exact_tolerance_m=1e-7,maximum_interval_width_m=.01,interval_numeric_tolerance_m=1e-6,midpoint_imputation=False,unknown_dry_imputation=False,
  supervision='Exact-only uses Exact Label Mask; interval-aware uses union of Exact and Interval Label Masks, with unchanged lower/upper targets. Same shared parent pool and frozen masks for every compared method.',
  informational_flags=['static subcell connectivity ambiguity','cross-ward contributing area','PLATEAU fallback fraction','uncertain road/structure classification'],
  flag_bits={'outside_support':1,'bounded_not_point':2,'sampled_negative_volume':4,'assumed_barrier_overtopping':8,'invalid_bounds':16,'wide_interval':32,'connectivity_ambiguity':64,'cross_ward':128,'fallback_terrain':256,'structure_uncertainty':512},
  predictor_leakage='Response-derived admission/quality masks are not predictor channels; raw responses remain separate.',
  evaluation='Report admitted-support scores plus quarantined/unknown area, full-domain potential failures and abstention coverage. Do not claim accuracy in excluded areas or optimize masks from test outcomes.',
  reference='regular10m labels only for this supervision package; adaptive bounds remain alternate numerical diagnostics',
  boundary='prescribed boundary rows have input context and typed links, never target labels or ordinary surface routing',
  release=dict(technical_existing_controls=True,formal_scenario_library=False,training=False,new_scenario_execution=False),
  unresolved=['review/recompute quarantined physical reference cells before full-coverage truth claims','freeze numerical mask policy on development before new held-out responses','validate generator and parent lineage','graph/tensor loading and normalization must preserve masks'])
 dump(dest/'quality_policy.json',policy)
 graphinfo=dict(nodes=len(grid_ids),undirected_surface_faces=int(core.sum()),directed_surface_edges=int(index.shape[1]),directed_road_edges=int(road_sel.sum()),receiver_links=int(receiver.sum()),prescribed_boundary_points=len(bnd),boundary_groups=groups,boundary_points_with_receiver_links=int(len(np.unique(links[1]))),blocked_reference_faces=int(blocked.sum()),surface_components=int(components),isolated_ground_nodes=int(np.sum(np.diff(graph.indptr)==0)),
  mapping='Node Grid Index maps row-major raster cells to graph nodes; exact raster feature correspondence. No ordinary edges between boundary nodes; Receiver Link Index is a distinct node namespace.',
  interpretation='Current-reference surface adjacency, not underground drainage, river channel network, observed street connectivity or a proof of mass conservation.')
 schema=dict(version='2026-10-15-v1',created_utc=datetime.now(timezone.utc).isoformat(),grid=dict(shape=[NR,NC],dx_m=10,epsg=6677,left=-11360,top=-32320,vertical='Source-declared T.P.; height epoch alignment uncertified'),
  static=dict(shape=list(tensor.shape),channels=names,channel_units=['m']*4+['1']*4+['s m^(-1/3)']+['1']*7+['m']*12,feature_valid_shape=list(feature_valid.shape),zero_placeholder='Only numeric storage where Feature Valid is false; not physical zero ground or imputed observation. Always provide validity to the model.',normalization='none fitted; raw units preserved'),
  dynamic=dict(units={'Event Time':'s','Physical Rainfall Rate':'mm/h','Effective Surface Rainfall Rate':'mm/h','River Stage':'m in working reference','Initial Water State':'m'},storage_roundoff_m=1e-5,event_time_seconds=clock.tolist(),physical_rainfall='time vector in mm/h broadcast exactly over spatially uniform physical storm; no areal variability inferred',effective_surface_rainfall_shape=[len(clock),NR,NC],river_stage_shape=[len(clock),len(bnd)],river_stage_group_shape=[len(clock),len(groups)],time_rule='Piecewise-linear common knot union preserving one-second ramps and all stage knots; event origin shared. This is not the300s response-output cadence.',
  input_file_zero='Original effective-rain NetCDF FillValue0 represents real no-rain and is read raw; units recovered from original singular unit attribute and source contract.',
  initial='Interior initial h verified exactly0 in every existing control. Zero outside Initial State Known is a masked placeholder; boundary point levels come from River Stage.'),
  graph=graphinfo,cases=CASES,existing_forcing_combinations=4,independent_events_claimed=0,controls_split='open development only; not sealed test',training=False,new_scenarios=False,
  source_sha256=hashes,quality_policy='quality_policy.json',ana_sop_hash_during_processing=ana_before,
  missing=['Numerical reference quarantine review','nonlinear/spatial forcing generator validation and actual unique parent forcing realizations','reserved family/split custody and near-duplicate checks','training-only normalization and graph/raster loaders','verified road length mapping and critical-region rule','domain/epoch/bathymetry/event validation for real-world claims','active-measure asset/response library later'],
  file_format='Geospatial arrays and typed graph stored as NPZ with readable English keys; no parquet or Excel conversion and no package installation.',elapsed_seconds=time.monotonic()-start)
 dump(dest/'input_schema.json',schema);dump(out/'decisions.json',dict(human_decision='MILI-D-20261015-004',schema=schema,quality_policy=policy));dump(out/'validation.json',dict(valid=True,checks=checks,graph=graphinfo,exact_open_area_match=True,source_label_hashes_verified=len(label_manifest),no_training=True,no_new_runs=True))
 csvwrite(out/'variable_list.csv',[dict(readable_name=name,role='static predictor',dtype='float32',shape=f'{NR}x{NC}',missing_percent=float((~feature_valid[i]).sum()*100/feature_valid[i].size),sample_values=tensor[i,valid][:3].tolist(),status='constructed technical input; final research admission pending') for i,name in enumerate(names)]+[dict(readable_name=x,role='dynamic input',dtype='float32',shape='see input_schema.json',missing_percent=0,sample_values='see case tensors',status='existing synthetic controls only') for x in ['Physical Rainfall Rate','Effective Surface Rainfall Rate','River Stage','Initial Water State']])
 assert sha(root/'docs/AnaSOP.md')==ana_before
 print(json.dumps(dict(static_tensor_shape=list(tensor.shape),graph=graphinfo,clock=clock.tolist(),validation=checks,elapsed_seconds=time.monotonic()-start),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
