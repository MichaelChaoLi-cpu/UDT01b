"""Audit two completed controls without running a solver or releasing labels.

Whole-domain diagnostics only. All timing is explicitly global every-step or local
300-second sampled. New peaks are integrated over original eligible fine ground.
"""
import os
os.environ['NUMBA_NUM_THREADS']='4'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from numba import njit,prange
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from reference_label_policy import POLICY,classify_labels
from prepare_sfincs_cpu_build import sha,dump
OUT='data/processed/three_ward_negative_volume_audit_v1'
REPORT='data/exp/data-preprocessing/three_ward_negative_volume_audit'
E2E='data/processed/three_ward_cpu_e2e_controls_v1'
CASES=['dry_raised','rain60_raised']
NR,NC=1012,770
SCOPES={'Whole domain':None,'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3,'Contributing area':0}

def csvwrite(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def readnp(p):
 with np.load(p,allow_pickle=False)as d:return {k:d[k]for k in d.files}
def arr(v):return np.ma.asarray(v,dtype='f8').filled(np.nan)
def structured(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf-8'))
def ratio(a,b):return float(a/b*100)if b else None
def quantile(a,q):return float(np.quantile(a,q))if len(a)else None

@njit(parallel=True)
def fine_peaks(z,cl,hyd,valid,peak,zmin):
 nr,nc=valid.shape
 ncase=peak.shape[0]
 lower=np.full((ncase,nr,nc),np.nan,np.float64);upper=lower.copy()
 ground=np.zeros((nr,nc),np.float64);road=ground.copy();barrier=np.full((nr,nc),np.inf,np.float64)
 for i in prange(nr*nc):
  r=i//nc;c=i%nc
  if not valid[r,c]:continue
  lo=np.zeros(ncase,np.float64);up=np.zeros(ncase,np.float64);n=0;roads=0
  for dr in range(20):
   for dc in range(20):
    rr=r*20+dr;cc=c*20+dc
    if cl[rr,cc]==1:
     if np.isfinite(hyd[rr,cc]):barrier[r,c]=min(barrier[r,c],hyd[rr,cc])
    elif np.isfinite(z[rr,cc]):
     n+=1
     if cl[rr,cc]==2:roads+=1
     for k in range(ncase):
      if np.isfinite(peak[k,r,c]):
       v=max(peak[k,r,c]-z[rr,cc],0.);lo[k]+=v;up[k]+=v
      else:up[k]+=max(zmin[k,r,c]+.01-z[rr,cc],0.)
  ground[r,c]=n*.25;road[r,c]=roads*.25
  if n:
   for k in range(ncase):lower[k,r,c]=lo[k]/n;upper[k,r,c]=up[k]/n
 return lower,upper,ground,road,barrier

def main():
 start=time.monotonic();out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen audit')
 out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 protected={};stats={};checks=[]
 def bind(p):
  p=Path(p);st=p.stat();key=str(p.relative_to(ROOT));protected[key]=sha(p);stats[key]=[st.st_size,st.st_mtime_ns];return p
 def check(name,ok,details=None):
  checks.append({'check':name,'pass':bool(ok),'details':details});print(('PASS 'if ok else'FAIL ')+name,flush=True)
 support=readnp(bind(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz'))
 tensor=readnp(bind(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz'))
 graph=readnp(bind(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz'))
 policy_path=bind(ROOT/'src/preprocessing/reference_label_policy.py')
 for rel in ['data/processed/three_ward_reference_contract_v1/reference_registry.json','data/processed/three_ward_minimum_scenarios_v1/run_manifest.json',E2E+'/freeze.json',E2E+'/run_manifest.json']:
  bind(ROOT/rel)
 priorfreeze=json.loads((ROOT/E2E/'freeze.json').read_text())['files_sha256']
 area=support['Ground Area'];road=support['Road Ground Area'];valid=support['Eligible Ground Mask'];ward=support['Ward Code'];tf=Affine(*support['Affine Transform'])
 features=tensor['Static Tensor'];names=list(tensor['Static Channel Names']);get=lambda n:features[names.index(n)]
 scopes={name:valid if code is None else valid&(ward>0)if code==-1 else valid&(ward==code)for name,code in SCOPES.items()}
 manifest=json.loads((ROOT/E2E/'run_manifest.json').read_text());runs={r['case_id']:r for r in manifest['runs']}
 static_a={k:v for k,v in runs[CASES[0]]['input_sha256'].items()if k!='roof_surface_precip.nc'}
 static_b={k:v for k,v in runs[CASES[1]]['input_sha256'].items()if k!='roof_surface_precip.nc'}
 check('paired_controls_identical_geometry_options_and_stage',static_a==static_b)
 run_data={};native_rows=[];temporal_rows=[];phase_rows=[];cohort_rows=[];coverage_rows=[];wet_rows=[];patch_rows=[];receiver_rows=[];first_rows=[]
 for case in CASES:
  folder=ROOT/runs[case]['folder'];print('READ EXISTING',case,flush=True)
  for name in ['sfincs_map.nc','udt_step_budget.csv','udt_native_ever_negative.csv','udt_output_clock.csv','sfincs.ind','sfincs.msk','sfincs.inp','sfincs.bzs','execution.json','udt_diagnostic_identity.txt']:
   p=bind(folder/name);check(case+'_frozen_'+name,protected[str(p.relative_to(ROOT))]==priorfreeze[str(p.relative_to(ROOT))])
  index=np.fromfile(folder/'sfincs.ind',dtype='<i4');n=int(index[0]);index=index[1:]-1
  sr=index%NR;cc=index//NR;rr=NR-1-sr
  native_mask=np.fromfile(folder/'sfincs.msk',dtype='u1')
  flags=structured(folder/'udt_native_ever_negative.csv');ever_native=flags[flags.dtype.names[1]]=='T'
  check(case+'_native_index_order_unique',len(index)==n==len(flags)and len(np.unique(index))==n and np.array_equal(flags['native_cell_index'],np.arange(1,n+1)))
  ever=np.zeros((NR,NC),bool);ever[rr,cc]=ever_native
  ledger=structured(folder/'udt_step_budget.csv');clock=structured(folder/'udt_output_clock.csv');maps=clock[clock['write_map']=='T']
  times=np.asarray(maps['actual_state_time'],dtype='f8');t=ledger['actual_state_time'];count=ledger['negative_native_count'];debt=ledger['negative_volume_sum']
  first=np.full((NR,NC),np.nan);last=first.copy();minimum=np.full((NR,NC),np.inf);sample_count=np.zeros((NR,NC),np.int16);sample_ever=np.zeros((NR,NC),bool)
  with Dataset(folder/'sfincs_map.nc')as d:
   mask=np.flipud(np.nan_to_num(arr(d['msk'][:]),nan=0)).astype('u1');peak=np.flipud(arr(d['zsmax'][0]));zmin=np.flipud(arr(d['zb'][:]))
   check(case+'_mapping_mask_coordinates_and_area_support',np.array_equal(mask[rr,cc],native_mask)and np.array_equal(mask==1,valid)and np.array_equal(ever&~valid,np.zeros_like(valid))and np.allclose(np.flipud(arr(d['y'][:])),-32320-(np.arange(NR)[:,None]+.5)*10)and np.allclose(arr(d['x'][:]),-11360+(np.arange(NC)[None,:]+.5)*10))
   initialvol=np.flipud(arr(d['subgrid_volume'][0]));check(case+'_initial_no_threshold_negative',not np.any(initialvol[valid]<-.001))
   for j,actual in enumerate(times):
    v=np.flipud(arr(d['subgrid_volume'][j]));zs=np.flipud(arr(d['zs'][j]));h=np.flipud(arr(d['h'][j]));negative=valid&(v<-.001)
    first[negative&~sample_ever]=actual;last[negative]=actual;sample_count[negative]+=1;sample_ever|=negative;minimum[valid]=np.minimum(minimum[valid],v[valid])
    step=int(maps['step_index'][j]);exactrow=ledger[step]
    check(case+f'_frame_{j}_negative_count_and_debt_match',int(negative.sum())==int(exactrow['negative_native_count'])and abs(float(-np.minimum(v[valid],0).sum())-exactrow['negative_volume_sum'])<1e-4)
    both=valid&np.isfinite(zs)&np.isfinite(h)
    if j==0:censor_ok=True;state_ok=True;maximum_peak_violation=0.
    state_ok &=bool(np.all(np.isfinite(v[valid]))and np.all(np.isfinite(np.flipud(arr(d['u'][j]))[valid]))and np.all(np.isfinite(np.flipud(arr(d['v'][j]))[valid])))
    censor_ok &=bool(np.all(np.isfinite(zmin[valid]))and (not both.any()or np.max(abs(h[both]-(zs[both]-zmin[both])))<1e-5))
    both_peak=valid&np.isfinite(zs)&np.isfinite(peak)
    if both_peak.any():maximum_peak_violation=max(maximum_peak_violation,float(np.max(zs[both_peak]-peak[both_peak])))
   check(case+'_finite_fields_and_shallow_semantics',state_ok and censor_ok and maximum_peak_violation<1e-5)
  check(case+'_snapshot_negative_subset_online',not np.any(sample_ever&~ever))
  check(case+'_native_ever_count_matches_completed_audit',int(ever.sum())==json.loads((ROOT/'data/exp/data-preprocessing/three_ward_cpu_e2e_controls/summary.json').read_text())['case_summaries'][case]['online_ever_negative_native_cells'])
  active=np.flatnonzero(count>0);i0=int(active[0]);imax=int(np.argmax(count));idmax=int(np.argmax(debt))
  summary={'online_ever_cells':int(ever.sum()),'sampled_ever_cells':int(sample_ever.sum()),'online_only_cells':int((ever&~sample_ever).sum()),'first_global_negative_time_s':float(t[i0]),'first_global_negative_step':int(ledger['step_index'][i0]),'previous_global_nonnegative_time_s':float(t[i0-1]),'max_simultaneous_negative_cells':int(count[imax]),'max_count_time_s':float(t[imax]),'max_debt_m3':float(debt[idmax]),'max_debt_time_s':float(t[idmax]),'minimum_online_volume_m3':float(min(ledger['minimum_native_volume_since_initialization'])),'final_negative_cells':int(count[-1]),'final_debt_m3':float(debt[-1]),'fraction_of_recorded_poststep_states_with_negatives_pct':float(np.mean(count[1:]>0)*100),'exact_per_cell_first_time_available':False,'exact_per_cell_duration_available':False,'per_cell_native_face_step_flux_available':False,'fields_finite_and_censoring_consistent':state_ok and censor_ok and maximum_peak_violation<1e-5}
  for a,b,name in [(0,3600,'Before forcing onset'),(3600,5400,'Rain and stage rise'),(5400,7201,'Late rain and stage change'),(7201,10800,'Rain-free recession and stage change'),(10800,14401,'Late recession')]:
   sel=(t>a)&(t<=b);phase_rows.append({'case':case,'phase':name,'from_s':a,'to_s':min(b,14400),'steps':int(sel.sum()),'steps_with_negative':int(np.sum(count[sel]>0)),'max_simultaneous_cells':int(max(count[sel])),'max_debt_m3':float(max(debt[sel]))})
  # Full native-step series, not a fitted or scheduled clock.
  for j in range(len(t)):
   temporal_rows.append({'case':case,'step':int(ledger['step_index'][j]),'actual_time_s':float(t[j]),'dt_s':float(ledger['actual_step_dt'][j]),'negative_cells':int(count[j]),'negative_debt_m3':float(debt[j]),'minimum_volume_since_start_m3':float(ledger['minimum_native_volume_since_initialization'][j]),'rain_applied_m3':float(ledger['rain_applied_cumulative'][j])})
  run_data[case]={'folder':folder,'rr':rr,'cc':cc,'ever':ever,'sampled':sample_ever,'first':first,'last':last,'min':minimum,'sample_count':sample_count,'peak':peak,'zmin':zmin,'mask':mask,'summary':summary}
 # Compute current labels and barrier-contact masks from immutable original fine ground.
 lo=np.full((2,NR,NC),np.nan);up=lo.copy();barrier=np.full((NR,NC),np.inf)
 base=ROOT/'data/processed/three_ward_sfincs_scenarios';con=ROOT/'data/processed/three_ward_urban_constraints'
 paths=[base/'terrain_sampling_0_50m.tif',con/'constraints_0_50m.tif',con/'hydraulic_obstacle_0_50m.tif']
 for p in paths:bind(p)
 peaks=np.array([run_data[c]['peak']for c in CASES]);mins=np.array([run_data[c]['zmin']for c in CASES])
 with rasterio.open(paths[0])as dem,rasterio.open(paths[1])as cl,rasterio.open(paths[2])as hyd:
  assert dem.transform==cl.transform==hyd.transform==tf*Affine.scale(.05,.05)
  for r in range(0,NR,32):
   n=min(32,NR-r);win=Window(0,r*20,NC*20,n*20)
   a,b,ga,ra,ba=fine_peaks(dem.read(1,window=win),cl.read(1,window=win),hyd.read(1,window=win),valid[r:r+n],peaks[:,r:r+n],mins[:,r:r+n])
   assert np.array_equal(ga,area[r:r+n])and np.array_equal(ra,road[r:r+n])
   lo[:,r:r+n]=a;up[:,r:r+n]=b;barrier[r:r+n]=ba
   if r%128==0:print('GROUND rows',r+n,'elapsed',round(time.monotonic()-start),flush=True)
 check('current_peaks_reconstructed_on_exact_original_ground',np.all(np.isfinite(lo[:,valid]))and np.all(up[:,valid]>=lo[:,valid])and np.all(lo[:,valid]>=0))
 boundary=np.argwhere(run_data[CASES[0]]['mask']==2);dist=cKDTree(boundary*10.).query(np.argwhere(valid)*10.)[0];boundary_distance=np.full((NR,NC),np.nan);boundary_distance[valid]=dist
 # A one-face structural-context summary is descriptive, never an expanded exclusion mask.
 openwidth=np.array([get(d+' Open Face Width')for d in ['East','West','North','South']]);widthvalid=np.array([tensor['Static Feature Valid'][names.index(d+' Open Face Width')]for d in ['East','West','North','South']])
 blocked=np.sum(widthvalid&(openwidth==0),axis=0)
 nodeflat=graph['Node Grid Index'];edges=graph['Surface Edge Index'];adj=coo_matrix((np.ones(edges.shape[1],np.uint8),(edges[0],edges[1])),shape=(len(nodeflat),len(nodeflat))).tocsr()
 receiver_link=graph['Receiver Link Index'];receiver_flat=graph['Node Grid Index'][receiver_link[0]]
 open_receiver=np.zeros((NR,NC),bool);open_receiver.ravel()[receiver_flat]=True
 cohorts={'Open receiver face connected':open_receiver,'Road present':get('Road Ground Fraction')>0,'Building mixed':get('Building Occupancy Fraction')>0,'Blocked face adjacent':blocked>0,'Direct boundary neighbor':boundary_distance<=10.001,'Within 50m boundary centre':boundary_distance<=50,'Connectivity ambiguity':get('Subcell Connectivity Ambiguity')>0,'Automatic refinement':get('Automatic Refinement Flag')>0,'PLATEAU fallback present':get('PLATEAU Ground Fraction')>0}
 summaries={};current_union=np.zeros((NR,NC),bool)
 for k,case in enumerate(CASES):
  x=run_data[case];ever=x['ever'];sample=x['sampled'];peak=x['peak'];bar=valid&np.isfinite(peak)&np.isfinite(barrier)&(peak>=barrier-.01)
  masks=classify_labels(lo[k],up[k],valid,ever,bar,interval_source_known=x['summary']['fields_finite_and_censoring_consistent'])
  sampled_masks=classify_labels(lo[k],up[k],valid,sample,bar,interval_source_known=x['summary']['fields_finite_and_censoring_consistent'])
  x['summary'].update({'online_negative_ground_km2':float(area[ever].sum()/1e6),'online_negative_road_area_km2':float(road[ever].sum()/1e6),'extra_quarantine_over_same_run_snapshots_km2':float(area[masks['Quarantine Mask']&~sampled_masks['Quarantine Mask']].sum()/1e6),'all_current_quarantine_km2':float(area[masks['Quarantine Mask']].sum()/1e6),'assumed_barrier_contact_km2':float(area[bar].sum()/1e6),'candidate_admitted_ground_km2':float(area[masks['Admitted Mask']].sum()/1e6),'formal_reference_release':False,'training_release':False})
  for group_index,code in enumerate(graph['Boundary Group Codes']):
   hits=graph['Boundary Group Index'][receiver_link[1]]==group_index;flat_ids=np.unique(receiver_flat[hits]);nflags=ever.ravel()[flat_ids]
   receiver_rows.append({'case':case,'water_group_code':int(code),'water_group_name':str(graph['Boundary Group Names'][group_index]),'open_receiver_ground_nodes':len(flat_ids),'negative_receiver_nodes':int(nflags.sum()),'negative_receiver_ground_m2':float(area.ravel()[flat_ids[nflags]].sum()),'negative_receiver_node_percent':ratio(nflags.sum(),len(flat_ids)),'counting_rule':'a cell linked to several groups may appear more than once'})
  for sample_time in np.unique(x['first'][sample]):
   sel=sample&(x['first']==sample_time);first_rows.append({'case':case,'first_sampled_negative_time_s':float(sample_time),'cells':int(sel.sum()),'ground_m2':float(area[sel].sum()),'interpretation':'first observed negative frame; not first ever crossing or duration'})
  first_rows.append({'case':case,'first_sampled_negative_time_s':None,'cells':int((ever&~sample).sum()),'ground_m2':float(area[ever&~sample].sum()),'interpretation':'online flag only; negative episode not seen at saved frames'})
  for name,scope in scopes.items():
   row={'case':case,'scope':name,'ground_km2':float(area[scope].sum()/1e6),'online_negative_cells':int((ever&scope).sum()),'online_negative_ground_km2':float(area[ever&scope].sum()/1e6),'online_negative_road_area_km2':float(road[ever&scope].sum()/1e6),'snapshot_negative_cells':int((sample&scope).sum()),'online_only_cells':int((ever&~sample&scope).sum()),'online_negative_ground_pct':ratio(area[ever&scope].sum(),area[scope].sum())}
   for n,m in masks.items():row[n+' km2']=float(area[scope&m].sum()/1e6)
   coverage_rows.append(row)
   for threshold in POLICY['peak_thresholds_m']:
    certain=scope&(lo[k]>threshold);possible=scope&(up[k]>threshold);negwet=certain&ever
    wet_rows.append({'case':case,'scope':name,'threshold_m':threshold,'certain_wet_km2':float(area[certain].sum()/1e6),'possible_wet_km2':float(area[possible].sum()/1e6),'negative_certain_wet_km2':float(area[negwet].sum()/1e6),'negative_certain_wet_pct':ratio(area[negwet].sum(),area[certain].sum()),'all_quarantine_certain_wet_pct':ratio(area[certain&masks['Quarantine Mask']].sum(),area[certain].sum()),'candidate_admitted_certain_wet_km2':float(area[certain&masks['Admitted Mask']].sum()/1e6)})
  for cohort,csel in cohorts.items():
   for name,sel in [('Online negative',ever),('Other eligible ground',valid&~ever)]:
    cohort_rows.append({'case':case,'cohort':cohort,'population':name,'population_cells':int(sel.sum()),'matching_cells':int((sel&csel).sum()),'matching_cells_pct':ratio((sel&csel).sum(),sel.sum()),'matching_ground_pct':ratio(area[sel&csel].sum(),area[sel].sum())})
  # Descriptive hot regions: peak-exceedance cells joined only over existing open faces.
  # These cell peaks need not be synchronous and are not realized flow trajectories.
  for th in [.1,.3]:
   wet=valid&(lo[k]>th);ids=np.flatnonzero(wet.ravel()[nodeflat]);sub=adj[ids][:,ids]
   ncomp,lab=connected_components(sub,directed=False);flat=nodeflat[ids];weights=area.ravel()[flat]
   sizes=np.bincount(lab,weights=weights,minlength=ncomp);negarea=np.bincount(lab,weights=weights*ever.ravel()[flat],minlength=ncomp)
   quarant=np.bincount(lab,weights=weights*masks['Quarantine Mask'].ravel()[flat],minlength=ncomp)
   patchid=np.zeros((NR,NC),np.int32)
   for rank,j in enumerate(np.argsort(sizes)[::-1][:10],1):
    ff=flat[lab==j];r,c=np.unravel_index(ff,(NR,NC));patchid.ravel()[ff]=rank
    original_nodes=ids[lab==j];keep=~masks['Quarantine Mask'].ravel()[ff];kept_nodes=original_nodes[keep]
    if len(kept_nodes):
     nc_after,cl_after=connected_components(adj[kept_nodes][:,kept_nodes],directed=False)
     retained_component_area=np.bincount(cl_after,weights=area.ravel()[ff[keep]],minlength=nc_after);largest_retained=float(retained_component_area.max())
    else:nc_after=0;largest_retained=0.
    patch_rows.append({'case':case,'threshold_m':th,'rank':rank,'retained_components_after_direct_quarantine':int(nc_after),'largest_retained_component_ground_pct_of_original':ratio(largest_retained,sizes[j]),'scope_of_connectivity_check':'label-support fragmentation only; not a hydraulic correction or actual route','open_face_component_id':int(j),'cells':len(ff),'ground_km2':float(sizes[j]/1e6),'negative_cells':int(ever.ravel()[ff].sum()),'negative_ground_pct':ratio(negarea[j],sizes[j]),'all_quarantine_ground_pct':ratio(quarant[j],sizes[j]),'centroid_x':float(np.average(-11360+(c+.5)*10,weights=area.ravel()[ff])),'centroid_y':float(np.average(-32320-(r+.5)*10,weights=area.ravel()[ff])),'Chiyoda_km2':float(area.ravel()[ff[ward.ravel()[ff]==1]].sum()/1e6),'Chuo_km2':float(area.ravel()[ff[ward.ravel()[ff]==2]].sum()/1e6),'Minato_km2':float(area.ravel()[ff[ward.ravel()[ff]==3]].sum()/1e6),'Contributing_km2':float(area.ravel()[ff[ward.ravel()[ff]==0]].sum()/1e6)})
   np.savez_compressed(out/(case+f'_top_regions_{th:g}m.npz'),**{'Rank':patchid,'Threshold':np.array(th)})
  # Per-cell evidence table includes unknown online-only timing as blank, never zero.
  for native_id,r,c in zip(np.arange(1,len(x['rr'])+1),x['rr'],x['cc']):
   if not ever[r,c]:continue
   native_rows.append({'case':case,'native_cell_id':int(native_id),'row_north':int(r),'col_east':int(c),'x_epsg6677':float(-11360+(c+.5)*10),'y_epsg6677':float(-32320-(r+.5)*10),'ward_code':int(ward[r,c]),'ground_area_m2':float(area[r,c]),'road_area_m2':float(road[r,c]),'snapshot_seen':bool(sample[r,c]),'first_observed_negative_time_s':float(x['first'][r,c])if sample[r,c]else None,'last_observed_negative_time_s':float(x['last'][r,c])if sample[r,c]else None,'negative_frames':int(x['sample_count'][r,c]),'sampled_minimum_volume_m3':float(x['min'][r,c]),'peak_mean_depth_lower_m':float(lo[k,r,c]),'peak_mean_depth_upper_m':float(up[k,r,c]),'building_fraction':float(get('Building Occupancy Fraction')[r,c]),'boundary_centre_distance_m':float(boundary_distance[r,c]),'blocked_adjacent_faces':int(blocked[r,c]),'assumed_barrier_contact':bool(bar[r,c]),'exact_first_crossing_s':None,'duration_s':None})
  current_union|=masks['Quarantine Mask']
  np.savez_compressed(out/(case+'_audit_masks.npz'),**{'Online Ever Negative':ever,'Sampled Ever Negative':sample,'Online Only Negative':ever&~sample,'First Sampled Negative Time':x['first'],'Last Sampled Negative Time':x['last'],'Sampled Minimum Volume':np.where(valid,x['min'],np.nan),'Negative Frame Count':x['sample_count'],'Peak Mean Depth Lower':lo[k],'Peak Mean Depth Upper':up[k],'Assumed Barrier Contact':bar,'Assumed Barrier Elevation':barrier,**masks})
  flag=ever.astype('u1')+2*sample.astype('u1')+4*bar.astype('u1')
  with rasterio.open(out/(case+'_quality_flags_10m.tif'),'w',driver='GTiff',height=NR,width=NC,count=1,dtype='uint8',crs=6677,transform=tf,compress='deflate')as d:
   d.write(flag,1);d.update_tags(bits='1 online ever negative;2 sampled negative;4 assumed barrier contact',outside_support='see companion support; zero is not a certified label')
  summaries[case]=x['summary']
  check(case+'_online_screen_only_removes_candidate_support',not np.any(masks['Admitted Mask']&~sampled_masks['Admitted Mask']))
  check(case+'_quarantine_disjoint_candidate_admission',not np.any(masks['Admitted Mask']&masks['Quarantine Mask'])and np.all(masks['Quarantine Mask'][ever]))
  print('SUMMARY',case,json.dumps(x['summary']),flush=True)
 csvwrite(report/'receiver_group_exposure.csv',receiver_rows);csvwrite(report/'sampled_first_detection.csv',first_rows)
 csvwrite(report/'negative_cell_evidence.csv',native_rows);csvwrite(report/'step_negative_timeline.csv',temporal_rows);csvwrite(report/'phase_summary.csv',phase_rows);csvwrite(report/'scope_coverage.csv',coverage_rows);csvwrite(report/'wet_area_exposure.csv',wet_rows);csvwrite(report/'static_cohort_comparison.csv',cohort_rows);csvwrite(report/'principal_peak_regions.csv',patch_rows)
 a=run_data[CASES[0]]['ever'];b=run_data[CASES[1]]['ever'];comparison={'shared_cells':int((a&b).sum()),'dry_only_cells':int((a&~b).sum()),'rain_only_cells':int((b&~a).sum()),'same_forcing_and_geometry_except_rain':True,'same_raised_stage':True,'independent_event_count':0}
 check('original_inputs_outputs_and_policies_unchanged',all(sha(ROOT/p)==h for p,h in protected.items()))
 source=ROOT/'data/processed/three_ward_cpu_solver_build_v1/context/source/src';evidence={}
 for f in ['sfincs_continuity.f90','sfincs_momentum.f90','sfincs_domain.f90','sfincs_quadtree.F90','udt_clock_budget.f90']:
  p=source/f;bind(p);text=p.read_text();evidence[f]={'sha256':sha(p),'relevant_lines':[{'line':i,'text':line.strip()}for i,line in enumerate(text.splitlines(),1)if any(key in line for key in ['z_volume(nm) = z_volume(nm) + dvol','z_volume(nm) <= 1.0e-6','z_volume(nm) < 0.0','z_volume(nmu) < 0.0','ever_negative=ever_negative.or.','minimum_volume=min(','index_v_nm(nm)   = nm','z_index_z_n(ip)'])]}
 dump(report/'source_mechanism_evidence.json',evidence)
 result={'cases':summaries,'comparison':comparison,'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'new_solver_runs':0,'training':0,'prospective_58_executed':0,'formal_reference_release':False,'policy_unchanged':True,'negative_threshold_m3':-.001,'local_timing_resolution_s':300,'global_timing':'actual every-step; native cell identity not available per step','hot_regions':'connected peak-exceedance cells over static open faces; not simultaneous pools or actual drainage paths','numerical_error_propagation_bounded':False,'diagnostic_only':True}
 dump(report/'summary.json',result);dump(report/'provenance.json',{'source_sha256':protected,'source_stats':stats,'human_record':'MILI-D-20261015-010','all_source_read_only':True});dump(out/'access_policy.json',{'diagnostic_only':True,'formal_reference_release':False,'training_release':False,'response_quality_flags_are_predictors':False,'original_registry_unchanged':True,'no_ad_hoc_spatial_buffer':True,'scope':'two already-open regular-grid controls only'})
 print('FINISH',len(checks),'checks',result['all_checks_passed'],'seconds',time.monotonic()-start,flush=True)
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
