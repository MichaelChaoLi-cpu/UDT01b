"""Actual crossing evidence and unchanged-input two-control numerical qualification."""
from pathlib import Path
import os,sys,json,csv,collections,itertools,argparse,shutil
os.environ['NUMBA_NUM_THREADS']='4'
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.windows import Window
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_negative_trace_controls import OUT,REPORT,CASES,PRIOR,INTERFACE
from prepare_sfincs_cpu_build import sha,dump
from reference_label_policy import POLICY,classify_labels
from audit_three_ward_negative_volumes import fine_peaks,readnp,csvwrite,arr
import audit_three_ward_cpu_e2e as budget_audit

def dictionary_rows(p):
 with p.open()as f:yield from csv.DictReader(f)
def parse_summary(p):
 dtype=[('native_cell_index','i4'),('initial_below_threshold','U1'),('first_below_endpoint_s','f8'),('last_below_endpoint_s','f8'),('minimum_volume_m3','f8'),('minimum_time_s','f8'),('minimum_step','i8'),('negative_poststep_count','i8'),('down_crossings','i8'),('up_crossings','i8'),('final_volume_m3','f8')]
 return np.genfromtxt(p,delimiter=',',names=True,dtype=dtype,encoding='utf-8')
def native_reduction_check(d,ledger):
 steps=d['minimum_step'];in_range=bool(np.all((steps>=0)&(steps<len(ledger))))
 timestamp_match=in_range and np.array_equal(d['minimum_time_s'],ledger['actual_state_time'][steps])
 sequential=float(np.cumsum(d['final_volume_m3'])[-1]);native=float(ledger['signed_storage'][-1])
 return bool(timestamp_match and sequential==native),{'minimum_times_match_exactly':bool(timestamp_match),'native_order_sum_m3':sequential,'ledger_storage_m3':native,'native_order_difference_m3':sequential-native,'pairwise_sum_minus_native_m3':float(np.sum(d['final_volume_m3']))-native,'formal_budget_gates_unchanged':True}

def recheck_native_reductions():
 report=ROOT/REPORT;out=ROOT/OUT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen controls')
 p=report/'summary.json';result=json.loads(p.read_text())
 backup=report/'audit-first-attempt.json'
 if not backup.exists():backup.write_bytes(p.read_bytes())
 manifest=json.loads((out/'run_manifest.json').read_text())
 for run in manifest['runs']:
  folder=ROOT/run['folder'];d=parse_summary(folder/'udt_negative_cell_summary.csv');ledger=np.genfromtxt(folder/'udt_step_budget.csv',delimiter=',',names=True);ok,details=native_reduction_check(d,ledger)
  check=next(c for c in result['checks']if c['case']==run['case_id']and c['check']=='minimum_times_and_final_volumes_match_actual_clock');check['pass']=ok;check['details']=details
 result['all_checks_passed']=all(c['pass']for c in result['checks']);dump(p,result)
 dump(report/'reduction-order-note.json',{'reason':'The final per-cell float64 values must be reduced in the same native interior-cell order as the pinned Fortran SUM. Pairwise NumPy reduction is reported separately. No clock, state, flux, source or formal budget tolerance is altered.','cases':{c['case']:c['details']for c in result['checks']if c['check']=='minimum_times_and_final_volumes_match_actual_clock'}})
 print('native reduction recheck',result['all_checks_passed'])
 if not result['all_checks_passed']:raise SystemExit(1)

def main(only=None,skip_budget=False):
 out=ROOT/OUT;report=ROOT/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen trace controls')
 manifest=json.loads((out/'run_manifest.json').read_text());runs=[r for r in manifest['runs']if only is None or r['case_id']==only]
 if not skip_budget:
  folder=report/'budget';folder.mkdir(exist_ok=True)
  provenance=json.loads((report/'provenance.json').read_text());old=json.loads((ROOT/'data/exp/data-preprocessing/three_ward_cpu_e2e_controls/provenance.json').read_text())
  provenance['existing_result_stat_fingerprints']=old['existing_result_stat_fingerprints'];dump(folder/'provenance.json',provenance)
  budget_audit.OUT=OUT;budget_audit.REPORT=REPORT+'/budget';budget_audit.main(only=only)
  for name in (['summary.json','validation.json']if only is None else[only+'_validation.json']):
   receipt=folder/name;content=json.loads(receipt.read_text());content['known_failed_attempts_retained']=0;content['compact_trace_enabled']=True;dump(receipt,content)
 checks=[];summaries={};group_rows=[];mechanism_rows=[];first_rows=[];datasets={}
 def check(case,name,ok,details=None):checks.append({'case':case,'check':name,'pass':bool(ok),'details':details});print(('PASS 'if ok else'FAIL ')+case+' '+name,flush=True)
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');area=support['Ground Area'];valid=support['Eligible Ground Mask'];ward=support['Ward Code'];schema=json.loads((ROOT/INTERFACE/'output_schema.json').read_text());codes=[int(k)for k in schema['groups']]
 for run in runs:
  case=run['case_id'];folder=ROOT/run['folder'];oldfolder=ROOT/run['previous_e2e_folder'];groupdest=report/case;groupdest.mkdir(exist_ok=True)
  receipt=(folder/'udt_negative_trace_receipt.txt').read_text();check(case,'real_runtime_trace_enabled','schema=negative-cell-face-v1'in receipt and'native_cells=368872'in receipt)
  d=parse_summary(folder/'udt_negative_cell_summary.csv');ids=d['native_cell_index'];native_n=368872;celllookup=np.full(native_n+1,-1,np.int32);celllookup[ids]=np.arange(len(ids))
  ledger=np.genfromtxt(folder/'udt_step_budget.csv',delimiter=',',names=True);times=ledger['actual_state_time'];nstep=len(times)-1
  ix=np.fromfile(folder/'sfincs.ind',dtype='<i4')[1:]-1;rr=1011-ix%1012;cc=ix//1012
  ever=np.zeros((1012,770),bool);nativeflag=np.zeros(native_n,bool);nativeflag[ids-1]=np.isfinite(d['first_below_endpoint_s']);ever[rr,cc]=nativeflag
  oldflag=np.genfromtxt(folder/'udt_native_ever_negative.csv',delimiter=',',names=True,dtype=None,encoding='utf-8');check(case,'all_interior_cells_and_legacy_ever_flags_agree',len(d)==int(valid.sum())and len(np.unique(ids))==len(d)and np.array_equal(nativeflag,oldflag[oldflag.dtype.names[1]]=='T'))
  reduction_ok,reduction_details=native_reduction_check(d,ledger)
  check(case,'minimum_times_and_final_volumes_match_actual_clock',reduction_ok,reduction_details)
  check(case,'no_initial_negative_state',np.all(d['initial_below_threshold']=='F'))
  links=collections.defaultdict(set)
  for row in dictionary_rows(folder/'udt_negative_cell_group_links.csv'):links[int(row['native_cell_index'])].add(int(row['water_group_code']))
  check(case,'complete_nonconflicting_water_group_membership',set(links)==set(map(int,ids))and all(gs<=set(codes)and not(0 in gs and len(gs)>1)for gs in links.values()))
  down=np.zeros(native_n+1,np.int64);up=down.copy();occupancy=down.copy();active_start=np.full(native_n+1,-1,np.int64);last_observed=np.full(native_n+1,np.nan);first=np.full(native_n+1,np.nan);lastdown=first.copy();lastup=first.copy();event_stats=collections.Counter();phase=collections.Counter();firststats=collections.Counter();facegroup=collections.defaultdict(lambda:collections.Counter());context=collections.defaultdict(lambda:collections.Counter());dominant=collections.defaultdict(lambda:collections.Counter())
  max_residual=max_horizontal_difference=max_face_reconstruction=0.;last_event=0
  with(folder/'udt_negative_crossing_faces.csv').open()as file:
   faces=csv.DictReader(file)
   for row in dictionary_rows(folder/'udt_negative_crossings.csv'):
    eid=int(row['event_id']);cell=int(row['native_cell_index']);kind=row['kind'];step=int(row['step_index']);dt=float(row['dt_s']);before=float(row['previous_volume_m3']);after=float(row['new_volume_m3']);horizontal=float(row['horizontal_applied_m3']);rain=float(row['rain_applied_m3']);increment=float(row['rain_postadd_increment_m3']);t=float(row['after_time_s']);ff=list(itertools.islice(faces,4))
    if eid!=last_event+1 or len(ff)!=4 or any(int(f['event_id'])!=eid or int(f['native_cell_index'])!=cell for f in ff)or[f['direction']for f in ff]!=['west','east','south','north']:raise ValueError('Unbound or missing crossing faces')
    last_event=eid
    if celllookup[cell]<0 or not(0<step<=nstep)or t!=times[step]or dt!=ledger['actual_step_dt'][step]or abs(float(row['before_time_s'])-times[step-1])>1e-8:raise ValueError('Crossing state clock mismatch')
    if not((kind=='down'and before>=-.001 and after<-.001)or(kind=='up'and before<-.001 and after>=-.001)):raise ValueError('Invalid crossing')
    signed=np.array([float(f['signed_inward_volume_m3'])for f in ff]);groups=np.array([int(f['boundary_group_code'])for f in ff]);q=np.array([float(f['q_m2_s'])for f in ff],dtype='f4');width=np.array([float(f['native_transverse_width_m'])for f in ff],dtype='f4')
    calc=(q*width*np.float32(dt)).astype('f8')*np.array([1,-1,1,-1]);max_face_reconstruction=max(max_face_reconstruction,float(max(abs(calc-signed))))
    if np.any(~np.isfinite(signed))or np.any(~np.isin(groups,codes)):raise ValueError('Invalid face values/group')
    total=float(signed.sum());reported=float(row['four_face_sum_m3']);rounding=float(row['horizontal_minus_four_faces_m3']);remainder=after-before-horizontal-increment
    if abs(total-reported)>1e-10*max(1,abs(total))or abs((horizontal-total)-rounding)>1e-10*max(1,abs(horizontal)):raise ValueError('Face accounting mismatch')
    max_residual=max(max_residual,abs(remainder-float(row['storage_update_remainder_m3'])));max_horizontal_difference=max(max_horizontal_difference,abs(horizontal-total))
    inward=np.maximum(signed,0);outward=np.maximum(-signed,0);bnd=np.array([int(f['neighbor_mask'])==2 for f in ff]);b_out=float(outward[bnd].sum());internal_out=float(outward[~bnd].sum());dominant_code=int(groups[int(np.argmax(outward))])if outward.max()>0 else -1
    source='Boundary outflow dominant'if b_out>internal_out else'Interior outflow dominant'if internal_out>b_out else'Tied outflow'
    first_down=kind=='down'and down[cell]==0
    if kind=='down':
     if active_start[cell]>=0:raise ValueError('Down crossing while already below threshold')
     active_start[cell]=step;down[cell]+=1;lastdown[cell]=t
     if first_down:first[cell]=t
     event_stats['down']+=1;event_stats['down_from_nonnegative']+=before>=0;event_stats['horizontal_depletion']+=horizontal<0;event_stats['nonnegative_rain']+=rain>=0 and increment>=0;event_stats['outflow_exceeds_previous_plus_inflow_and_source']+=float(outward.sum())>before+float(inward.sum())+increment;event_stats[source]+=1;event_stats['boundary_outflow_present']+=b_out>0;event_stats['only_internal_outflow']+=b_out==0 and internal_out>0
     for code in links[cell]:
      context[code]['down']+=1;context[code]['boundary_out_dominant']+=b_out>internal_out;context[code]['internal_out_dominant']+=internal_out>b_out
      if first_down:context[code]['first_down_cells']+=1
     dominant[dominant_code]['all_down']+=1
     if first_down:
      firststats[source]+=1;firststats['horizontal_depletion']+=horizontal<0;firststats['down_from_nonnegative']+=before>=0;dominant[dominant_code]['first_down']+=1
      first_rows.append({'case':case,'native_cell_index':cell,'row_north':int(rr[cell-1]),'col_east':int(cc[cell-1]),'ward_code':int(ward[rr[cell-1],cc[cell-1]]),'water_group_links':';'.join(map(str,sorted(links[cell]))),'before_time_s':float(row['before_time_s']),'first_below_endpoint_s':t,'before_volume_m3':before,'after_volume_m3':after,'horizontal_applied_m3':horizontal,'rain_postadd_increment_m3':increment,'gross_inward_m3':float(inward.sum()),'gross_outward_m3':float(outward.sum()),'boundary_outward_m3':b_out,'interior_outward_m3':internal_out,'dominant_outflow_group_code':dominant_code,'mechanism':source,'storage_update_remainder_m3':float(row['storage_update_remainder_m3'])})
    else:
     if active_start[cell]<0:raise ValueError('Recovery without an active negative episode')
     occupancy[cell]+=step-active_start[cell];active_start[cell]=-1;last_observed[cell]=times[step-1];up[cell]+=1;lastup[cell]=t;event_stats['up']+=1
    for i,f in enumerate(ff):
     code=int(groups[i]);key=(code,kind);facegroup[key]['records']+=1;facegroup[key]['inward']+=float(inward[i]);facegroup[key]['outward']+=float(outward[i]);facegroup[key]['closed_face_nonzero']+=f['face_exists']=='T'and float(f['open_width_m'])==0 and float(f['signed_inward_volume_m3'])!=0
    phase[(kind,min(int(t//1800),7))]+=1
   if next(faces,None)is not None:raise ValueError('Unconsumed extra crossing faces')
  check(case,'all_crossings_join_actual_steps_and_reconstruct_native_face_terms',max_face_reconstruction==0 and max_residual<1e-9,{'max_face_term_difference':max_face_reconstruction,'max_update_receipt_difference':max_residual,'max_grouped_vs_four_face_difference_m3':max_horizontal_difference})
  still=active_start>=0;occupancy[still]+=nstep-active_start[still]+1;last_observed[still]=times[-1]
  check(case,'per_cell_negative_step_counts_and_last_endpoints_reconstructed',np.array_equal(occupancy[ids],d['negative_poststep_count'])and np.array_equal(last_observed[ids],d['last_below_endpoint_s'],equal_nan=True))
  check(case,'down_up_counts_first_times_and_final_negative_state_agree',np.array_equal(down[ids],d['down_crossings'])and np.array_equal(up[ids],d['up_crossings'])and np.array_equal(first[ids],d['first_below_endpoint_s'],equal_nan=True)and np.array_equal((down[ids]-up[ids])==1,d['final_volume_m3']<-.001))
  # Every native-step count is independently reconstructed from crossing transitions.
  diffs=np.zeros(len(times),np.int64)
  for row in dictionary_rows(folder/'udt_negative_crossings.csv'):diffs[int(row['step_index'])]+=1 if row['kind']=='down'else -1
  check(case,'all_step_negative_counts_reconstructed_from_crossings',np.array_equal(np.cumsum(diffs),ledger['negative_native_count']))
  check(case,'global_minimum_matches_all_cell_minima',float(d['minimum_volume_m3'].min())==float(ledger['minimum_native_volume_since_initialization'][-1]))
  closed=sum(v['closed_face_nonzero']for v in facegroup.values());check(case,'no_nonzero_exchange_on_declared_closed_faces',closed==0,{'observations':closed})
  for code in codes:
   selected=[int(cell)for cell in ids if code in links[int(cell)]];neg=[cell for cell in selected if np.isfinite(first[cell])];mins=d['minimum_volume_m3'][celllookup[neg]]if neg else np.array([])
   group_rows.append({'case':case,'water_group_code':code,'water_group_name':schema['groups'][str(code)],'linked_interior_cells':len(selected),'negative_unique_cells':len(neg),'earliest_below_endpoint_s':float(np.nanmin(first[neg]))if neg else None,'minimum_volume_m3':float(mins.min())if len(mins)else None,'worst_volume_endpoint_s':float(d['minimum_time_s'][celllookup[neg]][np.argmin(mins)])if len(mins)else None,'latest_below_endpoint_s':float(np.nanmax(d['last_below_endpoint_s'][celllookup[neg]]))if neg else None,'repeated_crossing_cells':int(np.count_nonzero(down[neg]>1)),'down_crossings':int(down[selected].sum()),'up_crossings':int(up[selected].sum()),'negative_poststep_count':int(d['negative_poststep_count'][celllookup[selected]].sum()),'boundary_dominant_down':int(context[code]['boundary_out_dominant']),'interior_dominant_down':int(context[code]['internal_out_dominant']),'counting':'direct-open memberships, nonexclusive across groups'})
  for(code,kind),stats in facegroup.items():mechanism_rows.append({'case':case,'boundary_group_code':code,'water_group_name':schema['groups'][str(code)]if code else'Interior or absent face','kind':kind,**dict(stats),'scope':'crossing-face observations; not full-event water budget'})
  for code,stats in dominant.items():mechanism_rows.append({'case':case,'dominant_outflow_group_code':code,'water_group_name':schema['groups'].get(str(code),'No outward face'),'kind':'dominant_down',**dict(stats),'scope':'one largest outgoing face per crossing; first_down is unique-cell count'})
  comparison={};peaks_equal=True;states_equal=True;max_peak_diff=0.;max_state_diff=0.
  with Dataset(folder/'sfincs_map.nc')as new,Dataset(oldfolder/'sfincs_map.nc')as old:
   for name in ['zsmax','zb','msk']:
    a=arr(new[name][:]);b=arr(old[name][:]);equal=np.array_equal(a,b,equal_nan=True);comparison[name]=equal
    peaks_equal &=equal
    if name=='zsmax':both=np.isfinite(a)&np.isfinite(b);max_peak_diff=float(np.max(abs(a[both]-b[both])))
   for j in range(len(new['time'])):
    for name in ['zs','h','subgrid_volume','u','v']:
     a=arr(new[name][j]);b=arr(old[name][j]);states_equal &=bool(np.array_equal(a,b,equal_nan=True));both=np.isfinite(a)&np.isfinite(b)
     if both.any():max_state_diff=max(max_state_diff,float(np.max(abs(a[both]-b[both]))))
   peak=np.flipud(arr(new['zsmax'][0]));zmin=np.flipud(arr(new['zb'][:]))
  check(case,'diagnostics_leave_all_written_hydraulic_states_and_peak_unchanged',peaks_equal and states_equal,{'max_peak_difference_m':max_peak_diff,'max_any_state_field_difference':max_state_diff})
  repeated=int(np.count_nonzero(d['down_crossings']>1));summaries[case]={'native_cells':len(d),'ever_negative_cells':int(nativeflag.sum()),'down_crossing_events':int(event_stats['down']),'up_crossing_events':int(event_stats['up']),'repeated_crossing_cells':repeated,'final_negative_cells':int(np.count_nonzero(d['final_volume_m3']<-.001)),'first_negative_endpoint_s':float(np.nanmin(d['first_below_endpoint_s'])),'minimum_volume_m3':float(d['minimum_volume_m3'].min()),'most_negative_cell_minimum_time_s':float(d['minimum_time_s'][np.argmin(d['minimum_volume_m3'])]),'all_down_mechanisms':dict(event_stats),'first_down_mechanisms':dict(firststats),'max_face_reconstruction_difference_m3':max_face_reconstruction,'max_grouped_vs_face_sum_difference_m3':max_horizontal_difference,'all_written_states_and_peaks_identical_to_previous':peaks_equal and states_equal,'closed_face_nonzero_observations':int(closed),'exact_continuous_crossing_time':False,'propagated_error_bounded':False,'reference_release':False,'training':False}
  datasets[case]={'d':d,'rr':rr,'cc':cc,'ever':ever,'peak':peak,'zmin':zmin,'first':first,'min':d['minimum_volume_m3']}
  # Keep a georeferenced exact-endpoint diagnostic, separate from ground-depth labels.
  timegrid=np.full(valid.shape,np.nan);timegrid[rr[ids-1],cc[ids-1]]=d['first_below_endpoint_s'];mingrid=np.full(valid.shape,np.nan);mingrid[rr[ids-1],cc[ids-1]]=d['minimum_volume_m3']
  extra={}
  for column,label in [('minimum_time_s','Minimum Volume Actual Endpoint'),('last_below_endpoint_s','Last Below Actual Endpoint'),('negative_poststep_count','Negative Poststep Count')]:
   grid=np.full(valid.shape,np.nan);grid[rr[ids-1],cc[ids-1]]=d[column];extra[label]=grid
  np.savez_compressed(out/(case+'_actual_negative_times.npz'),**{'First Below Actual Endpoint':timegrid,'Minimum Signed Native Volume':mingrid,'Online Ever Negative':ever,**extra})
  print('ACTUAL TRACE SUMMARY',case,json.dumps(summaries[case]),flush=True)
 # Reconstruct current candidate label bounds on exactly the original fine ground.
 ordered=[r['case_id']for r in runs];peaks=np.array([datasets[c]['peak']for c in ordered]);mins=np.array([datasets[c]['zmin']for c in ordered]);lo=np.full(peaks.shape,np.nan);up=lo.copy();barrier=np.full(valid.shape,np.inf)
 groundbase=ROOT/'data/processed/three_ward_sfincs_scenarios';con=ROOT/'data/processed/three_ward_urban_constraints'
 with rasterio.open(groundbase/'terrain_sampling_0_50m.tif')as dem,rasterio.open(con/'constraints_0_50m.tif')as cls,rasterio.open(con/'hydraulic_obstacle_0_50m.tif')as hyd:
  for r in range(0,1012,32):
   n=min(32,1012-r);win=Window(0,r*20,770*20,n*20);a,b,ga,ra,ba=fine_peaks(dem.read(1,window=win),cls.read(1,window=win),hyd.read(1,window=win),valid[r:r+n],peaks[:,r:r+n],mins[:,r:r+n]);assert np.array_equal(ga,area[r:r+n]);lo[:,r:r+n]=a;up[:,r:r+n]=b;barrier[r:r+n]=ba
 coverage=[];wetrows=[];patchrows=[]
 graph=readnp(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz');nodeflat=graph['Node Grid Index'];edges=graph['Surface Edge Index'];adj=coo_matrix((np.ones(edges.shape[1],np.uint8),(edges[0],edges[1])),shape=(len(nodeflat),len(nodeflat))).tocsr()
 for k,case in enumerate(ordered):
  x=datasets[case];bar=valid&np.isfinite(x['peak'])&np.isfinite(barrier)&(x['peak']>=barrier-.01);masks=classify_labels(lo[k],up[k],valid,x['ever'],bar,interval_source_known=True)
  old=readnp(ROOT/'data/processed/three_ward_negative_volume_audit_v1'/(case+'_audit_masks.npz'))
  check(case,'new_peak_labels_online_quarantine_and_intervals_match_prior_evidence',np.array_equal(lo[k],old['Peak Mean Depth Lower'],equal_nan=True)and np.array_equal(up[k],old['Peak Mean Depth Upper'],equal_nan=True)and all(np.array_equal(masks[key],old[key])for key in masks))
  scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3),'Contributing area':valid&(ward==0)}
  for name,sel in scopes.items():
   coverage.append({'case':case,'scope':name,'ground_km2':float(area[sel].sum()/1e6),'negative_cells':int((sel&x['ever']).sum()),'negative_ground_km2':float(area[sel&x['ever']].sum()/1e6),'quarantine_ground_km2':float(area[sel&masks['Quarantine Mask']].sum()/1e6),'candidate_admitted_ground_km2':float(area[sel&masks['Admitted Mask']].sum()/1e6)})
   for th in POLICY['peak_thresholds_m']:
    wet=sel&(lo[k]>th);den=float(area[wet].sum());wetrows.append({'case':case,'scope':name,'threshold_m':th,'certain_wet_ground_km2':den/1e6,'negative_overlap_km2':float(area[wet&x['ever']].sum()/1e6),'negative_overlap_pct':float(area[wet&x['ever']].sum()/den*100)if den else None,'quarantine_overlap_pct':float(area[wet&masks['Quarantine Mask']].sum()/den*100)if den else None})
  for th in [.1,.3]:
   ids0=np.flatnonzero((lo[k].ravel()[nodeflat]>th));nc,labels=connected_components(adj[ids0][:,ids0],directed=False);flat=nodeflat[ids0];sizes=np.bincount(labels,weights=area.ravel()[flat],minlength=nc)
   for rank,comp in enumerate(np.argsort(sizes)[::-1][:10],1):
    ii=ids0[labels==comp];ff=nodeflat[ii];keep=~masks['Quarantine Mask'].ravel()[ff];ret=ii[keep];cn,cl=connected_components(adj[ret][:,ret],directed=False);largest=float(np.bincount(cl,weights=area.ravel()[ff[keep]],minlength=cn).max())if len(ret)else 0.
    patchrows.append({'case':case,'threshold_m':th,'rank':rank,'ground_km2':float(sizes[comp]/1e6),'negative_overlap_pct':float(area.ravel()[ff[x['ever'].ravel()[ff]]].sum()/sizes[comp]*100),'largest_retained_support_pct':largest/sizes[comp]*100,'meaning':'peak-exceedance support only; not a simultaneous pool or actual drainage route'})
  np.savez_compressed(out/(case+'_candidate_reference.npz'),**{'Peak Mean Depth Lower':lo[k],'Peak Mean Depth Upper':up[k],'Online Ever Negative':x['ever'],'Assumed Barrier Contact':bar,**masks})
 csvwrite(report/('water_group_cell_summary.csv'if only is None else only+'_water_group_cell_summary.csv'),group_rows);csvwrite(report/('crossing_face_mechanisms.csv'if only is None else only+'_crossing_face_mechanisms.csv'),mechanism_rows);csvwrite(report/('first_crossing_evidence.csv'if only is None else only+'_first_crossing_evidence.csv'),first_rows)
 for filename,data in [('scope_coverage.csv',coverage),('wet_support_exposure.csv',wetrows),('principal_peak_regions.csv',patchrows)]:csvwrite(report/((only+'_')+filename if only else filename),data)
 result={'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'cases':summaries,'hydrodynamic_runs':len(runs),'new_parent_events':0,'training':0,'batch_58':0,'formal_reference_release':False,'old_registry_unchanged':True,'policy_unchanged':True,'actual_crossing_endpoint_data_available':True,'real_world_flood_accuracy_verified':False}
 dump(report/('summary.json'if only is None else only+'_summary.json'),result)
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--only',choices=CASES);p.add_argument('--skip-budget',action='store_true');p.add_argument('--recheck-native-reductions',action='store_true');a=p.parse_args()
 if a.recheck_native_reductions:recheck_native_reductions()
 else:main(a.only,a.skip_budget)
