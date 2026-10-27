"""Read-only two-training-parent audit. No optimizer, fit loop or solver invocation.
Writes new diagnostic outputs only. Original sources, masks and parameters are immutable.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,hashlib,csv,html
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import arrays,read,write,sha,load_label
from load_three_ward_minimum_scenarios import find_case,load_forcing
from reference_label_policy import classify_labels
from train_three_ward_unet_exact_pilot import ConditionalUNet
from three_ward_surrogate_protocol_v2 import exact_loss,evaluate_depth
from run_three_ward_unet_controls_v2 import fingerprint,csv_write
IDS=['train-01','train-04']
NAME='three_ward_two_parent_alignment_condition_audit_v1'
OUT=ROOT/'data/processed'/NAME
REPORT=ROOT/'data/exp/estimation'/NAME
PRIOR=ROOT/'data/processed/three_ward_two_parent_fit_v1'
OLDREPORT=ROOT/'data/exp/estimation/three_ward_two_parent_fit_v1'
SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'

def mh(m):return hashlib.sha256(np.ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
def weighted(v,a):return float(np.average(v,weights=a))
def corr(x,y,a):
 x=np.asarray(x,dtype='f8');y=np.asarray(y,dtype='f8');xx=x-weighted(x,a);yy=y-weighted(y,a)
 den=np.sqrt(np.sum(a*xx*xx)*np.sum(a*yy*yy))
 return float(np.sum(a*xx*yy)/den) if den>0 else None

def mean_curve(t,v):
 # Independent cumulative piecewise-linear antiderivative at minute boundaries.
 dt=np.diff(t);slope=np.diff(v,axis=0)/dt[:,None]
 cum=np.vstack([np.zeros(v.shape[1]),np.cumsum((v[:-1]+v[1:])*dt[:,None]/2,axis=0)])
 q=np.arange(0,14401,60);ix=np.clip(np.searchsorted(t,q,side='right')-1,0,len(t)-2);d=q-t[ix]
 ant=cum[ix]+v[ix]*d[:,None]+slope[ix]*d[:,None]**2/2
 return np.diff(ant,axis=0)/60

def main():
 if (OUT/'freeze.json').exists():raise FileExistsError('Frozen audit; no in-place replay')
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 checks=[];protected={};access=[]
 def bind(p):
  p=Path(p);rel=str(p.relative_to(ROOT));protected[rel]=sha(p);return p
 def check(n,ok,details=None):
  checks.append({'check':n,'pass':bool(ok),'details':details})
  if not ok:raise AssertionError(n+': '+str(details))
  print('PASS',n,flush=True)
 oldp=read(bind(PRIOR/'protocol.json'));oldfreeze=read(bind(PRIOR/'freeze.json'))
 for rel,h in {**oldp['source_sha256'],**oldfreeze['files_sha256']}.items():
  check('frozen_'+rel,sha(ROOT/rel)==h);protected[rel]=h
 z=arrays(bind(SOURCE/'prepared_inputs.npz'));norm=read(bind(SOURCE/'normalization.json'));protocol=read(bind(SOURCE/'protocol.json'))
 records=[next(r for r in protocol['records'] if r['case_id']==cid) for cid in IDS]
 labels=[]
 for r in records:
  check(r['case_id']+'_original_train_partition',r['split']=='train' and r['case_id'] in oldp['formal_train_case_ids_unchanged'])
  labels.append(load_label(r['case_id'],'train'));access.append({'case_id':r['case_id'],'purpose':'train_diagnostic','path':r['label_path']});bind(ROOT/r['label_path'])
 support=arrays(bind(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz'))
 static=arrays(bind(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz'))
 graph=arrays(bind(ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz'))
 geom=arrays(bind(ROOT/'data/processed/three_ward_minimum_scenarios_v1/shared_geometry.npz'))
 area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];H,W=area.shape
 y=np.array([l['Peak Mean Depth Lower'].astype('f4') for l in labels]);masks=np.array([l['Exact Label Mask']&eligible for l in labels]);common=masks.all(0)&(ward>0)
 check('canonical_grid_affine_crs',list(support['Affine Transform'])==[10.,0.,-11360.,0.,-10.,-32320.] and int(support['EPSG'])==6677 and (H,W)==(1012,770))
 for k in ['Ground Area','Eligible Ground Mask','Ward Code']:check('prepared_static_support_'+k,np.array_equal(z[k],support[k]) and np.array_equal(z[k],static[k]))
 check('geometry_ground_area',np.array_equal(area,geom['Ground Area 10m']))
 ids=graph['Node Grid Index'];rr,cc=np.unravel_index(ids,(H,W));check('graph_node_index_and_position',np.array_equal(ids,np.flatnonzero(eligible)) and np.array_equal(graph['Node Ground Area'],area[eligible]) and np.max(abs(graph['Node Position']-np.c_[-11360+(cc+.5)*10,-32320-(rr+.5)*10]))<1e-5)
 check('graph_nodes_static_features',np.array_equal(graph['Node Features'],static['Static Tensor'][:,rr,cc].T) and np.array_equal(graph['Node Feature Valid'],static['Static Feature Valid'][:,rr,cc].T))
 br=graph['Boundary Point Row'];bc=graph['Boundary Point Column'];count=np.zeros_like(area);sums=np.zeros_like(area);np.add.at(count,(br,bc),1);np.add.at(sums,(br,bc),geom['Boundary Low Stage']);low=np.divide(sums,count,out=np.zeros_like(sums),where=count>0)
 check('boundary_point_order_and_group',np.array_equal(graph['Boundary Point Position'],geom['Boundary Point Position']) and np.array_equal(graph['Boundary Group Index'],geom['Boundary Group Index']))
 raw=np.concatenate([static['Static Tensor'].astype('f8'),low[None]],axis=0);valid=np.concatenate([static['Static Feature Valid'],(count>0)[None]],axis=0)
 rebuilt=np.where(valid,(raw-np.array(norm['static_mean'])[:,None,None])/np.array(norm['static_scale'])[:,None,None],0).astype('f4')
 check('all58_static_channels_reproduce',np.array_equal(rebuilt,z['Static Input'][:29]) and np.array_equal(valid.astype('f4'),z['Static Input'][29:]))
 check('normalization_original_train_only',norm['fit_case_ids']==oldp['formal_train_case_ids_unchanged'] and norm['excluded_validation'] and not norm['target_scaled'])
 manifest=read(bind(ROOT/'data/processed/three_ward_reference_setup_pilot_v1/pilot_manifest.json'));runs={r['case_id']:r for r in manifest['runs'] if r['case_id'] in IDS}
 peak=[];zmin=[];conditions=[];curves=[];encoding=[]
 for i,cid in enumerate(IDS):
  r=runs[cid];folder=ROOT/r['folder'];access.append({'case_id':cid,'purpose':'native_peak_spatial_reconstruction','path':str(folder.relative_to(ROOT))})
  with Dataset(bind(folder/'sfincs_map.nc')) as d:
   d.set_auto_mask(False)
   def ar(k,j=None):
    v=np.array(d[k][:] if j is None else d[k][j],dtype='f4');return np.where(v==getattr(d[k],'_FillValue',-9999),np.nan,v)
   x=ar('x');yy=ar('y');native=ar('msk');peak.append(np.flipud(ar('zsmax',0)));zmin.append(np.flipud(ar('zb')))
   check(cid+'_native_coordinate_orientation',np.max(abs(x-(-11360+(np.arange(W)[None,:]+.5)*10)))<1e-5 and np.max(abs(np.flipud(yy)-(-32320-(np.arange(H)[:,None]+.5)*10)))<1e-5 and np.array_equal(np.flipud(native)==1,eligible))
  ix=np.fromfile(bind(folder/'sfincs.ind'),'<i4');nr=H-1-(ix[1:]-1)%H;nc=(ix[1:]-1)//H;nm=np.fromfile(bind(folder/'sfincs.msk'),'u1')
  check(cid+'_native_sparse_to_north_row',len(nm)==ix[0] and len(np.unique(ix[1:]))==ix[0] and np.array_equal(np.flipud(native)[nr,nc],nm))
  case=find_case(cid);bind(ROOT/'data/processed/three_ward_minimum_scenarios_v1'/case['forcing_file']);f=load_forcing(cid);t=f['Event Time'];v=np.c_[f['Physical Rainfall Rate'],f['River Stage Increment By Group']];curve=mean_curve(t,v);curves.append(curve);j=list(z['Case ID']).index(cid);condition=((curve-norm['forcing_mean'])/norm['forcing_scale']).astype('f4').reshape(-1);conditions.append(z['Forcing Input'][j].copy())
  cerr=float(np.max(abs(curve-z['Minute Forcing Mean'][j])));eerr=float(np.max(abs(condition-conditions[-1])));inte=float(np.max(abs(curve.sum(0)*60-np.trapezoid(v,t,axis=0))))
  check(cid+'_minute_curve_independent_integral',cerr<1e-10 and inte<1e-7,{'curve_error':cerr,'integral_error':inte})
  check(cid+'_minute_channel_reshape_and_case_binding',eerr<2e-6 and j==[0,3][i],{'encoded_max_error':eerr,'prepared_index':j})
  with Dataset(bind(folder/'roof_surface_precip.nc')) as d:
   d.set_auto_mask(False);rt=np.array(d['time'][:])*60;rx=np.array(d['x'][:]);ry=np.array(d['y'][:]);rscalar=np.interp(rt,t,v[:,0]);err=0.
   for k in sorted({0,int(np.argmax(rscalar)),len(rt)//2}):
    actual=np.flipud(np.asarray(d['Precipitation'][k,1:-1,1:-1]));err=max(err,float(np.max(abs(actual-rscalar[k]*geom['Effective Rainfall Weight 10m']))))
   check(cid+'_rain_grid_corner_to_centre',np.max(abs(rx[1:-1]+5-x[0]))<1e-5 and np.max(abs(ry[1:-1]+5-yy[:,0]))<1e-5 and err<.01,{'sampled_fields_max_mm_h_error':err})
  stage=np.loadtxt(bind(folder/'sfincs.bzs'));expect=geom['Boundary Low Stage'][None]+np.stack([np.interp(stage[:,0],t,v[:,j+1]) for j in range(7)],axis=1)[:,geom['Boundary Group Index']]
  check(cid+'_stage_point_group_and_time_encoding',np.max(abs(stage[:,1:]-expect))<1e-8)
  encoding.append({'case_id':cid,'prepared_index':j,'rain_total_mm':float(np.trapezoid(v[:,0],t)/3600),'rain_peak_mm_h':float(v[:,0].max()),'stage_peak_m':float(v[:,1:].max()),'minute_error':cerr,'condition_error':eerr,'integral_error':inte,'physical_channels':8,'encoded_values':1920})
 conditions=np.array(conditions);curves=np.array(curves);check('distinct_finite_conditions',np.isfinite(conditions).all() and not np.array_equal(conditions[0],conditions[1]))
 # Independent vectorized fine-ground integral, not the original Numba function.
 lo=np.full((2,H,W),np.nan);up=lo.copy();rebuilt_area=np.zeros_like(area);terrain_error={k:0. for k in ['mean','minimum','maximum','road_fraction','building_fraction']};depth_errors=[]
 paths=[ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif',ROOT/'data/processed/three_ward_urban_constraints/constraints_0_50m.tif']
 for p in paths:bind(p)
 with rasterio.open(paths[0]) as dem,rasterio.open(paths[1]) as cl:
  check('fine_dem_classification_affine',dem.transform==cl.transform and tuple(dem.transform)[:6]==(.5,0.,-11360.,0.,-.5,-32320.) and dem.shape==(H*20,W*20) and dem.crs.to_epsg()==6677)
  for start in range(0,H,16):
   n=min(16,H-start);win=Window(0,start*20,W*20,n*20);zz=dem.read(1,window=win).reshape(n,20,W,20).transpose(0,2,1,3).reshape(n,W,400);cls=cl.read(1,window=win).reshape(n,20,W,20).transpose(0,2,1,3).reshape(n,W,400)
   ground=(cls!=1)&np.isfinite(zz);cnt=ground.sum(-1);ok=eligible[start:start+n];den=np.maximum(cnt,1);rebuilt_area[start:start+n]=np.where(ok,cnt*.25,0)
   vals={'mean':np.where(ground,zz,0).sum(-1,dtype='f8')/den,'minimum':np.where(ground,zz,np.inf).min(-1),'maximum':np.where(ground,zz,-np.inf).max(-1),'road_fraction':(cls==2).sum(-1)/400.,'building_fraction':(cls==1).sum(-1)/400.}
   for name,j in [('mean',0),('minimum',1),('maximum',2),('road_fraction',6),('building_fraction',5)]:terrain_error[name]=max(terrain_error[name],float(np.max(abs(vals[name][ok]-static['Static Tensor'][j,start:start+n][ok]))))
   for i in range(2):
    pp=peak[i][start:start+n];zm=zmin[i][start:start+n];finite=np.isfinite(pp);d=np.maximum(pp[:,:,None]-zz,0);dd=np.where(ground,d,0).sum(-1,dtype='f8')/den;upper=np.where(ground,np.maximum(zm.astype('f8')[:,:,None]+.01-zz,0),0).sum(-1)/den
    lo[i,start:start+n]=np.where(ok,np.where(finite,dd,0),np.nan);up[i,start:start+n]=np.where(ok,np.where(finite,dd,upper),np.nan)
   if start%160==0:print('FINE GROUND AUDIT',start,'/',H,flush=True)
 check('fine_ground_area_rebuilt',np.array_equal(rebuilt_area,area))
 check('fine_static_terrain_road_building_rebuilt',max(terrain_error.values())<3e-6,terrain_error)
 for i,cid in enumerate(IDS):
  err=max(float(np.max(abs(lo[i,eligible]-labels[i]['Peak Mean Depth Lower'][eligible]))),float(np.max(abs(up[i,eligible]-labels[i]['Peak Mean Depth Upper'][eligible]))));depth_errors.append({'case_id':cid,'max_bound_error_m':err,'exact_mask_sha256':mh(masks[i])})
  check(cid+'_full_domain_depth_bounds_independently_rebuilt',err<2e-6,err)
  rebuilt_masks=classify_labels(labels[i]['Peak Mean Depth Lower'],labels[i]['Peak Mean Depth Upper'],eligible,labels[i]['Online Ever Negative'],labels[i]['Assumed Barrier Contact'])
  check(cid+'_original_quality_masks_reproduce',all(np.array_equal(v,labels[i][k]) for k,v in rebuilt_masks.items()) and mh(masks[i])==oldp['selected_records'][i]['original_exact_eligible_mask_sha256'])
 write(REPORT/'alignment.json',{'grid':{'shape':[H,W],'epsg':6677,'row_direction':'north_to_south','column_direction':'west_to_east','pixel_center':'x=-11360+(c+0.5)*10; y=-32320-(r+0.5)*10'},'terrain_errors':terrain_error,'depth_reconstruction':depth_errors,'encoding':encoding,'subminute_limit':'Minute means conserve forcing integral; native within-minute shape is not recoverable from this ML input.'})
 # Frozen forward probes. No optimizer constructed. Only two existing conditions and
 # two rain/stage hybrids; hybrids are diagnostic OOD queries, not hydraulic truth.
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=ConditionalUNet().to(device);model.eval();xt=torch.from_numpy(z['Static Input'][None]).to(device);ct=torch.from_numpy(conditions).to(device);responses={};embeddings=[];films=[];cross=[];difference=[];saved={}
 regions={'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3}
 for step in [0,440,480]:
  state=torch.load(bind(PRIOR/f'step_{step:04}.pt'),map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);fp=fingerprint(model);preds=[]
  with torch.inference_mode():
   emb=model.condition(ct).cpu().numpy();embeddings.append({'step':step,'condition_input_l2':float(np.linalg.norm(conditions[1]-conditions[0])),'embedding_difference_l2':float(np.linalg.norm(emb[1]-emb[0])),'embedding_norm_01':float(np.linalg.norm(emb[0])),'embedding_norm_04':float(np.linalg.norm(emb[1]))})
   for name,block in [(f'enc.{j}',b) for j,b in enumerate(model.enc)]+[(f'dec.{j}',b) for j,b in enumerate(model.dec)]:
    v=block.film(torch.from_numpy(emb).to(device)).cpu().numpy();k=v.shape[1]//2;films.append({'step':step,'module':name,'scale_difference_rms':float(np.sqrt(np.mean((v[1,:k]-v[0,:k])**2))),'offset_difference_rms':float(np.sqrt(np.mean((v[1,k:]-v[0,k:])**2))),'film_weight_norm':float(torch.linalg.vector_norm(block.film.weight).cpu())})
   for i in range(2):preds.append(model(xt,ct[i:i+1])[0,0].cpu().numpy())
  preds=np.array(preds);responses[step]=preds;stored=arrays(PRIOR/({0:'initial_predictions.npz',440:'best_diagnostic_predictions.npz',480:'endpoint_predictions.npz'}[step]))['Peak Mean Depth'];check(f'step{step}_frozen_forward_replay',np.max(abs(preds[:,eligible]-stored[:,eligible]))<1e-5 and fp==fingerprint(model))
  saved['Prediction Step '+str(step)]=np.where(eligible[None],preds,np.nan)
  for src,cid in enumerate(IDS):
   for dest,tid in enumerate(IDS):
    for scope,region in regions.items():
     # Common exact masks for the 2x2 comparison, unchanged own mask separately.
     m=common&region;v=evaluate_depth(preds[src],y[dest],area,m);cross.append({'step':step,'condition_case':cid,'target_case':tid,'scope':scope,'support':'original_exact_intersection',**v})
  for scope,region in regions.items():
   m=common&region;a=area[m];dy=y[1,m].astype('f8')-y[0,m];dp=preds[1,m].astype('f8')-preds[0,m];wet=(y[0,m]>.1)|(y[1,m]>.1);sig=abs(dy)>.05
   difference.append({'step':step,'scope':scope,'common_support_km2':float(a.sum()/1e6),'reference_field_correlation':corr(y[0,m],y[1,m],a),'prediction_field_correlation':corr(preds[0,m],preds[1,m],a),'reference_mean_abs_difference_m':weighted(abs(dy),a),'prediction_mean_abs_difference_m':weighted(abs(dp),a),'difference_MAE_m':weighted(abs(dp-dy),a),'difference_correlation':corr(dp,dy,a),'difference_correlation_wet_union':corr(dp[wet],dy[wet],a[wet]),'direction_agreement_gt5cm':weighted((np.sign(dp[sig])==np.sign(dy[sig])).astype(float),a[sig]),'reference_gt5cm_difference_km2':float(a[sig].sum()/1e6)})
  check(f'step{step}_weights_unchanged',fp==fingerprint(model))
 # Rain-stage hybrid response: each physical channel standardized with its same
 # frozen original training scale, swap normalized columns only, never fit scaling.
 hybrids=[]
 with torch.inference_mode():
  for rain,stage in [(0,1),(1,0)]:
   h=conditions[stage].reshape(240,8).copy();h[:,0]=conditions[rain].reshape(240,8)[:,0];pred=model(xt,torch.from_numpy(h.reshape(1,-1)).to(device))[0,0].cpu().numpy();key=f'Hybrid Rain {IDS[rain]} Stage {IDS[stage]}';saved[key]=np.where(eligible,pred,np.nan)
   for scope,region in regions.items():
    m=common&region;a=area[m];hybrids.append({'rain_case':IDS[rain],'stage_case':IDS[stage],'scope':scope,'mean_depth_m':weighted(pred[m],a),'wet_area_km2':float(area[m&(pred>.1)].sum()/1e6),'mean_abs_change_vs_stage_donor_m':weighted(abs(pred[m]-responses[480][stage,m]),a),'mean_abs_change_vs_rain_donor_m':weighted(abs(pred[m]-responses[480][rain,m]),a),'interpretation':'OOD hybrid sensitivity only; no matching hydraulic label or causal attribution'})
 check('endpoint_weights_unchanged_after_hybrids',fp==fingerprint(model))
 # Zero-update gradients at the frozen endpoint: local objective compatibility,
 # not an optimization step, proof of historic interference or capacity failure.
 grads=[];gradient_losses=[]
 for i,cid in enumerate(IDS):
  model.zero_grad(set_to_none=True);pred=model(xt,ct[i:i+1])[0,0];loss,payload=exact_loss(pred,torch.from_numpy(y[i]).to(device),torch.from_numpy(area.astype('f4')).to(device),torch.from_numpy(masks[i]).to(device),'stratified_huber');loss.backward();grads.append({k:v.grad.detach().cpu().numpy().astype('f8') for k,v in model.named_parameters()});gradient_losses.append({'case_id':cid,**payload});check(cid+'_zero_update_gradient_finite',all(np.isfinite(v).all() for v in grads[-1].values()) and fp==fingerprint(model))
 gradient=[]
 for group in ['all','condition','spatial_convolutions','film','head']:
  def selected(k):return group=='all' or (group=='condition' and k.startswith('condition.')) or (group=='film' and '.film.' in k) or (group=='spatial_convolutions' and '.net.' in k) or (group=='head' and k.startswith('head.'))
  keys=[k for k in grads[0] if selected(k)];g=[np.concatenate([d[k].reshape(-1) for k in keys]) for d in grads];dot=float(np.dot(*g));norms=[float(np.linalg.norm(v)) for v in g];gradient.append({'group':group,'cosine':dot/(norms[0]*norms[1]) if min(norms)>0 else None,'norm_01':norms[0],'norm_04':norms[1],'dot':dot})
 model.zero_grad(set_to_none=True);check('zero_updates_and_weights_unchanged',fp==fingerprint(model))
 steps=[json.loads(s) for s in (OLDREPORT/'optimizer_steps.jsonl').read_text().splitlines()];telemetry=[]
 for cid in IDS:
  for first,last in [(1,80),(401,480)]:
   ss=[s for j,s in enumerate(steps,1) if s['case_id']==cid and first<=j<=last];telemetry.append({'case_id':cid,'update_window':[first,last],'observations':len(ss),'loss_mean_m2':float(np.mean([s['loss']['value_m2'] for s in ss])),'condition_gradient_median':float(np.median([s['gradient_pre_clip']['module_norms']['condition'] for s in ss])),'gradient_norm_median':float(np.median([s['gradient_pre_clip']['total_norm'] for s in ss])),'condition_updates_positive':sum(s['parameter_change']['condition']['update_norm']>0 for s in ss),'clipped_steps':sum(s['clipping_required'] for s in ss)})
 # Exact paired-error decomposition and fixed reference wetness cohorts.
 cohort=[];decomposition=[]
 for scope,region in regions.items():
  m=common&region;a=area[m];t=y[:,m];v=responses[480][:,m];e=v.astype('f8')-t
  shared=(e[0]+e[1])/2;contrast=(e[1]-e[0])/2;sm=weighted(shared**2,a);dm=weighted(contrast**2,a);total=weighted((e[0]**2+e[1]**2)/2,a)
  check(scope+'_paired_mse_identity',abs(sm+dm-total)<1e-12)
  decomposition.append({'scope':scope,'mean_parent_MSE_m2':total,'shared_error_MSE_m2':sm,'contrast_error_MSE_m2':dm,'shared_fraction':sm/total,'interpretation':'Algebraic paired-error identity, not causal mechanism attribution'})
  for name,g in [('both_nonwet',(t[0]<=.1)&(t[1]<=.1)),('only_train01_wet',(t[0]>.1)&(t[1]<=.1)),('only_train04_wet',(t[0]<=.1)&(t[1]>.1)),('both_wet',(t[0]>.1)&(t[1]>.1))]:
   for i,cid in enumerate(IDS):
    cohort.append({'scope':scope,'cohort':name,'case_id':cid,'support_km2':float(a[g].sum()/1e6),'reference_mean_m':weighted(t[i,g],a[g]),'prediction_mean_m':weighted(v[i,g],a[g]),'MAE_m':weighted(abs(e[i,g]),a[g]),'predicted_wet_km2':float(a[g&(v[i]>.1)].sum()/1e6),'reference_wet_km2':float(a[g&(t[i]>.1)].sum()/1e6)})
 csv_write(REPORT/'reference_wetness_cohorts.csv',cohort);write(REPORT/'paired_error_decomposition.json',decomposition)
 csv_write(REPORT/'cross_condition_target_metrics.csv',cross);csv_write(REPORT/'field_difference_metrics.csv',difference);csv_write(REPORT/'condition_embedding.csv',embeddings);csv_write(REPORT/'film_response.csv',films);csv_write(REPORT/'hybrid_sensitivity.csv',hybrids);csv_write(REPORT/'endpoint_gradient_compatibility.csv',gradient);write(REPORT/'optimization_telemetry.json',telemetry);write(REPORT/'endpoint_gradient_losses.json',gradient_losses)
 np.savez_compressed(OUT/'frozen_diagnostic_fields.npz',**saved,**{'Case ID':np.array(IDS),'Common Exact Mask':common})
 check('all_sources_unchanged_after_analysis',all(sha(ROOT/k)==h for k,h in protected.items()))
 write(REPORT/'provenance.json',{'decision':'MILI-D-20261016-010','source_sha256':protected,'response_access_log':access,'label_allowlist':IDS,'inference_checkpoints':[0,440,480],'optimizer_updates':0,'new_fits':0,'new_hydraulic_runs':0,'validation_response_access':False,'sealed_response_access':False,'formal_split_preserved':True,'quality_masks_preserved':True,'device':str(device),'script_sha256':sha(Path(__file__)),'netcdf_runtime':{'version':__import__('netCDF4').__version__,'module_path':__import__('netCDF4').__file__,'method':'Existing local uv cache exposed to this process; project environment and dependencies not modified'}})
 summary={'alignment_passed':True,'conditions_distinct_and_reconstructed':True,'spatial_reconstruction_max_bound_error_m':max(r['max_bound_error_m'] for r in depth_errors),'encoding':encoding,'embedding_response':embeddings,'endpoint_field_difference':next(r for r in difference if r['step']==480 and r['scope']=='three_wards'),'endpoint_cross_metrics':[r for r in cross if r['step']==480 and r['scope']=='three_wards'],'endpoint_gradient_compatibility':gradient,'hybrid_sensitivity':[r for r in hybrids if r['scope']=='three_wards'],'training_telemetry':telemetry,'reference_cohorts':[r for r in cohort if r['scope']=='three_wards'],'error_decomposition':decomposition,'optimizer_updates':0,'new_fits':0,'sealed_opened':False,'limits':['Read-only two-known-training-event diagnostics; no generic capacity or convergence result.','Hybrids have no hydraulic labels; only conditional sensitivity can be inferred.','Gradient cosine is endpoint-local and parameterization-dependent; it does not isolate historical interference.','Physical geometry accuracy, subminute shape loss and unobserved event generalization remain unverified.']}
 write(REPORT/'summary.json',summary);write(REPORT/'checks.json',{'all_checks_passed':True,'checks':checks});print(json.dumps(summary,indent=2),flush=True)
 make_report(summary,cross,difference,hybrids,gradient,encoding,curves,responses,y,common,area,ward)

def make_report(s,cross,difference,hybrids,gradient,encoding,curves,responses,y,common,area,ward):
 fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for i,cid in enumerate(IDS):
  ax[0,0].plot(np.arange(240)+.5,curves[i,:,0],label=cid)
  ax[0,1].plot(np.arange(240)+.5,curves[i,:,1],label=cid)
 for a,title,unit in [(ax[0,0],'Encoded minute rainfall','mm/h'),(ax[0,1],'Encoded stage increment (group 0)','m')]:a.set(title=title,xlabel='Event minute',ylabel=unit);a.legend();a.grid(alpha=.2)
 d=[r for r in difference if r['scope']=='three_wards'];ax[1,0].plot([r['step'] for r in d],[r['difference_correlation'] for r in d],marker='o');ax[1,0].set(title='Event-difference correlation (step 0 undefined)',xlabel='Stored update',ylabel='Area-weighted correlation',ylim=(0,1));ax[1,0].grid(alpha=.2)
 rows=[r for r in cross if r['step']==480 and r['scope']=='three_wards'];matrix=np.array([[next(r['iou_0.1'] for r in rows if r['condition_case']==a and r['target_case']==b) for b in IDS] for a in IDS]);im=ax[1,1].imshow(matrix,vmin=0,vmax=.3,cmap='Blues');ax[1,1].set(xticks=[0,1],xticklabels=IDS,yticks=[0,1],yticklabels=IDS,xlabel='Reference target',ylabel='Condition supplied',title='Frozen endpoint wet IoU, common exact support')
 for i in range(2):
  for j in range(2):ax[1,1].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
 fig.colorbar(im,ax=ax[1,1]);fig.savefig(REPORT/'condition_response.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(11,5),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32];dy=y[1]-y[0];dp=responses[480][1]-responses[480][0]
 for a,v,title in zip(axes,[dy,dp,dp-dy],['Reference: train-04 minus train-01','Prediction: train-04 minus train-01','Difference response error']):
  im=a.imshow(np.where(common,v,np.nan),origin='upper',extent=extent,cmap='RdBu',vmin=-.5,vmax=.5,interpolation='nearest');a.set(title=title,xlabel='Easting (km)');a.tick_params(labelsize=8)
 fig.colorbar(im,ax=axes,shrink=.7,label='Peak depth difference (m)',extend='both');fig.savefig(REPORT/'whole_ward_difference.png',dpi=160);plt.close(fig)
 def table(rows,cols):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(title)+'</th>' for k,title in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,title in cols)+'</tr>' for r in rows)+'</table></div>'
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-parent alignment and condition audit</title><style>body{font:15px/1.8 system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#172033;background:#fafbfd}h1{font-size:27px}h2{font-size:21px}.box{background:#edf5f2;padding:18px;border-left:5px solid #0f766e}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #cbd5e1}img{width:100%}a{color:#1762af}</style><h1>两父事件联合拟合失败：空间、条件与响应核查</h1><div class="box">只读已有输入、标签、原生峰值、权重和日志；仅对两个原训练父事件做冻结推理及零更新梯度诊断。新增训练 0、参数更新 0、扩样 0、水动力模拟 0；原 12＋4 划分、质量掩膜和归一化不变，验证响应与封存测试未打开。</div>'''
 body+='<h2>空间对齐：未发现接口错位</h2><p>独立核对 EPSG:6677、10 m 仿射、北至南行序、西至东列序、原生稀疏索引、图节点坐标及边界点顺序。全域 58 个输入通道可从原始静态量与固定归一化复现；逐块使用 0.50 m DEM 和原建筑分类独立复算两个事件的地面平均水深上下界、开放地面面积、地形统计、道路与建筑面积。</p><p>两个标签上下界的全域最大重建差为 '+f'{s["spatial_reconstruction_max_bound_error_m"]:.3g}'+' m。原质量掩膜可重现。这里核验的是数据在程序中的空间对应和定义，不能证明源地形及水动力结果的现实准确度。</p>'
 body+='<h2>条件编码：事件不同，接口可复现</h2>'+table(encoding,[('case_id','事件'),('prepared_index','原索引'),('rain_total_mm','总雨量 mm'),('rain_peak_mm_h','峰值 mm/h'),('stage_peak_m','水位增量 m'),('condition_error','编码最大误差')])
 body+='<p>从原秒级结点用独立的分段线性反导数重算 240×8 分钟均值，雨量与水位积分保持一致，按原训练集尺度编码并按分钟×通道顺序展开为 1920 项。两输入既不相同，也没有事件索引或通道混用。七水域的增量在当前情景内一致，是既有生成规则；模型保留组序，绝对低水位由静态边界通道提供。分钟均值仍压缩了分钟内部形状，不能据此认定时间信息完全无损。</p><img src="condition_response.png" alt="Frozen condition encoding and cross-target response">'
 body+='<h2>条件有响应，但空间响应是否对得上</h2>'+table([r for r in difference if r['scope']=='three_wards'],[('step','已存更新数'),('prediction_field_correlation','两预测场相关'),('reference_field_correlation','两参考场相关'),('prediction_mean_abs_difference_m','预测平均绝对变化 m'),('reference_mean_abs_difference_m','参考平均绝对变化 m'),('difference_correlation','事件差场相关'),('difference_MAE_m','事件差场 MAE m')])
 body+='<p>上述比较使用两个原精确掩膜的交集与 Ground Area 加权，只是配对诊断支持；没有改写正式 own-exact 掩膜、训练目标或评分。初始化 FiLM 为零时两事件预测一致是架构预期；训练后条件嵌入和各 FiLM 层已有差异，条件链路并未完全失效。冻结网络能区分事件，并不等于能定位条件引起的真实积水变化。</p><img src="whole_ward_difference.png" alt="Whole-three-ward event-difference fields"><p>显示范围为 ±0.5 m，图外饱和值仍全部计入指标。图中是四小时单元峰值的差异，不是同一时刻水量或地表排水路径。</p>'
 body+='<h2>主要误差：共同偏湿与事件响应放置不准并存</h2>'
 cohorts=s['reference_cohorts'];decomp=next(r for r in s['error_decomposition'] if r['scope']=='three_wards');body+='<p>两参考场同时不超过 0.10 m 的支持为 21.338 km²。其参考平均水深分别为 0.36 cm 与 1.27 cm，预测却为 6.80 cm 与 8.58 cm，仍产生 2.008 km² 与 5.381 km² 的误报。两事件都湿的区域中又存在明显低估。因此不能把问题简化成整体水深统一加减一个常数。</p>'+table(cohorts,[('cohort','固定参考分组'),('case_id','事件'),('support_km2','支持 km²'),('reference_mean_m','参考均值 m'),('prediction_mean_m','预测均值 m'),('predicted_wet_km2','预测湿区 km²')])
 body+='<p>成对误差按共同分量与差异分量作恒等分解，共同分量占平均平方误差 '+f'{decomp["shared_fraction"]*100:.1f}'+'%。这是误差结构描述，不能直接认定静态网络贡献了这个比例。虽然事件差场相关为 0.656，差场 MAE 为 7.28 cm；假设两事件完全不变时的零差场误差为 4.54 cm。相关改善没有带来足够的幅值与空间正确性。</p>'
 body+='<h2>原条件与目标的交叉评分</h2>'+table([r for r in cross if r['step']==480 and r['scope']=='three_wards'],[('condition_case','供给条件'),('target_case','对比标签'),('iou_0.1','湿区 IoU'),('wet_mae_m','湿区 MAE m'),('mae_m','全地面 MAE m')])
 body+='<p>交换原条件后与另一标签的相似度只用于检查身份与条件响应，不能据此重配标签或事后选择模型。雨与水位在两个父事件之间同时改变，交叉评分不能分离雨型、水量和水位的因果作用。</p>'
 body+='<h2>冻结混合条件敏感性与局部梯度</h2>'+table([r for r in hybrids if r['scope']=='three_wards'],[('rain_case','雨曲线来源'),('stage_case','水位来源'),('mean_abs_change_vs_stage_donor_m','换雨引起平均绝对变化 m'),('mean_abs_change_vs_rain_donor_m','换水位引起平均绝对变化 m'),('wet_area_km2','预测湿区 km²')])
 body+='<p>混合输入仅作训练点之外的敏感性探针，没有对应水动力真值，不能称为正确的物理响应或独立消融。没有新情景模拟。</p>'+table(gradient,[('group','参数组'),('cosine','两父事件梯度余弦'),('norm_01','train-01 梯度范数'),('norm_04','train-04 梯度范数')])
 body+='<p>全参数梯度余弦为 +0.131，并非整体相反；输出头为 −0.677，条件编码与 FiLM 分量接近正交。存在局部相反方向，但本次测量不支持“全网络严重梯度冲突已经解释联合失败”的结论。</p>'
 body+='<p>在原终点冻结参数分别求两个原损失的梯度，没有创建优化器或执行更新。负余弦只说明此点两事件的局部下降方向存在冲突；其数值依赖参数化，不能证明历史所有更新都冲突，也不能单独归因为损失、表征容量或共同训练顺序。</p>'
 body+='<h2>当前可支持的判断和下一步</h2><p>没有发现空间翻转、整体平移、事件标签串位、固定归一化或条件展开接口错误。条件链路有梯度、参数更新和冻结响应；主要未解决的是同一静态地理上，事件条件变化如何转化为正确的局部湿区位置和水深。当前证据更支持优先审查空间—条件联合表征及其损失/优化兼容性，不能将失败仅归因于样本不足，也尚不能确定是哪一个因素主导。</p><p>下一步如获授权，可先预登记单父事件拟合与双父事件联合拟合的匹配预算对照，区分共同空间可拟合性和事件间干扰；是否引入空间条件映射、坐标/多尺度通道或调整监督应待该对照。此处仅建议，没有执行新训练或变更正式方案参数。三区整体尺度不变。</p><p>AnaSOP 仅在第 5–7 节记录诊断证据与解释边界。第 3–4 节无需因本轮接口核查改变数据与变量；第 8 节独立泛化、观测和主动措施证据仍待完成。</p><ul>'
 for f in ['summary.json','alignment.json','cross_condition_target_metrics.csv','field_difference_metrics.csv','film_response.csv','condition_embedding.csv','hybrid_sensitivity.csv','endpoint_gradient_compatibility.csv','optimization_telemetry.json','reference_wetness_cohorts.csv','paired_error_decomposition.json','checks.json','provenance.json']:
  body+='<li><a href="'+f+'">'+f+'</a></li>'
 body+='</ul></html>';(REPORT/'review.html').write_text(body)

if __name__=='__main__':main()
