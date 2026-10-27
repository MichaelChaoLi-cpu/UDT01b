"""Scoped admission and training-only input preparation for the authorized exact-label pilot.
No hydraulic execution, old-registry mutation, or sealed response access.
"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from load_three_ward_minimum_scenarios import find_case, load_forcing, load_geometry
NAME='three_ward_unet_exact_pilot_v1'
OUT=ROOT/'data/processed'/NAME
REPORT=ROOT/'data/exp/estimation'/NAME

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def minute_means(t,v):
 """Exact integral of native piecewise-linear curves within each minute."""
 t=np.asarray(t);v=np.asarray(v)
 if v.ndim==1:v=v[:,None]
 assert t[0]==0 and t[-1]==14400 and np.all(np.diff(t)>0)
 out=[]
 for a in range(0,14400,60):
  x=np.r_[a,t[(t>a)&(t<a+60)],a+60]
  y=np.stack([np.interp(x,t,v[:,j]) for j in range(v.shape[1])],axis=1)
  out.append(np.sum((y[:-1]+y[1:])*.5*np.diff(x)[:,None],axis=0)/60)
 return np.array(out)

def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():raise FileExistsError('Pilot already prepared; preserve the frozen protocol')
 src=ROOT/'data/exp/data-preprocessing/three_ward_reference_expansion_12_4_v1'
 summary=read(src/'summary.json');assert summary['all16_numerically_qualified']
 assert read(src/'delivery_validation.json')['all_checks_passed']
 rows=list(csv.DictReader((src/'response_inventory.csv').open()))
 assert len(rows)==16 and sum(r['split']=='train' for r in rows)==12 and sum(r['split']=='validation' for r in rows)==4
 manifest=read(ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/expansion_manifest.json')
 reuse={x['case_id']:x['audit_path'] for x in manifest['reused']}
 hashes={}
 def protect(p):
  p=Path(p);r=str(p.relative_to(ROOT));hashes[r]=sha(p);return r
 for p in [src/'response_inventory.csv',src/'summary.json',src/'delivery_validation.json',ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/freeze.json',ROOT/'data/processed/three_ward_reference_contract_v1/reference_registry.json',ROOT/'src/preprocessing/load_three_ward_reference_labels.py',ROOT/'src/preprocessing/load_three_ward_minimum_scenarios.py',ROOT/'src/preprocessing/reference_label_policy.py',ROOT/'data/processed/three_ward_minimum_scenarios_v1/catalog.json',ROOT/'data/processed/three_ward_minimum_scenarios_v1/generator_config.json']:
  protect(p)
 records=[];curves=[];qc=[]
 for r in rows:
  case=find_case(r['case_id']);assert case['split']==r['split'] and case['parent_id']==r['parent_id'] and case['family_id']==case['parent_id']
  ap=ROOT/reuse.get(r['case_id'],str((src/'expansion'/r['case_id']/'audit.json').relative_to(ROOT)))
  a=read(ap);assert a['all_checks_passed'] and all(a['required_gates'].values()) and a['admission']['technical_reference']
  protect(ap);label=ROOT/r['label_path'];protect(label);d=arrays(label)
  m=d['Exact Label Mask'];lo=d['Peak Mean Depth Lower'];hi=d['Peak Mean Depth Upper']
  assert m.any() and np.isfinite(lo[m]).all() and np.array_equal(lo[m],hi[m]) and (lo[m]>=0).all()
  assert not (m&d['Quarantine Mask']).any() and not (m&d['Interval Label Mask']).any() and not (m&d['Online Ever Negative']).any()
  f=load_forcing(r['case_id']);curve=minute_means(f['Event Time'],np.column_stack([f['Physical Rainfall Rate'],f['River Stage Increment By Group']]))
  native=np.sum(np.diff(f['Event Time'])[:,None]*(np.column_stack([f['Physical Rainfall Rate'],f['River Stage Increment By Group']])[:-1]+np.column_stack([f['Physical Rainfall Rate'],f['River Stage Increment By Group']])[1:])*.5,axis=0)
  err=float(np.max(np.abs(curve.sum(axis=0)*60-native)));assert err<1e-7
  curves.append(curve);qc.append({'case_id':r['case_id'],'integral_abs_error_rate_seconds':err,'exact_cells':int(m.sum())})
  fp=ROOT/'data/processed/three_ward_minimum_scenarios_v1'/case['forcing_file'];protect(fp)
  records.append({k:case[k] for k in ['case_id','parent_id','family_id','split','forcing_sha256']}|{'label_path':r['label_path'],'label_sha256':sha(label),'audit_path':str(ap.relative_to(ROOT)),'purpose':r['split'],'source_forcing_file_sha256':sha(fp),'original_quality_masks_unchanged':True})
 assert len({r['family_id'] for r in records})==16
 sp=ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz';protect(sp);s=arrays(sp)
 gp=ROOT/'data/processed/three_ward_surrogate_inputs/surface_connectivity_graph.npz';protect(gp);g=arrays(gp)
 protect(ROOT/'data/processed/three_ward_minimum_scenarios_v1/shared_geometry.npz');geom=load_geometry()
 v=s['Static Feature Valid']&np.isfinite(s['Static Tensor']);x=s['Static Tensor'].astype('f8');names=list(s['Static Channel Names'].astype(str))
 # One shared geography is repeated in every training instance. Count it once;
 # fitting on 12 identical copies gives identical population moments.
 nrow,ncol=x.shape[1:];rr=g['Boundary Point Row'];cc=g['Boundary Point Column'];b=geom['Boundary Low Stage']
 count=np.zeros((nrow,ncol));sums=np.zeros_like(count);np.add.at(count,(rr,cc),1);np.add.at(sums,(rr,cc),b)
 base=np.divide(sums,count,out=np.zeros_like(sums),where=count>0)
 x=np.concatenate([x,base[None]],axis=0);v=np.concatenate([v,(count>0)[None]],axis=0);names+=['Prescribed Boundary Low Stage']
 means=np.array([c[k].mean() if k.any() else 0 for c,k in zip(x,v)])
 rawstd=np.array([c[k].std() if k.any() else 0 for c,k in zip(x,v)])
 scales=np.where(rawstd>1e-8,rawstd,1.)
 normalized=np.zeros(x.shape,'f4')
 for j in range(len(names)):normalized[j,v[j]]=(x[j,v[j]]-means[j])/scales[j]
 features=np.concatenate([normalized,v.astype('f4')],axis=0)
 ti=np.array([i for i,r in enumerate(records) if r['split']=='train']);curves=np.array(curves)
 fm=curves[ti].mean(axis=(0,1));fs=curves[ti].std(axis=(0,1));fs=np.where(fs>1e-8,fs,1.)
 cond=((curves-fm)/fs).astype('f4')
 np.savez_compressed(OUT/'prepared_inputs.npz',**{'Static Input':features,'Forcing Input':cond.reshape(16,-1),'Minute Forcing Mean':curves,'Case ID':np.array([r['case_id'] for r in records]),'Ground Area':s['Ground Area'],'Eligible Ground Mask':s['Eligible Ground Mask'],'Ward Code':s['Ward Code']})
 norm={'fit_case_ids':[records[i]['case_id'] for i in ti],'excluded_validation':True,'target_scaled':False,'static_names':names,'static_mean':means.tolist(),'static_raw_std':rawstd.tolist(),'static_scale':scales.tolist(),'static_valid_count':[int(k.sum()) for k in v],'static_fit_rule':'Shared fixed training geometry, known values only, counted once; equivalent to 12 identical copies. No response-dependent quality mask as predictor.','forcing_mean':fm.tolist(),'forcing_scale':fs.tolist(),'forcing_order':['Physical Rainfall Rate']+['River Stage Increment Group '+str(j) for j in range(7)],'conditioning_shape':[240,8],'forcing_rule':'Exact piecewise-linear mean in each 60-second bin. Integral preserved, sub-minute shape compressed only for the ML encoding; native forcing files unchanged.','boundary_raster':'Mean low stage of original prescribed boundary points within a 10m cell; known mask supplied; fixed input geometry, not a response.'}
 write(OUT/'normalization.json',norm)
 protocol={'version':NAME,'decision':'MILI-D-20261016-002','prediction':'Four-hour online peak common10m ground-area mean depth in metres','geography':'Full 1012 by 770 common10m grid, shared fixed geometry; no patch splitting','admission':'Scoped trial access only to original Exact Label Mask of 12 train and 4 validation parents. Other masks and original event gates unchanged. Historical registry and loader remain closed.','records':records,'sealed_access':False,'source_hashes':hashes,'seed':20261016,'device_preference':'mps if available else cpu','input_channels':58,'conditional_values':1920,'widths':[8,16,32,64,128],'architecture':'Four-level conditional U-Net, double convolution and GroupNorm, bilinear upsampling, FiLM at each encoder and decoder block, Softplus depth output','optimizer':'AdamW','learning_rate':.001,'weight_decay':.00001,'batch_size':1,'maximum_epochs':40,'validation_epochs':[1]+list(range(5,41,5)),'early_stop_validation_checks':4,'selection':'Validation parent-macro ground-area-weighted MAE on original exact support; strict improvement > 1e-6 m','loss':'Per-parent ground-area-weighted mean squared depth error on exact support only, uniform shuffled parent updates, no target filling or wet sampling','augmentation':False,'retrieval':'1-nearest training parent using mean squared normalized rain-curve distance (weight 1/2) plus mean squared seven-stage-curve distance (weight 1/2). Training diagnostics leave out own family. Never validation candidates. Only source-exact response values returned.','comparison':'Paired support = target exact AND selected retrieval source exact AND eligible ground; all three methods on same support. Also report each method on its own available target exact support and coverage.','wet_thresholds_m':[.05,.1,.3,.5],'metrics':'Parent-level area-weighted MAE, RMSE, bias, true-wet MAE, strict threshold IoU/recall/precision, missed and false wet area. Macro-average events. Empty denominator is undefined, not perfect.','limits':['One initialization and one bounded fit; no search and no independent test.','Peak emulation only; no dynamic path, mass-balance, road-length or observed-event accuracy claim.','Validation used for checkpoint selection; four events do not establish generalization.','No active measures, interval supervision, graph fitting or new hydraulic scenarios.','Fixed roof allocation, dry interior state, uniform rain and coherent river-stage increments.','No hydraulic speedup claim from historical interrupted runtime records.']}
 for p in [Path(__file__),ROOT/'src/analyses/train_three_ward_unet_exact_pilot.py']:
  protocol.setdefault('implementation_hashes',{})[str(p.relative_to(ROOT))]=sha(p)
 protocol['prepared_inputs_sha256']=sha(OUT/'prepared_inputs.npz');protocol['normalization_sha256']=sha(OUT/'normalization.json')
 write(OUT/'protocol.json',protocol);write(REPORT/'preparation_qc.json',{'integral_checks':qc,'records':len(records),'all_checks_passed':True,'protocol_sha256':sha(OUT/'protocol.json')})
 print(json.dumps({'records':len(records),'input_shape':features.shape,'output_shape':[nrow,ncol],'max_encoding_integral_error':max(q['integral_abs_error_rate_seconds'] for q in qc)}))

def load_label(case_id,purpose):
 """Purpose/split checked, hash-bound allowlist; no generic global release flag."""
 p=read(OUT/'protocol.json');matches=[r for r in p['records'] if r['case_id']==case_id]
 if len(matches)!=1:raise PermissionError('Case outside authorized pilot')
 r=matches[0]
 if purpose not in ('train','validation') or r['split']!=purpose:raise PermissionError('Wrong purpose or partition')
 lp=ROOT/r['label_path']
 if sha(lp)!=r['label_sha256']:raise ValueError('Frozen label changed')
 return arrays(lp)
if __name__=='__main__':main()
