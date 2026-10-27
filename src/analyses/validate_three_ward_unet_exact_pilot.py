"""Admission, leakage and evaluation checks for one scoped trial."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import OUT,REPORT,read,write,sha,arrays,load_label,minute_means
from train_three_ward_unet_exact_pilot import metrics
from load_three_ward_minimum_scenarios import find_case,load_forcing,response_access

def main():
 checks=[]
 def check(name,value):
  checks.append({'check':name,'pass':bool(value)})
  if not value:raise AssertionError(name)
 p=read(OUT/'protocol.json');z=arrays(OUT/'prepared_inputs.npz');norm=read(OUT/'normalization.json');records=p['records'];tr=[i for i,r in enumerate(records) if r['split']=='train'];va=[i for i,r in enumerate(records) if r['split']=='validation']
 check('exactly_12_training_and_4_validation_unique_families',len(tr)==12 and len(va)==4 and len({r['family_id'] for r in records})==16)
 check('frozen_source_bytes_unchanged',all(sha(ROOT/k)==v for k,v in p['source_hashes'].items()))
 check('source_code_matches_frozen_protocol',all(sha(ROOT/k)==v for k,v in p['implementation_hashes'].items()))
 check('train_only_fit_id_allowlist',set(norm['fit_case_ids'])=={records[i]['case_id'] for i in tr})
 check('prepared_feature_hash_and_normalizer',sha(OUT/'prepared_inputs.npz')==p['prepared_inputs_sha256'] and sha(OUT/'normalization.json')==p['normalization_sha256'])
 curves=z['Minute Forcing Mean'];fm=curves[tr].mean((0,1));fs=curves[tr].std((0,1));fs=np.where(fs>1e-8,fs,1)
 check('recomputed_training_only_forcing_moments',np.array_equal(fm,np.array(norm['forcing_mean'])) and np.array_equal(fs,np.array(norm['forcing_scale'])))
 source=arrays(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz');valid=source['Static Feature Valid']&np.isfinite(source['Static Tensor']);static=source['Static Tensor'].astype('f8')
 sm=np.array([c[k].mean() if k.any() else 0 for c,k in zip(static,valid)]);ss=np.array([c[k].std() if k.any() else 0 for c,k in zip(static,valid)])
 check('recomputed_known_training_static_moments',np.array_equal(sm,np.array(norm['static_mean'])[:28]) and np.array_equal(ss,np.array(norm['static_raw_std'])[:28]))
 check('finite_encoded_inputs',np.isfinite(z['Static Input']).all() and np.isfinite(z['Forcing Input']).all())
 check('feature_validity_preserved',set(np.unique(z['Static Input'][29:]))<={0.,1.} and np.all(z['Static Input'][:29][z['Static Input'][29:]==0]==0))
 masks=[]
 for r in records:
  d=load_label(r['case_id'],r['split']);m=d['Exact Label Mask'];masks.append(m)
  check(r['case_id']+'_original_partition_quality_and_finite_exact_target',find_case(r['case_id'])['split']==r['split'] and not (m&d['Quarantine Mask']).any() and not (m&d['Interval Label Mask']).any() and not (m&d['Online Ever Negative']).any() and np.array_equal(d['Peak Mean Depth Lower'][m],d['Peak Mean Depth Upper'][m]) and np.isfinite(d['Peak Mean Depth Lower'][m]).all())
 # Malformed purpose and all out-of-allowlist cases must be rejected before a read.
 for cid,purpose in [('validation-01','train'),('train-01','validation'),('sealed-test-01','train'),('challenge-01','validation')]:
  blocked=False
  try:load_label(cid,purpose)
  except PermissionError:blocked=True
  check('scope_denies_'+cid+'_'+purpose,blocked)
 closed=False
 try:response_access('train-01','train')
 except PermissionError:closed=True
 check('historical_response_loader_still_closed',closed)
 # Independent triangular analytic curve, including an intra-minute kink.
 t=np.array([0.,30.,60.,14400.]);v=np.array([0.,2.,0.,0.]);binmean=minute_means(t,v)
 check('analytic_subminute_integral',binmean[0,0]==1. and np.isclose(binmean.sum()*60,60.))
 a=np.array([1.,3.,5.]);y=np.array([0.,1.,np.nan]);pred=np.array([.2,.5,999.]);mask=np.array([True,True,False]);met=metrics(y,pred,a,mask)
 check('area_weighted_metric_and_missing_exclusion',np.isclose(met['mae_m'],.425) and np.isclose(met['rmse_m'],np.sqrt(.79/4)))
 dry=metrics(np.zeros(2),np.zeros(2),np.ones(2),np.ones(2,bool))
 check('dry_empty_union_not_perfect_and_zero_false_area',dry['iou_0.1'] is None and dry['false_km2_0.1']==0 and dry['missed_km2_0.1']==0)
 # Explicit zero-loss/zero-gradient support test: excluded targets never participate.
 import torch
 f=torch.tensor([1.,10.],requires_grad=True);w=torch.tensor([1.,0.]);safe=torch.tensor([2.,0.]);loss=((f-safe)**2*w).sum();loss.backward()
 check('zero_weight_excluded_cell_gradient',f.grad.tolist()==[-2.,0.])
 post=(REPORT/'completion.json').exists()
 if post:
  done=read(REPORT/'completion.json');pred=arrays(OUT/'predictions.npz');pp=pred['Conditional UNet Peak Mean Depth'];elig=z['Eligible Ground Mask']
  check('all16_full_domain_prediction_shapes',pp.shape==(16,1012,770) and np.array_equal(pred['Case ID'],z['Case ID']))
  check('nonnegative_finite_eligible_predictions',np.isfinite(pp[:,elig]).all() and (pp[:,elig]>=0).all() and np.isnan(pp[:,~elig]).all())
  check('single_bounded_fit_checkpoint_binding',done['fit_count']==1 and done['epochs_completed']<=40 and done['optimizer_steps']==12*done['epochs_completed'] and done['best_checkpoint_sha256']==sha(OUT/'best.pt') and done['protocol_sha256']==sha(OUT/'protocol.json') and done['predictions_sha256']==sha(OUT/'predictions.npz'))
  choices=list(csv.DictReader((REPORT/'retrieval_choices.csv').open()));byid={r['case_id']:i for i,r in enumerate(records)}
  for q in choices:
   i=byid[q['case_id']];j=byid[q['source_case_id']];c=z['Forcing Input'].reshape(16,240,8);cand=[k for k in tr if records[k]['family_id']!=records[i]['family_id']];ds=[.5*np.mean((c[i,:,0]-c[k,:,0])**2)+.5*np.mean((c[i,:,1:]-c[k,:,1:])**2) for k in cand]
   check(q['case_id']+'_nearest_training_family_only',j==cand[int(np.argmin(ds))] and records[j]['split']=='train' and records[j]['family_id']!=records[i]['family_id'])
  rows=list(csv.DictReader((REPORT/'event_metrics.csv').open()))
  for r in records:
   pair=[x for x in rows if x['case_id']==r['case_id'] and x['scope']=='domain' and x['support']=='paired_exact']
   check(r['case_id']+'_identical_paired_metric_support',len(pair)==3 and len({x['support_km2'] for x in pair})==1)
  area=z['Ground Area']
  for r in records:
   i=byid[r['case_id']];choice=next(q for q in choices if q['case_id']==r['case_id']);j=byid[choice['source_case_id']]
   native=load_label(r['case_id'],r['split'])['Peak Mean Depth Lower'];support=masks[i]&masks[j]&elig&(z['Ward Code']>0)
   aa=area[support];yy=native[support].astype('f4');uu=pp[i,support]
   expected_mae=float(np.sum(aa*np.abs(uu-yy))/aa.sum());wet=yy>.1;predwet=uu>.1
   union=float(aa[wet|predwet].sum());intersection=float(aa[wet&predwet].sum());expected_iou=intersection/union if union else None
   saved=next(x for x in rows if x['case_id']==r['case_id'] and x['scope']=='three_wards' and x['support']=='paired_exact' and x['method']=='conditional_unet')
   check(r['case_id']+'_saved_predictions_reproduce_MAE_and_IoU',np.isclose(expected_mae,float(saved['mae_m']),rtol=0,atol=1e-10) and ((expected_iou is None and saved['iou_0.1']=='') or np.isclose(expected_iou,float(saved['iou_0.1']),rtol=0,atol=1e-10)))
  macro=list(csv.DictReader((REPORT/'macro_metrics.csv').open()))
  check('parent_macro_not_pixel_pooling',all(np.isclose(float(v['mae_m']),np.mean([float(r['mae_m']) for r in rows if all(r[k]==v[k] for k in ['split','scope','support','method'])])) for v in macro))
 result={'phase':'delivery' if post else 'preflight','all_checks_passed':all(c['pass'] for c in checks),'checks':checks,'sealed_responses_read':False,'quality_policy_modified':False}
 write(REPORT/('delivery_validation.json' if post else 'preflight_validation.json'),result);print(json.dumps({'phase':result['phase'],'checks':len(checks),'all_checks_passed':result['all_checks_passed']}))
if __name__=='__main__':main()
