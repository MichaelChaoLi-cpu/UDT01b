"""Read-only whole-ward diagnosis of the frozen D-arm; no fitting or inference.
Uses only the 16 original allowed responses, selected predictions, and saved logs.
Label-distance/frequency strata are diagnostics only, never predictor/mask changes.
"""
from pathlib import Path
import csv, json, sys, hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,load_label
from three_ward_surrogate_protocol_v2 import evaluate_depth,checkpoint_key
NAME='three_ward_d_false_positive_diagnosis_v1'
OUT=ROOT/'data/processed'/NAME
REPORT=ROOT/'data/exp/estimation'/NAME
SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'
CONTROL=ROOT/'data/processed/three_ward_unet_controls_v2'
OLDREPORT=ROOT/'data/exp/estimation/three_ward_unet_controls_v2'
PLAN=ROOT/'data/processed/three_ward_unet_control_plan_v2'
ARM='D_huber_lower_lr'

def wc(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def mh(m):return hashlib.sha256(np.ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
def wq(v,a,q):
 if not len(v):return None
 order=np.argsort(v,kind='stable');v=v[order];c=np.cumsum(a[order]);return float(v[min(np.searchsorted(c,q*c[-1]),len(v)-1)])
def auc(p,wet,a):
 if not (wet.any() and (~wet).any()):return None
 order=np.argsort(p,kind='stable');pp=p[order];aw=np.where(wet[order],a[order],0);an=np.where(~wet[order],a[order],0)
 starts=np.r_[0,1+np.flatnonzero(pp[1:]!=pp[:-1])];ww=np.add.reduceat(aw,starts);nn=np.add.reduceat(an,starts)
 return float(np.sum(ww*(np.cumsum(nn)-.5*nn))/(ww.sum()*nn.sum()))
def macro(rows,groups):
 out=[];seen=sorted({tuple(r[g] for g in groups) for r in rows})
 for key in seen:
  rr=[r for r in rows if tuple(r[g] for g in groups)==key];res=dict(zip(groups,key));res['parents']=len(rr)
  for k in rr[0]:
   if k in groups or k in ('case_id','parent_id','support_sha256','split'):continue
   vv=[r[k] for r in rr if r.get(k) is not None]
   if vv and all(isinstance(x,(float,int,bool,np.floating,np.integer)) for x in vv):res[k]=float(np.mean(vv))
   elif not vv:res[k]=None
  out.append(res)
 return out

def main():
 if (OUT/'freeze.json').exists():raise FileExistsError('Preserve frozen diagnosis')
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);checks=[]
 def check(n,v):
  if not bool(v):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 frozen=read(CONTROL/'freeze.json');p=read(PLAN/'protocol.json');source=read(SOURCE/'protocol.json');records=p['records'];ids=[r['case_id'] for r in records];by={cid:i for i,cid in enumerate(ids)}
 check('original12train4validation_only',len(records)==16 and sum(r['split']=='train' for r in records)==12 and sum(r['split']=='validation' for r in records)==4)
 protected=dict(frozen['files_sha256']);protected[str((CONTROL/'freeze.json').relative_to(ROOT))]=sha(CONTROL/'freeze.json')
 protected.update(source['source_hashes']);protected.update(source['implementation_hashes'])
 for file in [SOURCE/'prepared_inputs.npz',SOURCE/'normalization.json',SOURCE/'protocol.json',PLAN/'protocol.json',ROOT/'src/analyses/three_ward_surrogate_protocol_v2.py']:
  protected[str(file.relative_to(ROOT))]=sha(file)
 check('all_source_hashes_match_before',all(sha(ROOT/k)==v for k,v in protected.items()))
 print(json.dumps({'phase':'source_hashes_verified','files':len(protected)}),flush=True)
 z=np.load(SOURCE/'prepared_inputs.npz',allow_pickle=False);area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];pr=np.load(CONTROL/ARM/'selected_predictions.npz',allow_pickle=False)
 check('case_order_unchanged',list(pr['Case ID'])==ids and list(z['Case ID'])==ids);pred=pr['Peak Mean Depth'];selected=read(OLDREPORT/ARM/'selected_metrics.json');check('selected_epoch36_preserved',selected['epoch']==36)
 labels=[load_label(r['case_id'],r['split']) for r in records];y=np.array([l['Peak Mean Depth Lower'].astype('f4') for l in labels]);own=np.array([l['Exact Label Mask']&eligible for l in labels]);paired=np.array([own[i]&own[by[r['retrieval_source_case_id']]]&(ward>0) for i,r in enumerate(records)])
 for i,r in enumerate(records):
  check(r['case_id']+'_own_mask_hash',mh(own[i])==r['original_exact_eligible_mask_sha256'])
  check(r['case_id']+'_paired_threeward_mask_hash',mh(paired[i])==r['paired_three_ward_mask_sha256'])
  check(r['case_id']+'_no_quarantine_interval_in_exact',not np.any(own[i]&(labels[i]['Quarantine Mask']|labels[i]['Interval Label Mask']|labels[i]['Online Ever Negative'])))
  check(r['case_id']+'_finite_all_scored',np.isfinite(pred[i][own[i]]).all() and np.isfinite(y[i][own[i]]).all())
 event=[];strata=[];gradient=[];lossrows=[]
 def addgroup(i,family,name,g):
  m=paired[i]&g;wet=y[i]>.1;pw=pred[i]>.1;fp=m&~wet&pw;n=m&~wet;mass=area[m].sum();nmass=area[n].sum();totalfp=area[paired[i]&~wet&pw].sum()
  strata.append({'case_id':ids[i],'split':records[i]['split'],'family':family,'stratum':name,'support_km2':float(mass/1e6),'reference_nonwet_km2':float(nmass/1e6),'false_km2':float(area[fp].sum()/1e6),'fpr':float(area[fp].sum()/nmass) if nmass else None,'fraction_of_event_false_area':float(area[fp].sum()/totalfp) if totalfp else None})
 raw=np.load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz',allow_pickle=False);sv=raw['Static Tensor'];valid=raw['Static Feature Valid'];names=list(raw['Static Channel Names'])
 feature_specs=[('Ground Elevation Mean',[-np.inf,0,5,15,30,np.inf]),('Building Occupancy Fraction',[-np.inf,.25,.5,.75,np.inf]),('Road Ground Fraction',[-np.inf,.1,.5,np.inf]),('Subcell Connectivity Ambiguity',[-np.inf,.5,np.inf]),('PLATEAU Ground Fraction',[-np.inf,0,.5,np.inf]),('Automatic Refinement Flag',[-np.inf,.5,np.inf])]
 tr=[i for i,r in enumerate(records) if r['split']=='train'];va=[i for i,r in enumerate(records) if r['split']=='validation'];traincommon=np.logical_and.reduce(paired[tr]);trainwet=np.sum((y[tr]>.1)&paired[tr],axis=0)
 freqsets=[('never_wet_in_12',traincommon&(trainwet==0)),('wet_in_1_to_3',traincommon&(trainwet>=1)&(trainwet<=3)),('wet_in_4_to_8',traincommon&(trainwet>=4)&(trainwet<=8)),('wet_in_9_to_12',traincommon&(trainwet>=9)),('incomplete_train_support',~traincommon)]
 for i,r in enumerate(records):
  m=paired[i];t=y[i];v=pred[i];wet=t>.1;pw=v>.1;fp=m&~wet&pw
  metrics=evaluate_depth(v,t,area,m);logged=next(x for x in selected['event_metrics'] if x['case_id']==r['case_id'] and x['scope']=='three_wards' and x['support']=='paired_exact')
  check(r['case_id']+'_reproduces_frozen_scores',all(abs(metrics[k]-logged[k])<1e-10 for k in ['mae_m','false_km2_0.1','balanced_extent_error','recall_0.1']))
  event.append({'case_id':ids[i],'parent_id':r['parent_id'],'split':r['split'],'scope':'three_wards','support':'paired_exact',**metrics,'wet_nonwet_rank_auc':auc(v[m],wet[m],area[m]),'false_prediction_q50_m':wq(v[fp],area[fp],.5),'false_prediction_q90_m':wq(v[fp],area[fp],.9),'false_reference_q50_m':wq(t[fp],area[fp],.5),'false_reference_q90_m':wq(t[fp],area[fp],.9),'mean_prediction_on_wet_m':float(np.average(v[m&wet],weights=area[m&wet])),'mean_prediction_on_nonwet_m':float(np.average(v[m&~wet],weights=area[m&~wet]))})
  for family,values,bounds in [('prediction_depth_m',v,[.1,.15,.3,.5,np.inf]),('reference_nonwet_depth_m',t,[-np.inf,.01,.05,.1])]:
   for lo,hi in zip(bounds[:-1],bounds[1:]):addgroup(i,family,f'({lo},{hi}]',(values>lo)&(values<=hi))
  for name,bounds in feature_specs:
   j=names.index(name);known=valid[j]&np.isfinite(sv[j])
   for lo,hi in zip(bounds[:-1],bounds[1:]):addgroup(i,name,f'({lo},{hi}]',known&(sv[j]>lo)&(sv[j]<=hi))
   addgroup(i,name,'unknown',~known)
  # Distance to a certified wet cell in the same event. Straight-line proximity,
  # not a hydraulic route and not an input to this or subsequent predictors.
  d=distance_transform_edt(~(m&wet),sampling=10)
  for lo,hi in [(0,20),(20,50),(50,100),(100,np.inf)]:addgroup(i,'distance_to_reference_wet_m',f'({lo},{hi}]',(d>lo)&(d<=hi))
  for name,g in freqsets:addgroup(i,'training_reference_wet_frequency',name,g)
  for code,name in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:addgroup(i,'ward',name,ward==code)
  # Analytic derivative along the existing shared final-head bias only. Exact
  # robust loss on actual full-domain target-exact support; no optimizer/model.
  m=own[i];a=area[m];tt=t[m];vv=v[m];ee=vv.astype('f8')-tt.astype('f8');abs_e=np.abs(ee);cell=np.where(abs_e<=.1,ee**2,.2*(abs_e-.05));dl=np.clip(2*ee,-.2,.2);gate=-np.expm1(-vv.astype('f8'))
  ss={'nonwet':tt<=.1,'ordinary_wet':(tt>.1)&(tt<=1),'deep':tt>1};coeff={'nonwet':.5,'ordinary_wet':.4,'deep':.1};active=sum(coeff[k] for k,b in ss.items() if b.any());total=0.;g=np.zeros(len(a));lossvec=np.zeros(len(a))
  for k,b in ss.items():
   w=coeff[k]/active if b.any() else 0.;den=a[b].sum();value=float(np.average(cell[b],weights=a[b])) if b.any() else None
   if b.any():g[b]=w*a[b]/den*dl[b]*gate[b];lossvec[b]=w*a[b]/den*cell[b];total+=w*value
   lossrows.append({'case_id':ids[i],'split':r['split'],'stratum':k,'area_fraction':float(den/a.sum()),'mixture_weight':w,'mean_loss_m2':value,'contribution_m2':w*value if b.any() else 0.,'head_bias_derivative':float(g[b].sum()),'absolute_head_bias_derivative':float(np.abs(g[b]).sum())})
  old=next(x for x in selected['event_metrics'] if x['case_id']==ids[i] and x['scope']=='domain' and x['support']=='own_exact')['fixed_model_loss_m2']
  check(ids[i]+'_numpy_robust_loss_matches_original_float32',abs(total-old)<max(3e-7,abs(old)*2e-5))
  if r['split']=='train':
   # Independent central finite difference of a hypothetical head-bias change;
   # compute in arrays only, without loading/changing the saved model.
   pp64=vv.astype('f8')
   with np.errstate(divide='ignore'):logit=pp64+np.log(-np.expm1(-pp64))
   def perturbed_loss(delta):
    changed=np.logaddexp(0,logit+delta);res=changed-tt.astype('f8');ae=np.abs(res);cl=np.where(ae<=.1,res**2,.2*(ae-.05))
    return sum(coeff[k]/active*np.average(cl[b],weights=a[b]) for k,b in ss.items() if b.any())
   fd=(perturbed_loss(.0001)-perturbed_loss(-.0001))/.0002
   check(ids[i]+'_head_bias_derivative_finite_difference',abs(fd-g.sum())<1e-7)
  cats={'false_wet':(tt<=.1)&(vv>.1),'true_nonwet':(tt<=.1)&(vv<=.1),'true_wet_ordinary':(tt>.1)&(tt<=1)&(vv>.1),'missed_wet_ordinary':(tt>.1)&(tt<=1)&(vv<=.1),'true_wet_deep':(tt>1)&(vv>.1),'missed_wet_deep':(tt>1)&(vv<=.1)}
  check(ids[i]+'_gradient_categories_partition_support',np.all(np.sum(list(cats.values()),axis=0)==1))
  for k,b in cats.items():gradient.append({'case_id':ids[i],'split':r['split'],'category':k,'area_km2':float(a[b].sum()/1e6),'loss_contribution_m2':float(lossvec[b].sum()),'head_bias_derivative':float(g[b].sum()),'positive_derivative':float(g[b][g[b]>0].sum()),'negative_derivative':float(g[b][g[b]<0].sum()),'absolute_derivative':float(np.abs(g[b]).sum()),'scope':'full_domain_own_exact','interpretation':'Positive derivative favors lowering the shared output logit; negative favors raising it. No parameter update or full gradient attribution.'})
  print(json.dumps({'phase':'selected_map_diagnostics','case_id':ids[i],'false_km2':metrics['false_km2_0.1'],'auc':event[-1]['wet_nonwet_rank_auc']}),flush=True)
 emacro=macro(event,['split','scope','support']);smacro=macro(strata,['split','family','stratum']);gmacro=macro(gradient,['split','category']);lmacro=macro(lossrows,['split','stratum'])
 for i in range(16):
  for fam in ['prediction_depth_m','reference_nonwet_depth_m','distance_to_reference_wet_m','training_reference_wet_frequency','ward']+[s[0] for s in feature_specs]:
   rr=[s for s in strata if s['case_id']==ids[i] and s['family']==fam];check(ids[i]+'_'+fam+'_FP_partition',abs(sum(s['false_km2'] for s in rr)-event[i]['false_km2_0.1'])<1e-10)
 # All 40 frozen score frames, no re-inference or retrospective promotion.
 trajectories=[];wardtrajectories=[];keys=[];frames=[]
 for epoch in range(1,41):
  f=read(OLDREPORT/ARM/f'epoch_{epoch:03}.jsonl');frames.append(f);ve=[r for r in f['event_metrics'] if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact'];expected={r['case_id']:r['paired_three_ward_mask_sha256'] for r in records if r['split']=='validation'};key=checkpoint_key(ve,epoch,p['validation_case_ids'],expected);keys.append(key)
  check(f'epoch{epoch}_frozen_key_reproduced',tuple(f['selection_key'])==key)
  check(f'epoch{epoch}_evaluation_did_not_update',f['optimizer_steps_before_eval']==f['optimizer_steps_after_eval']==12*epoch and f['model_fingerprint_before_eval']==f['model_fingerprint_after_eval'])
  for split in ['train','validation']:
   mm=next(v for v in f['parent_macro_metrics'] if v['split']==split and v['scope']=='three_wards' and v['support']=='paired_exact')['values'];domain=[r for r in f['event_metrics'] if r['split']==split and r['scope']=='domain' and r['support']=='own_exact'];lo=float(np.mean([v['fixed_model_loss_m2'] for v in domain]));deep=float(np.mean([v['fixed_loss_strata']['deep']['contribution_m2'] for v in domain]));nonwet=float(np.mean([v['fixed_loss_strata']['nonwet']['contribution_m2'] for v in domain]));trajectories.append({'epoch':epoch,'split':split,'fixed_domain_objective_m2':lo,'deep_loss_value_share':deep/lo,'nonwet_loss_value_share':nonwet/lo,**{k:mm[k] for k in ['balanced_extent_error','mae_m','wet_mae_m','deep_mae_m','false_km2_0.1','missed_km2_0.1','recall_0.1','precision_0.1','fpr_0.1','fnr_0.1','iou_0.1']}})
   for scope in ['Chiyoda','Chuo','Minato']:
    wm=next(v for v in f['parent_macro_metrics'] if v['split']==split and v['scope']==scope and v['support']=='paired_exact')['values'];wardtrajectories.append({'epoch':epoch,'split':split,'scope':scope,**{k:wm[k] for k in ['balanced_extent_error','false_km2_0.1','recall_0.1','mae_m']}})
 check('epoch36_remains_original_best_no_reselection',min(keys)[-1]==36)
 # Check ranking mechanics and show alternatives as retrospective diagnostics only.
 vals=[r for r in trajectories if r['split']=='validation'];diagnostic_points=[]
 for reason,rr in [('selected_original',vals[35]),('final_budget',vals[39]),('minimum_false_area_only',min(vals,key=lambda r:r['false_km2_0.1'])),('minimum_overall_MAE_only',min(vals,key=lambda r:r['mae_m'])),('minimum_training_objective_only',vals[min((r for r in trajectories if r['split']=='train'),key=lambda r:r['fixed_domain_objective_m2'])['epoch']-1])]:
  diagnostic_points.append({'reason':reason,'is_new_selected_model':False,**rr})
 steps=[json.loads(l) for l in (OLDREPORT/ARM/'optimizer_steps.jsonl').read_text().splitlines()];check('historical480steps_not_extended',len(steps)==480 and steps[-1]['optimizer_after']['steps']==[480]);check('historical_steps_finite_no_clip',all(x['gradient_pre_clip']['finite'] and not x['clipping_required'] for x in steps))
 # Spatial recurrence uses only cells certified in every relevant event; other
 # support remains excluded/unknown, never assigned a dry reference.
 maps={};recurrence=[]
 for split,ii in [('train',tr),('validation',va)]:
  common=np.logical_and.reduce(paired[ii]);wp=y[ii]>.1;pp=pred[ii]>.1;fp=paired[ii]&~wp&pp;count=fp.sum(0);allnonwet=common&~np.any(wp,axis=0);recurring=common&(count>=int(np.ceil(.75*len(ii))))
  maps[split+'_common_mask']=common;maps[split+'_false_frequency']=np.where(common,count/len(ii),np.nan).astype('f4');maps[split+'_reference_wet_frequency']=np.where(common,np.mean(wp,axis=0),np.nan).astype('f4')
  recurrence.append({'split':split,'parents':len(ii),'common_support_km2':float(area[common].sum()/1e6),'false_in_at_least75pct_events_km2':float(area[recurring].sum()/1e6),'never_reference_wet_but_false_in_at_least75pct_km2':float(area[recurring&allnonwet].sum()/1e6),'reference_nonwet_all_events_km2':float(area[allnonwet].sum()/1e6)})
  # Spatial discrimination: AUC plus conditional depth moments are above. This
  # does not test whether different model features would improve fit.
 wc(REPORT/'selected_event_diagnostics.csv',event);wc(REPORT/'selected_parent_macro.csv',emacro);wc(REPORT/'spatial_strata_by_event.csv',strata);wc(REPORT/'spatial_strata_macro.csv',smacro);wc(REPORT/'loss_strata_by_event.csv',lossrows);wc(REPORT/'loss_strata_macro.csv',lmacro);wc(REPORT/'head_bias_derivative_by_event.csv',gradient);wc(REPORT/'head_bias_derivative_macro.csv',gmacro);wc(REPORT/'forty_epoch_trajectories.csv',trajectories);wc(REPORT/'ward_trajectories.csv',wardtrajectories);wc(REPORT/'retrospective_tradeoffs.csv',diagnostic_points);wc(REPORT/'spatial_recurrence.csv',recurrence)
 np.savez_compressed(OUT/'diagnostic_spatial_aggregates.npz',**maps,**{'Ward Code':ward,'Ground Area':area})
 primary={r['split']:r for r in emacro};features={fam:[r for r in smacro if r['split']=='validation' and r['family']==fam] for fam in {s['family'] for s in smacro}};gtrain=[r for r in gmacro if r['split']=='train'];nonwet=primary['validation']['nonwet_area_km2'];wet=primary['validation']['wet_area_km2']
 summary={'status':'complete','decision':'MILI-D-20261016-006','selected_arm':ARM,'selected_epoch_preserved':36,'new_fit_count':0,'optimizer_updates':0,'inference_runs':0,'sealed_opened':False,'primary_support':'Original paired target/retrieval exact support AND eligible AND ward>0, separately per parent. Native float32 threshold comparisons before float64 area reductions.','loss_derivative_support':'Original full-domain target exact support, matching the actual training objective.','original_partitions_masks_and_normalization_preserved':True,'any_saved_D_epoch_passes_original_guard':any(f['baseline_readiness']['all_parents_pass'] for f in frames),'primary_metrics':primary,'validation_spatial_strata':features,'training_head_bias_directions':gtrain,'training_loss_strata':[r for r in lmacro if r['split']=='train'],'spatial_recurrence':recurrence,'retrospective_checkpoints_not_promoted':diagnostic_points,'selection_area_exchange_illustration':{'ratio_of_macro_nonwet_to_wet_areas':nonwet/wet,'per_parent_nonwet_to_wet_area_ratio':{e['case_id']:e['nonwet_area_km2']/e['wet_area_km2'] for e in event if e['split']=='validation'},'description':'For each individual event, BER gives equal marginal cost to FP/nonwet_area and FN/wet_area. One recovered wet square metre can offset nonwet_area/wet_area extra false square metres in that event. The macro-area ratio is not the exact macro exchange rate.','mean_per_parent_nonwet_to_wet_area_ratio':float(np.mean([e['nonwet_area_km2']/e['wet_area_km2'] for e in event if e['split']=='validation']))},'optimizer_evidence':{'old_steps':480,'new_steps':0,'clip_count':sum(s['clipping_required'] for s in steps),'all_gradients_finite':all(s['gradient_pre_clip']['finite'] for s in steps),'fixed_training_objective_first':trajectories[0]['fixed_domain_objective_m2'],'fixed_training_objective_final':next(r for r in trajectories if r['epoch']==40 and r['split']=='train')['fixed_domain_objective_m2'],'fixed_training_objective_selected':next(r for r in trajectories if r['epoch']==36 and r['split']=='train')['fixed_domain_objective_m2']},'causal_limits':['Poor training-set discrimination establishes a fit failure under this recipe, not a proof of intrinsic network-capacity failure.','Loss-value shares and the single shared-head-bias derivative are not full-parameter gradient attribution.','Euclidean proximity to wet reference cells is a label-side diagnostic, not surface drainage connectivity.','Static-feature associations are descriptive and confounded; no input occlusion, graph replacement, new architecture or longer training is run.','Four validation parents have been repeatedly exposed; all retrospective checkpoint comparisons are diagnostic, not independent evaluation or promotion.','No global calibration, threshold search, new masks, data admission or sealed-test response access.']}
 write(REPORT/'summary.json',summary)
 check('all_source_hashes_unchanged_after',all(sha(ROOT/k)==v for k,v in protected.items()))
 write(OUT/'diagnostic_protocol.json',{'decision':summary['decision'],'source_files_sha256':protected,'case_ids':ids,'splits':{r['case_id']:r['split'] for r in records},'new_fit_count':0,'optimizer_updates':0,'sealed_opened':False,'original_threshold_m':.1,'strata_rule':'Fixed physical bins and label-side distance/frequency summaries only; no fitting or threshold choice.','interpretation_notes':['FP share is the mean within-parent FP share; a ratio of macro areas need not equal that value.','FPR in a prediction-defined depth bin is conditional on predictions and is tautological; use its FP area and share, never compare it to input-feature conditional FPR.','All empty strata retain undefined rates.','The head-bias probe is an analytic snapshot derivative, checked by finite differences, not an optimizer update or complete parameter-gradient diagnosis.'],'new_script_sha256':sha(Path(__file__))})
 write(REPORT/'validation.json',{'all_checks_passed':True,'checks':checks,'source_files_checked':len(protected),'new_fit_count':0,'optimizer_updates':0,'sealed_opened':False})
 print(json.dumps({'phase':'complete','checks':len(checks),'primary':primary,'recurrence':recurrence},indent=2),flush=True)
if __name__=='__main__':main()
