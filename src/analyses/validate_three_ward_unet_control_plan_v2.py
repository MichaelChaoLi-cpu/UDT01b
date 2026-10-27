"""Verify proposed components, masks and telemetry with zero optimizer updates."""
from pathlib import Path
import sys,csv,json,hashlib,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import OUT as SOURCE,REPORT as SOURCE_REPORT,sha,read,write,arrays,load_label
from prepare_three_ward_unet_control_plan_v2 import OUT,REPORT,mask_sha
from train_three_ward_unet_exact_pilot import ConditionalUNet
from three_ward_surrogate_protocol_v2 import exact_loss,evaluate_depth,macro_events,checkpoint_key,baseline_readiness,OptimizationObserver,parameter_snapshot,write_epoch_record,REQUIRED_SCOPES,parameter_change,summarize_step_records

def main():
 if (REPORT/'validation.json').exists():raise FileExistsError('Preserve validated preflight')
 checks=[]
 def check(n,ok):
  if not ok:raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 def reject(n,fn,errors=(ValueError,RuntimeError,FloatingPointError)):
  try:fn()
  except errors:check(n,True)
  else:check(n,False)
 p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');old=read(SOURCE/'protocol.json')
 check('source_bindings_before',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 check('new_implementation_bindings',all(sha(ROOT/k)==v for k,v in p['implementation_sha256'].items()))
 check('original_frozen_trial_unchanged',all(sha(ROOT/k)==v for k,v in read(SOURCE/'freeze.json')['files_sha256'].items()))
 check('12_4_original_parent_partitions',[(r['case_id'],r['family_id'],r['split'],r['label_sha256']) for r in p['records']]==[(r['case_id'],r['family_id'],r['split'],r['label_sha256']) for r in old['records']])
 check('no_fit_authorization_or_execution',p['execution_authorized'] is False and p['fit_runs_executed']==p['optimizer_updates_executed']==0)
 check('exactly2by2_finite_proposed_arms',len(p['arms'])==4 and {(a['loss'],a['learning_rate']) for a in p['arms']}=={(k,lr) for k in ['mse','stratified_huber'] for lr in [.001,.0003]} and sum(a['maximum_optimizer_updates'] for a in p['arms'])==1920)
 order=read(OUT/'epoch_order.json');check('all40_orders_have_same12_train_once',len(order)==40 and all(set(r['case_order'])==set(p['train_case_ids']) and len(r['case_order'])==12 for r in order))
 check('shared_untrained_initialization_binding',sha(OUT/'shared_initialization.pt')==p['frozen_common_settings']['initialization_sha256'])
 init=torch.load(OUT/'shared_initialization.pt',map_location='cpu',weights_only=False);torch.manual_seed(p['frozen_common_settings']['seed']);untrained=ConditionalUNet();check('shared_state_is_untrained_and_seed_reproducible',not init['trained'] and init['optimizer_steps']==0 and all(torch.equal(v,untrained.state_dict()[k]) for k,v in init['state_dict'].items()))
 # Hand-computed exact-loss/derivative cases: masked NaN must never enter arithmetic.
 y=torch.tensor([0.,.1,.2,1.,2.,float('nan')],dtype=torch.float64);pred=torch.tensor([.2,.3,.4,1.3,3.,float('nan')],dtype=torch.float64,requires_grad=True);a=torch.tensor([1.,2.,3.,4.,5.,6.],dtype=torch.float64);m=torch.tensor([True]*5+[False])
 mse,md=exact_loss(pred,y,a,m,'mse');expected=float(np.sum(np.array([1,2,3,4,5])*np.array([.2,.2,.2,.3,1.])**2)/15)
 check('original_mse_matches_hand_area_weighted_result',np.isclose(float(mse.detach()),expected));mse.backward();check('masked_NaN_has_zero_loss_gradient',pred.grad[-1]==0 and torch.isfinite(pred.grad).all());pred.grad=None
 loss,detail=exact_loss(pred,y,a,m,'stratified_huber');expected=.5*.03+.4*((3*.03+4*.05)/7)+.1*.19
 check('stratified_2Huber_matches_manual_value',np.isclose(float(loss.detach()),expected));loss.backward();expected_grad=np.array([.5*.2/3,.5*.2*2/3,.4*.2*3/7,.4*.2*4/7,.1*.2,0])
 check('stratified_2Huber_matches_manual_gradient',np.allclose(pred.grad.numpy(),expected_grad));check('layer_contributions_sum_to_loss',np.isclose(sum(d['contribution_m2'] for d in detail['strata'].values()),float(loss.detach())))
 _,missing=exact_loss(pred[:3],y[:3],a[:3],m[:3],'stratified_huber');check('empty_deep_stratum_renormalizes_without_fabricated_target',missing['strata']['deep']['present'] is False and missing['strata']['deep']['mean_loss_m2'] is None and np.isclose(missing['strata']['ordinary_wet']['mixture_weight'],.4/.9))
 reject('empty_loss_event_refused',lambda:exact_loss(pred,y,a,torch.zeros_like(m),'mse'))
 reject('invalid_active_target_refused',lambda:exact_loss(pred,y,a,torch.ones_like(m),'mse'))
 # Confusion areas are 1,2,3,4: check area weighting and FPR versus false discovery.
 area=np.array([1.,2.,3.,4.]);target=np.array([.2,0.,.3,0.]);prediction=np.array([.2,.2,0.,0.]);mask=np.ones(4,bool);v=evaluate_depth(prediction,target,area,mask)
 check('area_weighted_confusion_rates',np.isclose(v['fnr_0.1'],.75) and np.isclose(v['fpr_0.1'],2/6) and np.isclose(v['false_discovery_0.1'],2/3) and np.isclose(v['balanced_extent_error'],.5*(.75+2/6)))
 check('strict_wet_threshold_preserved',evaluate_depth(np.array([.1,.1]),np.array([.1,.10001]),np.ones(2),np.ones(2,bool))['missed_km2_0.1']==1e-6)
 fp32=evaluate_depth(np.array([.1,.1],dtype='f4'),np.array([.1,.10001],dtype='f4'),np.ones(2),np.ones(2,bool));check('float32_threshold_classification_preserves_input_precision',fp32['missed_km2_0.1']==1e-6 and fp32['false_km2_0.1']==0.)
 perfect=evaluate_depth(target,target,area,mask);dry=evaluate_depth(np.zeros(4),target,area,mask);flood=evaluate_depth(np.ones(4),target,area,mask)
 check('dry_and_allwet_cannot_win_over_perfect',perfect['balanced_extent_error']==0 and dry['balanced_extent_error']==.5 and flood['balanced_extent_error']==.5)
 empty=evaluate_depth(np.zeros(4),np.zeros(4),area,mask);check('empty_wet_class_and_zero_precision_not_perfect',empty['balanced_extent_error'] is None and empty['precision_0.1'] is None and not empty['selection_valid'])
 reject('nonfinite_active_prediction_never_dropped',lambda:evaluate_depth(np.array([np.nan,0,0,0]),target,area,mask))
 # Four fixture parents carry real validation IDs but clearly synthetic metric inputs.
 def frames(metric):return [dict(metric,case_id=cid,split='validation',scope='three_wards',support='paired_exact') for cid in p['validation_case_ids']]
 em={cid:perfect['support_sha256'] for cid in p['validation_case_ids']};per=frames(perfect);dr=frames(dry);fl=frames(flood)
 check('selection_total_order_and_earliest_tie',checkpoint_key(per,2,p['validation_case_ids'],em)<checkpoint_key(dr,1,p['validation_case_ids'],em) and checkpoint_key(per,1,p['validation_case_ids'],em)<checkpoint_key(per,2,p['validation_case_ids'],em))
 check('undefined_class_checkpoint_unrankable',checkpoint_key(frames(empty),1,p['validation_case_ids'],em) is None)
 reject('incomplete_validation_checkpoint_refused',lambda:checkpoint_key(per[:-1],1,p['validation_case_ids'],em))
 reject('wrong_checkpoint_mask_refused',lambda:checkpoint_key(per,1,p['validation_case_ids'],{cid:'wrong' for cid in em}))
 agg=macro_events([dict(perfect,case_id='x',deep_mae_m=None),dict(v,case_id='y',deep_mae_m=.5)])
 check('event_macro_and_defined_deep_count',np.isclose(agg['values']['balanced_extent_error'],v['balanced_extent_error']/2) and agg['values']['deep_mae_m']==.5 and agg['defined_parents']['deep_mae_m']==1)
 check('readiness_dry_fails_and_perfect_can_pass',not baseline_readiness({r['case_id']:r for r in dr},{r['case_id']:r for r in dr},{r['case_id']:r for r in per})['all_parents_pass'] and baseline_readiness({r['case_id']:r for r in per},{r['case_id']:r for r in dr},{r['case_id']:r for r in per})['all_parents_pass'])
 # Original loss interfaces on actual frozen predictions; no forward/optimizer run.
 z=arrays(SOURCE/'prepared_inputs.npz');best=arrays(SOURCE/'predictions.npz')['Conditional UNet Peak Mean Depth'];last=arrays(ROOT/'data/processed/three_ward_unet_shallow_diagnosis_v1/diagnostic_predictions.npz')['last_epoch35'];records=p['records'];by={r['case_id']:i for i,r in enumerate(records)};labels=[load_label(r['case_id'],r['split']) for r in records];masks=[d['Exact Label Mask']&z['Eligible Ground Mask'] for d in labels];ys=[d['Peak Mean Depth Lower'].astype('f4') for d in labels];areas=z['Ground Area'];ward=z['Ward Code'];geos={'domain':np.ones_like(ward,bool),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0};expectedmask={r['case_id']:r['paired_three_ward_mask_sha256'] for r in records if r['split']=='validation'}
 fixed=[];baselines={'dry':{},'retrieval':{}};lastrows=[];audit=[];start=time.perf_counter()
 for i,r in enumerate(records):
  j=by[r['retrieval_source_case_id']];pair=masks[i]&masks[j]
  check(r['case_id']+'_exact_and_pair_masks_unchanged',mask_sha(masks[i])==r['original_exact_eligible_mask_sha256'] and mask_sha(pair)==r['paired_exact_mask_sha256'] and mask_sha(pair&(ward>0))==r['paired_three_ward_mask_sha256'])
  check(r['case_id']+'_retrieval_training_other_family',records[j]['split']=='train' and records[j]['family_id']!=r['family_id'])
  for scope,region in geos.items():
   for support,mask0 in [('own_exact',masks[i]&region),('paired_exact',pair&region)]:
    metric=evaluate_depth(best[i],ys[i],areas,mask0);fixed.append(metric|{'case_id':r['case_id'],'parent_id':r['parent_id'],'split':r['split'],'scope':scope,'support':support})
  if r['split']=='validation':
   mask0=pair&(ward>0)
   for method,pred0 in [('dry',np.zeros_like(ys[i])),('retrieval',ys[j])]:baselines[method][r['case_id']]=evaluate_depth(pred0,ys[i],areas,mask0)
   lastrows.append(evaluate_depth(last[i],ys[i],areas,mask0)|{'case_id':r['case_id'],'split':'validation','scope':'three_wards','support':'paired_exact'})
  for kind in ['mse','stratified_huber']:
   value,d=exact_loss(torch.from_numpy(best[i]),torch.from_numpy(ys[i]),torch.from_numpy(areas),torch.from_numpy(masks[i]),kind)
   check(r['case_id']+'_'+kind+'_finite_real_exact_loss',torch.isfinite(value) and np.isclose(sum(t['contribution_m2'] for t in d['strata'].values()),float(value)))
   audit.append({'case_id':r['case_id'],'split':r['split'],'loss':kind,'loss_value_m2':float(value),'detail':d})
 for row in fixed:
  if row['scope']=='domain' and row['support']=='own_exact':
   matching=next(q for q in audit if q['case_id']==row['case_id'] and q['loss']=='stratified_huber');row.update({'fixed_model_loss_m2':matching['loss_value_m2'],'loss_kind':'stratified_huber','fixed_loss_strata':matching['detail']['strata']})
 write(REPORT/'frozen_prediction_loss_probe.json',{'record_kind':'interface_probe','no_fitting':True,'note':'Values from existing predictions, not optimized new arms; loss scales/support mixtures differ. No claim that smaller cross-loss values mean better performance.','events':audit})
 val=[r for r in fixed if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact'];probes=[]
 for name,epoch,rows0 in [('original_saved_epoch15',15,val),('original_saved_epoch35',35,lastrows)]:
  key=checkpoint_key(rows0,epoch,p['validation_case_ids'],expectedmask);ready=baseline_readiness({r['case_id']:r for r in rows0},baselines['dry'],baselines['retrieval']);check(name+'_guard_reports_not_ready',not ready['all_parents_pass']);probes.append({'source':name,'record_kind':'retrospective_interface_probe','selection_key_if_this_rule_were_used':key,'readiness':ready,'model_promoted':False})
 write(REPORT/'selection_rule_probe.json',{'no_fit_or_model_promotion':True,'probes':probes})
 # Actual full-field backward interface check at an unchanged existing checkpoint.
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=ConditionalUNet().to(device);ck=torch.load(SOURCE/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['state_dict']);model.eval();before=parameter_snapshot(model)
 class NoOptimizerFixture:
  def state_dict(self):return {'state':{},'param_groups':[{'lr':.001}]}
 opt=NoOptimizerFixture();r=records[0];i=0;context={'arm':'no_fit_existing_checkpoint_probe','epoch':0,'source_checkpoint_epoch':15,'case_id':r['case_id'],'parent_id':r['parent_id'],'family_id':r['family_id'],'forcing_sha256':r['forcing_sha256'],'split':'train','seed':p['frozen_common_settings']['seed'],'mask_sha256':r['original_exact_eligible_mask_sha256'],'loss_kind':'stratified_huber','record_kind':'interface_probe','optimizer_instance':'none; state-only interface fixture','timing_scope':'Read-only forward/backward and logger check, not training'}
 observer=OptimizationObserver(REPORT/'step_interface_probe.jsonl',ph);observer.begin(model,opt,context);xx=torch.from_numpy(z['Static Input'][None]).to(device);cc=torch.from_numpy(z['Forcing Input'][i:i+1]).to(device);prediction=model(xx,cc)[0,0];loss,payload=exact_loss(prediction,torch.from_numpy(ys[i]).to(device),torch.from_numpy(areas.astype('f4')).to(device),torch.from_numpy(masks[i]).to(device),'stratified_huber');loss.backward();observer.after_backward(model,payload);torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);observer.after_clip(model);result=observer.finish(model,opt,expected_update=False)
 check('real_model_loss_and_logger_probe_no_parameter_change',all(torch.equal(before[k],v) for k,v in parameter_snapshot(model).items()) and result['optimizer_update_performed'] is False)
 summary0=summarize_step_records([result],[r['case_id']],record_kind='interface_probe');check('probe_aggregation_cannot_claim_updates',summary0['optimizer_updates']==0 and summary0['record_kind']=='interface_probe')
 model.zero_grad(set_to_none=True)
 # Known artificial gradients exercise clipping records without any learning step.
 dummy=torch.nn.Linear(1,1);obs=OptimizationObserver(REPORT/'synthetic_clipping_probe.jsonl',ph);ctx=context|{'arm':'synthetic_logger_fixture','case_id':'SYNTHETIC-FIXTURE','parent_id':'FIXTURE','family_id':'FIXTURE','forcing_sha256':'not_applicable','loss_kind':'synthetic_gradient_fixture','source_checkpoint_epoch':None};obs.begin(dummy,opt,ctx)
 for q in dummy.parameters():q.grad=torch.full_like(q,20.)
 obs.after_backward(dummy,{'kind':'synthetic','value_m2':0.});torch.nn.utils.clip_grad_norm_(dummy.parameters(),10,error_if_nonfinite=True);obs.after_clip(dummy);res=obs.finish(dummy,opt,expected_update=False);check('clipping_required_logged_without_optimizer_update',res['clipping_required'] and res['gradient_post_clip']['total_norm']<=10.0001 and not res['optimizer_update_performed'])
 reject('out_of_order_logger_call_refused',lambda:obs.finish(dummy,opt,expected_update=False))
 check('synthetic_parameter_delta_formula',parameter_change({'head.weight':torch.zeros(1)},{'head.weight':torch.tensor([3.])})['head']['update_norm']==3)
 # Pure in-memory synthetic clock dictionaries test future aggregation; no optimizer or history is created.
 import copy
 synthetic=[]
 for ii in range(2):
  rr=copy.deepcopy(res);rr.update({'arm':'SYNTHETIC-CLOCK-FIXTURE','case_id':'FIXTURE-'+str(ii),'record_kind':'training_update','optimizer_update_performed':True});rr['optimizer_before']['steps']=[ii];rr['optimizer_after']['steps']=[ii+1];synthetic.append(rr)
 ag=summarize_step_records(synthetic,['FIXTURE-0','FIXTURE-1']);check('synthetic_clock_aggregation_and_clipping_counts',ag['optimizer_updates']==2 and ag['clipping_count']==2)
 check('aggregation_binds_clock_and_loss_distribution',ag['optimizer_start_step']==0 and ag['optimizer_end_step']==2 and ag['online_preupdate_loss_distribution_m2']['maximum']==0.)
 check('probe_stratum_aggregate_preserved',set(summary0['mean_stratum_loss_contribution_m2'])=={'nonwet','ordinary_wet','deep'})
 synthetic[1]['optimizer_before']['steps']=[3];synthetic[1]['optimizer_after']['steps']=[4];reject('noncontiguous_optimizer_history_refused',lambda:summarize_step_records(synthetic,['FIXTURE-0','FIXTURE-1']))
 reject('interface_probe_not_summarized_as_training',lambda:summarize_step_records([result],[r['case_id']]))
 # Complete fixed-epoch metric frame from already saved predictions.
 ep={'loss_kind':'stratified_huber','step_summary':None,'arm':'no_fit_existing_checkpoint_probe','epoch':15,'record_kind':'interface_probe','optimizer_steps_before_eval':0,'optimizer_steps_after_eval':0,'model_fingerprint_before_eval':sha(SOURCE/'best.pt'),'model_fingerprint_after_eval':sha(SOURCE/'best.pt'),'checkpoint_path':str(SOURCE/'best.pt'),'checkpoint_sha256':sha(SOURCE/'best.pt'),'cumulative_training_seconds':0.,'cumulative_evaluation_seconds':time.perf_counter()-start,'cumulative_checkpoint_io_seconds':0.,'note':'Source epoch15 rescore only; this is not a new training epoch or optimization trace.'}
 epoch=write_epoch_record(REPORT/'epoch_interface_probe.jsonl',ph,ep,fixed,{r['case_id']:r['split'] for r in records},expectedmask,baselines);check('fixed_epoch_frame_all192_scopes_and_no_false_readiness',len(epoch['event_metrics'])==192 and not epoch['baseline_readiness']['all_parents_pass'])
 reject('epoch_update_during_eval_refused',lambda:write_epoch_record(REPORT/'must_not_exist.jsonl',ph,ep|{'optimizer_steps_after_eval':1},fixed,{r['case_id']:r['split'] for r in records},expectedmask,baselines))
 reject('missing_fixed_epoch_metric_frame_refused',lambda:write_epoch_record(REPORT/'must_not_exist.jsonl',ph,ep,fixed[:-1],{r['case_id']:r['split'] for r in records},expectedmask,baselines))
 check('no_invalid_epoch_log_created',not (REPORT/'must_not_exist.jsonl').exists())
 check('source_bindings_after',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()) and all(sha(ROOT/k)==v for k,v in old['source_hashes'].items()))
 check('original_trial_remains_frozen_after_gradient_probe',all(sha(ROOT/k)==v for k,v in read(SOURCE/'freeze.json')['files_sha256'].items()))
 write(REPORT/'validation.json',{'all_checks_passed':True,'checks':checks,'new_training_runs':0,'optimizer_updates':0,'source_unchanged':True,'sealed_opened':False,'probe_device':str(device),'original_quality_masks_preserved':True,'initialization_is_untrained':True});print(json.dumps({'checks':len(checks),'all_checks_passed':True,'new_fits':0,'optimizer_updates':0}),flush=True)
if __name__=='__main__':main()
