"""Verify real four-arm exposure, optimizer clocks, selection and stored maps."""
from pathlib import Path
import sys,json,csv,math,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import OUT as SOURCE,sha,read,write,arrays,load_label
from prepare_three_ward_unet_control_plan_v2 import OUT as PLAN,mask_sha
from train_three_ward_unet_exact_pilot import ConditionalUNet
from three_ward_surrogate_protocol_v2 import baseline_readiness
NAME='three_ward_unet_controls_v2';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def main():
 p=read(PLAN/'protocol.json');done=read(REPORT/'completion.json');ex=read(OUT/'execution_protocol.json');ph=sha(OUT/'execution_protocol.json');orders=read(PLAN/'epoch_order.json');z=arrays(SOURCE/'prepared_inputs.npz');records=p['records'];by={r['case_id']:i for i,r in enumerate(records)};baselines=read(REPORT/'frozen_baseline_metrics.json');checks=[]
 def check(name,value):
  if not value:raise AssertionError(name)
  checks.append({'check':name,'pass':True})
 check('four_authorized_fits_1920_real_updates_no_sealed',done['actual_fits']==4 and done['optimizer_updates']==1920 and done['sealed_opened'] is False and ex['decision']=='MILI-D-20261016-005')
 check('prepared_protocol_and_all_frozen_files_unchanged',ex['prepared_protocol_sha256']==sha(PLAN/'protocol.json') and all(sha(ROOT/k)==v for k,v in read(PLAN/'freeze.json')['files_sha256'].items()))
 check('all_original_sources_labels_normalization_unchanged',all(sha(ROOT/k)==v for k,v in read(SOURCE/'protocol.json')['source_hashes'].items()))
 check('runner_and_interface_bound',ex['runner_sha256']==sha(ROOT/'src/analyses/run_three_ward_unet_controls_v2.py') and ex['interface_sha256']==sha(ROOT/'src/analyses/three_ward_surrogate_protocol_v2.py'))
 check('same_shared_initialization_and_order_bound',ex['shared_initialization_sha256']==sha(PLAN/'shared_initialization.pt') and ex['epoch_order_sha256']==sha(PLAN/'epoch_order.json'))
 initfp=[];selectedevents=[];target={r['case_id']:load_label(r['case_id'],r['split']) for r in records};eligible=z['Eligible Ground Mask'];area=z['Ground Area'];ward=z['Ward Code'];actualsteps=0;selectedmodels=[]
 for armcfg in p['arms']:
  aid=armcfg['arm_id'];dest=OUT/aid;report=REPORT/aid;arm=read(report/'completion.json');ini=read(report/'initialization_receipt.json');initfp.append(ini['loaded_parameter_fingerprint']);steps=rows(report/'optimizer_steps.jsonl');actualsteps+=len(steps)
  check(aid+'_40epochs_480updates_no_failure',arm['epochs']==40 and arm['optimizer_updates']==480 and len(steps)==480 and not (report/'failure.json').exists())
  check(aid+'_same_untrained_start_and_approved_LR_loss',ini['shared_initialization_sha256']==ex['shared_initialization_sha256'] and ini['initial_optimizer_steps']==0 and ini['learning_rate']==armcfg['learning_rate'] and ini['loss']==armcfg['loss'])
  for epoch in range(1,41):
   ss=steps[(epoch-1)*12:epoch*12];order=[r['case_id'] for r in ss]
   check(aid+f'_epoch{epoch}_frozen_training_order_only',order==orders[epoch-1]['case_order'] and all(r['split']=='train' and r['epoch']==epoch and r['record_kind']=='training_update' and r['optimizer_update_performed'] for r in ss))
   ok=True
   for offset,row in enumerate(ss):
    rec=next(r for r in records if r['case_id']==row['case_id']);s=(epoch-1)*12+offset+1;before=max(row['optimizer_before']['steps'],default=0);after=row['optimizer_after']['steps'];loss=row['loss'];grad=row['gradient_pre_clip'];post=row['gradient_post_clip']
    ok &= before==s-1 and after==[s] and row['protocol_sha256']==ph and row['arm']==aid and row['seed']==ex['seed'] and row['mask_sha256']==rec['original_exact_eligible_mask_sha256'] and row['family_id']==rec['family_id'] and row['forcing_sha256']==rec['forcing_sha256']
    ok &= row['optimizer_before']['learning_rates']==[armcfg['learning_rate']] and row['optimizer_after']['learning_rates']==[armcfg['learning_rate']] and row['loss_kind']==armcfg['loss']
    ok &= grad['finite'] and post['finite'] and math.isfinite(grad['total_norm']) and post['total_norm']<=10.0001 and row['clipping_required']==(grad['total_norm']>10.)
    ok &= math.isfinite(loss['value_m2']) and abs(sum(v['contribution_m2'] for v in loss['strata'].values())-loss['value_m2'])<max(1e-7,abs(loss['value_m2'])*1e-5)
    ok &= sum(v['update_norm']**2 for v in row['parameter_change'].values())>0 and all(math.isfinite(v['update_norm']) for v in row['parameter_change'].values())
   check(aid+f'_epoch{epoch}_actual_clock_masks_gradients_updates',ok)
  epochs=[];keys=[]
  for epoch in range(1,41):
   path=report/f'epoch_{epoch:03}.jsonl';rr=rows(path);check(aid+f'_epoch{epoch}_one_complete_bound_score_frame',len(rr)==1 and rr[0]['protocol_sha256']==ph and len(rr[0]['event_metrics'])==192)
   e=rr[0];epochs.append(e);check(aid+f'_epoch{epoch}_frozen_model_and_real_step_summary',e['optimizer_steps_before_eval']==e['optimizer_steps_after_eval']==12*epoch and e['model_fingerprint_before_eval']==e['model_fingerprint_after_eval'] and e['step_summary']['optimizer_start_step']==12*(epoch-1) and e['step_summary']['optimizer_end_step']==12*epoch and e['step_summary']['parents']==12)
   check(aid+f'_epoch{epoch}_saved_checkpoint_hash',sha(dest/f'epoch_{epoch:03}.pt')==e['checkpoint_sha256'])
   vv=[r for r in e['event_metrics'] if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact'];check(aid+f'_epoch{epoch}_exact4_validation_support_bound',len(vv)==4 and {r['case_id'] for r in vv}==set(p['validation_case_ids']) and all(r['support_sha256']==next(s for s in records if s['case_id']==r['case_id'])['paired_three_ward_mask_sha256'] for r in vv))
   # Independent rank calculation from per-parent confusion rates, not cached key.
   B=float(np.mean([.5*(r['fnr_0.1']+r['fpr_0.1']) for r in vv]));wet=float(np.mean([r['wet_mae_m'] for r in vv]));mae=float(np.mean([r['mae_m'] for r in vv]));key=(round(B,6),round(wet,6),round(mae,6),epoch);keys.append(key)
   check(aid+f'_epoch{epoch}_selection_key_independent_reduction',tuple(e['selection_key'])==key)
  best=min(keys);check(aid+'_selected_checkpoint_is_global_minimum_frozen_key',tuple(arm['selection_key'])==best and arm['best_epoch']==best[-1])
  selected=read(report/'selected_metrics.json');check(aid+'_selected_score_matches_saved_epoch',selected==epochs[arm['best_epoch']-1]);vv={r['case_id']:r for r in selected['event_metrics'] if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact'};ready=baseline_readiness(vv,baselines['dry'],baselines['retrieval']);check(aid+'_baseline_guard_not_hidden',ready==selected['baseline_readiness'] and ready['all_parents_pass']==arm['baseline_all_parents_pass'])
  ck=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);same=torch.load(dest/f'epoch_{arm["best_epoch"]:03}.pt',map_location='cpu',weights_only=False);check(aid+'_best_weights_equal_selected_epoch',all(torch.equal(v,same['state_dict'][k]) for k,v in ck['state_dict'].items()) and ck['epoch']==best[-1] and ck['protocol_sha256']==ph)
  live=torch.load(dest/'live.pt',map_location='cpu',weights_only=False);opt=live['optimizer'];check(aid+'_retained_final_optimizer_480steps_and_rng',live['epoch']==40 and live['steps']==480 and live['phase']=='ready' and all(int(v['step'])==480 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in opt['state'].values()) and len(opt['state'])==96 and 'torch_rng' in live and 'numpy_rng' in live)
  pred=arrays(dest/'selected_predictions.npz');pp=pred['Peak Mean Depth'];check(aid+'_selected_map_shape_coverage_nonnegativity',pp.shape==(16,1012,770) and np.array_equal(pred['Case ID'],[r['case_id'] for r in records]) and np.isfinite(pp[:,eligible]).all() and (pp[:,eligible]>=0).all() and np.isnan(pp[:,~eligible]).all())
  check(aid+'_checkpoint_prediction_hashes_match_completion',sha(dest/'best.pt')==arm['best_checkpoint_sha256'] and sha(dest/'selected_predictions.npz')==arm['selected_predictions_sha256'])
  for cid,r in vv.items():
   i=by[cid];j=by[next(q for q in records if q['case_id']==cid)['retrieval_source_case_id']];m=target[cid]['Exact Label Mask']&target[records[j]['case_id']]['Exact Label Mask']&eligible&(ward>0);y=target[cid]['Peak Mean Depth Lower'].astype('f4')[m];v=pp[i,m];a=area[m];wet=y>.1;flood=v>.1;tp=a[wet&flood].sum();fp=a[~wet&flood].sum();fn=a[wet&~flood].sum();tn=a[~wet&~flood].sum();mae=np.sum(a*np.abs(v.astype('f8')-y.astype('f8')))/a.sum();B=.5*(fn/(tp+fn)+fp/(fp+tn));IoU=tp/(tp+fp+fn)
   check(aid+'_'+cid+'_stored_maps_reproduce_primary_metrics',abs(float(mae)-r['mae_m'])<1e-10 and abs(float(B)-r['balanced_extent_error'])<1e-10 and abs(float(IoU)-r['iou_0.1'])<1e-10 and mask_sha(m)==r['support_sha256'])
  selectedmodels.append((aid,ck,pp))
 check('all4_initial_parameter_fingerprints_identical',len(set(initfp))==1);check('actual1920steps_no_extra_arm_directories',actualsteps==1920 and {q.name for q in OUT.iterdir() if q.is_dir()}=={a['arm_id'] for a in p['arms']})
 # Re-infer one complete validation field per arm from selected weights, no fitting.
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');torch.set_num_threads(6);x=torch.from_numpy(z['Static Input'][None]).to(device);i=by[p['validation_case_ids'][0]];c=torch.from_numpy(z['Forcing Input'][i:i+1]).to(device)
 for aid,ck,pp in selectedmodels:
  model=ConditionalUNet().to(device);model.load_state_dict(ck['state_dict']);model.eval()
  with torch.inference_mode():a=model(x,c)[0,0].cpu().numpy()
  check(aid+'_selected_weights_reproduce_full_validation_map',float(np.max(np.abs(a[eligible]-pp[i,eligible])))<1e-5);del model
 check('all_frozen_preparation_still_unchanged_after_validation',all(sha(ROOT/k)==v for k,v in read(PLAN/'freeze.json')['files_sha256'].items()))
 write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'checks':checks,'actual_fits':4,'real_optimizer_updates':1920,'sealed_opened':False,'source_masks_and_partitions_preserved':True,'revalidation_performed_no_new_fit':True});print(json.dumps({'checks':len(checks),'all_checks_passed':True,'actual_fits':4,'real_updates':1920}))
if __name__=='__main__':main()
