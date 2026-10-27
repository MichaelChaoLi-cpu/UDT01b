"""Independent real-exposure, mask, arithmetic, selector and prediction checks."""
from pathlib import Path
import sys,json,hashlib,csv
from html.parser import HTMLParser
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from three_ward_spatial_fp_interfaces_v1 import SpatialConditionalUNet
from run_three_ward_unet_controls_v2 import fingerprint
NAME='three_ward_spatial_fp_controls_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
OLDPLAN=ROOT/'data/processed/three_ward_unet_control_plan_v2';OLD=ROOT/'data/processed/three_ward_unet_controls_v2';OLDREPORT=ROOT/'data/exp/estimation/three_ward_unet_controls_v2';SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'
def main():
 p=read(OUT/'protocol.json');done=read(REPORT/'completion.json');ex=read(OUT/'execution_protocol.json');ph=sha(OUT/'execution_protocol.json');orders=read(OLDPLAN/'epoch_order.json');z=arrays(SOURCE/'prepared_inputs.npz');records=p['records'];by={r['case_id']:i for i,r in enumerate(records)};labels=[load_label(r['case_id'],r['split']) for r in records];ys=np.array([v['Peak Mean Depth Lower'].astype('f4') for v in labels]);masks=np.array([v['Exact Label Mask']&z['Eligible Ground Mask'] for v in labels]);area=z['Ground Area'];ward=z['Ward Code'];eligible=z['Eligible Ground Mask'];bases=read(REPORT/'frozen_baseline_metrics.json');checks=[];selections={};allframes={};totalsteps=0
 def check(n,v):
  if not bool(v):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 check('exactly3newfits1440updates_no_expansion_or_test',done['new_fits']==3 and done['new_optimizer_updates']==1440 and not done['sample_expansion'] and not done['sealed_opened'])
 check('preflight_passed_before_fit_zero_updates',read(REPORT/'preflight.json')['all_checks_passed'] and read(REPORT/'preflight.json')['optimizer_updates']==0)
 check('execution_uses_bound_verified_interface',ex['interface_sha256']==sha(ROOT/'src/analyses/three_ward_spatial_fp_interfaces_v1.py') and ex['runner_sha256']==sha(ROOT/'src/analyses/run_three_ward_spatial_fp_controls_v1.py'))
 check('all_sources_labels_curves_norm_remain_unchanged',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 check('frozen_preflight_unchanged',all(sha(ROOT/k)==v for k,v in read(OUT/'preflight_freeze.json')['files_sha256'].items()))
 baselinefp=read(OLDREPORT/'D_huber_lower_lr/initialization_receipt.json')['loaded_parameter_fingerprint'];check('all_use_original_D_untrained_fingerprint',baselinefp==p['initial_parameter_fingerprint'])
 for aid in ['S0F0','S1F0','S0F1','S1F1']:
  reference=aid=='S0F0';cfg=next((q for q in p['new_arms'] if q['arm']==aid),{'retain_deep_detail':False,'fp_supervision':False});dest=OUT/aid;report=REPORT/aid
  if not reference:
   ini=read(report/'initialization_receipt.json');steps=[json.loads(l) for l in (report/'optimizer_steps.jsonl').read_text().splitlines()];totalsteps+=len(steps);check(aid+'_480realupdates_no_held_failure',len(steps)==480 and not (report/'failure.json').exists());check(aid+'_identical_full_initialization',ini['initial_parameter_fingerprint']==baselinefp and ini['initial_optimizer_steps']==0 and ini['parameters']==671513 and ini['shared_initialization_sha256']==sha(OLDPLAN/'shared_initialization.pt'))
   for epoch in range(1,41):
    ss=steps[(epoch-1)*12:epoch*12];check(aid+f'_epoch{epoch}_original_training_order', [s['case_id'] for s in ss]==orders[epoch-1]['case_order'])
    valid=True
    for j,s in enumerate(ss):
     r=records[by[s['case_id']]];n=(epoch-1)*12+j+1;valid &= s['record_kind']=='training_update' and s['optimizer_update_performed'] and s['split']=='train' and s['family_id']==r['family_id'] and s['forcing_sha256']==r['forcing_sha256'] and s['mask_sha256']==r['original_exact_eligible_mask_sha256'] and s['seed']==ex['seed'] and s['protocol_sha256']==ph
     valid &= max(s['optimizer_before']['steps'],default=0)==n-1 and s['optimizer_after']['steps']==[n] and s['optimizer_after']['learning_rates']==[.0003] and s['gradient_pre_clip']['finite'] and s['gradient_post_clip']['finite'] and s['gradient_post_clip']['total_norm']<=10.0001 and s['clipping_required']==(s['gradient_pre_clip']['total_norm']>10)
     valid &= s['retain_deep_detail']==cfg['retain_deep_detail'] and s['fp_supervision']==cfg['fp_supervision'] and sum(v['update_norm']**2 for v in s['parameter_change'].values())>0
     loss=s['loss'];valid &= abs(sum(v['contribution_m2'] for v in loss['strata'].values())-loss['base_value_m2'])<max(1e-7,abs(loss['base_value_m2'])*1e-5) and abs(loss['value_m2']-loss['base_value_m2']-loss['optimized_FP_contribution_m2'])<1e-7 and loss['fp_multiplier']==int(cfg['fp_supervision']) and abs(loss['optimized_FP_contribution_m2']-(loss['fp_penalty_m2'] if cfg['fp_supervision'] else 0))<1e-8
    check(aid+f'_epoch{epoch}_actual_clock_identity_loss_and_gradients',valid)
  frames=[];independent=[]
  for epoch in range(1,41):
   frame=read((OLDREPORT/'D_huber_lower_lr'/f'epoch_{epoch:03}.jsonl') if reference else report/f'epoch_{epoch:03}.json');frames.append(frame);ck=(OLD/'D_huber_lower_lr'/f'epoch_{epoch:03}.pt') if reference else dest/f'epoch_{epoch:03}.pt';check(aid+f'_epoch{epoch}_checkpoint_and_eval_clock_bound',sha(ck)==frame['checkpoint_sha256'] and frame['optimizer_steps_before_eval']==frame['optimizer_steps_after_eval']==12*epoch and frame['model_fingerprint_before_eval']==frame['model_fingerprint_after_eval'])
   ev=frame['event_metrics'];expected={(r['case_id'],r['split'],sc,su) for r in records for sc in ['domain','three_wards','Chiyoda','Chuo','Minato','cross_ward'] for su in ['paired_exact','own_exact']};actual={(r['case_id'],r['split'],r['scope'],r['support']) for r in ev};check(aid+f'_epoch{epoch}_complete192events_and_original_masks',len(ev)==192 and actual==expected and all(r['support_sha256']==records[by[r['case_id']]]['paired_three_ward_mask_sha256'] for r in ev if r['scope']=='three_wards' and r['support']=='paired_exact'))
   val=[v for v in ev if v['split']=='validation' and v['scope']=='three_wards' and v['support']=='paired_exact'];passed=True
   for v in val:
    b=bases['retrieval'][v['case_id']];dry=bases['dry'][v['case_id']];passed &= v['balanced_extent_error']<=b['balanced_extent_error']+1e-6 and v['mae_m']<=min(b['mae_m'],dry['mae_m'])+1e-6 and v['recall_0.1']>=b['recall_0.1']-1e-6 and v['false_km2_0.1']<=b['false_km2_0.1']+1e-6
   mean=lambda k:float(np.mean([v[k] for v in val]));key=(-round(mean('iou_0.1'),6),round(mean('wet_mae_m'),6),round(mean('mae_m'),6),epoch);bkey=(round(float(np.mean([.5*(v['fnr_0.1']+v['fpr_0.1']) for v in val])),6),key[1],key[2],epoch);independent.append({'epoch':epoch,'pass':passed,'key':key,'bkey':bkey})
  allframes[aid]=frames;passing=[v for v in independent if v['pass']];pool=passing or independent;sel=min(pool,key=lambda r:r['key']);shadow=min(pool,key=lambda r:r['bkey']);selection=done['reference_selection'] if reference else read(report/'final_selection.json');selections[aid]=selection
  check(aid+'_independent_IoU_guard_and_BER_pool_selection',selection['selected_epoch']==sel['epoch'] and selection['shadow_BER_epoch']==shadow['epoch'] and selection['candidate_pool_epochs']==[v['epoch'] for v in pool] and selection['passing_epochs']==[v['epoch'] for v in passing] and selection['status']==('baseline_guard_passed' if passing else 'not_ready'))
  for point,epoch,file in [('endpoint',40,'endpoint_predictions.npz'),('selected',selection['selected_epoch'],'selected_predictions.npz')]:
   zz=arrays(dest/file);pred=zz['Peak Mean Depth'];check(aid+'_'+point+'_all16predictions_original_coverage',pred.shape==(16,1012,770) and list(zz['Case ID'])==[r['case_id'] for r in records] and np.isfinite(pred[:,eligible]).all() and (pred[:,eligible]>=0).all() and np.isnan(pred[:,~eligible]).all())
   frame=frames[epoch-1]
   for i,r in enumerate(records):
    m=masks[i]&masks[by[r['retrieval_source_case_id']]]&(ward>0);t=ys[i][m];v=pred[i][m];a=area[m];w=t>.1;flood=v>.1;tp=a[w&flood].sum();fp=a[~w&flood].sum();fn=a[w&~flood].sum();tn=a[~w&~flood].sum();mae=np.sum(a*np.abs(v.astype('f8')-t.astype('f8')))/a.sum();B=.5*(fn/(tp+fn)+fp/(fp+tn));IoU=tp/(tp+fp+fn);record=next(q for q in frame['event_metrics'] if q['case_id']==r['case_id'] and q['scope']=='three_wards' and q['support']=='paired_exact')
    check(aid+'_'+point+'_'+r['case_id']+'_maps_match_frozen_score',abs(mae-record['mae_m'])<1e-7 and abs(B-record['balanced_extent_error'])<1e-10 and abs(fp/1e6-record['false_km2_0.1'])<1e-10 and abs(IoU-record['iou_0.1'])<1e-10)
    if not reference:
     mm=masks[i];tt=ys[i][mm];vv=pred[i][mm];aa=area[mm];nw=tt<=.1;penalty=.1*np.sum(aa[nw]*np.maximum(vv[nw]-.1,0))/aa[nw].sum() if nw.any() else 0.
     q=next(v for v in frame['event_metrics'] if v['case_id']==r['case_id'] and v['scope']=='domain' and v['support']=='own_exact')
     check(aid+'_'+point+'_'+r['case_id']+'_independent_whole_field_FP_penalty',abs(penalty-q['fixed_FP_penalty_m2'])<max(1e-7,abs(penalty)*2e-5) and abs(q['fixed_model_loss_m2']-q['fixed_base_loss_m2']-(q['fixed_FP_penalty_m2'] if cfg['fp_supervision'] else 0))<1e-7)

  if not reference:
   best=torch.load(dest/'best.pt',map_location='cpu',weights_only=False);epochstate=torch.load(dest/f"epoch_{selection['selected_epoch']:03}.pt",map_location='cpu',weights_only=False);live=torch.load(dest/'live.pt',map_location='cpu',weights_only=False)
   check(aid+'_best_exactly_selected_weights',best['epoch']==selection['selected_epoch'] and all(torch.equal(v,epochstate['state_dict'][k]) for k,v in best['state_dict'].items()))
   check(aid+'_retained_final_optimizer_and_RNG_state',live['epoch']==40 and live['steps']==480 and live['stage']=='epoch_complete' and live['selection'] is not None and all(int(v['step'])==480 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in live['optimizer']['state'].values()) and len(live['optimizer']['state'])==96 and all(k in live for k in ['torch_rng','numpy_rng','python_rng','mps_rng']))
 # Verify treatment tables against the actual bound per-parent score frames.
 derived=list(csv.DictReader((REPORT/'paired_contrasts.csv').open()));derived_ok=True
 for row in derived:
  lv=[]
  for aid in [row['left'],row['right']]:
   ep=40 if row['point']=='epoch40' else selections[aid]['selected_epoch'];fr=allframes[aid][ep-1];lv.append(next(v for v in fr['event_metrics'] if v['case_id']==row['case_id'] and v['scope']==row['scope'] and v['support']=='paired_exact'))
  for k,v in row.items():
   if not k.startswith('delta_') or k=='delta_AUC':continue
   metric=k[6:];left=lv[0].get(metric);right=lv[1].get(metric)
   derived_ok &= (v=='' if left is None or right is None else abs(float(v)-(left-right))<1e-12)
 check('all_paired_metric_deltas_recomputed_from_bound_frames',derived_ok and len(derived)==192)
 check('only3actualfits1440updates',totalsteps==1440 and {d.name for d in OUT.iterdir() if d.is_dir()}=={'S0F0','S1F0','S0F1','S1F1'})
 # One independently re-inferred complete validation field per new arm; no fit.
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');x=torch.from_numpy(z['Static Input'][None]).to(device);i=by[p['validation_case_ids'][0]];cond=torch.from_numpy(z['Forcing Input'][i:i+1]).to(device)
 for cfg in p['new_arms']:
  a=cfg['arm'];model=SpatialConditionalUNet(cfg['retain_deep_detail']).to(device);ck=torch.load(OUT/a/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);model.eval();before=fingerprint(model)
  with torch.inference_mode():pred=model(x,cond)[0,0].cpu().numpy()
  stored=arrays(OUT/a/'selected_predictions.npz')['Peak Mean Depth'][i];check(a+'_independent_selected_field_replay',float(np.max(np.abs(pred[eligible]-stored[eligible])))<1e-5 and fingerprint(model)==before);del model
 # Manifest and all result arithmetic are generated from actual records above.
 check('all_source_hashes_still_unchanged',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[]
  def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
 parser=Links();parser.feed((REPORT/'review.html').read_text());check('report_links_valid_or_this_validation_output',all('://' not in s and ((REPORT/s).is_file() or s=='delivery_validation.json') for s in parser.links))
 write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'checks':checks,'preflight_checks':len(read(REPORT/'preflight.json')['checks']),'independent_delivery_checks':len(checks),'actual_new_fits':3,'actual_new_updates':1440,'sealed_opened':False,'original_masks_and_lineage_preserved':True,'control_reused_without_fit':True,'selected_weight_field_replays':3});print(json.dumps({'all_checks_passed':True,'checks':len(checks),'real_new_fits':3,'real_updates':1440}))
if __name__=='__main__':main()
