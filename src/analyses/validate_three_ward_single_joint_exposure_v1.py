"""Independent fixed-budget, mask, endpoint score and frozen inference validation."""
from pathlib import Path
import sys,json,hashlib,collections
from html.parser import HTMLParser
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from train_three_ward_unet_exact_pilot import ConditionalUNet
from run_three_ward_unet_controls_v2 import fingerprint
NAME='three_ward_single_joint_exposure_fit_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1';JR=ROOT/'data/exp/estimation/three_ward_two_parent_fit_v1';JOINT=ROOT/'data/processed/three_ward_two_parent_fit_v1'
def mh(m):return hashlib.sha256(np.ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
def main():
 p=read(OUT/'protocol.json');done=read(REPORT/'completion.json');summary=read(REPORT/'summary.json');ex=read(OUT/'execution_protocol.json');reg=read(REPORT/'registration.json');checks=[]
 def check(n,ok):
  if not bool(ok):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 check('exactly_two_fits_and480_new_updates',done['actual_new_fits']==2 and done['actual_new_optimizer_updates']==480 and p['maximum_new_fits']==2 and p['maximum_new_updates_total']==480)
 check('before_forward_registration_and_source_bound',reg['registered_before_first_neural_execution'] and reg['optimizer_updates_at_registration']==0 and reg['protocol_sha256']==sha(OUT/'protocol.json') and p['runner_sha256']==sha(ROOT/'src/analyses/run_three_ward_single_joint_exposure_v1.py') and reg['registered_SOP_sha256']==sha(REPORT/'registered-AnaSOP.md'))
 check('no_validation_expansion_test_hydraulics',not any(done[k] for k in ['validation_response_access','sealed_opened','sample_expansion','new_hydraulic_runs']))
 check('old_sources_and_frozen_design_preserved',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 design=read(ROOT/'data/exp/research-design/three_ward_single_joint_exposure_plan_v1/protocol.json');check('original12plus4_and_recipe',p['formal_train_ids']==design['formal_train_ids'] and p['formal_validation_ids']==design['formal_validation_ids'] and p['fixed_recipe']==design['fixed_recipe'] and p['fitting_gates']==design['fitting_gates'])
 z=arrays(SOURCE/'prepared_inputs.npz');area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0};init=torch.load(ROOT/p['fixed_recipe']['initialization_path'],map_location='cpu',weights_only=False);jointinit=arrays(JOINT/'initial_predictions.npz');replay=[]
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 for arm,cid in p['arms'].items():
  d=OUT/arm;r=REPORT/arm;record=next(r for r in p['selected_records'] if r['case_id']==cid);label=load_label(cid,'train');y=label['Peak Mean Depth Lower'].astype('f4');mask=label['Exact Label Mask']&eligible;ir=read(r/'initialization_receipt.json');c=read(r/'completion.json');steps=[json.loads(s) for s in (r/'optimizer_steps.jsonl').read_text().splitlines()];zero=torch.load(d/'step_0000.pt',map_location='cpu',weights_only=False)
  check(arm+'_original_exact_mask',mh(mask)==record['original_exact_eligible_mask_sha256'] and not (mask&(label['Interval Label Mask']|label['Quarantine Mask']|label['Online Ever Negative'])).any())
  check(arm+'_common_all_tensor_initialization',all(torch.equal(v,init['state_dict'][k]) for k,v in zero['state_dict'].items()) and ir['initial_optimizer_steps']==0 and ir['shared_initialization_sha256']==p['fixed_recipe']['initialization_sha256'] and ir['parameters']==671513)
  check(arm+'_only_own_response240_real_updates',len(steps)==240 and {s['case_id'] for s in steps}=={cid} and c['actual_optimizer_updates']==c['per_parent_exposures']==240 and c['actual_evaluations']==25 and c['stop_reason']=='fixed_240_update_budget_completed' and ir['response_access_allowlist']==[cid])
  for n,s in enumerate(steps,1):
   ok=s['record_kind']=='training_update' and s['arm']==arm and s['split']=='train' and s['family_id']==record['family_id'] and s['forcing_sha256']==record['forcing_sha256'] and s['mask_sha256']==mh(mask) and s['seed']==p['fixed_recipe']['seed'] and s['epoch']==n and s['parent_exposure']==n
   ok &=max(s['optimizer_before']['steps'],default=0)==n-1 and s['optimizer_after']['steps']==[n] and s['optimizer_after']['learning_rates']==[.0003] and s['gradient_pre_clip']['finite'] and s['gradient_post_clip']['finite'] and s['clipping_required']==(s['gradient_pre_clip']['total_norm']>10) and s['gradient_post_clip']['total_norm']<=10.0001
   ok &=sum(v['update_norm']**2 for v in s['parameter_change'].values())>0 and abs(sum(v['contribution_m2'] for v in s['loss']['strata'].values())-s['loss']['value_m2'])<1e-7
   check(f'{arm}_real_update{n}',ok)
  streak=0;first=None
  for e in p['evaluation_exposures']:
   f=read(r/f'step_{e:04}.json');check(f'{arm}_exposure{e}_frozen_checkpoint_clock',f['step']==f['parent_exposures']==f['optimizer_steps_before_eval']==f['optimizer_steps_after_eval']==e and f['matched_joint_update']==2*e and f['fingerprint_before_eval']==f['fingerprint_after_eval'] and f['checkpoint_sha256']==sha(ROOT/f['checkpoint_path']))
   check(f'{arm}_exposure{e}_original_scopes_masks',len(f['event_metrics'])==6 and all(q['case_id']==cid and q['support']=='own_exact' and q['support_sha256']==p['expected_original_support_sha256'][cid][q['scope']] for q in f['event_metrics']))
   truths=[]
   for q in f['event_metrics']:
    if q['scope'] not in p['gate_scopes']:continue
    tests={}
    for k,(direction,limit) in p['fitting_gates'].items():
     if k=='deep_mae_m' and q['deep_area_km2']==0:continue
     v=q.get(k);tests[k]=v is not None and np.isfinite(v) and (v>=limit if direction=='min' else v<=limit)
    g=next(g for g in f['gates']['rows'] if g['scope']==q['scope']);truths.append(all(tests.values()));check(f'{arm}_exposure{e}_{q["scope"]}_actual_gates',g['checks']==tests and g['pass']==all(tests.values()))
   allpass=all(truths);streak=streak+1 if allpass and e>0 else 0
   if streak>=2 and first is None:first=e
   check(f'{arm}_exposure{e}_actual_success_history',f['gates']['all_parent_ward_gates_pass']==allpass and f['success_streak']==streak and f['first_confirmed_exposure']==first)
  check(arm+'_complete_budget_no_success_early_stop',c['first_two_consecutive_pass_exposure']==first and c['endpoint_consecutive_passes']==streak and c['endpoint_all_gates_pass']==allpass)
  final=torch.load(d/'final.pt',map_location='cpu',weights_only=False);check(arm+'_final_optimizer_clock_rng_finite',final['steps']==240 and final['parent_exposures']==240 and final['stage']=='complete' and all(int(v['step'])==240 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in final['optimizer']['state'].values()) and all(k in final for k in ['torch_rng','numpy_rng','python_rng','mps_rng']))
  pred=arrays(d/'endpoint_predictions.npz');v=pred['Peak Mean Depth'][0];check(arm+'_full_field_order_and_support',list(pred['Case ID'])==[cid] and v.shape==(1012,770) and np.isfinite(v[eligible]).all() and (v[eligible]>=0).all() and np.isnan(v[~eligible]).all())
  si=arrays(d/'initial_predictions.npz')['Peak Mean Depth'][0];ji=jointinit['Peak Mean Depth'][list(jointinit['Case ID']).index(cid)];check(arm+'_same_initial_fullfield_prediction',np.max(abs(si[eligible]-ji[eligible]))<1e-5)
  f=read(r/'endpoint_metrics.json')
  for scope,region in regions.items():
   m=mask&region;a=area[m];t=y[m];pv=v[m];err=pv.astype('f8')-t.astype('f8');w=t>.1;wet=pv>.1;tp=a[w&wet].sum();fp=a[~w&wet].sum();fn=a[w&~wet].sum();row=next(q for q in f['event_metrics'] if q['scope']==scope);mae=np.average(abs(err),weights=a)
   check(arm+'_'+scope+'_independent_endpoint_metrics',abs(row['mae_m']-mae)<1e-10 and abs(row['iou_0.1']-tp/(tp+fp+fn))<1e-10 and abs(row['false_km2_0.1']-fp/1e6)<1e-10 and abs(row['missed_km2_0.1']-fn/1e6)<1e-10 and abs(row['wet_mae_m']-np.average(abs(err[w]),weights=a[w]))<1e-10)
   if scope=='domain':
    loss=0.;ae=abs(err);cell=np.where(ae<=.1,err**2,.2*(ae-.05))
    for mass,g in [(.5,t<=.1),(.4,(t>.1)&(t<=1)),(.1,t>1)]:
     if g.any():loss+=mass*np.average(cell[g],weights=a[g])
    check(arm+'_independent_numpy_stratified_loss',abs(loss-row['fixed_model_loss_m2'])<max(1e-7,abs(loss)*2e-5))
  # Two fixed endpoint forwards only; no optimizer or extra fit is constructed.
  model=ConditionalUNet().to(device);model.load_state_dict(final['model']);model.eval();before=fingerprint(model);idx=list(z['Case ID']).index(cid)
  with torch.inference_mode():pv=model(torch.from_numpy(z['Static Input'][None]).to(device),torch.from_numpy(z['Forcing Input'][idx:idx+1]).to(device))[0,0].cpu().numpy()
  check(arm+'_independent_weight_condition_endpoint_replay',np.max(abs(pv[eligible]-v[eligible]))<1e-5 and before==fingerprint(model));replay.append({'arm':arm,'case_id':cid,'prepared_index':idx,'max_replay_error_m':float(np.max(abs(pv[eligible]-v[eligible]))),'parameters_unchanged':True})
  del model
  if device.type=='mps':torch.mps.empty_cache()
 import csv
 rows=list(csv.DictReader((REPORT/'endpoint_metrics.csv').open()));deltas=list(csv.DictReader((REPORT/'single_minus_joint.csv').open()))
 for d in deltas:
  a=next(r for r in rows if r['case_id']==d['case_id'] and r['scope']==d['scope'] and r['method']=='single');b=next(r for r in rows if r['case_id']==d['case_id'] and r['scope']==d['scope'] and r['method']=='joint_reused');original=next(r for r in read(JR/'step_0480.json')['event_metrics'] if r['case_id']==d['case_id'] and r['scope']==d['scope'])
  check('reused_joint_'+d['case_id']+'_'+d['scope'],float(b['iou_0.1'])==original['iou_0.1'] and b['support_sha256']==a['support_sha256']==original['support_sha256'] and abs(float(d['delta_iou_0.1'])-(float(a['iou_0.1'])-float(b['iou_0.1'])))<1e-10)
 check('unchanged_sources_after_endpoint_replays',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[]
  def handle_starttag(self,t,attrs):self.links += [v for k,v in attrs if k in ['href','src']]
 l=Links();l.feed((REPORT/'review.html').read_text());check('local_report_links',all('://' not in v and ((REPORT/v).exists() or v=='delivery_validation.json') for v in l.links))
 write(REPORT/'endpoint_replay.json',replay);write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'checks':checks,'independent_checks':len(checks),'actual_new_fits':2,'actual_optimizer_updates':480,'additional_verification_forwards':2,'new_optimizer_updates_during_validation':0,'sealed_opened':False});print(json.dumps({'all_checks_passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
