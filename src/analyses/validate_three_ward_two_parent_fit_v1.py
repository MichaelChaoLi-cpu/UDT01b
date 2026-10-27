"""Independent budget, original-support, gate and full-field replay checks."""
from pathlib import Path
import sys,json,hashlib,collections
from html.parser import HTMLParser
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from train_three_ward_unet_exact_pilot import ConditionalUNet
from run_three_ward_unet_controls_v2 import fingerprint
NAME='three_ward_two_parent_fit_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME;SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'
def mh(m):return hashlib.sha256(np.ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
def main():
 p=read(OUT/'protocol.json');ex=read(OUT/'execution_protocol.json');done=read(REPORT/'completion.json');ph=sha(OUT/'execution_protocol.json');checks=[]
 def check(n,v):
  if not bool(v):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 check('one_fit_within_predeclared480_update_cap',done['actual_fit_count']==1 and 0<done['actual_optimizer_updates']<=480 and not done['sample_expansion'] and not done['sealed_opened'])
 check('budget_gates_and_implementation_bound',ex['protocol_sha256']==sha(OUT/'protocol.json') and ex['runner_sha256']==sha(ROOT/'src/analyses/run_three_ward_two_parent_fit_v1.py') and ex['gate_interface_sha256']==sha(ROOT/'src/analyses/three_ward_two_parent_fit_interface_v1.py'))
 reg=read(REPORT/'registration.json');check('registration_before_first_neural_execution',reg['optimizer_updates_at_registration']==0 and reg['registered_before_neural_execution'] and ex['registered_SOP_sha256']==sha(REPORT/'registered-AnaSOP.md') and (REPORT/'registration.json').stat().st_mtime_ns<(OUT/'execution_protocol.json').stat().st_mtime_ns)
 old=read(ROOT/'data/processed/three_ward_unet_control_plan_v2/protocol.json');check('formal12plus4_membership_unchanged',p['formal_train_case_ids_unchanged']==old['train_case_ids'] and p['formal_validation_case_ids_unchanged']==old['validation_case_ids']);check('only_two_original_training_responses_used',p['label_access_allowlist']=={'train-01':'train','train-04':'train'} and set(p['selected_case_ids'])<=set(old['train_case_ids']) and not ex['validation_response_access'] and not ex['sealed_access'])
 check('all_prior_sources_and_quality_packages_unchanged',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 z=arrays(SOURCE/'prepared_inputs.npz');ids=p['selected_case_ids'];records=p['selected_records'];area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];labels=[load_label(cid,'train') for cid in ids];y=np.array([l['Peak Mean Depth Lower'].astype('f4') for l in labels]);masks=np.array([l['Exact Label Mask']&eligible for l in labels]);regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0}
 for i,r in enumerate(records):
  check(r['case_id']+'_original_full_mask',mh(masks[i])==r['original_exact_eligible_mask_sha256']);check(r['case_id']+'_no_interval_quarantine_negative_labels',not np.any(masks[i]&(labels[i]['Interval Label Mask']|labels[i]['Quarantine Mask']|labels[i]['Online Ever Negative'])))
 steps=[json.loads(l) for l in (REPORT/'optimizer_steps.jsonl').read_text().splitlines()];orders=read(OUT/'cycle_order.json');count=done['actual_optimizer_updates'];check('actual_step_ledger_matches_cap_and_cycle_plan',len(steps)==count and [r['case_id'] for r in steps]==[cid for r in orders for cid in r['case_order']][:count]);exposures=dict(collections.Counter(r['case_id'] for r in steps));check('actual_per_parent_exposures',exposures==done['per_parent_exposures'] and set(exposures)==set(ids))
 for cycle in range(1,done['actual_cycles']+1):
  rr=steps[(cycle-1)*2:cycle*2];ok=len(rr)==2
  for j,s in enumerate(rr):
   r=records[ids.index(s['case_id'])];n=(cycle-1)*2+j+1;ok &= s['record_kind']=='training_update' and s['split']=='train' and s['seed']==p['seed'] and s['family_id']==r['family_id'] and s['forcing_sha256']==r['forcing_sha256'] and s['mask_sha256']==r['original_exact_eligible_mask_sha256'] and s['protocol_sha256']==ph and s['epoch']==cycle
   ok &= max(s['optimizer_before']['steps'],default=0)==n-1 and s['optimizer_after']['steps']==[n] and s['optimizer_after']['learning_rates']==[.0003] and s['gradient_pre_clip']['finite'] and s['gradient_post_clip']['finite'] and s['gradient_post_clip']['total_norm']<=10.0001 and s['clipping_required']==(s['gradient_pre_clip']['total_norm']>10)
   ok &= s['loss']['kind']=='stratified_huber' and abs(sum(q['contribution_m2'] for q in s['loss']['strata'].values())-s['loss']['value_m2'])<max(1e-7,abs(s['loss']['value_m2'])*1e-5) and sum(q['update_norm']**2 for q in s['parameter_change'].values())>0
  check(f'cycle{cycle}_two_real_original_parent_updates',ok)
 frames=[];streak=0;first_confirmed=None;keys=[]
 for step in [s for s in p['evaluation_updates'] if s<=count]:
  f=read(REPORT/f'step_{step:04}.json');frames.append(f);check(f'step{step}_actual_checkpoint_clock_and_fingerprint',f['optimizer_steps_before_eval']==f['optimizer_steps_after_eval']==step and f['fingerprint_before_eval']==f['fingerprint_after_eval'] and sha(ROOT/f['checkpoint_path'])==f['checkpoint_sha256']);ev=f['event_metrics'];check(f'step{step}_complete12_original_scope_rows',len(ev)==12 and {(r['case_id'],r['scope']) for r in ev}=={(cid,s) for cid in ids for s in regions} and all(r['split']=='train' and r['support']=='own_exact' and r['support_sha256']==p['expected_support_sha256'][r['case_id']][r['scope']] for r in ev))
  gate_rows=[r for r in ev if r['scope'] in p['gate_scopes']];independent=[]
  for r in gate_rows:
   tests={}
   for k,(direction,limit) in p['success_gates'].items():
    if k=='deep_mae_m' and r['deep_area_km2']==0:continue
    v=r.get(k);tests[k]=v is not None and np.isfinite(v) and (v>=limit if direction=='min' else v<=limit)
   independent.append((r['case_id'],r['scope'],all(tests.values()),tests))
  check(f'step{step}_gates_independently_recomputed',all(any(x[0]==r['case_id'] and x[1]==r['scope'] and bool(x[2])==r['pass'] and x[3]==r['checks'] for x in independent) for r in f['gates']['rows']))
  passed=all(t[2] for t in independent);streak=streak+1 if step>0 and passed else 0
  if streak>=2 and first_confirmed is None:first_confirmed=step
  check(f'step{step}_consecutive_confirmation',f['success_streak']==streak and f['gates']['all_parent_ward_gates_pass']==passed)
  key=(-round(min(r['iou_0.1'] for r in gate_rows),6),round(max(r['wet_mae_m'] for r in gate_rows),6),round(max(r['mae_m'] for r in gate_rows),6),step);keys.append(key);check(f'step{step}_diagnostic_key_independent',tuple(f['diagnostic_key'])==key)
 check('no_early_stop_or_extension_outside_registered_rules',(first_confirmed==count and done['fit_demonstrated']) if first_confirmed is not None else (count==480 and not done['fit_demonstrated'] and done['stop_reason']=='budget_exhausted_without_demonstrated_fit'))
 check('diagnostic_best_not_substituted_for_endpoint',done['best_diagnostic_step']==min(keys)[-1] and done['final_step']==count)
 init=torch.load(ROOT/p['initialization_path'],map_location='cpu',weights_only=False);initial=torch.load(OUT/'step_0000.pt',map_location='cpu',weights_only=False);check('all_initial_tensors_equal_original_untrained_state',not init['trained'] and init['optimizer_steps']==0 and all(torch.equal(v,init['state_dict'][k]) for k,v in initial['state_dict'].items()))
 final=torch.load(OUT/'final.pt',map_location='cpu',weights_only=False);check('final_optimizer_RNG_and_exact_clock_retained',final['steps']==count and final['stage']=='complete' and all(int(v['step'])==count and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in final['optimizer']['state'].values()) and len(final['optimizer']['state'])==96 and all(k in final for k in ['torch_rng','numpy_rng','python_rng','mps_rng']))
 points=[('initial','initial_predictions.npz',frames[0]),('endpoint','endpoint_predictions.npz',frames[-1]),('diagnostic_best','best_diagnostic_predictions.npz',read(REPORT/'best_diagnostic_metrics.json'))]
 for point,file,f in points:
  zz=arrays(OUT/file);pred=zz['Peak Mean Depth'];check(point+'_exact2field_order_and_coverage',list(zz['Case ID'])==ids and pred.shape==(2,1012,770) and np.isfinite(pred[:,eligible]).all() and (pred[:,eligible]>=0).all() and np.isnan(pred[:,~eligible]).all())
  for i,cid in enumerate(ids):
   for scope,region in regions.items():
    m=masks[i]&region;a=area[m];t=y[i][m];v=pred[i][m];e=v.astype('f8')-t.astype('f8');w=t>.1;flood=v>.1;tp=a[w&flood].sum();fp=a[~w&flood].sum();fn=a[w&~flood].sum();mae=np.sum(a*np.abs(e))/a.sum();q=next(r for r in f['event_metrics'] if r['case_id']==cid and r['scope']==scope)
    check(point+'_'+cid+'_'+scope+'_original_mask_prediction_metrics',abs(mae-q['mae_m'])<1e-10 and abs(fp/1e6-q['false_km2_0.1'])<1e-10 and abs(fn/1e6-q['missed_km2_0.1'])<1e-10 and abs(tp/(tp+fp+fn)-q['iou_0.1'])<1e-10)
    if scope=='domain':
     ss=[t<=.1,(t>.1)&(t<=1),t>1];ww=[.5,.4,.1];active=sum(w0 for w0,m0 in zip(ww,ss) if m0.any());ae=np.abs(e);cell=np.where(ae<=.1,e**2,.2*(ae-.05));loss=sum(w0/active*np.average(cell[m0],weights=a[m0]) for w0,m0 in zip(ww,ss) if m0.any());check(point+'_'+cid+'_independent_numpy_loss',abs(loss-q['fixed_model_loss_m2'])<max(1e-7,abs(loss)*2e-5))
 # Replay both endpoint fields from source forcing indices and retained weights;
 # no optimizer is constructed and no response beyond selected2 is loaded.
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=ConditionalUNet().to(device);model.load_state_dict(final['model']);model.eval();before=fingerprint(model);x=torch.from_numpy(z['Static Input'][None]).to(device);stored=arrays(OUT/'endpoint_predictions.npz')['Peak Mean Depth'];manifest=[]
 for i,cid in enumerate(ids):
  j=list(z['Case ID']).index(cid);c=torch.from_numpy(z['Forcing Input'][j:j+1]).to(device)
  with torch.inference_mode():v=model(x,c)[0,0].cpu().numpy()
  check(cid+'_independent_endpoint_weight_and_forcing_replay',float(np.max(np.abs(v[eligible]-stored[i,eligible])))<1e-5 and before==fingerprint(model));manifest.append({'case_id':cid,'original_prepared_input_index':j,'forcing_tensor_sha256':hashlib.sha256(z['Forcing Input'][j].tobytes()).hexdigest(),'original_exact_mask_sha256':mh(masks[i])})
 write(REPORT/'input_binding_replay.json',{'case_bindings':manifest,'zero_optimizer_updates':True,'response_loading_allowlist':ids})
 check('all_prior_frozen_sources_unchanged_after_checks',all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[]
  def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'])
 parser=Links();parser.feed((REPORT/'review.html').read_text());check('report_links_local_and_present',all('://' not in link and ((REPORT/link).exists() or link=='delivery_validation.json') for link in parser.links))
 write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'checks':checks,'preflight_gate_checks':len(read(REPORT/'preflight.json')['checks']),'independent_checks':len(checks),'actual_fit_count':1,'actual_updates':count,'formal12plus4_preserved':True,'quality_masks_preserved':True,'validation_used_for_selection_or_scoring':False,'sealed_opened':False});print(json.dumps({'all_checks_passed':True,'checks':len(checks),'new_fit_count':1,'updates':count}))
if __name__=='__main__':main()
