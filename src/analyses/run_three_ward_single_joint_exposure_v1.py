"""Exactly two authorized single-parent fits, 240 updates each; reuse joint results.
No hydraulic simulation, validation responses, sealed tests, new normalization or best search.
"""
from pathlib import Path
import sys,json,time,random,os,traceback,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from train_three_ward_unet_exact_pilot import ConditionalUNet,synchronize
from run_three_ward_unet_controls_v2 import fingerprint,save_state,atomic_json,csv_write
from three_ward_surrogate_protocol_v2 import exact_loss,evaluate_depth,OptimizationObserver
from three_ward_two_parent_fit_interface_v1 import evaluate_fit_gates,update_success_streak,GATES
NAME='three_ward_single_joint_exposure_fit_v1'
OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
DESIGN=ROOT/'data/exp/research-design/three_ward_single_joint_exposure_plan_v1'
SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'

def mh(m):return hashlib.sha256(np.ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
def prepare():
 if (OUT/'protocol.json').exists():raise FileExistsError('Preserve prepared execution; no automatic restart')
 p=read(DESIGN/'protocol.json');frozen=read(DESIGN/'freeze.json')
 assert p['status']=='design_only' and not p['execution_authorized']
 assert all(sha(ROOT/k)==h for k,h in frozen['files_sha256'].items())
 protected={**p['source_sha256'],**frozen['files_sha256'],str((DESIGN/'freeze.json').relative_to(ROOT)):sha(DESIGN/'freeze.json')}
 for rel in ['src/analyses/train_three_ward_unet_exact_pilot.py','src/analyses/three_ward_surrogate_protocol_v2.py','src/analyses/three_ward_two_parent_fit_interface_v1.py','src/preprocessing/prepare_three_ward_unet_exact_pilot.py','src/analyses/run_three_ward_unet_controls_v2.py']:
  protected[rel]=sha(ROOT/rel)
 assert all(sha(ROOT/k)==h for k,h in protected.items())
 assert json.loads(json.dumps(GATES))==p['fitting_gates']
 OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
 protocol={'decision':'MILI-D-20261016-012','execution_authorized':True,'design_sha256':sha(DESIGN/'protocol.json'),'maximum_new_fits':2,'maximum_new_updates_total':480,'maximum_updates_per_arm':240,'arms':{'S01':'train-01','S04':'train-04'},'evaluation_exposures':p['exposure_axis'],'fixed_recipe':p['fixed_recipe'],'formal_train_ids':p['formal_train_ids'],'formal_validation_ids':p['formal_validation_ids'],'selected_records':p['selected_original_records'],'expected_original_support_sha256':p['expected_original_support_sha256'],'fitting_gates':p['fitting_gates'],'gate_scopes':p['gate_scopes'],'stopping':p['stopping'],'interpretation_limits':p['interpretation_limits'],'source_sha256':protected,'runner_sha256':sha(Path(__file__)),'new_hydraulic_runs':0,'sealed_opened':False,'validation_response_access':False,'sample_expansion':False,'best_checkpoint_search':False}
 write(OUT/'protocol.json',protocol)
 write(REPORT/'registration.json',{'registered_before_first_neural_execution':True,'optimizer_updates_at_registration':0,'decision':protocol['decision'],'protocol_sha256':sha(OUT/'protocol.json'),'design_sha256':protocol['design_sha256'],'future_arms':['S01','S04'],'registered_SOP_sha256':sha(REPORT/'registered-AnaSOP.md')})
 print('REGISTERED two fits x240 updates, no model forward yet',flush=True)

def run():
 p=read(OUT/'protocol.json');reg=read(REPORT/'registration.json');assert p['execution_authorized'] and p['runner_sha256']==sha(Path(__file__))
 assert reg['registered_before_first_neural_execution'] and reg['protocol_sha256']==sha(OUT/'protocol.json')
 if (OUT/'execution_protocol.json').exists():raise FileExistsError('Preserve execution; no automatic retry or retraining')
 assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items())
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');recipe=p['fixed_recipe'];seed=recipe['seed']
 ex={'protocol_sha256':sha(OUT/'protocol.json'),'runner_sha256':sha(Path(__file__)),'registered_SOP_sha256':reg['registered_SOP_sha256'],'maximum_fits':2,'maximum_new_updates':480,'device':str(device),'torch_version':torch.__version__,'decision':p['decision'],'validation_response_access':False,'sealed_opened':False};write(OUT/'execution_protocol.json',ex);ph=sha(OUT/'execution_protocol.json')
 z=arrays(SOURCE/'prepared_inputs.npz');eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];area=z['Ground Area'];xt=torch.from_numpy(z['Static Input'][None]).to(device);at=torch.from_numpy(area.astype('f4')).to(device)
 regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0};completions=[]
 for arm,cid in p['arms'].items():
  dest=OUT/arm;rep=REPORT/arm;dest.mkdir();rep.mkdir();record=next(r for r in p['selected_records'] if r['case_id']==cid);label=load_label(cid,'train');mask=label['Exact Label Mask']&eligible;y=label['Peak Mean Depth Lower'].astype('f4');assert mh(mask)==record['original_exact_eligible_mask_sha256'];j=list(z['Case ID']).index(cid);ct=torch.from_numpy(z['Forcing Input'][j:j+1]).to(device);yt=torch.from_numpy(y).to(device);mt=torch.from_numpy(mask).to(device)
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  if device.type=='mps':torch.mps.manual_seed(seed)
  init=torch.load(ROOT/recipe['initialization_path'],map_location='cpu',weights_only=False);assert not init['trained'] and init['optimizer_steps']==0
  model=ConditionalUNet().to(device);model.load_state_dict(init['state_dict'],strict=True);assert all(torch.equal(v.detach().cpu(),init['state_dict'][k]) for k,v in model.state_dict().items());optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);assert not optimizer.state
  write(rep/'initialization_receipt.json',{'arm':arm,'case_id':cid,'parameter_fingerprint':fingerprint(model),'shared_initialization_sha256':sha(ROOT/recipe['initialization_path']),'initial_optimizer_steps':0,'all_parameters_match_shared':True,'parameters':sum(t.numel() for t in model.parameters()),'seed':seed,'normalization_sha256':sha(SOURCE/'normalization.json'),'input_index':j,'mask_sha256':mh(mask),'response_access_allowlist':[cid]})
  step=0;stage='initializing';history=[];streak=0;first_confirmed=None;training_s=0.;eval_s=0.;io_s=0.;started=time.perf_counter();frame=None
  def snapshot():return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'steps':step,'parent_exposures':step,'stage':stage,'protocol_sha256':ph,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None,'history':history,'success_streak':streak,'first_confirmed_exposure':first_confirmed}
  def evaluate():
   nonlocal eval_s,io_s,stage,streak,first_confirmed,frame
   stage='fixed_evaluation';assert sha(OUT/'protocol.json')==ex['protocol_sha256'] and sha(ROOT/record['label_path'])==record['label_sha256'];tic=time.perf_counter();model.eval();before=fingerprint(model)
   with torch.inference_mode():
    pred=model(xt,ct)[0,0];loss,payload=exact_loss(pred,yt,at,mt,'stratified_huber');field=pred.cpu().numpy().copy()
   field[~eligible]=np.nan;events=[]
   for scope,region in regions.items():
    m=mask&region;assert mh(m)==p['expected_original_support_sha256'][cid][scope];row={'case_id':cid,'parent_id':record['parent_id'],'split':'train','scope':scope,'support':'own_exact',**evaluate_depth(field,y,area,m)}
    if scope=='domain':row.update({'fixed_model_loss_m2':payload['value_m2'],'fixed_loss_strata':payload['strata']})
    events.append(row)
   assert before==fingerprint(model);gate=evaluate_fit_gates(events,[cid],p['expected_original_support_sha256']);streak=update_success_streak(streak,gate,step)
   if streak>=2 and first_confirmed is None:first_confirmed=step
   eval_s+=time.perf_counter()-tic;tic=time.perf_counter();checkpoint=dest/f'step_{step:04}.pt';save_state(checkpoint,{'state_dict':model.state_dict(),'step':step,'parent_exposures':step,'case_id':cid,'arm':arm,'protocol_sha256':ph,'shared_initialization_sha256':recipe['initialization_sha256'],'parameter_fingerprint':before});io_s+=time.perf_counter()-tic
   frame={'schema':'single-parent-exposure-matched-evaluation-v1','arm':arm,'case_id':cid,'step':step,'parent_exposures':step,'matched_joint_update':step*2,'protocol_sha256':ph,'optimizer_steps_before_eval':step,'optimizer_steps_after_eval':step,'fingerprint_before_eval':before,'fingerprint_after_eval':fingerprint(model),'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),'event_metrics':events,'gates':gate,'success_streak':streak,'first_confirmed_exposure':first_confirmed,'validation_response_access':False,'training_seconds':training_s,'evaluation_seconds':eval_s,'checkpoint_IO_seconds':io_s};write(rep/f'step_{step:04}.json',frame)
   if step in [0,240]:np.savez_compressed(dest/('initial_predictions.npz' if step==0 else 'endpoint_predictions.npz'),**{'Peak Mean Depth':field[None],'Case ID':np.array([cid]),'Eligible Ground Mask':eligible})
   row=next(r for r in events if r['scope']=='three_wards');history.append({'exposure':step,'joint_update':step*2,'fixed_objective_m2':payload['value_m2'],'IoU':row['iou_0.1'],'recall':row['recall_0.1'],'precision':row['precision_0.1'],'wet_MAE_m':row['wet_mae_m'],'FP_km2':row['false_km2_0.1'],'FN_km2':row['missed_km2_0.1'],'gates_pass':gate['all_parent_ward_gates_pass'],'success_streak':streak});csv_write(rep/'history.csv',history);stage='evaluation_complete';save_state(dest/'latest.pt',snapshot());print(json.dumps({'arm':arm,**history[-1]}),flush=True)
  try:
   evaluate();observer=OptimizationObserver(rep/'optimizer_steps.jsonl',ph,clip_limit=recipe['gradient_clip'])
   for requested in range(1,241):
    stage='training';model.train();tic=time.perf_counter();optimizer.zero_grad(set_to_none=True);observer.begin(model,optimizer,{'arm':arm,'epoch':requested,'parent_exposure':requested,'case_id':cid,'parent_id':record['parent_id'],'family_id':record['family_id'],'forcing_sha256':record['forcing_sha256'],'split':'train','seed':seed,'mask_sha256':mh(mask),'loss_kind':'stratified_huber','record_kind':'training_update','purpose':'authorized_single_parent_known_field_fitting'})
    pred=model(xt,ct)[0,0];loss,payload=exact_loss(pred,yt,at,mt,'stratified_huber');assert torch.isfinite(loss);loss.backward();observer.after_backward(model,payload);torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'],error_if_nonfinite=True);observer.after_clip(model);optimizer.step();synchronize(device)
    assert all(bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    observer.finish(model,optimizer);step+=1;training_s+=time.perf_counter()-tic;assert step==requested and step<=240
    atomic_json(REPORT/'progress.json',{'status':'training','arm':arm,'case_id':cid,'step':step,'maximum_per_arm':240,'completed_arms':[c['arm'] for c in completions],'elapsed_arm_seconds':time.perf_counter()-started,'process_id':os.getpid()})
    if step in p['evaluation_exposures']:evaluate()
   assert step==240 and len(history)==25;stage='complete';save_state(dest/'final.pt',snapshot());write(rep/'endpoint_metrics.json',frame);result={'status':'complete','arm':arm,'case_id':cid,'actual_new_fits':1,'actual_optimizer_updates':step,'per_parent_exposures':step,'stop_reason':'fixed_240_update_budget_completed','first_two_consecutive_pass_exposure':first_confirmed,'endpoint_all_gates_pass':frame['gates']['all_parent_ward_gates_pass'],'endpoint_consecutive_passes':streak,'actual_evaluations':len(history),'training_seconds':training_s,'evaluation_seconds':eval_s,'checkpoint_IO_seconds':io_s,'total_process_seconds':time.perf_counter()-started,'validation_response_access':False,'sealed_opened':False,'sample_expansion':False};write(rep/'completion.json',result);completions.append(result);print(json.dumps(result),flush=True)
  except BaseException as err:
   save_state(dest/'held_state.pt',snapshot());write(rep/'failure.json',{'status':'held','arm':arm,'actual_steps':step,'stage':stage,'exception':repr(err),'traceback':traceback.format_exc(),'no_automatic_restart':True});raise
  del model,optimizer,yt,mt,ct,label
  if device.type=='mps':torch.mps.empty_cache()
 assert len(completions)==2 and sum(c['actual_optimizer_updates'] for c in completions)==480
 assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items())
 result={'status':'complete','decision':p['decision'],'actual_new_fits':2,'actual_new_optimizer_updates':480,'arms':completions,'joint_reference_reused':True,'new_hydraulic_runs':0,'formal12plus4_unchanged':True,'quality_masks_unchanged':True,'normalization_unchanged':True,'validation_response_access':False,'sealed_opened':False,'sample_expansion':False,'source_integrity_after_fit':True};write(REPORT/'completion.json',result);atomic_json(REPORT/'progress.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);args=parser.parse_args();prepare() if args.command=='prepare' else run()
