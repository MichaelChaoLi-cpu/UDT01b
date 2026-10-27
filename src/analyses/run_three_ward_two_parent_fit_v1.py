"""One pre-registered full-domain fit on two existing train parents; no validation."""
from pathlib import Path
import sys,time,json,os,random,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from run_three_ward_unet_controls_v2 import fingerprint,save_state,atomic_json,csv_write
from train_three_ward_unet_exact_pilot import ConditionalUNet,synchronize
from three_ward_surrogate_protocol_v2 import exact_loss,evaluate_depth,macro_events,OptimizationObserver
from three_ward_two_parent_fit_interface_v1 import evaluate_fit_gates,diagnostic_key,update_success_streak,GATES
from prepare_three_ward_two_parent_fit_v1 import masksha
NAME='three_ward_two_parent_fit_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME;SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'
def run():
 p=read(OUT/'protocol.json');assert p['fit_count']==1 and p['maximum_optimizer_updates']==480 and p['execution_authorized']
 if (OUT/'execution_protocol.json').exists():raise FileExistsError('Existing execution requires an audited continuation; no automatic restart')
 assert read(REPORT/'registration.json')['optimizer_updates_at_registration']==0
 assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items());assert p['interface_sha256']==sha(ROOT/'src/analyses/three_ward_two_parent_fit_interface_v1.py');assert json.loads(json.dumps(GATES))==p['success_gates'];assert read(REPORT/'preflight.json')['all_checks_passed']
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if device.type=='mps':torch.mps.manual_seed(seed)
 ex={'decision':p['decision'],'protocol_sha256':sha(OUT/'protocol.json'),'runner_sha256':sha(Path(__file__)),'gate_interface_sha256':p['interface_sha256'],'original_model_sha256':sha(ROOT/'src/analyses/train_three_ward_unet_exact_pilot.py'),'source_preparation_sha256':p['preparation_sha256'],'cycle_order_sha256':sha(OUT/'cycle_order.json'),'registered_SOP_sha256':sha(REPORT/'registered-AnaSOP.md'),'registration_sha256':sha(REPORT/'registration.json'),'device':str(device),'torch_version':torch.__version__,'seed':seed,'maximum_new_fits':1,'maximum_new_updates':480,'selected_original_training_parents':p['selected_case_ids'],'formal_split_unchanged':True,'validation_response_access':False,'sealed_access':False,'sample_expansion':False,'precision':'float32 model and native threshold classification, float64 metric reduction'}
 write(OUT/'execution_protocol.json',ex);ph=sha(OUT/'execution_protocol.json');z=arrays(SOURCE/'prepared_inputs.npz');indices=[list(z['Case ID']).index(cid) for cid in p['selected_case_ids']];records=p['selected_records'];by={r['case_id']:i for i,r in enumerate(records)};labels=[load_label(r['case_id'],'train') for r in records];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];area=z['Ground Area'];ys=np.array([d['Peak Mean Depth Lower'].astype('f4') for d in labels]);masks=np.array([d['Exact Label Mask']&eligible for d in labels]);regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0}
 for i,r in enumerate(records):assert masksha(masks[i])==r['original_exact_eligible_mask_sha256']
 x=torch.from_numpy(z['Static Input'][None]).to(device);cond=torch.from_numpy(z['Forcing Input'][indices]).to(device);target=torch.from_numpy(ys).to(device);ground=torch.from_numpy(area.astype('f4')).to(device);tmasks=torch.from_numpy(masks).to(device);init=torch.load(ROOT/p['initialization_path'],map_location='cpu',weights_only=False);assert not init['trained'] and init['optimizer_steps']==0
 model=ConditionalUNet().to(device);model.load_state_dict(init['state_dict'],strict=True);initialfp=fingerprint(model);assert initialfp==read(ROOT/'data/processed/three_ward_spatial_fp_controls_v1/protocol.json')['initial_parameter_fingerprint'];optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.00001);orders=read(OUT/'cycle_order.json');steps=0;cycle=0;cursor=0;frames=[];history=[];streak=0;bestkey=None;beststep=None;training_s=0.;evaluation_s=0.;io_s=0.;stage='initializing';started=time.perf_counter()
 write(REPORT/'initialization_receipt.json',{'parameter_fingerprint':initialfp,'shared_initialization_sha256':p['initialization_sha256'],'parameters':sum(v.numel() for v in model.parameters()),'initial_optimizer_steps':0,'seed':seed,'normalization_sha256':p['normalization_sha256'],'formal_split_unchanged':True})
 def snapshot():return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'steps':steps,'cycle':cycle,'cursor':cursor,'stage':stage,'success_streak':streak,'best_key':bestkey,'best_step':beststep,'protocol_sha256':ph,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None,'history':history}
 def store(path,pred):np.savez_compressed(path,**{'Peak Mean Depth':pred,'Case ID':np.array(p['selected_case_ids']),'Eligible Ground Mask':eligible})
 def evaluate():
  nonlocal stage,streak,bestkey,beststep,evaluation_s,io_s
  stage='fixed_evaluation';tic=time.perf_counter();assert sha(OUT/'protocol.json')==ex['protocol_sha256'] and sha(ROOT/'src/analyses/three_ward_two_parent_fit_interface_v1.py')==p['interface_sha256']
  for r in records:assert sha(ROOT/r['label_path'])==r['label_sha256']
  model.eval();before=fingerprint(model);preds=[];losses=[]
  with torch.inference_mode():
   for i in range(2):
    pred=model(x,cond[i:i+1])[0,0];v,payload=exact_loss(pred,target[i],ground,tmasks[i],'stratified_huber');a=pred.cpu().numpy().copy();a[~eligible]=np.nan;preds.append(a);losses.append(payload)
  preds=np.array(preds);events=[]
  for i,r in enumerate(records):
   for scope,region in regions.items():
    m=masks[i]&region;assert masksha(m)==p['expected_support_sha256'][r['case_id']][scope];q=evaluate_depth(preds[i],ys[i],area,m);q.update({'case_id':r['case_id'],'parent_id':r['parent_id'],'split':'train','scope':scope,'support':'own_exact'})
    if scope=='domain':q.update({'fixed_model_loss_m2':losses[i]['value_m2'],'fixed_loss_strata':losses[i]['strata']})
    events.append(q)
  assert before==fingerprint(model);gate=evaluate_fit_gates(events,p['selected_case_ids'],p['expected_support_sha256']);streak=update_success_streak(streak,gate,steps);key=diagnostic_key(events,steps);evaluation_s+=time.perf_counter()-tic;checkpoint=OUT/f'step_{steps:04}.pt';tic=time.perf_counter();save_state(checkpoint,{'state_dict':model.state_dict(),'step':steps,'cycle':cycle,'protocol_sha256':ph,'shared_initialization_sha256':p['initialization_sha256'],'parameter_fingerprint':before});io_s+=time.perf_counter()-tic
  frame={'schema':'two-parent-fixed-evaluation-v1','step':steps,'cycle':cycle,'record_kind':'initial_fixed_evaluation' if steps==0 else 'known_training_response_diagnostic','protocol_sha256':ph,'optimizer_steps_before_eval':steps,'optimizer_steps_after_eval':steps,'fingerprint_before_eval':before,'fingerprint_after_eval':fingerprint(model),'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),'event_metrics':events,'parent_macro_metrics':[{'scope':scope,**macro_events([r for r in events if r['scope']==scope])} for scope in regions],'gates':gate,'success_streak':streak,'diagnostic_key':key,'validation_response_access':False,'cumulative_training_seconds':training_s,'cumulative_evaluation_seconds':evaluation_s,'cumulative_checkpoint_IO_seconds':io_s}
  frames.append(frame);write(REPORT/f'step_{steps:04}.json',frame);tic=time.perf_counter();store(OUT/'endpoint_predictions.npz',preds)
  if steps==0:store(OUT/'initial_predictions.npz',preds)
  if bestkey is None or key<bestkey:
   bestkey=key;beststep=steps;save_state(OUT/'best_diagnostic.pt',torch.load(checkpoint,map_location='cpu',weights_only=False));store(OUT/'best_diagnostic_predictions.npz',preds);write(REPORT/'best_diagnostic_metrics.json',frame)
  io_s+=time.perf_counter()-tic;core=[r for r in events if r['scope'] in p['gate_scopes']];row={'step':steps,'cycle':cycle,'fixed_objective_m2':float(np.mean([r['value_m2'] for r in losses])),'worst_parent_ward_IoU':min(r['iou_0.1'] for r in core),'worst_parent_ward_recall':min(r['recall_0.1'] for r in core),'worst_parent_ward_precision':min((r['precision_0.1'] for r in core if r['precision_0.1'] is not None),default=None),'worst_parent_ward_wet_MAE_m':max(r['wet_mae_m'] for r in core),'worst_parent_ward_deep_MAE_m':max((r['deep_mae_m'] for r in core if r['deep_mae_m'] is not None),default=None),'all_gates_pass':gate['all_parent_ward_gates_pass'],'consecutive_passes':streak,'best_diagnostic_step_so_far':beststep,'training_seconds':training_s,'evaluation_seconds':evaluation_s,'checkpoint_IO_seconds':io_s}
  history.append(row);csv_write(REPORT/'history.csv',history);stage='evaluation_complete';save_state(OUT/'latest.pt',snapshot());print(json.dumps(row),flush=True);return frame
 try:
  evaluate()
  for order in orders:
   cycle=order['cycle'];model.train();stage='training';observer=OptimizationObserver(REPORT/'optimizer_steps.jsonl',ph)
   for cursor,cid in enumerate(order['case_order'],1):
    i=by[cid];r=records[i];tic=time.perf_counter();optimizer.zero_grad(set_to_none=True);observer.begin(model,optimizer,{'arm':'two_parent_D_recipe_diagnostic','epoch':cycle,'cycle':cycle,'case_id':cid,'parent_id':r['parent_id'],'family_id':r['family_id'],'forcing_sha256':r['forcing_sha256'],'split':'train','seed':seed,'mask_sha256':r['original_exact_eligible_mask_sha256'],'loss_kind':'stratified_huber','record_kind':'training_update','purpose':'bounded_known-training-response_diagnostic'})
    pred=model(x,cond[i:i+1])[0,0];loss,payload=exact_loss(pred,target[i],ground,tmasks[i],'stratified_huber');assert torch.isfinite(loss);loss.backward();observer.after_backward(model,payload);torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);observer.after_clip(model);optimizer.step();synchronize(device);observer.finish(model,optimizer);steps+=1;training_s+=time.perf_counter()-tic
    assert steps<=480;atomic_json(REPORT/'progress.json',{'status':'training','step':steps,'maximum_steps':480,'cycle':cycle,'parent_exposures_each_at_completed_cycle':cycle if cursor==2 else cycle-1,'current_case_id':cid,'success_streak':streak,'elapsed_seconds':time.perf_counter()-started,'process_id':os.getpid()})
   if steps%20==0:
    evaluate()
    if streak>=2:break
  reason='confirmed_two_consecutive_fitting_passes' if streak>=2 else 'budget_exhausted_without_demonstrated_fit';assert streak>=2 or steps==480;assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items());stage='complete';save_state(OUT/'final.pt',snapshot());write(REPORT/'endpoint_metrics.json',frames[-1]);counts={cid:sum(cid in q['case_order'] for q in orders[:cycle]) for cid in p['selected_case_ids']};result={'status':'complete','decision':p['decision'],'stop_reason':reason,'fit_demonstrated':streak>=2,'actual_fit_count':1,'actual_optimizer_updates':steps,'actual_cycles':cycle,'per_parent_exposures':counts,'evaluations':len(frames),'best_diagnostic_step':beststep,'final_step':steps,'final_consecutive_passes':streak,'all_endpoint_gates_pass':frames[-1]['gates']['all_parent_ward_gates_pass'],'training_seconds':training_s,'evaluation_seconds':evaluation_s,'checkpoint_IO_seconds':io_s,'total_process_seconds':time.perf_counter()-started,'original_formal12plus4_split_preserved':True,'original_quality_masks_preserved':True,'validation_response_access':False,'sealed_opened':False,'sample_expansion':False,'new_hydraulic_runs':0,'no_automatic_extension':True,'execution_protocol_sha256':ph};write(REPORT/'completion.json',result);atomic_json(REPORT/'progress.json',result);print(json.dumps(result),flush=True)
 except BaseException as err:
  save_state(OUT/'held_state.pt',snapshot());failed={'status':'held','stage':stage,'completed_observed_steps':steps,'cycle':cycle,'cursor':cursor,'exception':repr(err),'traceback':traceback.format_exc(),'no_automatic_restart':True};write(REPORT/'failure.json',failed);atomic_json(REPORT/'progress.json',failed);raise
if __name__=='__main__':run()
