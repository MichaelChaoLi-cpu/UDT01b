"""Execute only three authorized 480-update fits; reuse immutable D reference."""
from pathlib import Path
import sys,os,json,time,random,traceback,resource
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from run_three_ward_unet_controls_v2 import fingerprint,save_state,atomic_json,csv_write
from train_three_ward_unet_exact_pilot import synchronize
from three_ward_surrogate_protocol_v2 import OptimizationObserver,summarize_step_records,evaluate_depth,macro_events,REQUIRED_SCOPES
from three_ward_spatial_fp_interfaces_v1 import SpatialConditionalUNet,spatial_fp_loss,choose_checkpoints
NAME='three_ward_spatial_fp_controls_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1';OLDPLAN=ROOT/'data/processed/three_ward_unet_control_plan_v2';OLD=ROOT/'data/processed/three_ward_unet_controls_v2';OLDREPORT=ROOT/'data/exp/estimation/three_ward_unet_controls_v2'
def execute():
 p=read(OUT/'protocol.json');assert p['execution_authorized'] and p['maximum_new_fits']==3 and p['maximum_new_updates']==1440
 if (REPORT/'completion.json').exists():raise FileExistsError('Preserve completed fits')
 for aid in ['S1F0','S0F1','S1F1']:
  if (OUT/aid).exists():raise FileExistsError('Existing arm requires audited continuation; no automatic restart')
 assert read(REPORT/'preflight.json')['all_checks_passed']
 frozen=read(OUT/'preflight_freeze.json');assert all(sha(ROOT/k)==v for k,v in frozen['files_sha256'].items());assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items());assert all(sha(ROOT/k)==v for k,v in p['implementation_sha256'].items())
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');assert str(device)==p['preflight_device']
 ex={'decision':p['decision'],'protocol_sha256':sha(OUT/'protocol.json'),'runner_sha256':sha(Path(__file__)),'interface_sha256':sha(ROOT/'src/analyses/three_ward_spatial_fp_interfaces_v1.py'),'execution_authorized':True,'new_arms':p['new_arms'],'reused_arm':'S0F0 from immutable D_huber_lower_lr','maximum_new_fits':3,'maximum_new_updates':1440,'epochs_per_arm':40,'original12train4validation':True,'sealed_access':False,'sample_expansion':False,'device':str(device),'torch_version':torch.__version__,'seed':p['common_settings']['seed'],'cpu_threads':6,'precision':'float32, float64 metric aggregation; native threshold comparisons','memory_measurement':'Maximum of sampled MPS allocator values at step and evaluation boundaries, not exact allocation peak; host maxrss is process-cumulative.'}
 write(OUT/'execution_protocol.json',ex);ph=sha(OUT/'execution_protocol.json');records=p['records'];by={r['case_id']:i for i,r in enumerate(records)};orders=read(OLDPLAN/'epoch_order.json');seed=ex['seed'];init=torch.load(OLDPLAN/'shared_initialization.pt',map_location='cpu',weights_only=False);z=arrays(SOURCE/'prepared_inputs.npz');labels=[load_label(r['case_id'],r['split']) for r in records];eligible=z['Eligible Ground Mask'];area=z['Ground Area'];ward=z['Ward Code'];ys=np.array([d['Peak Mean Depth Lower'].astype('f4') for d in labels]);masks=np.array([d['Exact Label Mask']&eligible for d in labels]);pairs=np.array([masks[i]&masks[by[r['retrieval_source_case_id']]] for i,r in enumerate(records)])
 baselines=read(OLDREPORT/'frozen_baseline_metrics.json');write(REPORT/'frozen_baseline_metrics.json',baselines);expected_masks={r['case_id']:r['paired_three_ward_mask_sha256'] for r in records if r['split']=='validation'};regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0}
 reference_frames=[read(OLDREPORT/'D_huber_lower_lr'/f'epoch_{ep:03}.jsonl') for ep in range(1,41)];reference_selection=choose_checkpoints(reference_frames,p['validation_case_ids'],expected_masks,baselines);reference_binding={'arm':'S0F0','record_kind':'reused_reference_no_fit','source_arm':'D_huber_lower_lr','historical_selected_epoch':36,'prospective_selection':reference_selection,'frames':[{'epoch':f['epoch'],'frame_path':str((OLDREPORT/'D_huber_lower_lr'/f"epoch_{f['epoch']:03}.jsonl").relative_to(ROOT)),'frame_sha256':sha(OLDREPORT/'D_huber_lower_lr'/f"epoch_{f['epoch']:03}.jsonl"),'checkpoint_path':str((OLD/'D_huber_lower_lr'/f"epoch_{f['epoch']:03}.pt").relative_to(ROOT)),'checkpoint_sha256':f['checkpoint_sha256']} for f in reference_frames],'new_fits':0,'new_updates':0}
 (REPORT/'S0F0').mkdir(exist_ok=True);(OUT/'S0F0').mkdir(exist_ok=True);write(REPORT/'S0F0/reuse_binding.json',reference_binding)
 x=torch.from_numpy(z['Static Input'][None]).to(device);cond=torch.from_numpy(z['Forcing Input']).to(device);target=torch.from_numpy(ys).to(device);ground=torch.from_numpy(area.astype('f4')).to(device);tmasks=torch.from_numpy(masks).to(device);start=time.perf_counter();done=[]
 def store_predictions(path,preds):np.savez_compressed(path,**{'Peak Mean Depth':preds,'Case ID':z['Case ID'],'Eligible Ground Mask':eligible})
 for ai,cfg in enumerate(p['new_arms']):
  aid=cfg['arm'];dest=OUT/aid;report=REPORT/aid;dest.mkdir();report.mkdir();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  if device.type=='mps':torch.mps.manual_seed(seed)
  model=SpatialConditionalUNet(cfg['retain_deep_detail']).to(device);model.load_state_dict(init['state_dict'],strict=True);assert fingerprint(model)==p['initial_parameter_fingerprint'];optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.00001);frames=[];history=[];steps=0;training_s=0.;eval_s=0.;io_s=0.;selected_epoch=None;selection=None;epoch=0;cursor=0;maximum_sampled_mps_bytes=0;stage='initializing'
  write(report/'initialization_receipt.json',{'arm':aid,**cfg,'initial_parameter_fingerprint':fingerprint(model),'shared_initialization_sha256':sha(OLDPLAN/'shared_initialization.pt'),'initial_optimizer_steps':0,'parameters':sum(v.numel() for v in model.parameters()),'normalization_sha256':p['common_settings']['normalization_sha256'],'epoch_order_sha256':p['common_settings']['epoch_order_sha256'],'protocol_sha256':ph})
  def sample_memory():
   nonlocal maximum_sampled_mps_bytes
   if device.type=='mps':maximum_sampled_mps_bytes=max(maximum_sampled_mps_bytes,torch.mps.driver_allocated_memory())
  def state():return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'cursor':cursor,'steps':steps,'stage':stage,'protocol_sha256':ph,'arm':aid,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None,'history':history,'selection':selection}
  try:
   for epoch in range(1,41):
    model.train();step_records=[];stage='training';observer=OptimizationObserver(report/'optimizer_steps.jsonl',ph)
    for cursor,cid in enumerate(orders[epoch-1]['case_order'],1):
     i=by[cid];r=records[i];tic=time.perf_counter();optimizer.zero_grad(set_to_none=True);kind='stratified_huber_plus_FP' if cfg['fp_supervision'] else 'stratified_huber'
     observer.begin(model,optimizer,{'arm':aid,'epoch':epoch,'case_id':cid,'parent_id':r['parent_id'],'family_id':r['family_id'],'forcing_sha256':r['forcing_sha256'],'split':'train','seed':seed,'mask_sha256':r['original_exact_eligible_mask_sha256'],'loss_kind':kind,'record_kind':'training_update','retain_deep_detail':cfg['retain_deep_detail'],'fp_supervision':cfg['fp_supervision']})
     prediction=model(x,cond[i:i+1])[0,0];sample_memory();loss,payload=spatial_fp_loss(prediction,target[i],ground,tmasks[i],cfg['fp_supervision']);assert torch.isfinite(loss)
     loss.backward();sample_memory();observer.after_backward(model,payload);torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);observer.after_clip(model);optimizer.step();synchronize(device);row=observer.finish(model,optimizer);step_records.append(row);steps+=1;training_s+=time.perf_counter()-tic
     atomic_json(REPORT/'progress.json',{'status':'training','arm':aid,'new_arm_index':ai+1,'new_arms':3,'epoch':epoch,'update_in_epoch':cursor,'arm_updates':steps,'new_updates_completed':len(done)*480+steps,'elapsed_seconds':time.perf_counter()-start,'process_id':os.getpid()})
    assert steps==epoch*12;summary=summarize_step_records(step_records,p['train_case_ids']);summary['mean_base_loss_m2']=float(np.mean([r['loss']['base_value_m2'] for r in step_records]));summary['mean_FP_penalty_m2']=float(np.mean([r['loss']['fp_penalty_m2'] for r in step_records]));stage='fixed_evaluation';model.eval();before=fingerprint(model);tic=time.perf_counter();preds=[];fixed=[]
    with torch.inference_mode():
     for i in range(16):
      prediction=model(x,cond[i:i+1])[0,0];val,payload=spatial_fp_loss(prediction,target[i],ground,tmasks[i],cfg['fp_supervision']);fixed.append(payload);a=prediction.cpu().numpy().copy();a[~eligible]=np.nan;preds.append(a);sample_memory()
    preds=np.array(preds);events=[]
    for i,r in enumerate(records):
     for scope in REQUIRED_SCOPES:
      for support,mask in [('own_exact',masks[i]&regions[scope]),('paired_exact',pairs[i]&regions[scope])]:
       q=evaluate_depth(preds[i],ys[i],area,mask);q.update({'case_id':r['case_id'],'parent_id':r['parent_id'],'split':r['split'],'scope':scope,'support':support})
       if scope=='domain' and support=='own_exact':q.update({'fixed_model_loss_m2':fixed[i]['value_m2'],'fixed_base_loss_m2':fixed[i]['base_value_m2'],'fixed_FP_penalty_m2':fixed[i]['fp_penalty_m2'],'fixed_optimized_FP_contribution_m2':fixed[i]['optimized_FP_contribution_m2'],'fixed_loss_strata':fixed[i]['strata'],'loss_kind':kind})
       events.append(q)
    assert before==fingerprint(model);eval_s+=time.perf_counter()-tic;macros=[]
    for split in ['train','validation']:
     for scope in REQUIRED_SCOPES:
      for support in ['own_exact','paired_exact']:
       group=[v for v in events if v['split']==split and v['scope']==scope and v['support']==support];macros.append({'split':split,'scope':scope,'support':support,**macro_events(group)})
    checkpoint=dest/f'epoch_{epoch:03}.pt';tic=time.perf_counter();save_state(checkpoint,{'state_dict':model.state_dict(),'arm':aid,**cfg,'epoch':epoch,'optimizer_steps':steps,'protocol_sha256':ph,'parameter_fingerprint':before,'shared_initialization_sha256':sha(OLDPLAN/'shared_initialization.pt'),'normalization_sha256':p['common_settings']['normalization_sha256']});io_s+=time.perf_counter()-tic
    frame={'schema':'spatial-FP-fixed-checkpoint-v1','arm':aid,'epoch':epoch,**cfg,'protocol_sha256':ph,'record_kind':'training_epoch','loss_kind':kind,'step_summary':summary,'optimizer_steps_before_eval':steps,'optimizer_steps_after_eval':steps,'model_fingerprint_before_eval':before,'model_fingerprint_after_eval':fingerprint(model),'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'event_metrics':events,'parent_macro_metrics':macros,'cumulative_training_seconds':training_s,'cumulative_evaluation_seconds':eval_s,'cumulative_checkpoint_io_seconds':io_s,'sealed_access':False}
    frames.append(frame);selection=choose_checkpoints(frames,p['validation_case_ids'],expected_masks,baselines);frame['selection_through_this_epoch']=selection;write(report/f'epoch_{epoch:03}.json',frame)
    if selection['selected_epoch']!=selected_epoch:
     assert selection['selected_epoch']==epoch,'Selection cannot jump to an older nonselected checkpoint as the pool grows'
     selected_epoch=epoch;tic=time.perf_counter();save_state(dest/'best.pt',torch.load(checkpoint,map_location='cpu',weights_only=False));store_predictions(dest/'selected_predictions.npz',preds);write(report/'selected_metrics.json',frame);io_s+=time.perf_counter()-tic
    if epoch==40:store_predictions(dest/'endpoint_predictions.npz',preds)
    vm=next(q['values'] for q in macros if q['split']=='validation' and q['scope']=='three_wards' and q['support']=='paired_exact');tm=next(q['values'] for q in macros if q['split']=='train' and q['scope']=='domain' and q['support']=='own_exact')
    history.append({'arm':aid,'epoch':epoch,'optimizer_steps':steps,'fixed_train_base_loss_m2':tm['fixed_base_loss_m2'],'fixed_train_FP_penalty_m2':tm['fixed_FP_penalty_m2'],'fixed_train_total_objective_m2':tm['fixed_model_loss_m2'],'validation_IoU':vm['iou_0.1'],'validation_false_km2':vm['false_km2_0.1'],'validation_recall':vm['recall_0.1'],'validation_precision':vm['precision_0.1'],'validation_BER':vm['balanced_extent_error'],'validation_MAE_m':vm['mae_m'],'selected_epoch_so_far':selection['selected_epoch'],'shadow_BER_epoch_so_far':selection['shadow_BER_epoch'],'readiness':selection['status'],'clipping_count':summary['clipping_count'],'gradient_max':summary['gradient_preclip_norm']['maximum'],'cumulative_training_seconds':training_s,'cumulative_evaluation_seconds':eval_s,'cumulative_checkpoint_io_seconds':io_s})
    csv_write(report/'history.csv',history);stage='epoch_complete';tic=time.perf_counter();save_state(dest/'live.pt',state());io_s+=time.perf_counter()-tic;print(json.dumps(history[-1]),flush=True)
   assert steps==480 and len(frames)==40
   selection=choose_checkpoints(frames,p['validation_case_ids'],expected_masks,baselines);write(report/'final_selection.json',selection)
   result={'arm':aid,'status':'complete',**cfg,'epochs':40,'optimizer_updates':steps,'selection':selection,'initial_parameter_fingerprint':p['initial_parameter_fingerprint'],'training_seconds':training_s,'evaluation_seconds':eval_s,'checkpoint_IO_seconds':io_s,'maximum_sampled_mps_driver_allocated_bytes':maximum_sampled_mps_bytes,'host_process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'memory_scope':ex['memory_measurement'],'selected_checkpoint_sha256':sha(dest/'best.pt'),'selected_predictions_sha256':sha(dest/'selected_predictions.npz'),'endpoint_predictions_sha256':sha(dest/'endpoint_predictions.npz'),'sealed_opened':False};write(report/'completion.json',result);done.append(result);print(json.dumps({'arm_completed':aid,'selection_epoch':selection['selected_epoch'],'readiness':selection['status']}),flush=True)
  except BaseException as err:
   stage_at_failure=stage;save_state(dest/'failure_state.pt',state());failure={'arm':aid,'status':'held','epoch':epoch,'cursor':cursor,'steps':steps,'stage':stage_at_failure,'exception':repr(err),'traceback':traceback.format_exc(),'no_automatic_restart':True};write(report/'failure.json',failure);atomic_json(REPORT/'progress.json',failure);raise
  finally:
   del model,optimizer
   if device.type=='mps':torch.mps.empty_cache()
 # Re-infer only the frozen D endpoint and common-selector checkpoint, no optimizer.
 reference_model=SpatialConditionalUNet(False).to(device);reference_model.eval();reference_epochs=sorted({40,reference_selection['selected_epoch']});reference_exports=[]
 for ep in reference_epochs:
  ck=torch.load(OLD/'D_huber_lower_lr'/f'epoch_{ep:03}.pt',map_location='cpu',weights_only=False);reference_model.load_state_dict(ck['state_dict']);before=fingerprint(reference_model);preds=[]
  with torch.inference_mode():
   for i in range(16):a=reference_model(x,cond[i:i+1])[0,0].cpu().numpy().copy();a[~eligible]=np.nan;preds.append(a)
  assert fingerprint(reference_model)==before;preds=np.array(preds)
  if ep==40:store_predictions(OUT/'S0F0/endpoint_predictions.npz',preds)
  if ep==reference_selection['selected_epoch']:store_predictions(OUT/'S0F0/selected_predictions.npz',preds)
  reference_exports.append({'epoch':ep,'cases':16,'parameter_fingerprint_before_after':before,'new_optimizer_updates':0,'source_checkpoint_sha256':sha(OLD/'D_huber_lower_lr'/f'epoch_{ep:03}.pt')})
 write(REPORT/'S0F0/reference_export.json',{'read_only_exports':reference_exports,'selection':reference_selection,'new_fits':0,'new_updates':0});assert len(done)==3 and sum(a['optimizer_updates'] for a in done)==1440;assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items())
 result={'status':'complete','decision':p['decision'],'new_fits':3,'new_optimizer_updates':1440,'reused_reference':'S0F0 from existing D','reference_selection':reference_selection,'arms':done,'execution_protocol_sha256':ph,'elapsed_seconds':time.perf_counter()-start,'original_data_and_masks_unchanged':True,'sealed_opened':False,'sample_expansion':False};write(REPORT/'completion.json',result);atomic_json(REPORT/'progress.json',{'status':'complete','new_fits':3,'new_updates':1440,'sealed_opened':False});print('THREE_SPATIAL_FP_FITS_COMPLETE',flush=True)
if __name__=='__main__':execute()
