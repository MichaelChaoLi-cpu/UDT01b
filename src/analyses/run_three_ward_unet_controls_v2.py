"""Execute the human-authorized frozen four-arm protocol, without sealed data.
Each arm starts from the shared untrained state. Retain every epoch, real step
telemetry and fixed-model scores. No settings are adapted to outcomes.
"""
from pathlib import Path
import os,sys,time,json,hashlib,random,traceback,csv
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import OUT as SOURCE,sha,read,write,arrays,load_label
from prepare_three_ward_unet_control_plan_v2 import OUT as PLAN,mask_sha
from train_three_ward_unet_exact_pilot import ConditionalUNet,synchronize
from three_ward_surrogate_protocol_v2 import exact_loss,evaluate_depth,macro_events,checkpoint_key,baseline_readiness,OptimizationObserver,parameter_snapshot,write_epoch_record,REQUIRED_SCOPES,summarize_step_records
NAME='three_ward_unet_controls_v2';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def fingerprint(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():
  a=p.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def atomic_json(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');write(tmp,value);tmp.replace(path)
def save_state(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def csv_write(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def execute():
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 p=read(PLAN/'protocol.json');freeze=read(PLAN/'freeze.json')
 assert all(sha(ROOT/k)==v for k,v in freeze['files_sha256'].items()),'Prepared protocol changed'
 assert all(sha(ROOT/k)==v for k,v in p['source_sha256'].items()),'Original data/code changed'
 assert all(sha(ROOT/k)==v for k,v in p['implementation_sha256'].items()),'Frozen interfaces changed'
 if (REPORT/'completion.json').exists():raise FileExistsError('Completed experiment cannot be repeated in place')
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');seed=p['frozen_common_settings']['seed']
 ep_path=OUT/'execution_protocol.json'
 if not ep_path.exists():
  execution={'decision':'MILI-D-20261016-005','prepared_protocol_sha256':sha(PLAN/'protocol.json'),'prepared_freeze_sha256':sha(PLAN/'freeze.json'),'execution_authorized':True,'authorized_arms':p['arms'],'training_case_ids':p['train_case_ids'],'validation_case_ids':p['validation_case_ids'],'source_records':p['records'],'shared_initialization_sha256':sha(PLAN/'shared_initialization.pt'),'epoch_order_sha256':sha(PLAN/'epoch_order.json'),'normalization_sha256':p['frozen_common_settings']['normalization_sha256'],'runner_sha256':sha(Path(__file__)),'interface_sha256':sha(ROOT/'src/analyses/three_ward_surrogate_protocol_v2.py'),'torch_version':torch.__version__,'python_version':sys.version,'device':str(device),'cpu_threads':6,'precision':'float32, no mixed precision','seed':seed,'max_epochs_per_arm':40,'maximum_total_fits':4,'maximum_total_optimizer_steps':1920,'sealed_access':False,'scientific_settings_changed_from_prepared_plan':False,'reproducibility':'Paired shared initialization and order; Metal kernel bitwise determinism not certified.'}
  write(ep_path,execution)
 else:
  execution=read(ep_path);assert execution['runner_sha256']==sha(Path(__file__)) and execution['device']==str(device),'Runtime version/device changed'
 ph=sha(ep_path);records=p['records'];by={r['case_id']:i for i,r in enumerate(records)};tr=[by[c] for c in p['train_case_ids']];va=[by[c] for c in p['validation_case_ids']];orders=read(PLAN/'epoch_order.json')
 init=torch.load(PLAN/'shared_initialization.pt',map_location='cpu',weights_only=False);assert not init['trained'] and init['optimizer_steps']==0
 z=arrays(SOURCE/'prepared_inputs.npz');labels=[load_label(r['case_id'],r['split']) for r in records];area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];ys=np.array([d['Peak Mean Depth Lower'].astype('f4') for d in labels]);masks=np.array([d['Exact Label Mask']&eligible for d in labels]);pairs=np.array([masks[i]&masks[by[r['retrieval_source_case_id']]] for i,r in enumerate(records)])
 for i,r in enumerate(records):assert mask_sha(masks[i])==r['original_exact_eligible_mask_sha256'] and mask_sha(pairs[i])==r['paired_exact_mask_sha256']
 expected={r['case_id']:r['split'] for r in records};expected_masks={r['case_id']:r['paired_three_ward_mask_sha256'] for r in records if r['split']=='validation'};regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0}
 baselines={'dry':{},'retrieval':{}}
 for i in va:
  r=records[i];mask=pairs[i]&(ward>0);j=by[r['retrieval_source_case_id']]
  baselines['dry'][r['case_id']]=evaluate_depth(np.zeros_like(ys[i]),ys[i],area,mask);baselines['retrieval'][r['case_id']]=evaluate_depth(ys[j],ys[i],area,mask)
 write(REPORT/'frozen_baseline_metrics.json',baselines)
 x=torch.from_numpy(z['Static Input'][None]).to(device);cond=torch.from_numpy(z['Forcing Input']).to(device);targets=torch.from_numpy(ys).to(device);ground=torch.from_numpy(area.astype('f4')).to(device);tmasks=torch.from_numpy(masks).to(device)
 results=[];global_start=time.perf_counter()
 for ai,arm in enumerate(p['arms']):
  aid=arm['arm_id'];dest=OUT/aid;report=REPORT/aid;dest.mkdir(exist_ok=True);report.mkdir(exist_ok=True)
  if (report/'completion.json').exists():results.append(read(report/'completion.json'));continue
  if (report/'failure.json').exists():raise RuntimeError('Held failed arm requires a separate audited continuation, never silently restart')
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  if device.type=='mps':torch.mps.manual_seed(seed)
  model=ConditionalUNet().to(device);model.load_state_dict(init['state_dict']);optimizer=torch.optim.AdamW(model.parameters(),lr=arm['learning_rate'],weight_decay=.00001)
  live=dest/'live.pt';epoch=0;steps=0;phase='ready';cursor=0;step_records=[];training_s=0.;evaluation_s=0.;checkpoint_s=0.;best_key=None;best_epoch=None;history=[]
  start_fingerprint=fingerprint(model)
  if live.exists():
   ck=torch.load(live,map_location='cpu',weights_only=False);assert ck['protocol_sha256']==ph and ck['arm']==aid
   model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);epoch=ck['epoch'];steps=ck['steps'];phase=ck['phase'];cursor=ck['cursor'];step_records=ck['step_records'];training_s=ck['training_s'];evaluation_s=ck['evaluation_s'];checkpoint_s=ck['checkpoint_s'];best_key=tuple(ck['best_key']) if ck['best_key'] is not None else None;best_epoch=ck['best_epoch'];history=ck['history']
   torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
   if device.type=='mps':torch.mps.set_rng_state(ck['mps_rng'])
  else:
   write(report/'initialization_receipt.json',{'arm':aid,'shared_initialization_sha256':sha(PLAN/'shared_initialization.pt'),'loaded_parameter_fingerprint':start_fingerprint,'initial_optimizer_steps':0,'seed':seed,'learning_rate':arm['learning_rate'],'loss':arm['loss'],'protocol_sha256':ph})
  def snapshot(state_phase):
   state={'protocol_sha256':ph,'arm':aid,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'steps':steps,'phase':state_phase,'cursor':cursor,'step_records':step_records,'training_s':training_s,'evaluation_s':evaluation_s,'checkpoint_s':checkpoint_s,'best_key':best_key,'best_epoch':best_epoch,'history':history,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
   if device.type=='mps':state['mps_rng']=torch.mps.get_rng_state()
   return state
  try:
   while epoch<40 or phase=='trained_pending_evaluation':
    if phase=='ready':epoch+=1;cursor=0;step_records=[];phase='training'
    order=orders[epoch-1]['case_order'];assert set(order)==set(p['train_case_ids'])
    if phase=='training':
     model.train();observer=OptimizationObserver(report/'optimizer_steps.jsonl',ph)
     for position in range(cursor,12):
      cid=order[position];i=by[cid];r=records[i];optimizer.zero_grad(set_to_none=True);tic=time.perf_counter()
      observer.begin(model,optimizer,{'arm':aid,'epoch':epoch,'case_id':cid,'parent_id':r['parent_id'],'family_id':r['family_id'],'forcing_sha256':r['forcing_sha256'],'split':'train','seed':seed,'mask_sha256':r['original_exact_eligible_mask_sha256'],'loss_kind':arm['loss'],'record_kind':'training_update','timing_scope':'Forward/backward, finite checks, clipping, update and observer transfers; evaluation and checkpoint I/O separate'})
      pred=model(x,cond[i:i+1])[0,0];loss,payload=exact_loss(pred,targets[i],ground,tmasks[i],arm['loss'])
      if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
      loss.backward();observer.after_backward(model,payload);torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);observer.after_clip(model);optimizer.step();synchronize(device)
      row=observer.finish(model,optimizer,expected_update=True);step_records.append(row);steps+=1;cursor=position+1;training_s+=time.perf_counter()-tic
      atomic_json(REPORT/'progress.json',{'status':'training','arm':aid,'arm_index':ai+1,'arms':4,'epoch':epoch,'epochs':40,'update_in_epoch':cursor,'arm_optimizer_steps':steps,'completed_arms':len(results),'elapsed_current_process_seconds':time.perf_counter()-global_start,'process_id':os.getpid()})
     assert steps==12*epoch;phase='trained_pending_evaluation';tic=time.perf_counter();save_state(live,snapshot(phase));checkpoint_s+=time.perf_counter()-tic
    if phase=='trained_pending_evaluation':
     summary=summarize_step_records(step_records,p['train_case_ids']);model.eval();before=fingerprint(model);opt_before=steps;tic=time.perf_counter();preds=[];fixed_losses=[]
     with torch.inference_mode():
      for i in range(16):
       pred=model(x,cond[i:i+1])[0,0];value,payload=exact_loss(pred,targets[i],ground,tmasks[i],arm['loss']);fixed_losses.append(payload);pp=pred.cpu().numpy().copy();pp[~eligible]=np.nan;preds.append(pp)
     preds=np.array(preds);events=[]
     for i,r in enumerate(records):
      for scope in REQUIRED_SCOPES:
       for support,mask0 in [('own_exact',masks[i]&regions[scope]),('paired_exact',pairs[i]&regions[scope])]:
        metric=evaluate_depth(preds[i],ys[i],area,mask0);metric.update({'case_id':r['case_id'],'parent_id':r['parent_id'],'split':r['split'],'scope':scope,'support':support})
        if scope=='domain' and support=='own_exact':metric.update({'fixed_model_loss_m2':fixed_losses[i]['value_m2'],'loss_kind':arm['loss'],'fixed_loss_strata':fixed_losses[i]['strata']})
        events.append(metric)
     after=fingerprint(model);evaluation_s+=time.perf_counter()-tic;assert after==before
     checkpoint=dest/f'epoch_{epoch:03}.pt';tic=time.perf_counter()
     if not checkpoint.exists():save_state(checkpoint,{'state_dict':model.state_dict(),'epoch':epoch,'optimizer_steps':steps,'arm':aid,'protocol_sha256':ph,'normalization_sha256':p['frozen_common_settings']['normalization_sha256'],'parameter_fingerprint':after,'shared_initialization_sha256':sha(PLAN/'shared_initialization.pt')})
     ch=sha(checkpoint);checkpoint_s+=time.perf_counter()-tic
     context={'arm':aid,'epoch':epoch,'record_kind':'training_epoch','loss_kind':arm['loss'],'step_summary':summary,'optimizer_steps_before_eval':opt_before,'optimizer_steps_after_eval':steps,'model_fingerprint_before_eval':before,'model_fingerprint_after_eval':after,'checkpoint_path':str(checkpoint),'checkpoint_sha256':ch,'cumulative_training_seconds':training_s,'cumulative_evaluation_seconds':evaluation_s,'cumulative_checkpoint_io_seconds':checkpoint_s}
     epochfile=report/f'epoch_{epoch:03}.jsonl'
     if epochfile.exists():
      event=read_jsonl(epochfile)[0];assert event['checkpoint_sha256']==ch and event['protocol_sha256']==ph
     else:event=write_epoch_record(epochfile,ph,context,events,expected,expected_masks,baselines)
     key=tuple(event['selection_key']) if event['selection_key'] is not None else None
     selected=key is not None and (best_key is None or key<best_key)
     if selected:
      best_key=key;best_epoch=epoch;save_state(dest/'best.pt',torch.load(checkpoint,map_location='cpu',weights_only=False));np.savez_compressed(dest/'selected_predictions.npz',**{'Peak Mean Depth':preds,'Case ID':np.array([r['case_id'] for r in records]),'Eligible Ground Mask':eligible})
      write(report/'selected_metrics.json',event)
     macro=next(v for v in event['parent_macro_metrics'] if v['split']=='validation' and v['scope']=='three_wards' and v['support']=='paired_exact')['values'];trainmacro=next(v for v in event['parent_macro_metrics'] if v['split']=='train' and v['scope']=='domain' and v['support']=='own_exact')['values']
     row={'arm':aid,'epoch':epoch,'optimizer_steps':steps,'online_mean_loss_m2':summary['online_preupdate_mean_loss_m2'],'fixed_train_loss_m2':trainmacro['fixed_model_loss_m2'],'validation_balanced_extent_error':macro['balanced_extent_error'],'validation_MAE_m':macro['mae_m'],'validation_wet_MAE_m':macro['wet_mae_m'],'validation_deep_MAE_m':macro['deep_mae_m'],'validation_IoU':macro['iou_0.1'],'validation_recall':macro['recall_0.1'],'validation_false_km2':macro['false_km2_0.1'],'clipping_count':summary['clipping_count'],'gradient_max':summary['gradient_preclip_norm']['maximum'],'baseline_all_parents_pass':event['baseline_readiness']['all_parents_pass'],'best_epoch_so_far':best_epoch,'cumulative_training_seconds':training_s,'cumulative_evaluation_seconds':evaluation_s,'cumulative_checkpoint_io_seconds':checkpoint_s}
     if not history or history[-1]['epoch']!=epoch:history.append(row)
     csv_write(report/'history.csv',history);phase='ready';tic=time.perf_counter();save_state(live,snapshot(phase));checkpoint_s+=time.perf_counter()-tic
     print(json.dumps(row),flush=True)
     atomic_json(REPORT/'progress.json',{'status':'epoch_complete','arm':aid,'arm_index':ai+1,'arms':4,'epoch':epoch,'epochs':40,'arm_optimizer_steps':steps,'best_epoch':best_epoch,'selected_key':best_key,'completed_arms':len(results),'elapsed_current_process_seconds':time.perf_counter()-global_start,'process_id':os.getpid()})
   assert steps==480 and epoch==40 and len(history)==40
   best=read(report/'selected_metrics.json');bm=next(v for v in best['parent_macro_metrics'] if v['split']=='validation' and v['scope']=='three_wards' and v['support']=='paired_exact')
   done={'arm':aid,'status':'complete','loss':arm['loss'],'learning_rate':arm['learning_rate'],'seed':seed,'epochs':40,'optimizer_updates':480,'best_epoch':best_epoch,'selection_key':best_key,'baseline_all_parents_pass':best['baseline_readiness']['all_parents_pass'],'validation_three_ward_paired_macro':bm,'training_seconds_with_observer':training_s,'evaluation_seconds':evaluation_s,'checkpoint_io_seconds':checkpoint_s,'protocol_sha256':ph,'best_checkpoint_sha256':sha(dest/'best.pt'),'selected_predictions_sha256':sha(dest/'selected_predictions.npz'),'initialization_fingerprint':start_fingerprint,'sealed_opened':False}
   write(report/'completion.json',done);results.append(done);print(json.dumps({'arm_completed':aid,'best_epoch':best_epoch,'selected_B':bm['values']['balanced_extent_error']}),flush=True)
  except BaseException as err:
   synchronize(device);failure={'arm':aid,'status':'held','epoch':epoch,'optimizer_steps_completed':steps,'cursor':cursor,'phase':phase,'exception':repr(err),'traceback':traceback.format_exc(),'protocol_sha256':ph,'no_automatic_restart':True};write(report/'failure.json',failure);save_state(dest/'failure_state.pt',snapshot(phase));atomic_json(REPORT/'progress.json',failure);raise
  finally:
   del model,optimizer
   if device.type=='mps':torch.mps.empty_cache()
 assert len(results)==4 and sum(r['optimizer_updates'] for r in results)==1920
 assert all(sha(ROOT/k)==v for k,v in freeze['files_sha256'].items())
 assert all(sha(ROOT/k)==v for k,v in read(SOURCE/'protocol.json')['source_hashes'].items())
 write(REPORT/'completion.json',{'status':'complete','decision':'MILI-D-20261016-005','arms':results,'actual_fits':4,'optimizer_updates':1920,'sealed_opened':False,'original_sources_and_prepared_plan_unchanged':True,'execution_protocol_sha256':ph,'current_process_elapsed_seconds':time.perf_counter()-global_start})
 atomic_json(REPORT/'progress.json',{'status':'complete','completed_arms':4,'optimizer_updates':1920,'sealed_opened':False});print('ALL_FOUR_ARMS_COMPLETE',flush=True)
if __name__=='__main__':execute()
