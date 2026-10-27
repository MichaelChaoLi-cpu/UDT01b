"""Preflight accepted controls using original allowlist. Zero optimizer steps."""
from pathlib import Path
import sys,json,copy
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,load_label,arrays
from train_three_ward_unet_exact_pilot import ConditionalUNet
from run_three_ward_unet_controls_v2 import fingerprint
from three_ward_spatial_fp_interfaces_v1 import SpatialConditionalUNet,spatial_fp_loss,choose_checkpoints
from three_ward_surrogate_protocol_v2 import exact_loss,evaluate_depth
NAME='three_ward_spatial_fp_controls_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
DESIGN=ROOT/'data/exp/research-design/three_ward_spatial_fp_control_plan_v1';SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1';OLDPLAN=ROOT/'data/processed/three_ward_unet_control_plan_v2';OLDREPORT=ROOT/'data/exp/estimation/three_ward_unet_controls_v2';OLD=ROOT/'data/processed/three_ward_unet_controls_v2'
def main():
 if (OUT/'preflight_freeze.json').exists():raise FileExistsError('Preserve preflight')
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);checks=[]
 def check(n,v):
  if not bool(v):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 design=read(DESIGN/'design.json');p=read(OLDPLAN/'protocol.json');sources=dict(design['source_files_sha256']);sources.update(read(SOURCE/'protocol.json')['source_hashes'])
 for file in DESIGN.iterdir():
  if file.is_file():sources[str(file.relative_to(ROOT))]=sha(file)
 check('all_frozen_sources_match',all(sha(ROOT/k)==v for k,v in sources.items()))
 check('unchanged_original16records',design['fixed']['original_records_and_mask_bindings']==p['records'])
 z=arrays(SOURCE/'prepared_inputs.npz');records=p['records'];labels=[load_label(r['case_id'],r['split']) for r in records];by={r['case_id']:i for i,r in enumerate(records)};masks=np.array([l['Exact Label Mask']&z['Eligible Ground Mask'] for l in labels]);mh=lambda m:__import__('hashlib').sha256(np.ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
 for i,r in enumerate(records):
  check(r['case_id']+'_original_exact_mask',mh(masks[i])==r['original_exact_eligible_mask_sha256'])
  check(r['case_id']+'_original_paired_mask',mh(masks[i]&masks[by[r['retrieval_source_case_id']]]&(z['Ward Code']>0))==r['paired_three_ward_mask_sha256'])
  check(r['case_id']+'_no_interval_or_quarantine_training',not np.any(masks[i]&(labels[i]['Interval Label Mask']|labels[i]['Quarantine Mask']|labels[i]['Online Ever Negative'])))
 init=torch.load(OLDPLAN/'shared_initialization.pt',map_location='cpu',weights_only=False);check('original_initialization_untrained',not init['trained'] and init['optimizer_steps']==0 and sha(OLDPLAN/'shared_initialization.pt')==design['fixed']['initialization_sha256'])
 torch.set_num_threads(6);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');check('original_device_reusable',str(device)==read(OLD/'execution_protocol.json')['device'])
 ref=ConditionalUNet().to(device);ref.load_state_dict(init['state_dict']);ref.eval();fp0=fingerprint(ref);x=torch.from_numpy(z['Static Input'][None]).to(device);c=torch.from_numpy(z['Forcing Input'][0:1]).to(device)
 with torch.inference_mode():rp=ref(x,c).cpu()
 shape_records=[]
 for detail in [False,True]:
  model=SpatialConditionalUNet(detail).to(device);model.load_state_dict(init['state_dict'],strict=True);model.eval();before=fingerprint(model);captured=[];hook=model.enc[-1].register_forward_hook(lambda mod,inp,out:captured.append(list(out.shape[-2:])))
  original_pool=torch.nn.functional.avg_pool2d
  with patch('torch.nn.functional.avg_pool2d',wraps=original_pool) as pool:
   with torch.inference_mode():pred=model(x,c).cpu()
   count=pool.call_count
  hook.remove();check(f'S{int(detail)}_identical_full_parameter_fingerprint',before==fp0 and fingerprint(model)==before);check(f'S{int(detail)}_same_parameter_count',sum(t.numel() for t in model.parameters())==671513);check(f'S{int(detail)}_output_shape_finite_nonnegative',list(pred.shape)==[1,1,1012,770] and torch.isfinite(pred).all() and bool((pred>=0).all()))
  check(f'S{int(detail)}_only_declared_pool_count',count==(3 if detail else 4));check(f'S{int(detail)}_declared_bottleneck',captured==[[128,98] if detail else [64,49]])
  if not detail:check('S0_reproduces_original_forward',float(torch.max(torch.abs(pred-rp)))<1e-6)
  shape_records.append({'detail':detail,'pool_calls':count,'bottleneck_shape':captured[0],'initial_parameter_fingerprint':before});del model
 # Independent expected added loss/derivative on masked cells, including native
 # float32 threshold ties and an excluded sentinel with extreme depth.
 y=torch.tensor([[0.,.1,.1,.08,.5,2.,0.]],dtype=torch.float32,device=device);a=torch.tensor([[1.,2.,3.,4.,5.,6.,7.]],device=device);m=torch.tensor([[True,True,True,True,True,True,False]],device=device);v0=[[.2,.1,.3,.05,.8,1.,999.]]
 for enabled in [False,True]:
  v=torch.tensor(v0,device=device,requires_grad=True);l,d=spatial_fp_loss(v,y,a,m,enabled);base,bmeta=exact_loss(v,y,a,m,'stratified_huber');expected=.1*(1*.1+3*.2)/10
  check(f'F{int(enabled)}_independent_loss_value',abs(d['fp_penalty_m2']-expected)<1e-7 and abs(d['value_m2']-(bmeta['value_m2']+(expected if enabled else 0)))<1e-7)
  g=torch.autograd.grad(l,v,retain_graph=True)[0];gb=torch.autograd.grad(base,v)[0];ge=torch.tensor([[.01,0.,.03,0.,0.,0.,0.]],device=device) if enabled else torch.zeros_like(v)
  check(f'F{int(enabled)}_added_gradient_mask_and_sign',float(torch.max(torch.abs(g-gb-ge)))<1e-7)
 for target in [torch.full_like(y,.5),torch.full_like(y,2.)]:
  vv=torch.ones_like(y,requires_grad=True);l,d=spatial_fp_loss(vv,target,a,m,True);check('absent_nonwet_'+str(float(target[0,0]))+'_no_fabricated_penalty',not d['nonwet_present'] and d['fp_penalty_m2']==0 and torch.isfinite(l))
 try:spatial_fp_loss(torch.ones_like(y),y,a,torch.zeros_like(m),True);empty=False
 except ValueError:empty=True
 check('empty_exact_rejected',empty)
 # Every retained D frame/checkpoint is a real fixed-model, original-mask score.
 baselines=read(OLDREPORT/'frozen_baseline_metrics.json');frames=[];expected_masks={r['case_id']:r['paired_three_ward_mask_sha256'] for r in records if r['split']=='validation'}
 for ep in range(1,41):
  f=read(OLDREPORT/'D_huber_lower_lr'/f'epoch_{ep:03}.jsonl');check(f'D_epoch{ep}_bound_checkpoint',sha(OLD/'D_huber_lower_lr'/f'epoch_{ep:03}.pt')==f['checkpoint_sha256']);check(f'D_epoch{ep}_exact_clock_and_fingerprint',f['optimizer_steps_before_eval']==f['optimizer_steps_after_eval']==ep*12 and f['model_fingerprint_before_eval']==f['model_fingerprint_after_eval']);frames.append(f)
 selected=choose_checkpoints(frames,p['validation_case_ids'],expected_masks,baselines);check('D_reuse_all40_failed_guard_not_promoted',selected['status']=='not_ready' and selected['passing_epochs']==[] and not selected['sealed_access_authorized']);check('D_historical_selection_unchanged',read(OLDREPORT/'D_huber_lower_lr/selected_metrics.json')['epoch']==36)
 # Selector fixtures are explicitly synthetic, not measured model performance.
 ids=['fixture'];keymask={'fixture':'synthetic'}
 def metrics(B,IoU,mae=.01,wet=.02,recall=.9,fp=.1):return {'case_id':'fixture','parent_id':'synthetic','split':'validation','scope':'three_wards','support':'paired_exact','support_sha256':'synthetic','support_km2':1.,'selection_valid':True,'balanced_extent_error':B,'iou_0.1':IoU,'wet_mae_m':wet,'mae_m':mae,'recall_0.1':recall,'false_km2_0.1':fp}
 bs={'dry':{'fixture':metrics(.5,0,mae=.1)},'retrieval':{'fixture':metrics(.2,.7,mae=.03,recall=.8,fp=.2)}}
 fixture=[{'epoch':1,'event_metrics':[metrics(.15,.5)]},{'epoch':2,'event_metrics':[metrics(.1,.4)]},{'epoch':3,'event_metrics':[metrics(.3,.9)]}]
 sel=choose_checkpoints(fixture,ids,keymask,bs);check('selector_guard_first_IoU_and_shadow_same_pool',sel['candidate_pool_epochs']==[1,2] and sel['selected_epoch']==1 and sel['shadow_BER_epoch']==2)
 tied=copy.deepcopy(fixture[:1]);tied.append({'epoch':2,'event_metrics':[metrics(.15,.5)]});check('selector_earliest_tie',choose_checkpoints(tied,ids,keymask,bs)['selected_epoch']==1)
 bad=copy.deepcopy(fixture[:1]);bad[0]['event_metrics'][0]['support_sha256']='wrong'
 try:choose_checkpoints(bad,ids,keymask,bs);caught=False
 except ValueError:caught=True
 check('selector_rejects_changed_mask',caught)
 bad=copy.deepcopy(fixture[:1]);bad[0]['event_metrics'][0]['split']='train'
 try:choose_checkpoints(bad,ids,keymask,bs);caught=False
 except ValueError:caught=True
 check('selector_rejects_split_leakage',caught)
 del ref,x,c,rp
 if device.type=='mps':torch.mps.empty_cache()
 check('all_frozen_sources_unchanged_after_preflight',all(sha(ROOT/k)==v for k,v in sources.items()))
 implementation={str((ROOT/'src/analyses'/n).relative_to(ROOT)):sha(ROOT/'src/analyses'/n) for n in ['three_ward_spatial_fp_interfaces_v1.py','prepare_three_ward_spatial_fp_controls_v1.py']}
 protocol={'decision':'MILI-D-20261016-008','execution_authorized':True,'design_path':str((DESIGN/'design.json').relative_to(ROOT)),'design_sha256':sha(DESIGN/'design.json'),'design_unchanged':True,'source_sha256':sources,'implementation_sha256':implementation,'records':records,'train_case_ids':p['train_case_ids'],'validation_case_ids':p['validation_case_ids'],'common_settings':design['fixed'],'new_arms':[{'arm':'S1F0','retain_deep_detail':True,'fp_supervision':False},{'arm':'S0F1','retain_deep_detail':False,'fp_supervision':True},{'arm':'S1F1','retain_deep_detail':True,'fp_supervision':True}],'reused_arm':'S0F0','reused_source_arm':'D_huber_lower_lr','preflight_device':str(device),'torch_version':torch.__version__,'initial_parameter_fingerprint':fp0,'checkpoint_policy':design['checkpoint'],'maximum_new_fits':3,'maximum_new_updates':1440,'sealed_access':False,'sample_expansion':False,'preflight_optimizer_updates':0}
 write(OUT/'protocol.json',protocol);write(REPORT/'preflight.json',{'all_checks_passed':True,'checks':checks,'new_fit_count':0,'optimizer_updates':0,'zero_update_full_field_forward_checks':3,'shape_records':shape_records,'D_reuse_selection':selected,'fixture_scope':'Synthetic selector examples test interface behavior only, not model performance.','sealed_opened':False})
 write(OUT/'preflight_freeze.json',{'status':'passed_before_training','files_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [OUT/'protocol.json',REPORT/'preflight.json']},'implementation_sha256':implementation,'zero_optimizer_updates':True})
 print(json.dumps({'all_checks_passed':True,'checks':len(checks),'device':str(device),'D_prospective_selection':selected,'zero_optimizer_updates':True}),flush=True)
if __name__=='__main__':main()
