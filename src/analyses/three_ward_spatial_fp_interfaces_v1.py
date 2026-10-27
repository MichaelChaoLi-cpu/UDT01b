"""Versioned spatial toggle, exact-only FP loss and common selector. No fitting."""
import torch
from torch.nn import functional as F
from train_three_ward_unet_exact_pilot import ConditionalUNet
from three_ward_surrogate_protocol_v2 import exact_loss,macro_events,checkpoint_key,baseline_readiness

class SpatialConditionalUNet(ConditionalUNet):
 def __init__(self,retain_deep_detail=False):
  super().__init__();self.retain_deep_detail=bool(retain_deep_detail)
 def forward(self,x,c):
  if not self.retain_deep_detail:return super().forward(x,c)
  h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%16,0,(-h)%16));c=self.condition(c);skips=[]
  for i,b in enumerate(self.enc):
   if i and i!=4:x=F.avg_pool2d(x,2)
   x=b(x,c);skips.append(x)
  for b,s in zip(self.dec,reversed(skips[:-1])):
   x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False);x=b(torch.cat([x,s],dim=1),c)
  return F.softplus(self.head(x))[...,:h,:w]

def spatial_fp_loss(pred,target,area,mask,fp_supervision=False):
 base,meta=exact_loss(pred,target,area,mask,'stratified_huber');nonwet=mask&(target<=.1);present=bool(nonwet.any())
 penalty=.1*(area[nonwet]*torch.relu(pred[nonwet]-.1)).sum()/area[nonwet].sum() if present else base*0
 total=base+penalty if fp_supervision else base
 return total,{'kind':'stratified_huber_plus_FP' if fp_supervision else 'stratified_huber','value_m2':float(total.detach().cpu()),'base_value_m2':meta['value_m2'],'fp_penalty_m2':float(penalty.detach().cpu()),'optimized_FP_contribution_m2':float(penalty.detach().cpu()) if fp_supervision else 0.,'fp_multiplier':int(bool(fp_supervision)),'nonwet_present':present,'strata':meta['strata']}

def score_candidate(frame,validation_ids,expected_masks,baselines):
 events=[r for r in frame['event_metrics'] if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact']
 legacy=checkpoint_key(events,frame['epoch'],validation_ids,expected_masks)
 readiness=baseline_readiness({r['case_id']:r for r in events},baselines['dry'],baselines['retrieval'])
 macro=macro_events(events);m=macro['values'];iou=m.get('iou_0.1')
 key=(-round(iou,6),round(m['wet_mae_m'],6),round(m['mae_m'],6),int(frame['epoch'])) if legacy is not None and iou is not None else None
 return {'epoch':frame['epoch'],'iou_key':key,'BER_key':legacy,'baseline_readiness':readiness,'macro':macro}

def choose_checkpoints(frames,validation_ids,expected_masks,baselines):
 if len({f['epoch'] for f in frames})!=len(frames):raise ValueError('Duplicate epoch')
 candidates=[score_candidate(f,validation_ids,expected_masks,baselines) for f in frames];valid=[r for r in candidates if r['iou_key'] is not None]
 if not valid:raise ValueError('No numerical-valid complete checkpoint')
 passing=[r for r in valid if r['baseline_readiness']['all_parents_pass']];pool=passing or valid
 primary=min(pool,key=lambda r:r['iou_key']);shadow=min(pool,key=lambda r:r['BER_key'])
 return {'status':'baseline_guard_passed' if passing else 'not_ready','selected_epoch':primary['epoch'],'selected_key':primary['iou_key'],'shadow_BER_epoch':shadow['epoch'],'shadow_BER_key':shadow['BER_key'],'candidate_pool_epochs':[r['epoch'] for r in pool],'pool_kind':'all_parent_guard_passed' if passing else 'numerical_valid_diagnostic_only','passing_epochs':[r['epoch'] for r in passing],'selected_readiness':primary['baseline_readiness'],'shadow_readiness':shadow['baseline_readiness'],'sealed_access_authorized':False}
