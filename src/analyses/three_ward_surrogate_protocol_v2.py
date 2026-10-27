"""Loss, metrics, checkpoint and telemetry primitives. Contains no fitting loop.
Original inputs/labels are immutable; these are proposed development rules.
"""
from pathlib import Path
import json,math,time,hashlib
import numpy as np
import torch

LOSS_WEIGHTS={'nonwet':.5,'ordinary_wet':.4,'deep':.1}
THRESHOLDS=(.05,.1,.3,.5)

def _loss_values(pred,target,area,mask):
 if not (pred.shape==target.shape==area.shape==mask.shape):raise ValueError('Shape mismatch')
 if mask.dtype!=torch.bool:raise TypeError('Mask must be boolean')
 if not bool(mask.any()):raise ValueError('Empty exact support')
 p=pred[mask];y=target[mask];a=area[mask]
 if not bool(torch.isfinite(p).all() and torch.isfinite(y).all() and torch.isfinite(a).all() and (a>0).all() and (y>=0).all()):raise ValueError('Invalid prediction/target/area on original exact support; do not silently drop cells')
 return p,y,a

def exact_loss(pred,target,area,mask,kind='mse'):
 """Area-weighted per-event loss on exact support; returns differentiable loss.
2*Huber(delta=.10m) matches MSE for residual magnitude<=.10m, linear tails.
Fixed nonwet/ordinary/deep mixture weights .5/.4/.1, normalized over present
strata. Metadata missing/empty strata are explicit. No target replacement.
"""
 p,y,a=_loss_values(pred,target,area,mask);err=p-y
 strata={'nonwet':y<=.1,'ordinary_wet':(y>.1)&(y<=1.),'deep':y>1.}
 if kind=='mse':cell=err.square()
 elif kind=='stratified_huber':
  e=err.abs();cell=torch.where(e<=.1,err.square(),.2*(e-.05))
 else:raise ValueError(kind)
 entries={};active=sum(LOSS_WEIGHTS[n] for n,m in strata.items() if bool(m.any()));total=cell.sum()*0
 for name,m in strata.items():
  present=bool(m.any());mass=a[m].sum();fraction=float((mass/a.sum()).detach().cpu()) if present else 0.
  coeff=(LOSS_WEIGHTS[name]/active if kind=='stratified_huber' else fraction) if present else 0.
  value=(cell[m]*a[m]).sum()/mass if present else None
  if present:total=total+coeff*value
  entries[name]={'present':present,'area_fraction':fraction,'mixture_weight':coeff,'mean_loss_m2':float(value.detach().cpu()) if present else None,'contribution_m2':float((coeff*value).detach().cpu()) if present else 0.}
 return total,{'kind':kind,'value_m2':float(total.detach().cpu()),'strata':entries,'empty_stratum_rule':'Renormalize fixed weights over present strata; never fabricate a target.'}

def evaluate_depth(pred,target,area,mask):
 """No silent removal of invalid active cells; empty metrics are undefined."""
 if not (pred.shape==target.shape==area.shape==mask.shape):raise ValueError('Shape mismatch')
 if mask.dtype!=np.bool_:raise TypeError('Mask must be boolean')
 if not mask.any():return {'support_km2':0.,'mae_m':None,'wet_mae_m':None,'balanced_extent_error':None,'selection_valid':False}
 pn=np.asarray(pred[mask]);yn=np.asarray(target[mask]);p=pn.astype('f8');y=yn.astype('f8');a=np.asarray(area[mask],dtype='f8')
 if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(a).all() and (p>=0).all() and (y>=0).all() and (a>0).all()):raise ValueError('Invalid active support: refuse a coverage-improving omission')
 den=a.sum();e=p-y;wet=yn>.1;nonwet=~wet
 out={'support_sha256':hashlib.sha256(np.ascontiguousarray(mask,dtype='u1').tobytes()).hexdigest(),'support_km2':float(den/1e6),'mae_m':float(np.sum(a*np.abs(e))/den),'rmse_m':float(np.sqrt(np.sum(a*e**2)/den)),'bias_m':float(np.sum(a*e)/den),'selection_valid':bool(wet.any() and nonwet.any())}
 for name,m in [('nonwet',nonwet),('ordinary_wet',wet&(yn<=1)),('deep',yn>1),('wet',wet)]:
  out[name+'_area_km2']=float(a[m].sum()/1e6);out[name+'_mae_m']=float(np.sum(a[m]*np.abs(e[m]))/a[m].sum()) if m.any() else None
 for t in THRESHOLDS:
  w=yn>t;v=pn>t;tp=float(a[w&v].sum());fp=float(a[~w&v].sum());fn=float(a[w&~v].sum());tn=float(a[~w&~v].sum());k=f'{t:g}'
  out.update({f'tp_km2_{k}':tp/1e6,f'false_km2_{k}':fp/1e6,f'missed_km2_{k}':fn/1e6,f'tn_km2_{k}':tn/1e6,f'iou_{k}':tp/(tp+fp+fn) if tp+fp+fn else None,f'recall_{k}':tp/(tp+fn) if tp+fn else None,f'precision_{k}':tp/(tp+fp) if tp+fp else None,f'fnr_{k}':fn/(tp+fn) if tp+fn else None,f'fpr_{k}':fp/(fp+tn) if fp+tn else None,f'false_discovery_{k}':fp/(tp+fp) if tp+fp else None})
 out['balanced_extent_error']=.5*(out['fnr_0.1']+out['fpr_0.1']) if out['selection_valid'] else None
 return out

def macro_events(items):
 """Parents have equal weight, with metric-specific defined-event counts."""
 if not items:raise ValueError('No events')
 if len({r['case_id'] for r in items})!=len(items):raise ValueError('Duplicate parent observations')
 values={};counts={};meta={'case_id','parent_id','split','scope','method','support','checkpoint'}
 for key in set.intersection(*(set(r) for r in items))-meta:
  vv=[r[key] for r in items if r[key] is not None]
  if vv and all(isinstance(v,(float,int,bool,np.floating)) for v in vv):values[key]=float(np.mean(vv));counts[key]=len(vv)
  elif not vv:values[key]=None;counts[key]=0
 return {'parents':len(items),'values':values,'defined_parents':counts,'all_selection_valid':all(r.get('selection_valid',False) for r in items)}

def checkpoint_key(events,epoch,expected_ids,expected_masks):
 """Fixed three-ward paired support. Lower is better. No implicit fallback."""
 if set(r['case_id'] for r in events)!=set(expected_ids) or len(events)!=len(expected_ids):raise ValueError('Incomplete or duplicate validation parents')
 if any(r.get('split')!='validation' or r.get('scope')!='three_wards' or r.get('support')!='paired_exact' for r in events):raise ValueError('Wrong checkpoint evaluation support or split')
 if any(r.get('support_sha256')!=expected_masks.get(r['case_id']) for r in events):raise ValueError('Checkpoint mask differs from frozen comparison support')
 agg=macro_events(events)
 if not agg['all_selection_valid']:return None
 v=agg['values'];keys=['balanced_extent_error','wet_mae_m','mae_m']
 if any(v.get(k) is None or not math.isfinite(v[k]) for k in keys):return None
 return tuple(round(v[k],6) for k in keys)+(int(epoch),)

def baseline_readiness(candidate,dry,retrieval):
 """Development baseline guard, separate from checkpoint ranking.
Every validation parent must satisfy the four comparisons. This is not physical
accuracy certification nor authority to open a test. No tolerance is learned.
"""
 if not (set(candidate)==set(dry)==set(retrieval)):raise ValueError('Different event sets')
 rows=[]
 for cid,c in candidate.items():
  d=dry[cid];r=retrieval[cid]
  if not (c.get('support_km2')==d.get('support_km2')==r.get('support_km2') and c.get('support_sha256')==d.get('support_sha256')==r.get('support_sha256')):raise ValueError('Unpaired baseline support')
  finite=all(c.get(k) is not None and r.get(k) is not None and math.isfinite(c[k]) and math.isfinite(r[k]) for k in ['balanced_extent_error','mae_m','recall_0.1','false_km2_0.1'])
  checks={'balanced_error_no_worse_than_retrieval':finite and c['balanced_extent_error']<=r['balanced_extent_error']+1e-6,'MAE_no_worse_than_both_baselines':finite and c['mae_m']<=min(d['mae_m'],r['mae_m'])+1e-6,'recall_no_worse_than_retrieval':finite and c['recall_0.1']>=r['recall_0.1']-1e-6,'false_area_no_worse_than_retrieval':finite and c['false_km2_0.1']<=r['false_km2_0.1']+1e-6}
  rows.append({'case_id':cid,'pass':all(checks.values()),'checks':checks})
 return {'all_parents_pass':bool(rows) and all(r['pass'] for r in rows),'events':rows,'test_access_authorized':False,'meaning':'Proposed development baseline guard only; failed candidates remain diagnostic, never silently pass.'}

def _group(n):return 'condition' if n.startswith('condition') else '.'.join(n.split('.')[:2]) if n.startswith(('enc.','dec.')) else 'head'
def parameter_snapshot(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
def tensor_norm(v):return math.sqrt(sum(float((x.detach().cpu().double()**2).sum()) for x in v))
def gradient_summary(model):
 groups={};missing=[];finite=True
 for name,p in model.named_parameters():
  if p.grad is None:missing.append(name);continue
  g=p.grad.detach().cpu();finite=finite and bool(torch.isfinite(g).all());groups.setdefault(_group(name),[]).append(g)
 return {'finite':finite,'total_norm':tensor_norm([g for gg in groups.values() for g in gg]),'module_norms':{n:tensor_norm(v) for n,v in groups.items()},'missing_gradient_parameters':missing}
def parameter_change(before,after):
 if before.keys()!=after.keys():raise ValueError('Parameter set changed')
 groups={}
 for name in before:groups.setdefault(_group(name),[]).append(name)
 return {k:{'update_norm':tensor_norm([after[n]-before[n] for n in names]),'before_norm':tensor_norm([before[n] for n in names])} for k,names in groups.items()}
def optimizer_clock(optimizer):
 state=optimizer.state_dict();steps=sorted({int(v['step']) for v in state['state'].values() if 'step' in v})
 return {'steps':steps,'learning_rates':[float(v['lr']) for v in state['param_groups']]}

class OptimizationObserver:
 """Call around externally authorized backward/clip/update operations.
This observer never calls backward, clip, step or changes parameters itself.
Logs written only to its supplied new-run file; immutable sources stay separate.
"""
 def __init__(self,path,protocol_sha,clip_limit=10.):
  self.path=Path(path);self.protocol_sha=protocol_sha;self.clip_limit=clip_limit;self.phase='idle'
 def begin(self,model,optimizer,context):
  if self.phase!='idle':raise RuntimeError('Unfinished step')
  required=['arm','epoch','case_id','parent_id','family_id','forcing_sha256','split','seed','mask_sha256','loss_kind','record_kind']
  if any(k not in context for k in required) or context['split']!='train':raise ValueError('Missing context or nontraining event')
  self.context=dict(context);self.before=parameter_snapshot(model);self.before_clock=optimizer_clock(optimizer);self.start=time.perf_counter();self.phase='begun'
 def after_backward(self,model,loss_payload):
  if self.phase!='begun':raise RuntimeError('Backward telemetry out of order')
  self.pre=gradient_summary(model)
  if not self.pre['finite']:raise FloatingPointError('Nonfinite gradient: stop, retain diagnostic')
  self.loss_payload=loss_payload;self.phase='backward'
 def after_clip(self,model):
  if self.phase!='backward':raise RuntimeError('Clip telemetry out of order')
  self.post=gradient_summary(model)
  if not self.post['finite'] or self.post['total_norm']>self.clip_limit+1e-4:raise FloatingPointError('Gradient not finite/bounded after declared clipping')
  self.phase='clipped'
 def finish(self,model,optimizer,expected_update=True):
  if self.phase!='clipped':raise RuntimeError('Update telemetry out of order')
  after=parameter_snapshot(model)
  if not all(bool(torch.isfinite(v).all()) for v in after.values()):raise FloatingPointError('Nonfinite parameter after update: stop')
  change=parameter_change(self.before,after);clock=optimizer_clock(optimizer)
  if (expected_update and self.context['record_kind']!='training_update') or (not expected_update and self.context['record_kind']!='interface_probe'):raise ValueError('Update kind disagrees with actual operation')
  if expected_update:
   old=max(self.before_clock['steps'],default=0)
   if clock['steps']!=[old+1]:raise RuntimeError('Optimizer did not advance exactly one step')
  elif clock!=self.before_clock or any(q['update_norm']!=0 for q in change.values()):raise RuntimeError('Read-only probe changed parameters or optimizer')
  result={'schema':'optimization-step-v2','protocol_sha256':self.protocol_sha,**self.context,'loss':self.loss_payload,'gradient_pre_clip':self.pre,'gradient_post_clip':self.post,'clip_limit':self.clip_limit,'clipping_required':self.pre['total_norm']>self.clip_limit,'parameter_change':change,'optimizer_before':self.before_clock,'optimizer_after':clock,'optimizer_update_performed':expected_update,'elapsed_seconds':time.perf_counter()-self.start}
  self.path.parent.mkdir(parents=True,exist_ok=True)
  with self.path.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
  self.phase='idle';self.before=None;return result

REQUIRED_SCOPES=('domain','three_wards','Chiyoda','Chuo','Minato','cross_ward')
def write_epoch_record(path,protocol_sha,context,events,expected_cases,expected_masks,baselines):
 """Fixed-model score/log interface. Requires complete split/support coverage.
The caller measures real optimization, evaluation and checkpoint I/O clocks.
An interface probe must identify itself and cannot count as a training epoch.
 """
 needed=['arm','epoch','record_kind','optimizer_steps_before_eval','optimizer_steps_after_eval','model_fingerprint_before_eval','model_fingerprint_after_eval','checkpoint_path','checkpoint_sha256','cumulative_training_seconds','cumulative_evaluation_seconds','cumulative_checkpoint_io_seconds']
 if any(k not in context for k in needed):raise ValueError('Incomplete epoch provenance')
 if context['record_kind']=='training_epoch':
  if context['optimizer_steps_before_eval']!=12*int(context['epoch']):raise ValueError('Incomplete epoch optimizer clock')
  summary=context.get('step_summary',{})
  if summary.get('optimizer_updates')!=12 or summary.get('parents')!=12 or summary.get('record_kind')!='training_update':raise ValueError('Missing true training-step summary')
  if summary.get('optimizer_end_step')!=context['optimizer_steps_before_eval'] or summary.get('epoch')!=context['epoch'] or summary.get('arm')!=context['arm'] or summary.get('protocol_sha256')!=protocol_sha:raise ValueError('Step summary is not this checkpoint epoch')
 elif context['record_kind']=='interface_probe':
  if context['optimizer_steps_before_eval']!=0:raise ValueError('Interface probe cannot claim training updates')
 else:raise ValueError('Unknown epoch record kind')
 if context['optimizer_steps_before_eval']!=context['optimizer_steps_after_eval'] or context['model_fingerprint_before_eval']!=context['model_fingerprint_after_eval']:raise ValueError('Model updated during fixed-checkpoint evaluation')
 pp=Path(context['checkpoint_path'])
 if hashlib.sha256(pp.read_bytes()).hexdigest()!=context['checkpoint_sha256']:raise ValueError('Checkpoint binding mismatch')
 actual=[(r['case_id'],r.get('split'),r.get('scope'),r.get('support')) for r in events]
 required={(cid,split,scope,support) for cid,split in expected_cases.items() for scope in REQUIRED_SCOPES for support in ('paired_exact','own_exact')}
 if len(actual)!=len(set(actual)) or set(actual)!=required:raise ValueError('Incomplete/duplicate fixed-checkpoint metrics')
 fixed=[r for r in events if r['scope']=='domain' and r['support']=='own_exact']
 if any(not isinstance(r.get('fixed_model_loss_m2'),(float,int)) or not math.isfinite(r['fixed_model_loss_m2']) or r.get('loss_kind')!=context.get('loss_kind') for r in fixed):raise ValueError('Missing fixed-model per-event objective, separate from online loss')
 validation=[r for r in events if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact']
 valids=[cid for cid,split in expected_cases.items() if split=='validation'];key=checkpoint_key(validation,context['epoch'],valids,expected_masks)
 ready=baseline_readiness({r['case_id']:r for r in validation},baselines['dry'],baselines['retrieval'])
 macros=[]
 for split in ('train','validation'):
  for scope in REQUIRED_SCOPES:
   for support in ('paired_exact','own_exact'):
    group=[r for r in events if r['split']==split and r['scope']==scope and r['support']==support]
    macros.append({'split':split,'scope':scope,'support':support,**macro_events(group)})
 payload={'schema':'fixed-checkpoint-v2','protocol_sha256':protocol_sha,**context,'event_metrics':events,'parent_macro_metrics':macros,'selection_key':key,'baseline_readiness':ready,'test_access_authorized':False}
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('a') as f:f.write(json.dumps(payload,allow_nan=False)+'\n')
 return payload

def summarize_step_records(records,expected_case_ids,record_kind='training_update'):
 """Aggregate a complete epoch without confusing probes with training updates."""
 if len(records)!=len(expected_case_ids) or {r['case_id'] for r in records}!=set(expected_case_ids):raise ValueError('Incomplete/duplicate epoch exposure')
 if len({(r['arm'],r['epoch'],r['protocol_sha256']) for r in records})!=1:raise ValueError('Mixed arm/epoch/protocol')
 if record_kind not in ('training_update','interface_probe'):raise ValueError('Unknown step record kind')
 for r in records:
  if r['record_kind']!=record_kind or r['optimizer_update_performed']!=(record_kind=='training_update'):raise ValueError('Probe/update custody mismatch')
 if record_kind=='training_update':
  clocks=[]
  for r in records:
   old=max(r['optimizer_before']['steps'],default=0);after=r['optimizer_after']['steps']
   if after!=[old+1]:raise ValueError('Step clock is not a single increment')
   clocks.append((old,old+1))
  if any(clocks[i+1][0]!=clocks[i][1] for i in range(len(clocks)-1)):raise ValueError('Noncontiguous optimizer clock')
 def distribution(v):
  return {'minimum':float(np.min(v)),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'maximum':float(np.max(v))}
 summaries={'arm':records[0]['arm'],'epoch':records[0]['epoch'],'protocol_sha256':records[0]['protocol_sha256'],'optimizer_start_step':clocks[0][0] if record_kind=='training_update' else None,'optimizer_end_step':clocks[-1][1] if record_kind=='training_update' else None,'online_preupdate_loss_distribution_m2':distribution([r['loss']['value_m2'] for r in records]),'record_kind':record_kind,'parents':len(records),'optimizer_updates':len(records) if record_kind=='training_update' else 0,'online_preupdate_mean_loss_m2':float(np.mean([r['loss']['value_m2'] for r in records])),'gradient_preclip_norm':distribution([r['gradient_pre_clip']['total_norm'] for r in records]),'gradient_postclip_norm':distribution([r['gradient_post_clip']['total_norm'] for r in records]),'clipping_count':sum(r['clipping_required'] for r in records),'timing_scope':'Sum of observer-covered step/probe elapsed; separate fixed evaluation and checkpoint I/O clocks','observer_elapsed_seconds':float(sum(r['elapsed_seconds'] for r in records)),'module_statistics':{}}
 strata=set().union(*(set(r['loss'].get('strata',{})) for r in records))
 summaries['mean_stratum_loss_contribution_m2']={k:float(np.mean([r['loss']['strata'][k]['contribution_m2'] for r in records if k in r['loss'].get('strata',{})])) for k in strata}
 modules=set.intersection(*(set(r['parameter_change']) for r in records))
 for name in sorted(modules):
  summaries['module_statistics'][name]={'update_norm':distribution([r['parameter_change'][name]['update_norm'] for r in records]),'gradient_preclip_norm':distribution([r['gradient_pre_clip']['module_norms'].get(name,0.) for r in records])}
 return summaries
