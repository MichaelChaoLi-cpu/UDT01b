"""Read-only diagnosis of the completed first U-Net trial.
No optimizer construction/steps, fitting, forcing changes or sealed response access.
Gradient probes are instantaneous loss derivatives at frozen checkpoints, not
reconstructed historical optimizer traces. Initial state is seed-reconstructed.
"""
from pathlib import Path
import csv,json,sys,time,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import OUT as SOURCE,REPORT as SOURCE_REPORT,sha,read,write,arrays,load_label
from train_three_ward_unet_exact_pilot import ConditionalUNet,metrics,writecsv,synchronize
NAME='three_ward_unet_shallow_diagnosis_v1'
OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
STRATA=[('at_most_0.01',-np.inf,.01),('0.01_to_0.10',.01,.1),('0.10_to_0.30',.1,.3),('0.30_to_1.00',.3,1.),('above_1.00',1.,np.inf)]

def quantiles(values,weights,qs):
 v=np.asarray(values).ravel();w=np.asarray(weights).ravel();ok=np.isfinite(v)&np.isfinite(w)&(w>0);v=v[ok];w=w[ok]
 if not len(v):return [None]*len(qs)
 order=np.argsort(v,kind='stable');v=v[order];w=w[order];c=np.cumsum(w);return [float(v[min(int(np.searchsorted(c,q*c[-1],side='left')),len(v)-1)]) for q in qs]
def group(name):return 'condition' if name.startswith('condition') else '.'.join(name.split('.')[:2]) if name.startswith(('enc.','dec.')) else 'head'
def model_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def norms(vs):return math.sqrt(sum(float((v.detach().cpu().double()**2).sum()) for v in vs))

def main():
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 if (REPORT/'summary.json').exists():raise FileExistsError('Preserve completed diagnosis')
 protocol=read(SOURCE/'protocol.json');frozen=read(SOURCE/'freeze.json');assert all(sha(ROOT/k)==v for k,v in frozen['files_sha256'].items())
 assert all(sha(ROOT/k)==v for k,v in protocol['source_hashes'].items());assert all(sha(ROOT/k)==v for k,v in protocol['implementation_hashes'].items())
 records=protocol['records'];assert len(records)==16;tr=[i for i,r in enumerate(records) if r['split']=='train'];va=[i for i,r in enumerate(records) if r['split']=='validation'];assert len(tr)==12 and len(va)==4
 z=arrays(SOURCE/'prepared_inputs.npz');source_prediction=arrays(SOURCE/'predictions.npz')['Conditional UNet Peak Mean Depth']
 labels=[load_label(r['case_id'],r['split']) for r in records];area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code']
 y=np.array([a['Peak Mean Depth Lower'] for a in labels],dtype='f4');mask=np.array([a['Exact Label Mask'] for a in labels])&eligible[None]
 weights=mask*area[None];weights/=weights.sum((1,2))[:,None,None]
 safe=np.where(mask,y,0).astype('f4')
 torch.set_num_threads(6);torch.manual_seed(protocol['seed']);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 if device.type=='mps':torch.mps.manual_seed(protocol['seed'])
 model=ConditionalUNet().to(device);initial=model_state(model)
 best=torch.load(SOURCE/'best.pt',map_location='cpu',weights_only=False);last=torch.load(SOURCE/'last.pt',map_location='cpu',weights_only=False)
 assert best['epoch']==15 and last['epoch']==35 and last['steps']==420 and best['protocol_sha256']==sha(SOURCE/'protocol.json')
 checkpoints={'seed_reconstructed_initial':initial,'best_epoch15':best['state_dict'],'last_epoch35':last['model']}
 sourcefiles={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'freeze.json',SOURCE/'protocol.json',SOURCE/'normalization.json',SOURCE/'prepared_inputs.npz',SOURCE/'predictions.npz',SOURCE/'best.pt',SOURCE/'last.pt',SOURCE_REPORT/'training_history.csv',SOURCE_REPORT/'completion.json',ROOT/'src/analyses/train_three_ward_unet_exact_pilot.py',ROOT/'src/preprocessing/prepare_three_ward_unet_exact_pilot.py']}
 plan={'torch_version':torch.__version__,'device':str(device),'python_version':sys.version,'gradient_dtype':'float32; norm reduction float64 on CPU','decision':'MILI-D-20261016-003','source_files_sha256':sourcefiles,'authorized_case_ids':[r['case_id'] for r in records],'partitions':{r['case_id']:r['split'] for r in records},'optimizer_steps':0,'new_fit_count':0,'sealed_response_access':False,'mask_change':False,'score_support':'Original target exact AND eligible; paired baseline scores remain in original report. Diagnosis compares checkpoints on identical original exact masks.','checkpoint_scope':['Seed-reconstructed initialization (not an actually saved initial checkpoint)','Saved best epoch15','Saved last epoch35'],'gradient_scope':'Training-only, instantaneous full-field MSE backward at both saved checkpoints, zero optimizer steps; per-stratum analytic derivative of an additive output logit bias for MSE and MAE. Not a historical clipping or gradient record.','strata_bounds':'(lower, upper], wet strictly above0.10m; deep strictly above1m','no_new_operating_threshold':'All trial thresholds preserved; no threshold search, calibration or new selected model.','missing_diagnostics':['No intermediate checkpoints except best and last','No per-step historical loss/gradient/clipping/update log','No per-epoch wet recall/IoU or fixed-checkpoint training scores','No multiple-seed or longer-fit response, so convergence/capacity effects unisolated']}
 write(OUT/'diagnostic_plan.json',plan)
 # Training distribution is event-balanced exactly as in the original loss.
 tv=np.concatenate([y[i][mask[i]] for i in tr]);tw=np.concatenate([weights[i][mask[i]]/len(tr) for i in tr])
 dist={'training_parent_count':12,'event_balanced_depth_mean_m':float(np.sum(tv*tw)/tw.sum()),'event_balanced_depth_quantiles_m':dict(zip(['q25','q50','q75','q90','q95','q99'],quantiles(tv,tw,[.25,.5,.75,.9,.95,.99]))),'training_wet_area_fraction_event_macro':float(np.mean([weights[i][y[i]>.1].sum() for i in tr])),'training_above1m_area_fraction_event_macro':float(np.mean([weights[i][y[i]>1].sum() for i in tr])),'note':'These are descriptive training-label moments, not a newly fitted predictor or alternative selected model.'}
 write(REPORT/'training_distribution.json',dist)
 x=torch.from_numpy(z['Static Input'][None]).to(device);c=torch.from_numpy(z['Forcing Input']).to(device);targets=torch.from_numpy(safe[:,None]).to(device)
 # The production loss normalizes float32 weights; reproduce its rounding for probes.
 production_weights=(mask*area[None]).astype('f4');production_weights/=production_weights.sum((1,2))[:,None,None]
 wt=torch.from_numpy(production_weights[:,None]).to(device)
 eventrows=[];stratarows=[];gradrows=[];biasrows=[];paramrows=[];predictions={};checks=[]
 def check(name,value):
  if not value:raise AssertionError(name)
  checks.append({'check':name,'pass':bool(value)})
 check('original_trial_frozen_files_unchanged_before',all(sha(ROOT/k)==v for k,v in frozen['files_sha256'].items()))
 for cname,state in checkpoints.items():
  model.load_state_dict(state);model.eval();before=model_state(model);pr=[];start=time.perf_counter()
  for i,r in enumerate(records):
   with torch.inference_mode():pp=model(x,c[i:i+1])[0,0].cpu().numpy().copy()
   check(cname+'_'+r['case_id']+'_finite_nonnegative',np.isfinite(pp[eligible]).all() and (pp[eligible]>=0).all());pp[~eligible]=np.nan;pr.append(pp)
   for scope,region in [('domain',np.ones_like(eligible)),('three_wards',ward>0),('Chiyoda',ward==1),('Chuo',ward==2),('Minato',ward==3)]:
    support=mask[i]&region;v=metrics(y[i],pp,area,support);w=area[support];t=y[i,support];p=pp[support];wet=t>.1
    q=quantiles(p,w,[.5,.9,.99,1]);qw=quantiles(p[wet],w[wet],[.5,.9,.99])
    eventrows.append({'checkpoint':cname,'case_id':r['case_id'],'split':r['split'],'scope':scope,'mse_m2':v['rmse_m']**2,'prediction_mean_m':float(np.sum(w*p)/w.sum()),'reference_mean_m':float(np.sum(w*t)/w.sum()),'prediction_q50_m':q[0],'prediction_q90_m':q[1],'prediction_q99_m':q[2],'prediction_max_m':q[3],'prediction_on_reference_wet_q50_m':qw[0],'prediction_on_reference_wet_q90_m':qw[1],'prediction_on_reference_wet_q99_m':qw[2],**v})
   # Objective decomposition on the actual whole-domain training/evaluation support.
   m=mask[i];w=weights[i,m];t=y[i,m].astype('f8');p=pp[m].astype('f8');e=p-t;gate=-np.expm1(-p);sq=e*e;ab=np.abs(e);total_mse=float(np.sum(w*sq));total_mae=float(np.sum(w*ab));mse_g=2*w*e*gate;mae_g=w*np.sign(e)*gate
   for name,lo,hi in STRATA:
    k=(t>lo)&(t<=hi);mass=float(w[k].sum());sse=float(np.sum(w[k]*sq[k]));sae=float(np.sum(w[k]*ab[k]));g2=float(mse_g[k].sum());g1=float(mae_g[k].sum())
    stratarows.append({'checkpoint':cname,'case_id':r['case_id'],'split':r['split'],'stratum':name,'cell_count':int(k.sum()),'area_fraction':mass,'area_km2':float(area[m][k].sum()/1e6),'mse_contribution_m2':sse,'mae_contribution_m':sae,'fraction_of_mse':sse/total_mse if total_mse else None,'fraction_of_mae':sae/total_mae if total_mae else None,'within_stratum_mae_m':sae/mass if mass else None,'mse_logit_bias_gradient':g2,'mae_logit_bias_gradient':g1,'mse_gradient_L1':float(np.abs(mse_g[k]).sum()),'mae_gradient_L1':float(np.abs(mae_g[k]).sum())})
   biasrows.append({'checkpoint':cname,'case_id':r['case_id'],'split':r['split'],'mse_logit_bias_gradient':float(mse_g.sum()),'mae_logit_bias_gradient':float(mae_g.sum()),'softplus_derivative_mean':float(np.sum(w*gate)),'softplus_derivative_on_wet_mean':float(np.sum(w[t>.1]*gate[t>.1])/w[t>.1].sum()),'MSE_wants_uniform_logit_raise':bool(mse_g.sum()<0),'MAE_wants_uniform_logit_lower':bool(mae_g.sum()>0)})
   if cname!='seed_reconstructed_initial' and i in tr:
    # Pure backward probe: do not create/use any optimizer and do not change state.
    model.zero_grad(set_to_none=True);out=model(x,c[i:i+1]);loss=((out-targets[i:i+1]).square()*wt[i:i+1]).sum();loss.backward()
    gs={}
    for n,param in model.named_parameters():
     check(cname+'_'+r['case_id']+'_'+n+'_finite_gradient',param.grad is not None and bool(torch.isfinite(param.grad).all()))
     gs.setdefault(group(n),[]).append(param.grad)
    total=norms([v.grad for v in model.parameters()]);bias=float(model.head.bias.grad.detach().cpu()[0])
    check(cname+'_'+r['case_id']+'_analytic_bias_gradient',abs(bias-float(mse_g.sum()))<max(1e-6,abs(bias)*1e-4))
    for name,g in gs.items():gradrows.append({'checkpoint':cname,'case_id':r['case_id'],'split':'train','module':name,'gradient_l2':norms(g),'total_gradient_l2':total,'would_exceed_original_clip10_at_probe':total>10.,'head_bias_MSE_gradient_autograd':bias,'probe_only_no_update':True})
   model.zero_grad(set_to_none=True)
  predictions[cname]=np.array(pr);after=model_state(model);check(cname+'_zero_parameter_or_buffer_updates',all(torch.equal(before[k],after[k]) for k in before))
  print(json.dumps({'checkpoint':cname,'scored':len(pr),'seconds':time.perf_counter()-start}),flush=True)
  for module in sorted({group(k) for k in state}):
   names=[k for k in state if group(k)==module];vs=[state[k] for k in names];delta=[state[k].cpu()-initial[k].cpu() for k in names]
   ini=norms([initial[k] for k in names]);paramrows.append({'checkpoint':cname,'module':module,'parameter_l2':norms(vs),'change_from_seed_reconstructed_initial_l2':norms(delta),'relative_change':norms(delta)/ini if ini else None})
 check('best_checkpoint_reproduces_frozen_prediction',np.nanmax(np.abs(predictions['best_epoch15']-source_prediction))<1e-5)
 np.savez_compressed(OUT/'diagnostic_predictions.npz',**{k:v for k,v in predictions.items()},**{'Case ID':np.array([r['case_id'] for r in records]),'Eligible Ground Mask':eligible})
 writecsv(REPORT/'event_metrics.csv',eventrows);writecsv(REPORT/'loss_strata.csv',stratarows);writecsv(REPORT/'gradient_probes.csv',gradrows);writecsv(REPORT/'bias_gradient_probes.csv',biasrows);writecsv(REPORT/'parameter_changes.csv',paramrows)
 macro=[]
 for cname in checkpoints:
  for split in ['train','validation']:
   for scope in ['domain','three_wards','Chiyoda','Chuo','Minato']:
    rr=[r for r in eventrows if r['checkpoint']==cname and r['split']==split and r['scope']==scope];keys=[k for k in rr[0] if k not in ['checkpoint','case_id','split','scope']]
    sums={};counts={}
    for key in keys:
     vv=[r[key] for r in rr if r[key] is not None];sums[key]=float(np.mean(vv)) if vv else None;counts['defined_'+key]=len(vv)
    macro.append({'checkpoint':cname,'split':split,'scope':scope,'parents':len(rr),**sums,**counts})
 writecsv(REPORT/'macro_metrics.csv',macro)
 stratum_macro=[]
 for cname in checkpoints:
  for split in ['train','validation']:
   for name,_,_ in STRATA:
    rr=[r for r in stratarows if r['checkpoint']==cname and r['split']==split and r['stratum']==name];keys=[k for k in rr[0] if k not in ['checkpoint','case_id','split','stratum']]
    agg={k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in keys}
    stratum_macro.append({'checkpoint':cname,'split':split,'stratum':name,**agg})
 writecsv(REPORT/'loss_strata_macro.csv',stratum_macro)
 # Historical online epoch loss is a different estimand from fixed-checkpoint scores.
 hist=list(csv.DictReader((SOURCE_REPORT/'training_history.csv').open()));steps=[int(r['steps']) for r in hist]
 check('exactly35_epochs_and420_original_updates',len(hist)==35 and steps==list(range(12,421,12)) and last['steps']==420)
 check('last_history_matches_original_log',last['history'][-1]['epoch']==35 and abs(last['history'][-1]['train_mse_m2']-float(hist[-1]['train_mse_m2']))<1e-12)
 for cname,epoch in [('best_epoch15',15),('last_epoch35',35)]:
  v=next(r for r in macro if r['checkpoint']==cname and r['split']=='validation' and r['scope']=='domain')['mae_m'];old=float(hist[epoch-1]['validation_mae_m']);check(cname+'_validation_clock_and_metric_reproduce',abs(v-old)<1e-7)
 states=last['optimizer']['state'];op={ 'parameter_state_count':len(states),'unique_step_counts':sorted({int(v['step']) for v in states.values()}),'all_moments_finite':all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() for s in states.values()),'group_learning_rates':[g['lr'] for g in last['optimizer']['param_groups']],'historical_gradient_norm_available':False,'historical_clipping_frequency_available':False,'historical_update_norm_available':False}
 check('optimizer_state_has420steps_and_finite_moments',op['unique_step_counts']==[420] and op['all_moments_finite']);write(REPORT/'optimizer_state.json',op)
 check('source_files_unchanged_after_probes',all(sha(ROOT/k)==v for k,v in sourcefiles.items()))
 check('frozen_input_curves_and_labels_unchanged_after',all(sha(ROOT/k)==v for k,v in protocol['source_hashes'].items()))
 bestmetrics={s:next(r for r in macro if r['checkpoint']=='best_epoch15' and r['split']==s and r['scope']=='three_wards') for s in ['train','validation']};lastmetrics={s:next(r for r in macro if r['checkpoint']=='last_epoch35' and r['split']==s and r['scope']=='three_wards') for s in ['train','validation']}
 bstat=[r for r in stratum_macro if r['checkpoint']=='best_epoch15' and r['split']=='train'];bgrad=[r for r in biasrows if r['checkpoint']=='best_epoch15' and r['split']=='train'];gprobe=[r for r in gradrows if r['module']=='head']
 summary={'loss_share_definition':'Existing MSE share fields are the arithmetic mean of within-parent shares, not the ratio of mean contributions; objective contribution ratios are also provided.', 'best_train_objective_MSE_share_above0_1m':sum(r['mse_contribution_m2'] for r in bstat if r['stratum'] in ['0.10_to_0.30','0.30_to_1.00','above_1.00'])/sum(r['mse_contribution_m2'] for r in bstat),'best_train_objective_MSE_share_above1m':next(r['mse_contribution_m2'] for r in bstat if r['stratum']=='above_1.00')/sum(r['mse_contribution_m2'] for r in bstat),'decision':'MILI-D-20261016-003','new_training_runs':0,'optimizer_steps':0,'sealed_opened':False,'original_partitions_and_masks_preserved':True,'checkpoints_actually_saved':[15,35],'initialization_reconstructed_not_observed':True,'training_distribution':dist,'best_metrics':bestmetrics,'last_metrics':lastmetrics,'best_train_MSE_share_above0_1m':float(sum(r['fraction_of_mse'] for r in bstat if r['stratum'] in ['0.10_to_0.30','0.30_to_1.00','above_1.00'])),'best_train_MSE_share_above1m':next(r['fraction_of_mse'] for r in bstat if r['stratum']=='above_1.00'),'best_train_area_share_above1m':dist['training_above1m_area_fraction_event_macro'],'best_train_bias_objective_conflict_count':sum(r['MSE_wants_uniform_logit_raise'] and r['MAE_wants_uniform_logit_lower'] for r in bgrad),'gradient_probe_total_norm_range':[min(r['total_gradient_l2'] for r in gprobe),max(r['total_gradient_l2'] for r in gprobe)],'current_probes_above_clip10':sum(r['would_exceed_original_clip10_at_probe'] for r in gprobe),'original_online_train_MSE_first':float(hist[0]['train_mse_m2']),'original_online_train_MSE_last':float(hist[-1]['train_mse_m2']),'historical_missing':plan['missing_diagnostics'],'claim_limits':['Checkpoint comparison covers saved epoch15 and epoch35 only, not other validation epochs.','No causal isolation of loss weighting, model capacity or training budget without a new controlled experiment.','Small validation set reused for diagnostics; do not treat a retrospective alternative checkpoint as a newly validated model.','Counterfactual metric gradients are diagnostic only, not a newly applied selection policy.']}
 write(REPORT/'summary.json',summary);write(REPORT/'validation.json',{'all_checks_passed':True,'checks':checks,'source_freeze_hash':sha(SOURCE/'freeze.json'),'new_fit_count':0,'optimizer_steps':0,'sealed_opened':False})
 print(json.dumps({k:v for k,v in summary.items() if k not in ['best_metrics','last_metrics']},indent=2),flush=True)
if __name__=='__main__':main()
