"""Independent numerical/custody checks of the frozen-checkpoint diagnosis."""
from pathlib import Path
import ast,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import OUT as SOURCE,REPORT as SOURCE_REPORT,sha,read,write,arrays,load_label
NAME='three_ward_unet_shallow_diagnosis_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def csvrows(name):return list(csv.DictReader((REPORT/name).open()))
def main():
 checks=[]
 def check(name,ok):
  checks.append({'check':name,'pass':bool(ok)})
  if not ok:raise AssertionError(name)
 p=read(SOURCE/'protocol.json');plan=read(OUT/'diagnostic_plan.json');freeze=read(SOURCE/'freeze.json');s=read(REPORT/'summary.json');findings=read(REPORT/'findings.json');z=arrays(SOURCE/'prepared_inputs.npz');pr=arrays(OUT/'diagnostic_predictions.npz');rows=csvrows('event_metrics.csv');strata=csvrows('loss_strata.csv');gr=csvrows('gradient_probes.csv');bias=csvrows('bias_gradient_probes.csv')
 check('original_trial_entire_frozen_delivery_unchanged',all(sha(ROOT/k)==v for k,v in freeze['files_sha256'].items()))
 check('original_input_curves_and_masks_hashes_unchanged',all(sha(ROOT/k)==v for k,v in p['source_hashes'].items()))
 check('diagnostic_source_manifest_unchanged',all(sha(ROOT/k)==v for k,v in plan['source_files_sha256'].items()))
 check('same12train4validation_no_other_ids',len(p['records'])==16 and sum(r['split']=='train' for r in p['records'])==12 and sum(r['split']=='validation' for r in p['records'])==4 and plan['authorized_case_ids']==[r['case_id'] for r in p['records']])
 check('prediction_ids_match_approved_order',np.array_equal(pr['Case ID'],[r['case_id'] for r in p['records']]))
 code=ast.parse((ROOT/'src/analyses/diagnose_three_ward_unet_shallow.py').read_text());bad=[n for n in ast.walk(code) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','train')]
 check('analysis_contains_no_optimizer_step_or_train_mode_call',not bad)
 check('no_new_fit_or_sealed_access_declared',s['new_training_runs']==0 and s['optimizer_steps']==0 and not s['sealed_opened'] and plan['new_fit_count']==0 and plan['optimizer_steps']==0 and not plan['sealed_response_access'])
 area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];bounds=[(-np.inf,.01),(.01,.1),(.1,.3),(.3,1),(1,np.inf)]
 for i,r in enumerate(p['records']):
  d=load_label(r['case_id'],r['split']);mask=d['Exact Label Mask']&eligible;y=d['Peak Mean Depth Lower'].astype('f4');support=mask&(ward>0)
  for checkpoint in ['seed_reconstructed_initial','best_epoch15','last_epoch35']:
   pred=pr[checkpoint][i];check(checkpoint+'_'+r['case_id']+'_valid_prediction_support',pred.shape==(1012,770) and np.isfinite(pred[eligible]).all() and (pred[eligible]>=0).all() and np.isnan(pred[~eligible]).all())
   a=area[support];t=y[support];v=pred[support];err=v-t;wet=t>.1;pwet=v>.1;den=a.sum();mae=np.sum(a*np.abs(err))/den;mse=np.sum(a*(err.astype('f8')**2))/den;tp=a[wet&pwet].sum();union=a[wet|pwet].sum();iou=tp/union if union else 0
   saved=next(q for q in rows if q['checkpoint']==checkpoint and q['case_id']==r['case_id'] and q['scope']=='three_wards')
   check(checkpoint+'_'+r['case_id']+'_reproduce_depth_and_extent',np.isclose(mae,float(saved['mae_m']),atol=1e-10) and np.isclose(mse,float(saved['mse_m2']),atol=1e-8) and np.isclose(iou,float(saved['iou_0.1']),atol=1e-10))
   st=[q for q in strata if q['checkpoint']==checkpoint and q['case_id']==r['case_id']];full=next(q for q in rows if q['checkpoint']==checkpoint and q['case_id']==r['case_id'] and q['scope']=='domain')
   check(checkpoint+'_'+r['case_id']+'_strata_partition_original_exact_support',sum(int(q['cell_count']) for q in st)==int(mask.sum()) and np.isclose(sum(float(q['area_fraction']) for q in st),1.) and all(np.sum(mask&(y>lo)&(y<=hi))==int(q['cell_count']) for q,(lo,hi) in zip(st,bounds)))
   check(checkpoint+'_'+r['case_id']+'_loss_decomposition_adds_up',np.isclose(sum(float(q['mse_contribution_m2']) for q in st),float(full['mse_m2']),atol=1e-8) and np.isclose(sum(float(q['mae_contribution_m']) for q in st),float(full['mae_m']),atol=1e-8))
   b=next(q for q in bias if q['checkpoint']==checkpoint and q['case_id']==r['case_id']);check(checkpoint+'_'+r['case_id']+'_analytic_derivative_strata_add_up',np.isclose(sum(float(q['mse_logit_bias_gradient']) for q in st),float(b['mse_logit_bias_gradient']),atol=1e-10) and np.isclose(sum(float(q['mae_logit_bias_gradient']) for q in st),float(b['mae_logit_bias_gradient']),atol=1e-10))
 check('gradient_probes_training_only_both_frozen_checkpoints',len({(q['checkpoint'],q['case_id']) for q in gr})==24 and all(q['split']=='train' and q['checkpoint'] in ('best_epoch15','last_epoch35') and q['case_id'] in [r['case_id'] for r in p['records'] if r['split']=='train'] and q['probe_only_no_update']=='True' for q in gr))
 old=arrays(SOURCE/'predictions.npz')['Conditional UNet Peak Mean Depth'];check('recomputed_best_reproduces_original_maps',np.nanmax(np.abs(pr['best_epoch15']-old))<1e-5)
 hist=list(csv.DictReader((SOURCE_REPORT/'training_history.csv').open()));scored=[q for q in hist if q['validation_mae_m']];best=min(scored,key=lambda q:float(q['validation_mae_m']));check('historical_selected_epoch15_reproduces_original_rule',int(best['epoch'])==15)
 b=[q for q in strata if q['checkpoint']=='best_epoch15' and q['split']=='train'];total=sum(float(q['mse_contribution_m2']) for q in b);wet=sum(float(q['mse_contribution_m2']) for q in b if q['stratum'] in ('0.10_to_0.30','0.30_to_1.00','above_1.00'))/total;deep=sum(float(q['mse_contribution_m2']) for q in b if q['stratum']=='above_1.00')/total
 check('objective_loss_shares_use_ratio_of_mean_contributions',np.isclose(wet,s['best_train_objective_MSE_share_above0_1m']) and np.isclose(deep,s['best_train_objective_MSE_share_above1m']) and np.isclose(deep,findings['objective_loss_shares']['deep_above_1m']))
 check('incomplete_history_not_claimed_as_exhaustive_checkpoint_search',s['checkpoints_actually_saved']==[15,35] and s['initialization_reconstructed_not_observed'] and bool(s['historical_missing']))
 write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'checks':checks,'new_fit_count':0,'optimizer_steps':0,'source_unchanged':True,'sealed_opened':False});print(json.dumps({'all_checks_passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
