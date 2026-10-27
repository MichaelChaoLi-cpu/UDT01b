"""Independent saved-field reductions and scope checks; no neural training/inference."""
from pathlib import Path
import sys,json,ast
from html.parser import HTMLParser
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,arrays,load_label
from load_three_ward_minimum_scenarios import find_case
NAME='three_ward_two_parent_alignment_condition_audit_v1';REPORT=ROOT/'data/exp/estimation'/NAME;OUT=ROOT/'data/processed'/NAME
IDS=['train-01','train-04']
def main():
 import csv
 checks=[]
 def check(n,ok):
  if not ok:raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 s=read(REPORT/'summary.json');p=read(REPORT/'provenance.json');z=arrays(ROOT/'data/processed/three_ward_unet_exact_pilot_v1/prepared_inputs.npz');d=arrays(OUT/'frozen_diagnostic_fields.npz');labels=[load_label(c,'train') for c in IDS];y=np.array([l['Peak Mean Depth Lower'].astype('f4') for l in labels]);area=z['Ground Area'];ward=z['Ward Code'];common=(labels[0]['Exact Label Mask']&labels[1]['Exact Label Mask']&z['Eligible Ground Mask']&(ward>0));regions={'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3}
 check('only_two_original_training_parents',p['label_allowlist']==IDS and all(r['case_id'] in IDS for r in p['response_access_log']) and all(find_case(c)['split']=='train' for c in IDS))
 check('no_updates_fits_simulation_or_heldout_responses',p['optimizer_updates']==p['new_fits']==p['new_hydraulic_runs']==0 and not p['validation_response_access'] and not p['sealed_response_access'])
 check('original_formal_roles_masks_normalization_retained',p['formal_split_preserved'] and p['quality_masks_preserved'] and np.array_equal(d['Common Exact Mask'],common))
 check('audit_script_bound',sha(ROOT/'src/analyses/audit_three_ward_two_parent_alignment_v1.py')==p['script_sha256'])
 check('audit_checks_all_pass',read(REPORT/'checks.json')['all_checks_passed'] and all(c['pass'] for c in read(REPORT/'checks.json')['checks']))
 tree=ast.parse((ROOT/'src/analyses/audit_three_ward_two_parent_alignment_v1.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 check('audit_has_no_optimizer_or_step_call',not any((isinstance(n.func,ast.Attribute) and n.func.attr in ['step','Adam','AdamW','SGD']) for n in calls))
 check('source_protection_hashes_hold',all(sha(ROOT/k)==h for k,h in p['source_sha256'].items()))
 check('initial_predictions_condition_independent',np.array_equal(d['Prediction Step 0'][0,common],d['Prediction Step 0'][1,common]))
 check('trained_condition_response_not_collapsed',not np.array_equal(d['Prediction Step 480'][0,common],d['Prediction Step 480'][1,common]))
 manifest=read(ROOT/'data/processed/three_ward_reference_setup_pilot_v1/pilot_manifest.json');orig=read(ROOT/'data/processed/three_ward_unet_exact_pilot_v1/protocol.json')
 for cid in IDS:
  r=next(r for r in manifest['runs'] if r['case_id']==cid);rec=next(r for r in orig['records'] if r['case_id']==cid);f=ROOT/r['folder'];check(cid+'_native_run_parent_forcing_binding',r['original_forcing_sha256']==rec['forcing_sha256'] and r['parent_id']==rec['parent_id'] and r['family_id']==rec['family_id'] and r['split']=='train')
  check(cid+'_native_input_hashes_unchanged',all(sha(f/k)==h for k,h in r['input_sha256'].items()))
  check(cid+'_independent_full_domain_bounds_within_roundoff',next(r for r in read(REPORT/'alignment.json')['depth_reconstruction'] if r['case_id']==cid)['max_bound_error_m']<2e-6)
 cross=list(csv.DictReader((REPORT/'cross_condition_target_metrics.csv').open()));diff=list(csv.DictReader((REPORT/'field_difference_metrics.csv').open()));coh=list(csv.DictReader((REPORT/'reference_wetness_cohorts.csv').open()));decomp=read(REPORT/'paired_error_decomposition.json')
 for step in [0,440,480]:
  pred=d['Prediction Step '+str(step)]
  check(f'step{step}_finite_on_eligible_and_nan_outside',np.isfinite(pred[:,z['Eligible Ground Mask']]).all() and np.isnan(pred[:,~z['Eligible Ground Mask']]).all())
  for scope,reg in regions.items():
   m=common&reg;a=area[m];dy=y[1,m].astype('f8')-y[0,m];dp=pred[1,m].astype('f8')-pred[0,m];dd=next(r for r in diff if int(r['step'])==step and r['scope']==scope)
   check(f'{step}_{scope}_difference_MAE_independent',abs(float(dd['difference_MAE_m'])-np.sum(a*abs(dp-dy))/a.sum())<1e-10)
   for i,cid in enumerate(IDS):
    for j,target in enumerate(IDS):
     r=next(r for r in cross if int(r['step'])==step and r['scope']==scope and r['condition_case']==cid and r['target_case']==target);t=y[j,m];v=pred[i,m];w=t>.1;wet=v>.1;tp=a[w&wet].sum();fp=a[~w&wet].sum();fn=a[w&~wet].sum();iou=tp/(tp+fp+fn);mae=np.sum(a*abs(v.astype('f8')-t))/a.sum()
     check(f'{step}_{scope}_{cid}_{target}_metrics_independent',abs(float(r['iou_0.1'])-iou)<1e-10 and abs(float(r['mae_m'])-mae)<1e-10 and abs(float(r['false_km2_0.1'])-fp/1e6)<1e-10)
   if step==480:
    e=pred[:,m].astype('f8')-y[:,m];mean=(e[0]+e[1])/2;change=(e[1]-e[0])/2;dr=next(r for r in decomp if r['scope']==scope);v=np.average(mean**2,weights=a);total=np.average(np.mean(e*e,axis=0),weights=a)
    check(scope+'_error_decomposition_independent',abs(v-dr['shared_error_MSE_m2'])<1e-10 and abs(total-dr['mean_parent_MSE_m2'])<1e-10)
    for cname,g in [('both_nonwet',(y[0,m]<=.1)&(y[1,m]<=.1)),('only_train01_wet',(y[0,m]>.1)&(y[1,m]<=.1)),('only_train04_wet',(y[0,m]<=.1)&(y[1,m]>.1)),('both_wet',(y[0,m]>.1)&(y[1,m]>.1))]:
     for i,cid in enumerate(IDS):
      r=next(r for r in coh if r['case_id']==cid and r['scope']==scope and r['cohort']==cname);v=pred[i,m];check(scope+'_'+cid+'_'+cname+'_independent',abs(float(r['support_km2'])-a[g].sum()/1e6)<1e-10 and abs(float(r['prediction_mean_m'])-np.average(v[g],weights=a[g]))<1e-10 and abs(float(r['predicted_wet_km2'])-a[g&(v>.1)].sum()/1e6)<1e-10)
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[]
  def handle_starttag(self,t,attrs):self.links += [v for k,v in attrs if k in ['href','src']]
 parser=Links();parser.feed((REPORT/'review.html').read_text());check('all_report_links_present',all('://' not in link and (REPORT/link).exists() for link in parser.links))
 write(REPORT/'delivery_validation.json',{'all_checks_passed':True,'independent_checks':len(checks),'source_and_analysis_checks':len(read(REPORT/'checks.json')['checks']),'checks':checks,'new_optimizer_updates':0,'new_fits':0,'sealed_opened':False});print(json.dumps({'all_checks_passed':True,'checks':len(checks)}))
if __name__=='__main__':main()
