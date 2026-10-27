"""Pre-register and verify the one-fit diagnostic before neural execution."""
from pathlib import Path
import sys,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,sha,load_label,arrays
from three_ward_two_parent_fit_interface_v1 import GATES,GATE_SCOPES,evaluate_fit_gates,update_success_streak
NAME='three_ward_two_parent_fit_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1';OLDPLAN=ROOT/'data/processed/three_ward_unet_control_plan_v2'
def masksha(m):return hashlib.sha256(__import__('numpy').ascontiguousarray(m,dtype='u1').tobytes()).hexdigest()
def main():
 import numpy as np
 if (OUT/'protocol.json').exists():raise FileExistsError('Preserve registered protocol')
 OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);p=read(OLDPLAN/'protocol.json');catalogpath=ROOT/'data/processed/three_ward_minimum_scenarios_v1/catalog.json';c=read(catalogpath)['cases'];training=set(p['train_case_ids']);chosen=[next(r for r in c if r['case_id'] in training and r['parameters']['shape_class']==shape) for shape in ['early','multiple_peak']];ids=[r['case_id'] for r in chosen];assert ids==['train-01','train-04'];records=[next(r for r in p['records'] if r['case_id']==cid) for cid in ids]
 protected=dict(read(SOURCE/'protocol.json')['source_hashes']);protected.update(read(SOURCE/'protocol.json')['implementation_hashes'])
 for path in [ROOT/'data/processed/three_ward_spatial_fp_controls_v1/freeze.json',ROOT/'data/processed/three_ward_unet_controls_v2/freeze.json']:
  protected.update(read(path)['files_sha256']);protected[str(path.relative_to(ROOT))]=sha(path)
 for path in [OLDPLAN/'protocol.json',OLDPLAN/'shared_initialization.pt',OLDPLAN/'epoch_order.json',SOURCE/'prepared_inputs.npz',SOURCE/'normalization.json',SOURCE/'protocol.json',catalogpath,ROOT/'src/analyses/three_ward_surrogate_protocol_v2.py']:
  protected[str(path.relative_to(ROOT))]=sha(path)
 assert all(sha(ROOT/k)==v for k,v in protected.items())
 # Only the two original training responses are loaded; no validation/other labels.
 with np.load(SOURCE/'prepared_inputs.npz',allow_pickle=False) as z:eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];area=z['Ground Area']
 regions={'domain':np.ones_like(eligible),'three_wards':ward>0,'Chiyoda':ward==1,'Chuo':ward==2,'Minato':ward==3,'cross_ward':ward==0};expected={};support=[];checks=[]
 def check(n,v):
  if not bool(v):raise AssertionError(n)
  checks.append({'check':n,'pass':True})
 for r in records:
  d=load_label(r['case_id'],'train');mask=d['Exact Label Mask']&eligible;y=d['Peak Mean Depth Lower'].astype('f4');check(r['case_id']+'_original_full_exact_mask',masksha(mask)==r['original_exact_eligible_mask_sha256']);check(r['case_id']+'_no_interval_quarantine_negative',not np.any(mask&(d['Interval Label Mask']|d['Quarantine Mask']|d['Online Ever Negative'])));check(r['case_id']+'_finite_original_exact_targets',np.isfinite(y[mask]).all() and (y[mask]>=0).all());expected[r['case_id']]={}
  for scope,g in regions.items():
   m=mask&g;expected[r['case_id']][scope]=masksha(m);support.append({'case_id':r['case_id'],'scope':scope,'original_exact_cells':int(m.sum()),'support_km2':float(area[m].sum()/1e6),'wet_area_km2':float(area[m&(y>.1)].sum()/1e6),'deep_area_km2':float(area[m&(y>1)].sum()/1e6),'support_sha256':masksha(m)})
   if scope in GATE_SCOPES:check(r['case_id']+'_'+scope+'_both_reference_classes_present',(m&(y>.1)).any() and (m&(y<=.1)).any())
 orders=read(OLDPLAN/'epoch_order.json');rounds=[{'cycle':i+1,'source_order_epoch':i%40+1,'case_order':[cid for cid in orders[i%40]['case_order'] if cid in ids]} for i in range(240)];check('every_cycle_exactly_two_original_training_parents',all(len(r['case_order'])==2 and set(r['case_order'])==set(ids) for r in rounds));write(OUT/'cycle_order.json',rounds)
 # Synthetic fixtures test gate mechanics; these are not fitted score observations.
 fixtures=[]
 for cid in ids:
  for scope in GATE_SCOPES:fixtures.append({'case_id':cid,'split':'train','scope':scope,'support':'own_exact','support_sha256':expected[cid][scope],'iou_0.1':.9,'precision_0.1':.95,'recall_0.1':.95,'wet_mae_m':.01,'deep_mae_m':.1,'deep_area_km2':.01,'mae_m':.01})
 gate=evaluate_fit_gates(fixtures,ids,expected);check('synthetic_complete_pass',gate['all_parent_ward_gates_pass']);check('zero_step_never_counts_as_success',update_success_streak(0,gate,0)==0);check('two_consecutive_required',update_success_streak(update_success_streak(0,gate,20),gate,40)==2)
 for metric in GATES:
  broken=copy.deepcopy(fixtures);broken[0][metric]=0.5 if metric in ['wet_mae_m','deep_mae_m'] else .5;bad=evaluate_fit_gates(broken,ids,expected);check('gate_rejects_'+metric,not bad['all_parent_ward_gates_pass'] and update_success_streak(1,bad,40)==0)
 broken=copy.deepcopy(fixtures);broken[0]['precision_0.1']=None;check('undefined_precision_fails',not evaluate_fit_gates(broken,ids,expected)['all_parent_ward_gates_pass'])
 broken=copy.deepcopy(fixtures);broken[0]['deep_area_km2']=0;broken[0]['deep_mae_m']=None;check('empty_deep_is_NA_not_zero',evaluate_fit_gates(broken,ids,expected)['all_parent_ward_gates_pass'])
 for change in ['split','mask','missing']:
  broken=copy.deepcopy(fixtures)
  if change=='split':broken[0]['split']='validation'
  elif change=='mask':broken[0]['support_sha256']='wrong'
  else:broken.pop()
  try:evaluate_fit_gates(broken,ids,expected);caught=False
  except ValueError:caught=True
  check('gate_rejects_'+change,caught)
 protocol={'decision':'MILI-D-20261016-009','execution_authorized':True,'purpose':'Joint fitting/memorization sanity check on two known full-field training events; not validation, formal model admission, learning-curve evidence or sample expansion.','settings_ownership':'User authorized scope, finite budget and predeclared stopping. Exact selection rule, 480-step budget and numeric diagnostic gates are agent implementation choices fixed before this fit, not independently human-specified accuracy standards.','selection_rule':'First catalog entry of early and multiple_peak shape among the original12 train IDs, without choosing by response difficulty. Different stage/amplitude/duration also vary, so this is not a causal rainfall-shape contrast.','selected_case_ids':ids,'selected_records':records,'selected_metadata':[{k:r[k] for k in ['case_id','parent_id','family_id','split','parameters']} for r in chosen],'formal_train_case_ids_unchanged':p['train_case_ids'],'formal_validation_case_ids_unchanged':p['validation_case_ids'],'fit_count':1,'maximum_optimizer_updates':480,'maximum_two_parent_cycles':240,'parent_exposures_at_cap':240,'evaluation_interval_updates':20,'evaluation_updates':[0]+list(range(20,481,20)),'success_gates':GATES,'gate_scopes':GATE_SCOPES,'success_rule':'All 2parents x4scopes pass every applicable gate at two consecutive post-update evaluations. Deep gate is not applicable only when its original support is empty. Never stop on one favorable checkpoint or a macro average.','stop_rules':['Success after two consecutive passing evaluations','Budget exhaustion at480actual optimizer updates','Immediate held stop on nonfinite active predictions, gradients, parameters or changed source/mask bindings; preserve failure state, no automatic restart'],'automatic_extension_or_new_fit':False,'diagnostic_key':'highest worst parent/ward IoU, then lowest worst wetMAE, lowest worst overallMAE, earliest update; round metrics to6decimals. Diagnostic best is separate from the actual stopping endpoint and does not determine success.','architecture':'Original D ConditionalUNet with four pooling operations; all671513parameters','initialization_path':str((OLDPLAN/'shared_initialization.pt').relative_to(ROOT)),'initialization_sha256':sha(OLDPLAN/'shared_initialization.pt'),'normalization_sha256':sha(SOURCE/'normalization.json'),'prepared_inputs_sha256':sha(SOURCE/'prepared_inputs.npz'),'optimizer':'AdamW','learning_rate':.0003,'weight_decay':.00001,'gradient_clip':10.,'seed':20261016,'loss':'Original D stratified_huber: nonwet/ordinary/deep .5/.4/.1, residual transition0.10m; no extra FP penalty or new head','input_channels':58,'forcing_values':1920,'normalization':'Reuse original training12-only scaling unchanged, not recomputed from selected2 or validation','training_support':'Each original full-domain own exact AND eligible mask, no patch, crop, wet sampling or output clipping','evaluation_support':'Same own exact support by original geographic regions; not historical paired retrieval support','expected_support_sha256':expected,'cycle_order_sha256':sha(OUT/'cycle_order.json'),'label_access_allowlist':{cid:'train' for cid in ids},'no_validation_response_loading':True,'sealed_access':False,'sample_expansion':False,'formal_split_change':False,'source_sha256':protected,'interface_sha256':sha(ROOT/'src/analyses/three_ward_two_parent_fit_interface_v1.py'),'preparation_sha256':sha(Path(__file__)),'interpretation_limits':['Passing only demonstrates fit under this recipe on two known synthetic events, not generalization or real-world flood accuracy.','Failure within480steps does not prove intrinsic model incapacity or data misalignment; it leaves current encoding, loss and optimization jointly unresolved.','The original D comparison uses own exact scores for the same two training parents and has different per-parent exposure; it is context, not an equal-data ablation.','No automatic sample expansion, additional seeds, scheduler/loss changes or sealed-test access after either verdict.']}
 write(OUT/'protocol.json',protocol);write(REPORT/'preflight.json',{'all_checks_passed':True,'checks':checks,'full_support_inventory':support,'responses_loaded':ids,'optimizer_updates':0,'synthetic_gate_fixtures_only':True,'sealed_opened':False});print(json.dumps({'registered':ids,'max_updates':480,'preflight_checks':len(checks),'no_updates_yet':True}),flush=True)
if __name__=='__main__':main()
