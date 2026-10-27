"""Analytical and delivered-product tests for numerical reference admission. No runs."""
from pathlib import Path
import sys,json,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from reference_label_policy import POLICY,CONTRACT,classify_labels,integrate_rate_at,balance_check,event_admission
from audit_three_ward_reference_contract import OUT,REPORT,INPUT,LABEL,read_npz,sha,dump

def main(root=ROOT):
 root=Path(root);checks=[]
 def check(name,ok,evidence=None):
  checks.append({'check':name,'pass':bool(ok),'evidence':evidence});print(('PASS ' if ok else 'FAIL ')+name,flush=True)
 def rejects(fn,kind=ValueError):
  try:fn()
  except kind:return True
  return False
 # Independent finite-ground example: area1 and3m2, heights0 and2, level3 gives mean depth1.5m.
 ground=np.array([0.,2.]);area=np.array([1.,3.]);known=float(np.dot(area,np.maximum(3-ground,0))/area.sum())
 check('ground_depth_semantics_differ_from_native_minimum',known==1.5 and known!=3-ground.min())
 # A triangular rate peaking at2 over4s has exact cumulative volume0.5,2,3.5,4.
 got=integrate_rate_at([0,2,4],[0,2,0],[1,2,3,4])
 check('analytical_piecewise_linear_rain_integral',np.allclose(got,[.5,2.,3.5,4.],atol=1e-14,rtol=0))
 check('invalid_and_out_of_support_time_rejected',rejects(lambda:integrate_rate_at([0,0],[1,1],[0])) and rejects(lambda:integrate_rate_at([0,2],[1,1],[3])))
 lo=np.array([.2,0.,.1,0.,.3,np.nan,.2,.1]);hi=np.array([.2,.01,.111,.01,.3,1.,.19,.1]);elig=np.ones(8,bool);neg=np.array([0,0,0,1,0,0,0,0],bool);bar=np.array([0,0,0,0,1,0,0,0],bool)
 m=classify_labels(lo,hi,elig,neg,bar)
 check('exact_interval_wide_and_direct_quarantine',np.array_equal(np.flatnonzero(m['Exact Label Mask']),[0,7]) and np.array_equal(np.flatnonzero(m['Interval Label Mask']),[1]) and np.array_equal(np.flatnonzero(m['Wide Interval Mask']),[2]) and np.array_equal(np.flatnonzero(m['Quarantine Mask']),[3,4,5,6]))
 check('intervals_preserved_not_midpoint_or_dry_imputed',lo[1]==0 and hi[1]==.01 and m['Interval Label Mask'][1] and not m['Exact Label Mask'][1])
 unproved=classify_labels(lo,hi,elig,neg,bar,interval_source_known=False)
 check('unproved_shallow_mask_quarantined',unproved['Quarantine Mask'][1] and not unproved['Admitted Mask'][1])
 check('strict_wet_exceedance_equality_and_interval_ambiguity',not(lo[7]>.1) and hi[7]<=.1 and lo[1]<=.005<hi[1])
 ok={k:True for k in ['completion','identity','geometry','fields','time','clock_alignment','boundary','rain','budget','censoring']}
 state=event_admission(ok,m)
 check('partial_support_admission_never_training_release',state['state']=='partial_snapshot_screened' and not state['training_release'])
 check('missing_clock_holds_entire_event',event_admission({k:v for k,v in ok.items() if k!='clock_alignment'},m)['state']=='hold_event')
 check('budget_failure_cannot_be_rescued_by_good_cell_mask',event_admission(ok|{'budget':False},m)['state']=='hold_event')
 b=balance_check([0,2],[0,4],[0,4],[0,0],[0,0])
 check('analytical_closed_flat_rain_balance',b['endpoint_pass'] and b['prefix_pass'] and b['residual'][-1]==0)
 back=balance_check([0,10],[0,80],[0,0],[0,0],[0,80])
 check('zero_rain_boundary_inflow_balance',back['endpoint_pass'] and back['residual'][-1]==0 and back['scale'][-1]==80)
 signed=balance_check([0,1],[0,-10],[0,0],[0,10],[0,0]);clipped=balance_check([0,1],[0,0],[0,0],[0,10],[0,0])
 check('signed_storage_required_clipping_changes_balance',signed['endpoint_pass'] and not clipped['endpoint_pass'])
 pulse=balance_check([0,1,2],[0,10,0],[0,0,0],[0,0,0],[0,0,0])
 check('prefix_failure_detected_despite_perfect_final_closure',pulse['endpoint_pass'] and not pulse['prefix_pass'])
 check('nan_flux_never_becomes_zero',rejects(lambda:balance_check([0,1],[0,0],[0,0],[0,np.nan],[0,0])))
 q=np.array([50.,-40.]);check('gross_exchange_before_group_cancellation',np.maximum(q,0).sum()==50 and np.maximum(-q,0).sum()==40 and q.sum()==10)
 check('unit_absolute_floor_handles_dry_zero_exchange',balance_check([0,1],[0,0],[0,0],[0,0],[0,0])['endpoint_gate']==1)
 # Delivered observations may legitimately fail policy. Validate faithful reporting, not universal passing.
 report=root/REPORT;out=root/OUT;summary=json.loads((report/'summary.json').read_text());diag=json.loads((report/'run_diagnostics.json').read_text());prov=json.loads((report/'provenance.json').read_text())
 check('eight_existing_runs_only',len(diag)==8 and summary['new_simulations']==summary['training']==0)
 check('required_failures_force_event_holds',all(v['label_admission']['state']=='hold_event' for k,v in diag.items() if k.endswith('_corrected_10m') and not all(v['checks'].values())))
 checkpoint_ok=True;stored=[]
 import csv
 for tag,d in diag.items():
  rows=list(csv.DictReader((report/(tag+'_budget_checkpoints.csv')).open()));x=lambda k:np.array([float(r[k]) for r in rows])
  b=balance_check(x('time_s'),x('signed_storage_m3'),x('rain_m3'),x('boundary_out_m3'),x('boundary_in_m3'))
  checkpoint_ok &= len(rows)==49 and abs(b['residual'][-1]-d['end_residual_m3'])<1e-6 and b['endpoint_pass']==d['nominal_endpoint_budget_pass'] and b['prefix_pass']==d['nominal_prefix_budget_pass']
 check('all_49_checkpoint_budgets_reproducible',checkpoint_ok)
 cell_partition=True;originals=True;states=[]
 support=read_npz(root/LABEL/'common_support_10m.npz');v=support['Eligible Ground Mask']
 for case in ['rain60_low','rain100_low','rain60_raised','dry_raised']:
  z=read_npz(out/(case+'_reference_masks.npz'));o=read_npz(root/INPUT/(case+'_label_admission.npz'))
  cell_partition &= np.all(sum(z[k].astype('uint8') for k in ['Exact Label Mask','Interval Label Mask','Quarantine Mask','Wide Interval Mask'])[v]==1)
  originals &= all(np.array_equal(z[k],o[k],equal_nan=True) for k in ['Label Lower','Label Upper','Exact Label Mask','Interval Label Mask','Quarantine Mask'])
 check('four_cell_classes_partition_all_eligible_ground',cell_partition)
 check('old_label_bounds_and_quarantine_unchanged',originals)
 protected=all(sha(root/p)==h for p,h in prov['protected_sha256'].items())
 check('frozen_forcing_inputs_and_existing_labels_unmodified',protected)
 # The geometry/history binding must retain two identically named water groups separately.
 binding=json.loads((out/'boundary_section_binding.json').read_text());codes=set(CONTRACT.get('unused',[]) or [17,51,71,79,106,110,111])
 check('boundary_bindings_preserve_seven_codes',all(set(s['boundary_group_code'] for s in b)==codes and len({s['section'] for s in b})==len(b) for b in binding.values()))
 check('sampling_and_dynamic_limits_explicit',not CONTRACT['temporal_profiles']['dynamic_release'] and not summary['dynamic_profile_ready'] and all(not d['continuous_state_nonnegativity_proved'] and not d['dynamic_face_flux_profile_complete'] for d in diag.values()))
 from load_three_ward_reference_labels import load_reference
 check('default_loader_blocks_every_held_control',all(rejects(lambda c=c:load_reference(c,root),PermissionError) for c in ['rain60_low','rain100_low','rain60_raised','dry_raised']))
 check('explicit_diagnostic_loader_preserves_bounds',np.array_equal(load_reference('rain60_low',root,diagnostic_only=True)['Label Lower'],read_npz(root/INPUT/'rain60_low_label_admission.npz')['Label Lower'],equal_nan=True))
 check('unregistered_or_sealed_response_loading_blocked',rejects(lambda:load_reference('sealed_test-01',root),KeyError))
 futures=json.loads((out/'future_run_contract_bindings.json').read_text());old_runs=json.loads((root/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs']
 check('58_unrun_contract_bindings_preserve_parent_splits',len(futures['runs'])==58 and not futures['execution_authorized'] and all(all(a[k]==b[k] for k in b) and not a['response_available'] for a,b in zip(futures['runs'],old_runs)))
 import rasterio
 geo_ok=True
 for case in ['rain60_low','rain100_low','rain60_raised','dry_raised']:
  z=read_npz(out/(case+'_reference_masks.npz'))
  with rasterio.open(out/(case+'_reference_class_10m.tif')) as ds:
   a=ds.read(1);geo_ok &= ds.crs.to_epsg()==6677 and ds.shape==(1012,770) and tuple(ds.transform)[:6]==tuple(support['Affine Transform']) and np.array_equal(a==3,z['Quarantine Mask']) and np.array_equal(a==2,z['Interval Label Mask']) and np.array_equal(a>0,v)
 check('four_georeferenced_quarantine_class_maps',geo_ok)
 check('contract_policy_files_match_executed_settings',json.loads((out/'label_policy.json').read_text())==POLICY and json.loads((out/'output_contract.json').read_text())==CONTRACT)
 result={'all_checks_passed':all(x['pass'] for x in checks),'checks':checks,'observed_run_holds_are_not_test_failures':True,'new_simulations':0,'training':0,'source_outputs_read_only':True}
 dump(report/'validation.json',result);print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed'],'observed_run_holds':summary['event_holds']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
