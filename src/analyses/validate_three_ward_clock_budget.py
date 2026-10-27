"""Independent arithmetic and fault tests for the clock/budget interface audit."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from sfincs_clock_budget_interface import decode_legacy_fews_minutes,candidate_integer_second_decoder,imported_rate,scheduler_curve,forcing_only_capped_replay,validate_step_ledger,LEDGER_SCHEMA
from reference_label_policy import integrate_rate_at
from audit_three_ward_clock_budget import OUT,REPORT,LIB,OLD,sha,dump


def main(root=ROOT):
 root=Path(root);out=root/OUT;report=root/REPORT;summary=json.loads((report/'summary.json').read_text());checks=[]
 def check(name,value,evidence=None):
  checks.append({'check':name,'pass':bool(value),'evidence':evidence});print(('PASS ' if value else 'FAIL ')+name,flush=True)
 def rejects(fn):
  try:fn()
  except ValueError:return True
  return False
 source=json.loads((out/'source_manifest.json').read_text())
 good=True
 for x in source:
  b=(out/'source_snapshots'/x['file']).read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();good &= blob==x['git_blob_sha1'] and sha(out/'source_snapshots'/x['file'])==x['sha256']
 check('ten_official_source_blobs_verified',good and len(source)==10)
 receipt=json.loads((report/'fortran_probe_receipt.json').read_text())
 check('unmodified_official_date_function_compiled_and_exercised',receipt['source_unmodified'] and receipt['official_function_invocations']==60 and receipt['all_python_legacy_decodes_equal_compiled_function'])
 intended=np.array([0,3600,3601,7200,7201,14400.]);decoded=decode_legacy_fews_minutes(intended/60)
 check('minute_fraction_truncation_reproduced',np.array_equal(decoded,[0,3600,3600,7200,7200,14400]))
 check('fortran_and_python_legacy_anchor_decode_match',np.array_equal(decoded,receipt['legacy_decoded_seconds']))
 check('seconds_unit_rename_not_a_fix',receipt['seconds_unit_rename_probe_decoded_seconds']==[0,216060])
 check('integer_second_candidate_restores_anchor_knots',np.array_equal(candidate_integer_second_decoder(intended/60),intended))
 check('duplicate_times_use_last_value',np.array_equal(imported_rate([0,3600,3600,7200,7200,14400],[0,0,60,60,0,0],[3599,3600,7199,7200]),[0,60,60,0]))
 t,r=scheduler_curve(decoded,[0,0,60,60,0,0]);check('one_second_meteo_lookahead_ramp_identified',np.interp(3599,t,r)==0 and np.interp(3599.5,t,r)==30 and np.interp(3600,t,r)==60)
 check('continuous_imported_rain_before_nominal_onset',abs(integrate_rate_at(t,r,[3600])[0]-30)<1e-10)
 check('legacy_full_rain_integral_preserved_despite_shift',abs(integrate_rate_at(t,r,[14400])[0]/3600-60)<1e-12)
 # Without float-roundoff, capped0.4s endpoint samples of a unit1s ramp sum0.72.
 check('analytical_right_endpoint_ramp_integral',abs(.4*(.2+.6+1)-.72)<1e-14)
 replay=forcing_only_capped_replay()
 check('source_dtype_replay_matches_analytical_limit',abs(replay['unit_applied_rain_seconds']-.72)<.0001 and replay['hydrodynamics_executed'] is False and not replay['actual_step_trace_recovered'])
 onset=list(csv.DictReader((report/'onset_residual_explanation.csv').open()));rainy=[x for x in onset if float(x['rain_peak_mm_h'])>0]
 check('six_whole_domain_onset_residuals_reproduced_conditionally',len(rainy)==6 and all(abs(float(x['remaining_after_forcing_and_boundary_rule_m3']))<.001 for x in rainy),{'max_remaining_m3':summary['onset_kernel_plus_boundary_rule_max_remaining_m3'],'not_full_event_clock_recovery':True})
 check('onset_boundary_quadrature_difference_kept_separate',all(abs(float(x['remaining_after_forcing_and_boundary_rule_m3'])-(float(x['onset_residual_minus_conditional_rain_m3'])+float(x['boundary_right_minus_trapezoid_at_onset_m3'])))<1e-9 for x in onset))
 check('dry_controls_reproduce_zero_source',all(float(x['conditional_capped_step_rain_m3'])==0 for x in onset if float(x['rain_peak_mm_h'])==0))
 prospective=list(csv.DictReader((report/'prospective_export_impact.csv').open()));catalog=json.loads((root/LIB/'catalog.json').read_text());expected={c['case_id'] for c in catalog['cases'] if not c.get('existing_completed')}
 check('all_50_pending_forcings_audited',len(prospective)==50 and {x['case_id'] for x in prospective}==expected)
 check('prospective_collision_and_mass_counts_reproducible',sum(int(x['decoded_collisions'])>0 for x in prospective)==summary['prospective_cases_with_decoded_collisions']==25 and sum(float(x['relative_depth_change'])>1e-6 for x in prospective)==summary['prospective_cases_with_relative_depth_change_gt_1e_6']==48)
 check('candidate_restores_all_50_integer_second_curves',all(x['integer_second_candidate_restores_knots']=='True' for x in prospective) and summary['candidate_integer_second_knots_all_restored'])
 check('candidate_not_deployed_or_qualified_in_hydrodynamic_image',not LEDGER_SCHEMA['operational_status'].startswith('Deployed') and not summary['new_forcing_curves_or_partitions_modified'])
 # Synthetic accounting fixtures are interface unit tests, never Tokyo responses.
 zero=np.zeros(3)
 fixture={k:zero.copy() for k in LEDGER_SCHEMA['checkpoint_fields']}
 fixture.update({'nominal_output_time':np.array([0.,1.,2.]),'actual_state_time':np.array([0.,1.1,2.1]),'actual_step_dt':np.array([0.,.4,.4]),'step_index':np.array([0,3,6]),'signed_storage':np.array([0.,5.,10.]),'rain_applied_cumulative':np.array([0.,5.,10.])})
 check('valid_same_clock_ledger_passes_arithmetic_only',validate_step_ledger(fixture)['balance_pass'] and not validate_step_ledger(fixture)['formal_reference_release'])
 f={k:v for k,v in fixture.items() if k!='actual_state_time'};check('missing_actual_clock_rejected',rejects(lambda:validate_step_ledger(f)))
 f=fixture|{'actual_state_time':np.array([0.,1.6,2.6])};check('clock_outside_actual_step_rejected',rejects(lambda:validate_step_ledger(f)))
 f=fixture|{'rain_applied_cumulative':np.array([0.,5.,np.nan])};check('unknown_applied_rain_is_not_zero',rejects(lambda:validate_step_ledger(f)))
 f=fixture|{'boundary_out_cumulative':np.array([0.,3.,2.])};check('nonmonotone_cumulative_exchange_rejected',rejects(lambda:validate_step_ledger(f)))
 f=fixture|{'signed_storage':np.array([0.,8.,10.])};check('perfect_endpoint_cannot_hide_prefix_error',not validate_step_ledger(f)['balance_pass'])
 f=fixture|{'negative_native_count':np.array([0.,-.1,0.])};check('invalid_negative_diagnostic_rejected',rejects(lambda:validate_step_ledger(f)))
 f=fixture|{'minimum_native_volume_since_initialization':np.array([0.,-1.,0.])};check('online_minimum_may_not_reset',rejects(lambda:validate_step_ledger(f)))
 clock=json.loads((report/'existing_clock_inventory.json').read_text())
 check('misnamed_average_dt_cannot_reconstruct_actual_steps',len(clock)==8 and all(x['netcdf_average_dt_value']>10000 and x['log_average_dt_s']<1 for x in clock.values()))
 status=json.loads((out/'transport_readiness.json').read_text());check('all_58_run_descendants_remain_held',status['pending_hydraulic_runs_held']==58 and not status['policy_or_reference_release_changed'])
 prov=json.loads((report/'provenance.json').read_text());check('old_policies_labels_and_forcing_library_unmodified',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('existing_result_files_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==fp for p,fp in prov['existing_result_stat_fingerprints'].items()))
 registry=json.loads((root/'data/processed/three_ward_reference_contract_v1/reference_registry.json').read_text())
 check('previous_strict_admission_holds_preserved',all(c['state']=='hold_event' and not c['technical_reference'] for c in registry['controls'].values()))
 check('no_model_execution_training_or_sealed_response_reads',summary['new_hydrodynamic_runs']==summary['training']==summary['sealed_responses_read']==0 and receipt['no_hydrodynamic_model_called'])
 result={'all_checks_passed':all(x['pass'] for x in checks),'checks':checks,'new_hydrodynamic_runs':0,'training':0,'no_reference_release':True}
 dump(report/'validation.json',result);print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
