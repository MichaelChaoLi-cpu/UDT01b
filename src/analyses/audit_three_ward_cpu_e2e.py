"""Audit the two authorized whole-domain controls at actual native step times."""
from pathlib import Path
import sys,json,csv,hashlib,argparse
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_cpu_e2e import OUT,REPORT,CASES,BUILD
from prepare_sfincs_cpu_build import sha,dump
from reference_label_policy import balance_check,integrate_rate_at

def structured(p):return np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf-8'))
def csvwrite(p,rows):
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def arr(v):return np.ma.asarray(v,dtype='f8').filled(np.nan)
def main(root=ROOT,only=None):
 root=Path(root);out=root/OUT;report=root/REPORT
 if(out/'freeze.json').exists():raise FileExistsError('Frozen; validate delivery instead')
 manifest=json.loads((out/'run_manifest.json').read_text());checks=[];summaries={};checkpoint_rows=[]
 def check(case,name,ok,evidence=None):
  checks.append({'case_id':case,'check':name,'pass':bool(ok),'evidence':evidence});print(('PASS 'if ok else'FAIL ')+case+' '+name,flush=True)
 for run in manifest['runs']:
  cid=run['case_id']
  if only and cid!=only:continue
  folder=root/run['folder'];execution=json.loads((folder/'execution.json').read_text());log=(folder/'sfincs.log').read_text()
  ledger=structured(folder/'udt_step_budget.csv');clock=structured(folder/'udt_output_clock.csv');final=structured(folder/'udt_final_clock.csv')[0]
  get=lambda k:np.array(ledger[k],dtype='f8');t=get('actual_state_time');dt=get('actual_step_dt');step=get('step_index')
  check(cid,'clean_completion_and_real_final_state',execution['returncode']==0 and execution['completion_marker']and final['solver_error']==0 and t[-1]>=14400 and t[-1]-14400<=dt[-1]+1e-8)
  check(cid,'every_step_actual_clock_and_true_initial_row',step[0]==0 and t[0]==dt[0]==0 and np.array_equal(np.diff(step),np.ones(len(step)-1))and np.all(np.diff(t)>0)and max(abs(np.diff(t)-dt[1:]))<1e-8)
  check(cid,'all_recorded_native_quantities_finite',all(np.all(np.isfinite(get(k)))for k in ledger.dtype.names))
  check(cid,'final_receipt_matches_last_accounted_step',int(final['step_index'])==int(step[-1])and final['actual_final_time']==t[-1]and final['actual_last_dt']==dt[-1])
  bal=balance_check(t,get('signed_storage'),get('rain_applied_cumulative'),get('boundary_out_cumulative'),get('boundary_in_cumulative'),get('other_source_cumulative'),get('other_sink_cumulative'))
  check(cid,'same_step_budget_endpoint_and_all_prefixes',bal['endpoint_pass']and bal['prefix_pass'],{'end_residual_m3':float(bal['residual'][-1]),'max_abs_prefix_m3':float(max(abs(bal['residual']))),'max_prefix_scaled_residual':bal['max_prefix_relative_residual']})
  check(cid,'exported_residual_is_reproducible_not_forced_zero',max(abs(get('budget_residual')-bal['residual']))<1e-7)
  cumulative=['rain_applied_cumulative','boundary_out_cumulative','boundary_in_cumulative','other_source_cumulative','other_sink_cumulative']
  check(cid,'gross_exchange_and_sources_monotone_with_explicit_baseline_zeros',all(get(k)[0]==0 and np.min(np.diff(get(k)))>=-1e-9 for k in cumulative)and np.all(get('other_source_cumulative')==0)and np.all(get('other_sink_cumulative')==0))
  source_add_roundoff=get('rain_dvol_increment_cumulative')-get('rain_applied_cumulative')
  storage_double_remainder=get('signed_storage')-get('signed_storage')[0]-get('horizontal_applied_cumulative')-get('rain_dvol_increment_cumulative')
  exported_horizontal=get('horizontal_applied_cumulative')+get('boundary_out_cumulative')-get('boundary_in_cumulative')
  check(cid,'horizontal_and_source_roundoff_retained_with_double_storage_remainder',np.array_equal(exported_horizontal,get('internal_flux_roundoff_residual'))and np.all(np.isfinite(storage_double_remainder)),{'max_double_storage_remainder_m3':float(max(abs(storage_double_remainder))),'formal_balance_gates_unchanged':True})
  observed=get('meteo_bracket_observed');m0=get('meteo_bracket_start');m1=get('meteo_bracket_end')
  check(cid,'actual_one_second_linear_meteo_brackets',observed[0]==0 and np.all(observed[1:]==1)and np.all(m1[1:]-m0[1:]==1)and np.all(get('meteo_update_interval')[1:]==1)and np.all(t[1:]>=m0[1:]-1e-8)and np.all(t[1:]<=m1[1:]+1e-8))
  # Exact source knots are checked independently of step-integrated rain and output labels.
  trace=np.loadtxt(folder/'udt_rain_decode.csv',delimiter=',',skiprows=2,ndmin=2)
  with Dataset(folder/'roof_surface_precip.nc')as rainnc:
   rainnc.set_auto_mask(False);rawtime=np.array(rainnc['time'][:],dtype='f8');units=rainnc['time'].units
   raw_fields=np.array(rainnc['Precipitation'][:],dtype='f4');rain_x=np.array(rainnc['x'][:],dtype='f8');rain_y=np.array(rainnc['y'][:],dtype='f8');original_rain_units=getattr(rainnc['Precipitation'],'units',getattr(rainnc['Precipitation'],'unit',''))
  intended=np.rint(rawtime*60)
  check(cid,'original_seconds_and_raw_units_preserved',np.array_equal(trace[:,0],rawtime)and np.array_equal(trace[:,1],intended)and np.all(np.diff(trace[:,1])>0)and units=='minutes since 2020-01-01')
  check(cid,'no_applied_rain_before_3600_seconds',np.all(get('rain_applied_cumulative')[t<=3600]==0))
  if cid=='dry_raised':check(cid,'dry_control_has_zero_rain_all_steps',np.all(get('rain_applied_cumulative')==0))
  else:check(cid,'rain_control_has_positive_applied_rain',get('rain_applied_cumulative')[-1]>0)
  idx=get('step_index').astype(int);by_step={int(k):i for i,k in enumerate(idx)}
  associated=np.array([by_step[int(x)]for x in clock['step_index']],dtype=int)
  check(cid,'every_output_clock_binds_the_same_accounted_state',np.array_equal(clock['actual_state_time'],t[associated])and np.array_equal(clock['actual_step_dt'],dt[associated]))
  nominal=np.array(clock['nominal_output_time'],dtype='f8');actual=np.array(clock['actual_state_time'],dtype='f8');odt=np.array(clock['actual_step_dt'],dtype='f8')
  check(cid,'nominal_labels_within_the_actual_captured_step',np.min(actual-nominal)>=-1e-7 and np.max(actual-nominal-odt)<=1e-6)
  maps=clock[clock['write_map']=='T'];hist=clock[clock['write_history']=='T'];peaks=clock[clock['write_peak']=='T'];map_rows=[]
  with Dataset(folder/'sfincs_map.nc')as nc:
   mask=arr(nc['msk'][:]);inside=mask==1;ts=arr(nc['time'][:]);nr=len(ts)
   check(cid,'map_record_count_geometry_and_required_exports',nr==len(maps)and mask.shape==(1012,770)and all(k in nc.variables for k in ['subgrid_volume','zs','h','zsmax','zb','u','v']))
   check(cid,'map_record_numbers_and_nominal_times_match',np.array_equal(maps['map_record'],np.arange(1,nr+1))and np.array_equal(ts,maps['nominal_output_time'].astype(nc['time'].dtype).astype('f8')))
   check(cid,'full_event_snapshots_have_no_more_than_300s_spacing',nr>=49 and max(np.diff(ts))<=300.001 and abs(ts[0])<1e-7 and abs(ts[-1]-14400)<1)
   all_finite=True;all_storage=True;negative_written=np.zeros(mask.shape,bool);max_delta=0.;max_bound=0.;negative_written_count=0
   for i,row in enumerate(maps):
    j=by_step[int(row['step_index'])];volume=arr(nc['subgrid_volume'][i]);v=volume[inside];all_finite &= bool(np.all(np.isfinite(v)))
    storage=float(np.sum(v,dtype='f8'));delta=storage-get('signed_storage')[j]
    # Original NetCDF map volumes are float32; account for write quantization separately from budget gates.
    vf=v.astype('f4');bound=float(np.sum(np.maximum(np.abs(np.nextafter(vf,np.float32(np.inf))-vf),np.abs(vf-np.nextafter(vf,np.float32(-np.inf)))).astype('f8')*.5))+1e-7
    all_storage &= abs(delta)<=bound;max_delta=max(max_delta,abs(delta));max_bound=max(max_bound,bound)
    negative_written |= inside&(volume<-.001)
    negatives=int(np.sum(v<-.001));negative_written_count=max(negative_written_count,negatives)
    map_rows.append({'case_id':cid,'map_record':i+1,'nominal_time_s':float(row['nominal_output_time']),'actual_time_s':float(row['actual_state_time']),'step_index':int(row['step_index']),'last_dt_s':float(row['actual_step_dt']),'signed_ledger_storage_m3':float(get('signed_storage')[j]),'NetCDF_float32_storage_sum_m3':storage,'storage_export_difference_m3':delta,'float32_rounding_bound_m3':bound,'rain_applied_m3':float(get('rain_applied_cumulative')[j]),'boundary_out_m3':float(get('boundary_out_cumulative')[j]),'boundary_in_m3':float(get('boundary_in_cumulative')[j]),'same_step_budget_residual_m3':float(bal['residual'][j]),'prefix_gate_m3':float(bal['prefix_gate'][j]),'negative_native_cells':int(get('negative_native_count')[j])})
   check(cid,'all_map_native_storages_finite_and_match_same_step_with_write_quantization',all_finite and all_storage,{'max_difference_m3':max_delta,'max_quantization_bound_m3':max_bound})
   peak_times=arr(nc['timemax'][:]);check(cid,'peak_records_bind_actual_accumulation_windows',len(peaks)==len(peak_times)and np.array_equal(peaks['peak_record'],np.arange(1,len(peaks)+1))and np.array_equal(peak_times,peaks['nominal_output_time'].astype(nc['timemax'].dtype).astype('f8')))
   positive=peaks[(peaks['actual_peak_end']-peaks['actual_peak_start'])>1e-7]
   check(cid,'a_complete_positive_length_peak_window_exists',any(abs(float(p['actual_peak_start']))<1e-9 and float(p['actual_peak_end'])>=14400 for p in positive))
   check(cid,'rain_corner_coordinates_align_native_grid_points',np.max(abs(rain_x[1:-1]+5-arr(nc['x'][:])[0]))<1e-7 and np.max(abs(rain_y[1:-1]+5-arr(nc['y'][:])[:,0]))<1e-7)
   rates=np.sum(raw_fields[:,1:-1,1:-1][:,inside],axis=1,dtype='f8')*100/3.6e6
   input_volume=float(np.trapezoid(rates,intended));expected_continuous=integrate_rate_at(intended,rates,np.minimum(t,intended[-1]))
   rain_difference=get('rain_applied_cumulative')-expected_continuous
  with Dataset(folder/'sfincs_his.nc')as nc:
   ht=arr(nc['time'][:]);check(cid,'history_record_numbers_and_times_match_captured_step',len(ht)==len(hist)and np.array_equal(hist['history_record'],np.arange(1,len(ht)+1))and np.array_equal(ht,hist['nominal_output_time'].astype(nc['time'].dtype).astype('f8')))
   flux_ok='crosssection_discharge'in nc.variables and nc['crosssection_discharge'].shape[1]>0
   if flux_ok:
    for first in range(0,len(ht),256):
     flux_ok &= bool(np.all(np.isfinite(arr(nc['crosssection_discharge'][first:first+256,:]))))
   check(cid,'boundary_flux_sections_exported_and_finite',flux_ok)
  negative=structured(folder/'udt_native_ever_negative.csv');native=np.fromfile(folder/'sfincs.ind',dtype='<i4');flags=negative['ever_volume_below_minus_0_001_m3']=='T'
  check(cid,'all_native_cells_have_online_ever_negative_flags',len(negative)==native[0]and np.array_equal(negative['native_cell_index'],np.arange(1,len(negative)+1))and int(np.sum(flags))>=int(np.max(get('negative_native_count'))))
  check(cid,'online_minimum_and_negative_summaries_monotone',np.max(np.diff(get('minimum_native_volume_since_initialization')))<=0 and np.min(get('negative_volume_sum'))>=0)
  identity_bytes=(folder/'udt_run_identity.txt').read_bytes();check(cid,'run_identity_export_matches_staging_and_new_image',identity_bytes==(folder/'udt_diagnostic_identity.txt').read_bytes()and run['identity']['solver_digest']==execution['solver_image_digest'])
  check(cid,'all_staged_input_bytes_unchanged_during_run',all(sha(folder/name)==h for name,h in run['input_sha256'].items()))
  check(cid,'no_astronomical_tide_dataset_loaded','reading tidal components'not in log and 'Astro boundary'not in log)
  check(cid,'rain_and_waterlevel_original_files_unchanged',all(sha(folder/name)==run['original_source_input_sha256'][name]for name in ['roof_surface_precip.nc','sfincs.bzs']))
  checkpoint_rows+=map_rows;csvwrite(report/(cid+'_map_checkpoints.csv'),map_rows)
  csvwrite(report/(cid+'_peak_windows.csv'),[{'peak_record':int(x['peak_record']),'nominal_time_s':float(x['nominal_output_time']),'actual_start_s':float(x['actual_peak_start']),'actual_end_s':float(x['actual_peak_end']),'duration_s':float(x['actual_peak_end']-x['actual_peak_start']),'usable_complete_event_window':bool(abs(x['actual_peak_start'])<1e-9 and x['actual_peak_end']>=14400)}for x in peaks])
  rainsteps=np.flatnonzero(np.diff(get('rain_applied_cumulative'))>0)
  summaries[cid]={'image_id':run['image_id'],'actual_steps':len(t)-1,'actual_mean_dt_s':float((t[-1]-t[0])/(len(t)-1)),'startup_dt_s':float(dt[1]),'minimum_post_startup_dt_s':float(min(dt[2:])),'maximum_dt_s':float(max(dt[1:])),'actual_final_time_s':float(t[-1]),'nominal_horizon_s':14400.,'final_overshoot_s':float(t[-1]-14400),'last_dt_s':float(dt[-1]),'runtime_seconds':execution['runtime_seconds'],'true_initial_storage_m3':float(get('signed_storage')[0]),'final_signed_storage_m3':float(get('signed_storage')[-1]),'actual_applied_rain_m3':float(get('rain_applied_cumulative')[-1]),'continuous_encoded_input_rain_m3':input_volume,'final_applied_minus_continuous_rain_m3':float(rain_difference[-1]),'max_applied_minus_continuous_rain_m3':float(max(abs(rain_difference))),'first_positive_rain_step_time_s':float(t[rainsteps[0]+1])if len(rainsteps)else None,'boundary_out_m3':float(get('boundary_out_cumulative')[-1]),'boundary_in_m3':float(get('boundary_in_cumulative')[-1]),'end_budget_residual_m3':float(bal['residual'][-1]),'max_abs_step_budget_residual_m3':float(max(abs(bal['residual']))),'max_double_storage_reduction_remainder_m3':float(max(abs(storage_double_remainder))),'end_relative_budget_residual':bal['end_relative_residual'],'max_prefix_scaled_residual':bal['max_prefix_relative_residual'],'endpoint_gate_m3':float(bal['endpoint_gate']),'budget_endpoint_pass':bal['endpoint_pass'],'budget_prefix_pass':bal['prefix_pass'],'online_ever_negative_native_cells':int(np.sum(flags)),'max_negative_volume_sum_m3':float(max(get('negative_volume_sum'))),'minimum_native_volume_m3':float(min(get('minimum_native_volume_since_initialization'))),'written_negative_cell_union':int(np.sum(negative_written)),'map_records':len(maps),'history_records':len(hist),'peak_records':len(peaks),'zero_length_final_peak_records':int(np.sum(peaks['actual_peak_end']-peaks['actual_peak_start']<=1e-7)),'max_map_nominal_time_float32_export_error_s':float(max(abs(ts-maps['nominal_output_time']))),'max_history_nominal_time_float32_export_error_s':float(max(abs(ht-hist['nominal_output_time']))),'max_map_storage_export_difference_m3':max_delta,'max_map_storage_quantization_bound_m3':max_bound,'physical_event_accuracy_verified':False,'training_release':False,'reference_label_release':False}
  near=np.flatnonzero((t>=3598)&(t<=3603))
  csvwrite(report/(cid+'_onset_actual_steps.csv'),[{'step':int(step[i]),'actual_time_s':float(t[i]),'dt_s':float(dt[i]),'applied_rain_m3':float(get('rain_applied_cumulative')[i]),'continuous_file_rain_m3':float(expected_continuous[i]),'budget_residual_m3':float(bal['residual'][i]),'boundary_out_m3':float(get('boundary_out_cumulative')[i]),'boundary_in_m3':float(get('boundary_in_cumulative')[i])}for i in near])
  dump(report/(cid+'_summary.json'),summaries[cid])
 prov=json.loads((report/'provenance.json').read_text())
 check('global','all_original_inputs_policies_and_libraries_unchanged',all(sha(root/p)==h for p,h in prov['protected_sha256'].items()))
 check('global','all_prior_hydrodynamic_results_unchanged',all([(root/p).stat().st_size,(root/p).stat().st_mtime_ns]==v for p,v in prov['existing_result_stat_fingerprints'].items()))
 result={'all_checks_passed':all(x['pass']for x in checks),'checks':checks,'case_summaries':summaries,'completed_full_control_runs':len(summaries),'new_parent_events':0,'pending_58_executed':0,'training':0,'sealed_responses_read':0,'no_reference_release':True,'known_failed_attempts_retained':2}
 dump(report/('validation.json'if only is None else only+'_validation.json'),result)
 if only is None:dump(report/'summary.json',result)
 print(json.dumps({k:v for k,v in result.items()if k not in ['checks']},indent=2))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--only',choices=CASES);a=p.parse_args();main(only=a.only)
