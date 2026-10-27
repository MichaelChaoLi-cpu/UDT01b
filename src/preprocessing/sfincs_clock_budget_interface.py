"""Pinned Galibier timing interface probes only; no hydrodynamic state evolution.

The forcing-only replay is conditional on a declared capped step. It does not
recover unknown later adaptive steps or grant reference-label admission.
"""
import numpy as np


def decode_legacy_fews_minutes(values):
    """Exact pinned converter arithmetic after NetCDF reads time into real*4."""
    x=np.asarray(values,dtype='f4')
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite input clock')
    return (np.trunc(x).astype('i8')*60).astype('f4').astype('f8')


def candidate_integer_second_decoder(values):
    """UNDEPLOYED candidate, integer-second inputs only; not the official reader."""
    x=np.asarray(values,dtype='f4')
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite input clock')
    # Source integers are nonnegative in this study; Fortran NINT rounds away
    # from zero at ties. None of this library's intended seconds are half ties.
    return np.floor(x.astype('f8')*60+.5)


def imported_rate(times,rates,query):
    """Match update_ampr_data: last knot <= t, then next knot, including duplicates."""
    t=np.asarray(times,dtype='f8');r=np.asarray(rates,dtype='f8');q=np.asarray(query,dtype='f8')
    if len(t)<2 or r.shape!=t.shape or np.any(np.diff(t)<0) or not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)):raise ValueError('Invalid imported time/rate vectors')
    x=np.clip(q,t[0],t[-1]);i=np.clip(np.searchsorted(t,x,side='right')-1,0,len(t)-1);j=np.minimum(i+1,len(t)-1)
    w=(np.minimum(x,t[j])-t[i])/np.maximum(t[j]-t[i],1e-6)
    return r[i]*(1-w)+r[j]*w


def scheduler_curve(times,rates,horizon=14400,update_s=1):
    """Meteo endpoint resampling, then linear interpolation each actual step."""
    clock=np.arange(0,horizon+update_s+1e-9,update_s,dtype='f8')
    return clock,imported_rate(times,rates,clock)


def forcing_only_capped_replay(nominal_time=3600.,rate_scale=1.,alpha=.4,dtmax=1.):
    """No water solver: integrate the reconstructed unit rainfall rate only.

    Capped dry-start steps are a declared hypothesis for the onset diagnostic.
    Real*4 products reproduce the source arithmetic before float64 reduction.
    """
    a=np.float32(alpha);dt=np.float32(a*np.float32(dtmax))
    t=float(np.float32(a*np.float32(1e-6)));mass_seconds=0.;steps=1
    if nominal_time<0 or dt<=0:raise ValueError('Invalid replay configuration')
    while t<nominal_time:
        t+=float(dt);steps+=1
        # The imported duplicate step becomes the meteo3599..3600 ramp.
        fraction=np.float32(np.clip(t-3599,0,1))
        mass_seconds+=float(np.float32(fraction*dt))
    return {'assumed_actual_time_s':t,'assumed_dt_s':float(dt),'assumed_step_count':steps,
            'unit_applied_rain_seconds':mass_seconds,'scaled_applied_rain':mass_seconds*rate_scale,
            'hydrodynamics_executed':False,'actual_step_trace_recovered':False}


LEDGER_SCHEMA={
 'version':'clock-step-budget-diagnostics-v1','status':'specified_and_validator_tested_not_exported_by_existing_engine',
 'mandatory_identity':['parent_id','family_id','partition','forcing_sha256','geometry_sha256','solver_digest','source_commit','diagnostic_schema_version'],
 'clock_dtype':'float64 seconds from declared event origin; step counter int64',
 'accounting_profile':'Current baseline has no infiltration, evaporation, spinup loss, external coupling or active measures. rain_applied is actual net-meteo volume; never subtract an already-netted loss twice. Enabling another process requires a separately qualified decomposition and profile.',
 'checkpoint_fields':{
  'nominal_output_time':'preserve existing time label for compatibility',
  'actual_state_time':'actual sfincs_lib.t at state capture, not inferred from output schedule',
  'actual_step_dt':'actual continuity step dt at capture',
  'step_index':'monotone solver step counter',
  'signed_storage':'float64 reduction of native signed interior z_volume; never clip negative values',
  'rain_applied_cumulative':'sum of the actual per-cell netprcp*cell_area*dt volume added by continuity',
  'boundary_out_cumulative':'positive outward face volumes integrated each continuity step before group aggregation',
  'boundary_in_cumulative':'positive inward face volumes integrated each continuity step before group aggregation',
  'other_source_cumulative':'explicit baseline zero, never infer missing term as zero',
  'other_sink_cumulative':'explicit baseline zero; include only sinks not already netted into netprcp; enabled nonbaseline processes require an explicit new accounting profile',
  'minimum_native_volume_since_initialization':'online minimum at every step',
  'negative_native_count':'count at capture; every-step ever-negative bitmask separately',
  'negative_volume_sum':'signed deficit magnitude at capture'},
 'forcing_decode_trace':['raw_time_values','raw_time_units','decoded_solver_seconds','decoded_strictly_increasing','raw_curve_integral','meteo_resampled_curve_integral','meteo_t0','meteo_t1','dtwnd'],
 'initial_checkpoint':'real pre-step0 dry state, not a first-step state relabeled0',
 'capture_policy':'same step as each map; add actual checkpoints bracketing forcing/stage knots and initial/final state, keep cadence<=300s for peak profile',
 'group_exchange':'preserve all seven group codes; split gross signs per face before summing; optional group cumulative totals',
 'maximum_window':'record actual native-peak accumulator start/end; scheduled four-hour timemax alone does not certify exact end',
 'reduction':'float64 sums, record actual source arithmetic and residual; do not force exact cancellation or hide cast-rounding error',
 'minimal_size':'O(checkpoints*global_fields + checkpoints*7groups), plus one native-cell ever-negative mask; full internal-face cubes not needed for global peak-reference budget',
 'later_dynamic_extension':'time-integrated internal faces and source/sink increments only for dynamic conservation or action targets',
 'no_new_simulation_or_training':True,
 'export_hooks':[
  {'source':'sfincs_lib.f90','hook':'after compute_water_levels, immediately before write_output','purpose':'capture public t/dt together with scheduled tout and step index; output API currently forwards only tout'},
  {'source':'sfincs_continuity.f90','hook':'same signed boundary and netprcp volume terms used in dvol before z_volume update','purpose':'accumulate applied step volumes, including coarse/fine factors and non-rain terms'},
  {'source':'sfincs_ncoutput.F90','hook':'write same-step scalar ledger beside map/history records','purpose':'write actual clock and matched-step cumulative terms'},
  {'source':'sfincs_ncinput.F90 / sfincs_date.f90','hook':'after convert_fewsdate and before update_ampr_data','purpose':'persist decoded forcing knots and reject unintended collisions'}],
 'operational_status':'No solver patch, image rebuild, BMI execution or new model run performed.'}


def validate_step_ledger(d):
    """Fail-closed structural and same-clock conservation check for future exports.

    Does not certify exporter provenance or physical accuracy. Identity and
    metadata must additionally bind to the frozen run manifest before release.
    """
    fields=list(LEDGER_SCHEMA['checkpoint_fields'])
    missing=[k for k in fields if k not in d]
    if missing:raise ValueError('Missing checkpoint fields: '+','.join(missing))
    n=len(np.asarray(d['actual_state_time']));a={k:np.asarray(d[k],dtype='f8') for k in fields}
    if n<2 or any(v.shape!=(n,) or not np.all(np.isfinite(v)) for v in a.values()):raise ValueError('Nonfinite or misaligned checkpoints')
    t=a['actual_state_time'];nom=a['nominal_output_time'];step=a['step_index'];dt=a['actual_step_dt']
    if t[0]!=0 or step[0]!=0 or dt[0]!=0 or np.any(np.diff(t)<=0) or np.any(np.diff(nom)<=0) or np.any(np.diff(step)<=0) or np.any(step!=np.rint(step)) or np.any(dt[1:]<=0):raise ValueError('Invalid actual/nominal clock or step index')
    if np.any(t<nom-1e-6) or np.any(t-nom>dt+1e-3):raise ValueError('Scheduled and actual clocks disagree with captured step')
    count=a['negative_native_count'];deficit=a['negative_volume_sum'];minimum=a['minimum_native_volume_since_initialization']
    if np.any(count<0) or np.any(count!=np.rint(count)) or np.any(deficit<0) or np.any(np.diff(minimum)>1e-9):raise ValueError('Invalid every-step negative-volume diagnostics')
    cum=['rain_applied_cumulative','boundary_out_cumulative','boundary_in_cumulative','other_source_cumulative','other_sink_cumulative']
    if any(abs(a[k][0])>1e-9 or np.any(a[k]<0) or np.any(np.diff(a[k])<-1e-9) for k in cum):raise ValueError('Invalid cumulative applied volumes')
    v=a['signed_storage'];res=v-v[0]+a[cum[1]]-a[cum[2]]-a[cum[0]]-a[cum[3]]+a[cum[4]]
    scale=np.maximum.reduce([a[cum[0]]+a[cum[3]]+a[cum[2]],a[cum[1]]+a[cum[4]],np.full(n,abs(v[0])),np.ones(n)])
    gates=np.maximum(1,.01*scale);gates[-1]=max(1,.001*scale[-1])
    return {'balance_pass':bool(np.all(abs(res)<=gates)),'residual':res,'gate':gates,'actual_clock_fields_present':True,'formal_reference_release':False}
