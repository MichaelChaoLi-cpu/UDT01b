"""Versioned numerical reference rules. No training, solver execution or data edits."""
import numpy as np
VERSION='reference-admission-v2'
POLICY={
 'version':VERSION,'status':'frozen_engineering_screen_not_observational_validation',
 'exact_width_m':1e-7,'maximum_interval_width_m':.01,'interval_roundoff_m':1e-6,
 'negative_bookkeeping_volume_threshold_m3':-.001,'barrier_contact_margin_m':.01,
 'hard_cell_quarantine':['invalid or missing bounds','sampled native bookkeeping volume below -0.001m3','native peak reaches assumed building barrier minus0.01m'],
 'wide_interval':'withhold from primary regression, retain interval and flag; do not midpoint-fill',
 'no_automatic_exclusion':['cross-ward location','PLATEAU fallback','subcell connectivity ambiguity','uncertain structure class'],
 'parent_pool':'same parents with nonzero exact support for exact and interval methods; interval-only parents reported separately',
 'comparison_support':'same per-event label masks for all methods; matched-grid diagnostic comparisons use union of both numerical quarantine masks',
 'spatial_extent':'direct affected common cell only; not a proven bound on error propagation; no ad hoc buffer or test-informed mask editing',
 'unknown_zero_imputation':False,'interval_midpoint_imputation':False,
 'peak_thresholds_m':[.05,.1,.3,.5],'wet_rule':'lower > threshold = certainly wet; upper <= threshold = certainly dry; otherwise ambiguous',
 'response_flags_are_predictors':False,
 'event_hard_hold':['incomplete execution or horizon','unverified actual state/flux integration clock','identity/hash/geometry mismatch','missing/invalid required state or boundary flux','failed water balance','unproved shallow censoring','no usable label support'],
 'partial_reference':'admitted cells are conditional numerical labels only; report quarantined and interval area plus certainly-wet coverage by ward and domain',
 'continuous_negative_detection':'current native volumes sampled every300s; no claim that unsampled negative states are excluded',
 'formal_reference_release':False,'training_release':False,'sealed_response_release':False,
 'decision_record':'MILI-D-20261015-006','threshold_authority':'Agent-specified versioned engineering gates; not user-supplied accuracy standards.'}
CONTRACT={
 'version':'hydraulic-output-v1','frozen_before_new_responses':True,
 'scope':'fixed whole three-ward and adjoining model domain; no action; physical stage/epoch validity unestablished',
 'identity':['parent_id','family_id','partition','forcing_semantic_sha256','geometry_sha256','solver_image_digest','native_mapping_sha256','ground_support_sha256','policy_sha256','output_contract_sha256'],
 'existing_control_adapter':'four already-inspected controls share CONTROL-LEGACY; historical metadata retained without retroactive event inflation',
 'response_time_origin':'synthetic2020-01-01 clock, not measurement date',
 'horizon_s':14400,'time_tolerance_s':.001,'coordinate_tolerance_m':.001,
 'required_completion':'returncode0, completion marker, map and history status0, full0..14400 horizon and verified input hashes',
 'required_peak_profile':{
  'native_state_max_gap_s':300.001,
  'actual_clock_requirement':'Record actual solver integration times for states and boundary flux, or verified equivalent step-integrated budgets at state checkpoints. Nominal output time alone cannot certify prefix closure. Do not shift clocks to force closure.','native_peak_time_s':14400,'native_peak_windows':1,
  'map_variables':{'msk':'native interior1/boundary2','zb':'m working datum, native minimum','zs':'m; censored below source huthresh, keep mask','zsmax':'m; online maximum over full horizon, not max of snapshots','subgrid_volume':'signed m3 bookkeeping storage, never clip for budget','h':'m above native minimum, NOT eligible-ground mean depth','hmax':'solver whole-cell/storage definition, NOT training label','u':'m s-1, instantaneous diagnostic','v':'m s-1, instantaneous diagnostic','cumprcp':'m, final cumulative report; rounding diagnostic only'},
  'history_max_gap_s':.401,'history_variables':{'crosssection_discharge':'m3 s-1 instantaneous signed discharge','crosssection_name':'exact geometric boundary section order','status':'0 complete'},
  'boundary_requirement':'all interior-to-prescribed-boundary faces exactly once with outward-positive sign; zero-width faces retained; group codes not names; no missing-flux-to-zero conversion',
  'label_target':'max over time of original eligible fine-ground positive depth volume divided by common10m eligible-ground area',
  'shallow_censoring':'only prove using pinned writer, active mask, finite native minimum and huthresh0.01; missing variable/file is not shallow censoring',
  'regular_peak':'native-aligned10m monotone ground integration at online zsmax; censored peak retains [lower,upper]',
  'adaptive_peak':'synchronous aggregate lower bound and native-peak upper bound; preserve intervals, never mix as primary truth',
  'minimum_observability':'snapshot-screened negative volume; online every-step minima/cumulative diagnostics recommended for stronger certification, not inferred from sampled frames'},
 'budget':{
  'rain_volume_relative_tolerance':1e-6,'endpoint_relative_tolerance':.001,'prefix_relative_tolerance':.01,'absolute_floor_m3':1.,
  'equation':'storage(t)-storage(0)+boundary_out(t)-boundary_in(t)-rain(t)-other_sources(t)+other_sinks(t)',
  'scale':'max(rain+other_sources+gross_boundary_in, gross_boundary_out+other_sinks, abs(initial_storage), 1m3)',
  'gate':'abs(residual) <= max(1m3, relative_tolerance*scale); final0.001, every written state prefix0.01',
  'gross_exchange':'split positive and negative on each boundary face BEFORE summing groups; avoids cancelling simultaneous outflow and inflow',
  'storage':'signed native bookkeeping volume on interior cells; do not discard quarantined cells or negative storage to manufacture closure',
  'rain':'integrate exact physical/effective forcing knots; roof already included; no second roof source',
  'instantaneous_flux_integral':'trapezoid over actual written times, not mean dt; retain right-rectangle alternative and difference as sampling diagnostic',
  'quadrature_difference_gate':'final trapezoid/right-rectangle difference <= max(1m3,0.001*scale); not a rigorous unsampled-flux error bound',
  'zero_rain':'include incoming boundary water and signed storage; never divide by rain alone',
  'other_sources_and_sinks':'none in baseline; unknown/nonzero unaccounted pipes, pumps, infiltration, evaporation or discharges => hold and explicit contract extension',
  'cumprcp':'audit reported final cumulative precipitation separately; float32 accumulation roundoff cannot replace intended rain volume',
  'unit_of_balance':'whole hydraulic domain only; per-ward closure needs cross-ward internal-face flux, which is absent'},
 'temporal_profiles':{
  'peak_only':'online native peaks plus max300s native states and max0.401s complete boundary discharge; four-hour target only',
  'coarse_evolution':'existing300s frames diagnostic only; cannot certify onset, wet duration, particle paths or full recession',
  'dynamic_candidate':'future max60s synchronized full fields, compare30s on preselected development cases; not automatically qualified by cadence',
  'dynamic_conservation_control':'requires interval-integrated every-native-face flux and rain/source/sink increments, storage at matched endpoints, connectivity and optional restart; cell-centre velocities and boundary-only histories insufficient',
  'dynamic_release':False,'temporal_tolerance_for_future_timing_metrics':'must be specified in later authorized dynamic study before interpreting onset/duration',
  'no_new_output_generation':True},
 'split_custody':'all descendants inherit parent family; no new sealed/challenge response reads in development qualification; fail a held-out event using pre-frozen rule, never retune or silently discard it',
 'hold_behavior':'missing diagnostics = hold, failure = hold whole event; retain raw results and error reasons; never auto-repair inputs or rerun',
 'training_authorized':False,'new_simulations_authorized':False,'physical_event_validation':False}

def classify_labels(lower,upper,eligible,negative,barrier,*,interval_source_known=True):
 lower=np.asarray(lower,dtype='f8');upper=np.asarray(upper,dtype='f8');eligible=np.asarray(eligible,bool)
 for a in (upper,eligible,negative,barrier):
  if np.shape(a)!=lower.shape:raise ValueError('Label/mask shapes must match')
 finite=eligible&np.isfinite(lower)&np.isfinite(upper)&(lower>=0)&(upper>=lower)
 width=upper-lower
 unproved=finite&(width>POLICY['exact_width_m'])&~np.broadcast_to(np.asarray(interval_source_known,bool),lower.shape)
 quarantine=eligible&(~finite|np.asarray(negative,bool)|np.asarray(barrier,bool)|unproved)
 exact=finite&(width<=POLICY['exact_width_m'])&~quarantine
 interval=finite&(width>POLICY['exact_width_m'])&(width<=POLICY['maximum_interval_width_m']+POLICY['interval_roundoff_m'])&~quarantine
 wide=finite&(width>POLICY['maximum_interval_width_m']+POLICY['interval_roundoff_m'])&~quarantine
 return {'Exact Label Mask':exact,'Interval Label Mask':interval,'Wide Interval Mask':wide,'Quarantine Mask':quarantine,'Invalid Bounds Mask':eligible&~finite,'Unproved Interval Mask':unproved,'Admitted Mask':exact|interval}

def integrate_rate_at(times,rates,queries):
 """Exact integral of a piecewise-linear rate; time in seconds, arbitrary units."""
 t=np.asarray(times,dtype='f8');r=np.asarray(rates,dtype='f8');q=np.asarray(queries,dtype='f8')
 if t.ndim!=1 or r.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0) or not np.all(np.isfinite(r)):raise ValueError('Invalid rate series')
 if np.any(q<t[0]) or np.any(q>t[-1]) or not np.all(np.isfinite(q)):raise ValueError('Query outside forcing support')
 cumulative=np.r_[0,np.cumsum(np.diff(t)*(r[:-1]+r[1:])*.5)]
 i=np.clip(np.searchsorted(t,q,side='right')-1,0,len(t)-2);dt=q-t[i]
 return cumulative[i]+r[i]*dt+.5*(r[i+1]-r[i])/np.diff(t)[i]*dt*dt

def balance_check(time,storage,rain,outward,inward,other_source=None,other_sink=None):
 vectors=[np.asarray(x,dtype='f8') for x in (time,storage,rain,outward,inward)]
 t,s,r,o,i=vectors
 if any(x.shape!=t.shape for x in vectors) or len(t)<2 or not all(np.all(np.isfinite(x)) for x in vectors) or not np.all(np.diff(t)>0):raise ValueError('Missing/nonfinite or misaligned balance series')
 a=np.zeros(t.shape) if other_source is None else np.asarray(other_source,dtype='f8');b=np.zeros(t.shape) if other_sink is None else np.asarray(other_sink,dtype='f8')
 if a.shape!=t.shape or b.shape!=t.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):raise ValueError('Invalid source/sink series')
 if any(np.any(x < -1e-8) or np.any(np.diff(x)<-1e-8) for x in (r,o,i,a,b)):raise ValueError('Cumulative amounts must be nonnegative and monotone')
 if any(abs(x[0])>1e-8 for x in (r,o,i,a,b)):raise ValueError('Cumulative amounts must start at0')
 residual=s-s[0]+o-i-r-a+b
 scale=np.maximum.reduce([r+a+i,o+b,np.full(t.shape,abs(s[0])),np.ones(t.shape)])
 rules=CONTRACT['budget'];prefix_gate=np.maximum(rules['absolute_floor_m3'],rules['prefix_relative_tolerance']*scale);end_gate=max(rules['absolute_floor_m3'],rules['endpoint_relative_tolerance']*scale[-1])
 return {'residual':residual,'scale':scale,'prefix_gate':prefix_gate,'endpoint_gate':end_gate,
 'endpoint_pass':bool(abs(residual[-1])<=end_gate),'prefix_pass':bool(np.all(abs(residual)<=prefix_gate)),
 'end_relative_residual':float(abs(residual[-1])/scale[-1]),'max_prefix_relative_residual':float(np.max(abs(residual)/scale))}

def event_admission(required_checks,masks):
 missing=[k for k in ['completion','identity','geometry','fields','time','clock_alignment','boundary','rain','budget','censoring'] if required_checks.get(k) is not True]
 if missing:return {'state':'hold_event','reasons':missing,'technical_reference':False,'training_release':False}
 if not np.any(masks['Admitted Mask']):return {'state':'hold_event','reasons':['no admitted support'],'technical_reference':False,'training_release':False}
 state='partial_snapshot_screened' if np.any(masks['Quarantine Mask']|masks['Wide Interval Mask']) else 'snapshot_screened'
 return {'state':state,'reasons':[],'technical_reference':True,'core_exact_pool':bool(np.any(masks['Exact Label Mask'])),'training_release':False}
