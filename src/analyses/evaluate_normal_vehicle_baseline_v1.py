"""Evaluate conditional shortest-distance passages against held-out census screenlines.
No OD, routing, capacity, hydraulic or survey-source edits. Scalar fits are diagnostics.
"""
from pathlib import Path
import os,json,csv,hashlib,datetime as dt
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1'
R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_baseline_v1'
B=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1'
M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
PERIODS=['daily','hour8','hour10','hour17','daytime12h']
BRANCHES=['base_no_remote_remote','plus_conditional_chord_through']
def dump(p,x):
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()if isinstance(v,np.generic)else str(v))+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def metric(g):
 x=g.predicted.to_numpy();y=g.observed.to_numpy();n=len(g)
 return {'n':n,'mae':float(np.mean(abs(x-y)))if n else None,'wape':float(abs(x-y).sum()/y.sum())if n and y.sum()>0 else None,'prediction_observation_ratio':float(x.sum()/y.sum())if n and y.sum()>0 else None,'spearman':float(spearmanr(x,y).statistic)if n>2 and np.ptp(x)>0 and np.ptp(y)>0 else None,'zero_prediction_fraction':float(np.mean(x<1e-9))if n else None}
def main():
 policy={'policy_write_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'primary_target':'Observed small-vehicle 7–19h totals on qualified unique paired-direction screenlines','secondary':'All-vehicle count, true24h counts and three hourly periods; class and clock mismatches retained','diagnostic_fit':'One nonnegative least-squares scalar per branch on calibration small-vehicle12h units only; freeze for validation and other periods; never apply to source OD or raw modeled flux','split':'Reuse saved SHA256 observation-section split; no re-selection based on prediction','claim_limit':'Conditional source-conserving demand and distance assignment; no speed or congestion model; survey unit holdout is not independent geographic holdout'}
 
 if not(P/'evaluation_policy.json').exists():dump(P/'evaluation_policy.json',policy)
 cross=pd.read_csv(P/'census_screenline_crosswalk.csv',dtype={'count_authority_code':str,'count_unit_id':str,'observation_section_id':str,'source_road_type':str})
 profiles=pd.read_csv(B/'hourly_profile_quality.csv',dtype={'count_authority_code':str,'count_unit_id':str,'direction_code':str,'vehicle_class_code':str,'observed_estimated_code':str})
 paired={k:len(g)==4 and set(zip(g.direction_code,g.vehicle_class_code))=={('1','1'),('1','2'),('2','1'),('2','2')}for k,g in profiles.groupby(['count_authority_code','count_unit_id'])}
 cross['paired_directions_and_classes']=[paired[(a,b)]for a,b in zip(cross.count_authority_code,cross.count_unit_id)]
 eligible=cross.matching_status.eq('qualified_route_gated_screenline')&cross.all_profiles_observed.eq(True)&~cross.duplicate_observation_unit.eq(True)&cross.paired_directions_and_classes
 cross['eligible_observation_comparison']=eligible
 z=load(P/'pilot_arc_flows.npz');pr=load(M/'road_projection.npz');inv=pd.read_csv(M/'road_inventory.csv');roadflow=np.array([[np.bincount(pr['arc_to_road'],weights=z['arc_flow'][b,t],minlength=len(inv))for t in range(5)]for b in range(2)])
 rows=[]
 for r in cross.itertuples():
  if not r.eligible_observation_comparison:continue
  idx=np.array([int(x)for x in str(r.matched_road_indices).split('|')]);assert len(idx)==len(set(idx))
  for b,branch in enumerate(BRANCHES):
   for t,period in enumerate(PERIODS):
    predicted=float(roadflow[b,t,idx].sum())
    for cls in ['small','all']:
     observed=getattr(r,'observed_'+cls+'_'+period)
     if np.isfinite(observed):rows.append({'count_authority_code':r.count_authority_code,'count_unit_id':r.count_unit_id,'observation_section_id':r.observation_section_id,'source_route_name':r.source_route_name,'source_road_type':r.source_road_type,'split':r.split,'branch':branch,'period':period,'comparison_class':cls,'predicted':predicted,'observed':float(observed),'n_physical_roads':len(idx)})
 raw=pd.DataFrame(rows);factors=[];metrics=[];adjusted=[]
 for branch in BRANCHES:
  cal=raw[raw.branch.eq(branch)&raw.period.eq('daytime12h')&raw.comparison_class.eq('small')&raw.split.eq('calibration')];x=cal.predicted.to_numpy();y=cal.observed.to_numpy();factor=max(0.,float(np.dot(x,y)/np.dot(x,x)))if np.dot(x,x)>0 else 0.
  factors.append({'branch':branch,'fit_class':'small','fit_period':'daytime12h','fit_split':'calibration','calibration_units':len(cal),'scalar_diagnostic':factor,'input_od_modified':False})
  scaled=raw[raw.branch.eq(branch)].copy();scaled['predicted']*=factor;adjusted.append(scaled)
 for kind,df in [('raw',raw),('calibration_scalar_diagnostic',pd.concat(adjusted,ignore_index=True))]:
  for key,g in df.groupby(['branch','period','comparison_class','split'],sort=True):metrics.append(dict(zip(['branch','period','comparison_class','split'],key),prediction_kind=kind,**metric(g)))
 raw.to_csv(P/'census_passage_comparison_raw.csv',index=False,encoding='utf-8-sig');pd.concat(adjusted,ignore_index=True).to_csv(P/'census_passage_comparison_scalar_diagnostic.csv',index=False,encoding='utf-8-sig');pd.DataFrame(metrics).to_csv(P/'census_passage_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(factors).to_csv(P/'calibration_scalar_diagnostics.csv',index=False,encoding='utf-8-sig');cross.to_csv(P/'census_comparison_eligibility.csv',index=False,encoding='utf-8-sig')
 # Descriptive class stratification after the predeclared comparison; no new fit.
 classmetrics=[]
 for branch in BRANCHES:
  subset=raw[raw.branch.eq(branch)&raw.period.eq('daytime12h')&raw.comparison_class.eq('small')].copy()
  subset['road_class']=np.where(subset.source_road_type.isin(['1','2']),'expressway','ordinary')
  for (roadclass,split),g in subset.groupby(['road_class','split']):classmetrics.append({'branch':branch,'road_class':roadclass,'split':split,**metric(g)})
 pd.DataFrame(classmetrics).to_csv(P/'census_road_class_diagnostics.csv',index=False,encoding='utf-8-sig')
 graph=load(M/'source_motor_graph.npz');scc=graph['strong_component'];maincomp=int(np.argmax(np.bincount(scc)));highway=inv.highway.to_numpy()[pr['arc_to_road']];express=np.char.startswith(highway.astype(str),'motorway');inmain=(scc[pr['from_index']]==maincomp)&(scc[pr['to_index']]==maincomp);usage=[]
 for branch,b in zip(BRANCHES,range(2)):
  km=z['arc_flow'][b,4]*pr['length_m']/1000
  for h in sorted(set(highway)):usage.append({'branch':branch,'highway':h,'daytime12h_candidate_passage_km':float(km[highway==h].sum()),'used_directed_arcs':int((z['arc_flow'][b,4,highway==h]>1e-9).sum()),'source_directed_arcs':int((highway==h).sum())})
 pd.DataFrame(usage).to_csv(P/'road_class_assignment_usage.csv',index=False,encoding='utf-8-sig')
 structural={'source_motorway_and_link_directed_arcs':int(express.sum()),'motorway_and_link_arcs_with_both_ends_in_main_scc':int((express&inmain).sum()),'main_scc_node_count':int((scc==maincomp).sum()),'interpretation':'All demand access representatives were selected in the main SCC. Most motorway/ramp arcs are outside that two-way component in the conservative source graph. This establishes a baseline support limitation, not the real-world closure or exact cause of each missing connection.'}
 # Disjoint routing ledger: remote source quantities are not called held core demand.
 dem=load(P/'routing_demands.npz');same=dem['same_representative'];local=dem['local_base_candidate'];through=dem['chord_through_candidate'];categories={'routed_at_least_one_local_representative':local&~same,'held_local_same_representative':local&same,'conditional_remote_through':~local&through&~same,'outside_remote_same_portal':~local&same,'outside_remote_other':~local&~same&~through};account=[]
 fine=pd.read_csv(P/'conditional_fine_driver_od.csv',dtype={'origin_code':str,'destination_code':str});corecodes=set(pd.read_csv(B/'pt_2018_core_zone_codes.csv',dtype=str).planning_zone);coretouch=fine.origin_code.isin(corecodes)|fine.destination_code.isin(corecodes)
 for name,mask in categories.items():account.append({'category':name,'od_pairs':int(mask.sum()),'daily_candidate':float(dem['period_weights'][mask,0].sum()),'core_touching_daily_candidate':float(dem['period_weights'][mask&coretouch.to_numpy(),0].sum())})
 pd.DataFrame(account).to_csv(P/'routing_demand_account.csv',index=False,encoding='utf-8-sig')
 # Speeds: source attributes and native measurement flags, not modeled speed validation.
 nativefile=ROOT/'data/processed/three_ward_traffic_data_search_v1/sources/mlit_tokyo_2021_sections.csv'
 with nativefile.open(encoding='cp932')as f:rr=list(csv.reader(f))
 native={r[0]:r for r in rr[1:]};speed=[]
 for r in cross.itertuples():
  sid=str(r.observation_section_id);file=P/'sources'/f'{sid}.json'
  if not file.exists()or sid not in native:continue
  attrs=json.loads(file.read_text(encoding='utf-8-sig'));nat=native[sid]
  for period,entries in [('peak',[(1,72,75),(2,79,82)]),('offpeak',[(1,86,89),(2,93,96)])]:
   for direction,flagcol,valuecol in entries:
    try:v=float(nat[valuecol].replace(',',''))
    except ValueError:v=np.nan
    speed.append({'count_authority_code':r.count_authority_code,'count_unit_id':r.count_unit_id,'observation_section_id':sid,'speed_survey_unit':attrs.get('旅行速度調査単位区間番号',''),'eligible_flow_comparison':r.eligible_observation_comparison,'period':period,'direction':direction,'native_measurement_flag':nat[flagcol],'source_speed_kmh':v})
 speeds=pd.DataFrame(speed);speeds.to_csv(P/'source_speed_evidence.csv',index=False,encoding='utf-8-sig');available=speeds[speeds.eligible_flow_comparison&speeds.source_speed_kmh.gt(0)];speedcoverage={'qualified_flow_units':int(eligible.sum()),'with_any_positive_speed_attribute':int(available[['count_authority_code','count_unit_id']].drop_duplicates().shape[0]),'distinct_source_speed_survey_units':int(available.speed_survey_unit.replace('',np.nan).dropna().nunique()),'native_speed_flag_counts':available.native_measurement_flag.value_counts().to_dict(),'source_peak_direction_median_kmh':float(available.loc[available.period.eq('peak'),'source_speed_kmh'].median()),'source_offpeak_direction_median_kmh':float(available.loc[available.period.eq('offpeak'),'source_speed_kmh'].median()),'modeled_speeds':False,'speed_calibration':False,'capacity_calibration':False,'speed_unit_warning':'Speed survey section may differ from flow observation section and multiple count units may share one speed section.'}
 # File bindings checked against the state prepared before the pilot.
 bindings=json.loads((P/'input_bindings.json').read_text())['sha256'];hashchecks={name:sha(ROOT/name)==expected for name,expected in bindings.items()}
 calids=set(raw.loc[raw.split.eq('calibration'),'observation_section_id']);valids=set(raw.loc[raw.split.eq('validation'),'observation_section_id']);pilot=json.loads((R/'pilot_validation.json').read_text());prep=json.loads((R/'preparation_summary.json').read_text());sums=np.sum([m.astype(int)for m in categories.values()],axis=0)
 checks={'source_bindings_unchanged':all(hashchecks.values()),'pilot_numerical_checks':pilot['passed'],'middle_zone_conservation':prep['checks']['middle_controls_conserved'],'routing_categories_are_disjoint_and_complete':bool((sums==1).all()),'allocated_source_total_reconciles':abs(sum(r['daily_candidate']for r in account)-prep['allocated_fine_driver_candidate_daily'])<1e-6,'native_region_total_reconciles':abs(prep['allocated_fine_driver_candidate_daily']+prep['held_no_fine_seed_daily']+prep['unlocated_or_other_source_geography_daily']-prep['whole_region_native_driver_candidate_daily'])<1e-6,'no_calibration_validation_source_overlap':not bool(calids&valids),'observations_only':bool(raw.observed.notna().all()and cross.loc[eligible,'all_profiles_observed'].all()),'one_qualified_cross_section_per_observation':not cross.loc[eligible,'observation_section_id'].duplicated().any(),'both_directions_two_classes':bool(cross.loc[eligible,'paired_directions_and_classes'].all()),'nonnegative_finite_passages':bool(np.isfinite(raw.predicted).all()and(raw.predicted>=0).all()),'validation_not_used_for_scalar_fit':all(f['fit_split']=='calibration'for f in factors),'raw_demand_not_rescaled':all(not f['input_od_modified']for f in factors)}
 dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks,'source_hash_checks':hashchecks})
 primary=[m for m in metrics if m['period']=='daytime12h'and m['comparison_class']=='small'];splitcounts=raw[raw.branch.eq(BRANCHES[0])&raw.period.eq('daytime12h')&raw.comparison_class.eq('small')].split.value_counts().to_dict()
 summary={'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'stage':'Candidate normal-driver demand and distance assignment checked; not released as calibrated congestion baseline','qualified_observed_12h_units':int(sum(splitcounts.values())),'split_units':splitcounts,'primary_small_12h_metrics':primary,'scalar_diagnostics':factors,'demand_account':account,'core_touching_candidate_total_daily':float(dem['period_weights'][coretouch.to_numpy(),0].sum()),'source_speed_coverage':speedcoverage,'road_class_diagnostics':classmetrics,'network_structure_diagnostic':structural,'through_branch_added_daily':float(z['routed'][1,0]-z['routed'][0,0]),'checks_passed':sum(checks.values()),'checks_total':len(checks),'calibrated_speed_or_capacity':False,'new_hydraulic_cases':0,'affected_person_counts':False,'limitations':['2018 driver-candidate proxy,2021 normal census counts and2026 road snapshot differ in time and vehicle coverage','One point per local PT zone,conditional remote portals,all-or-nothing shortest distance and person-clock proxy are assumptions','Same-zone local trips are held; source unknowns and unsupported fine-seed controls remain held','Census observation-unit split does not guarantee independent geographic,route or event generalization','No speed/capacity response or traffic interaction in distance pilot; zero-flow roads do not establish no actual traffic']}
 dump(R/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
 if not all(checks.values()):raise SystemExit('Candidate evaluation integrity check failed')
if __name__=='__main__':main()
