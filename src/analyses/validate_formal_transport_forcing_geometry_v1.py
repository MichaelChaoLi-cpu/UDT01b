"""Independent product-level qualification of the completed factorial analysis."""
from pathlib import Path
import json,csv,hashlib
from html.parser import HTMLParser
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_formal_transport_experiments_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
R=ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1'
Q=P/'factor_comparison_v1';O=R/'B2_B3_forcing_geometry_comparison'
def read(p):
 with p.open()as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 checks=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)))
  if not v:raise AssertionError(name)
 runs=json.loads((P/'run_manifest.json').read_text())['runs'];s=np.load(Q/'common_support.npz');mask=s['common_admitted'];ward=s['ward_code'];area=s['ground_area'];road=s['road_area'];expected=s['eligible'].copy();arrays={}
 for run in runs:
  case=run['case_id'];old=run.get('original_case_id',case);base=H if run['reused']else P
  label=np.load(base/'execution'/(old+'_candidate_reference.npz'));good=s['eligible']&label['Admitted Mask']&~label['Quarantine Mask']&np.isfinite(label['Peak Mean Depth Lower'])&np.isfinite(label['Peak Mean Depth Upper']);expected &= good
  z=np.load(Q/(case+'_peak_bounds.npz'));arrays[case]={k:z[k]for k in z.files}
  check(case+'_original_ground_bounds_preserved',np.array_equal(z['ground_lower'],label['Peak Mean Depth Lower'],equal_nan=True)and np.array_equal(z['ground_upper'],label['Peak Mean Depth Upper'],equal_nan=True))
 check('exact_intersection_of_all_twelve_original_masks',np.array_equal(expected,mask))
 rows=read(O/'case_peak_extent.csv');lookup={(x['case_id'],x['scope'],x['support'],float(x['threshold_m'])):x for x in rows}
 check('complete_fixed_peak_extent_matrix',len(rows)==12*4*2*3 and len(lookup)==len(rows))
 for case,z in arrays.items():
  for name,weights in [('ground',area),('road',road)]:
   lo=z[name+'_lower'];up=z[name+'_upper'];sel=mask&(ward>0)&(weights>0)
   check(case+name+'_finite_ordered_bounds',np.isfinite(lo[sel]).all()and np.isfinite(up[sel]).all()and np.all(up[sel]>=lo[sel]))
   for threshold in [.1,.3,.5]:
    row=lookup[case,'Three wards',name,threshold];low=weights[sel&(lo>threshold)].sum()/1e6;high=weights[sel&(up>threshold)].sum()/1e6
    check(f'{case}_{name}_{threshold}_extent',abs(low-float(row['certain_km2']))<1e-10 and abs(high-float(row['possible_km2']))<1e-10)
    check(f'{case}_{name}_{threshold}_ward_partition',abs(sum(float(lookup[case,w,name,threshold]['certain_km2'])for w in ['Chiyoda','Chuo','Minato'])-low)<1e-10)
 effects=read(O/'conditional_forcing_contrasts.csv')
 check('forcing_contrast_matrix_complete',len(effects)==2*2*5*4*2*3)
 for rain in ['R1','R125']:
  for arm in ['T','U']:
   for name in ['ground','road']:
    for threshold in [.1,.3,.5]:
     these={x['effect']:x for x in effects if x['rainfall']==rain and x['arm']==arm and x['scope']=='Three wards'and x['support']==name and float(x['threshold_m'])==threshold}
     # Interval algebra: both decompositions must contain the total difference.
     a=lookup[f'{rain}__H__{arm}','Three wards',name,threshold];b=lookup[f'R0__L__{arm}','Three wards',name,threshold]
     total_lo=float(a['certain_km2'])-float(b['possible_km2']);total_hi=float(a['possible_km2'])-float(b['certain_km2'])
     lo=sum(float(these[k]['difference_lower_km2'])for k in ['rain_at_low_stage','stage_without_rain','rain_stage_interaction']);hi=sum(float(these[k]['difference_upper_km2'])for k in ['rain_at_low_stage','stage_without_rain','rain_stage_interaction'])
     check(f'{rain}{arm}{name}{threshold}_factor_reconstruction',lo<=total_lo+1e-10 and hi>=total_hi-1e-10)
     check(f'{rain}{arm}{name}{threshold}_conditional_path_reconstruction',float(these['stage_without_rain']['difference_lower_km2'])+float(these['rain_at_raised_stage']['difference_lower_km2'])<=total_lo+1e-10 and float(these['stage_without_rain']['difference_upper_km2'])+float(these['rain_at_raised_stage']['difference_upper_km2'])>=total_hi-1e-10)
 stability=read(O/'geometry_stability.csv')
 for row in stability:
  if row['scope']!='Three wards':continue
  stage=row['stage'];name=row['support'];threshold=float(row['threshold_m']);w=area if name=='ground'else road;sel=mask&(ward>0)&(w>0)
  z1=arrays[f'R1__{stage}__U'];t1=arrays[f'R1__{stage}__T'];z2=arrays[f'R125__{stage}__U'];t2=arrays[f'R125__{stage}__T']
  a=(z1[name+'_lower']>threshold)&(t1[name+'_upper']<=threshold);b=(z2[name+'_lower']>threshold)&(t2[name+'_upper']<=threshold)
  both=w[sel&a&b].sum()/1e6;union=w[sel&(a|b)].sum()/1e6
  check(stage+name+str(threshold)+'_independent_added_overlap',abs(both-float(row['added_retained_km2']))<1e-10 and(not union or abs(both/union-float(row['added_pattern_jaccard']))<1e-10))
 transitions=read(O/'threshold_transition_explanation.csv')
 check('threshold_transition_matrix',len(transitions)==4)
 for row in transitions:
  total=float(row['original_U_only_wet_km2']);parts=[float(row[k])for k in ['remains_U_only_wet_km2','both_arms_wet_under_stronger_rain_km2','both_arms_dry_under_stronger_rain_km2','reversed_to_T_only_wet_km2','uncertain_km2']]
  check(row['stage']+row['support']+'_transition_partition',min(parts)>=0 and abs(sum(parts)-total)<1e-10)
  match=next(x for x in stability if x['stage']==row['stage']and x['support']==row['support']and x['scope']=='Three wards'and float(x['threshold_m'])==float(row['threshold_m']))
  check(row['stage']+row['support']+'_transition_matches_persistence',abs(parts[0]-float(match['added_retained_km2']))<1e-10)
 budget=read(O/'full_domain_water_budget.csv');groups=read(O/'water_group_exchange.csv')
 for row in budget:
  balance=float(row['rain_applied_m3'])+float(row['gross_in_m3'])-float(row['gross_out_m3'])-float(row['storage_change_m3'])
  check(row['case_id']+'_full_domain_budget',abs(balance)<=float(row['max_abs_budget_residual_m3'])+1e-7)
  parts=[x for x in groups if x['case_id']==row['case_id']]
  check(row['case_id']+'_directional_group_totals',len(parts)==7 and all(abs(sum(float(x[k])for x in parts)-float(row[k]))<1e-5 for k in ['gross_in_m3','gross_out_m3']))
 times=read(O/'full_domain_budget_at_times.csv');check('actual_sample_clocks',len(times)==144 and max(abs(float(x['actual_time_s'])-float(x['scheduled_time_s']))for x in times)<.001)
 stations=read(O/'station_exposure_summary.csv');detail=read(O/'station_exposure_detail.csv');check('12_fixed_times_and_separate_peak',len(stations)==156 and len(detail)==156*675)
 check('shared_station_quality_cohort',len({x['paired_quality_points']for x in stations})==1)
 for case in arrays:
  q=[x for x in stations if x['case_id']==case];check(case+'_station_bounds_and_peak',all(int(x['trigger_lower_points'])<=int(x['trigger_upper_points'])for x in q)and int(next(x for x in q if x['frame_id']=='peak_envelope')['trigger_lower_points'])>=max(int(x['trigger_lower_points'])for x in q if x['frame_id']!='peak_envelope'))
 original=json.loads((R/'B1_geometry_comparison/source_bindings.json').read_text())
 check('B1_bound_sources_unchanged',all(sha(ROOT/k)==v for k,v in original.items()))
 bind=json.loads((O/'source_bindings.json').read_text());check('new_sources_unchanged',all(sha(ROOT/k)==v for k,v in bind.items()))
 summary=json.loads((O/'summary.json').read_text());check('no_simulation_or_population_or_network_inference',summary['new_hydraulic_simulations']==0 and all(summary[k]is False for k in ['people_counted','training','sealed_tests_opened','transport_accessibility_computed','real_world_accuracy_verified']))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.paths=[]
  def handle_starttag(self,t,a):
   for k,v in a:
    if k in ['href','src']and v and not v.startswith(('http:','https:','#','data:')):self.paths.append(v)
 parser=Links();parser.feed((O/'review.html').read_text());future={'independent_validation.json','delivery_manifest.json'}
 check('html_assets_available',all((O/p).exists()or p in future for p in parser.paths))
 result=dict(all_passed=True,checks=checks,simulations_started=0);(O/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 sources=[x for x in O.iterdir()if x.is_file()and x.name!='delivery_manifest.json']+list(Q.iterdir())+[ROOT/'src/analyses/compare_formal_transport_forcing_geometry_v1.py',Path(__file__)]
 (O/'delivery_manifest.json').write_text(json.dumps(dict(files={str(x.relative_to(ROOT)):sha(x)for x in sources}),indent=2)+'\n')
 print('INDEPENDENT VALIDATION',len(checks),'PASSED')
if __name__=='__main__':main()
