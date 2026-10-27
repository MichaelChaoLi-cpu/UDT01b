"""Independent checks on stored station risk-screen outputs and bindings."""
from pathlib import Path
import csv,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'data/exp/data-preprocessing/three_ward_station_surface_risk_v1';P=ROOT/'data/processed/three_ward_station_surface_risk_v1'
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 results={};m=json.loads((R/'delivery_manifest.json').read_text());results['all_product_and_script_hashes_match']=all(sha(ROOT/r['file'])==r['sha256'] for r in m['products']+m['scripts']);rr=rows(P/'entrance_scenario_surface_screen.csv');groups=collections.defaultdict(dict)
 for r in rr:groups[(r['case_id'],r['osm_node_id'])][int(r['radius_m'])]=r
 monotone=True;areaok=True;ordered=True
 for g in groups.values():
  for r in g.values():
   a=float(r['admitted_open_ground_m2']);ordered &= r['neighborhood_peak_mean_depth_lower_max_m']=='' or float(r['neighborhood_peak_mean_depth_lower_max_m'])<=float(r['neighborhood_peak_mean_depth_upper_max_m'])+1e-9
   for t in ['0p1','0p3','0p5']:areaok &= 0<=float(r['ground_area_lower_ge_'+t+'_m2'])<=float(r['ground_area_upper_ge_'+t+'_m2'])+1e-7<=a+1e-6
  for small,big in [(10,20),(20,30)]:
   for key in ['neighborhood_peak_mean_depth_lower_max_m','neighborhood_peak_mean_depth_upper_max_m']:
    if g[small][key]!='':monotone &= g[big][key]!='' and float(g[big][key])+1e-9>=float(g[small][key])
 results['radius_expansion_preserves_or_increases_nearby_peak']=monotone;results['exceedance_areas_ordered_within_admitted_support']=areaok;results['all_stored_bounds_ordered']=ordered;results['all_16_events_have_all_675_points_and_3_radii']=len(rr)==16*675*3 and len(groups)==16*675
 events=rows(P/'scenario_bindings.csv');results['only_open_train_validation_labels']=all(r['split'] in ['train','validation'] for r in events) and len(events)==16;results['all_label_bindings_unchanged']=all(sha(ROOT/r['label_path'])==r['label_sha256'] for r in events)
 s=rows(P/'station_screen_summary.csv');e=rows(P/'entrance_screen_summary.csv');results['station_identity_units_and_core_entries_preserved']=len(s)==114 and sum(int(r['station_line_records']) for r in s)==154 and sum(r['inside_core']=='True' for r in e)==673;results['no_station_ingress_or_shutdown_inferred']=all(r['interior_water_ingress']=='unknown' and r['station_shutdown']=='not_assessed' for r in s)
 assert all(results.values()),results
 previous=json.loads((R/'delivery_validation.json').read_text());previous.update({'independent_checks':results,'all_independent_checks_passed':True});(R/'delivery_validation.json').write_text(json.dumps(previous,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
