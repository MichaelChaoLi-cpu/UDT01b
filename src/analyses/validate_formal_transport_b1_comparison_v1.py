"""Independent saved-product checks for B1; no hydraulic execution."""
from pathlib import Path
import sys,json,csv,hashlib
from html.parser import HTMLParser
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1'
P=ROOT/'data/processed/three_ward_formal_transport_experiments_v1'
O=R/'B1_geometry_comparison'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open()as f:return list(csv.DictReader(f))
def main():
 checks=[]
 def check(n,v):checks.append(dict(check=n,pass_=bool(v)));assert v,n
 bindings=json.loads((P/'execution_code_binding.json').read_text())
 check('original_execution_code_still_bound',all(sha(ROOT/k)==v for k,v in bindings.items()))
 check('frozen_protocol_still_bound',sha(ROOT/'data/exp/research-design/three_ward_formal_transport_matrix_v1/protocol.json')==json.loads((P/'run_manifest.json').read_text())['protocol_sha256'])
 check('comparison_sources_still_bound',all(sha(ROOT/k)==v for k,v in json.loads((O/'source_bindings.json').read_text()).items()))
 support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');ward=support['Ward Code'];valid=support['Eligible Ground Mask']
 metrics=read(O/'paired_wet_extent.csv')
 for stage in ['L','H']:
  z=np.load(O/f'{stage}_paired_peak_bounds.npz');good=z['paired_admitted'];check(stage+'_support_only_within_eligible',not np.any(good&~valid))
  for name,prefix,area in [('Open ground','ground',z['ground_area']),('Classified ground roads','road',z['road_area'])]:
   lo=z[prefix+'_lower'];up=z[prefix+'_upper'];sel=good&(ward>0)&(area>0)
   for arm,i in [('T',0),('U',1)]:
    check(stage+name+arm+'_ordered_finite_bounds',np.isfinite(lo[i,sel]).all()and np.isfinite(up[i,sel]).all()and np.all(lo[i,sel]<=up[i,sel]))
    for threshold in [.1,.3,.5]:
     row=next(x for x in metrics if x['stage']==stage and x['support']==name and x['scope']=='Three wards'and float(x['threshold_m'])==threshold)
     lower=area[sel&(lo[i]>threshold)].sum()/1e6;upper=area[sel&(up[i]>threshold)].sum()/1e6
     check(f'{stage}_{name}_{arm}_{threshold}_reported_extent',abs(lower-float(row[arm+'_certain_km2']))<1e-10 and abs(upper-float(row[arm+'_possible_km2']))<1e-10)
     wardrows=[x for x in metrics if x['stage']==stage and x['support']==name and x['scope']in['Chiyoda','Chuo','Minato']and float(x['threshold_m'])==threshold]
     check(f'{stage}_{name}_{arm}_{threshold}_ward_partition',len(wardrows)==3 and abs(sum(float(x[arm+'_certain_km2'])for x in wardrows)-lower)<1e-10)
 exchange=read(O/'paired_group_exchange.csv');budgets=read(O/'paired_water_budget.csv')
 for row in budgets:
  grouped=[x for x in exchange if x['case_id']==row['case_id']]
  check(row['case_id']+'_seven_water_groups_preserved',len(grouped)==7 and len(set(x['water_group']for x in grouped))==7)
  check(row['case_id']+'_directional_budget_reduction',all(abs(sum(float(x[k])for x in grouped)-float(row[k]))<1e-5 for k in ['gross_out_m3','gross_in_m3']))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.paths=[]
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in ['href','src']and v and not v.startswith(('http:','https:','#','data:')):self.paths.append(v)
 parser=Links();parser.feed((O/'review.html').read_text())
 check('report_local_assets_resolve',all((O/p).exists()for p in parser.paths))
 station=read(O/'entrance_exposure_summary.csv');check('two_stages_twelve_synchronous_frames_and_separate_peak',len(station)==26 and all(sum(x['stage']==stage and x['simultaneous']=='True'for x in station)==12 for stage in ['L','H']))
 detail=read(O/'entrance_exposure_detail.csv');check('675_source_entrances_kept_per_frame',len(detail)==26*675)
 check('report_scope_flags',all(json.loads((O/'summary.json').read_text())[k]is False for k in ['accuracy_improvement_established','transport_accessibility_computed','new_hydraulics','people_counted','training','sealed_tests_opened']))
 result=dict(all_passed=True,checks=checks,solver_launched=False);(O/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 files=list(O.iterdir())+[ROOT/'src/analyses/compare_formal_transport_b1_geometry_v1.py',Path(__file__)]
 out={str(f.relative_to(ROOT)):sha(f)for f in files if f.is_file()and f.name!='delivery_manifest.json'}
 (O/'delivery_manifest.json').write_text(json.dumps(dict(files=out),indent=2)+'\n');print('INDEPENDENT CHECKS',len(checks),'PASSED')
if __name__=='__main__':main()
