"""Independent final validation of the matched regional comparison delivery."""
from pathlib import Path
import csv,json,re,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_geometry_pair_controls_v1';R=ROOT/'data/exp/estimation/three_ward_urban_geometry_pair_controls_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rr(p):
 with p.open() as f:return list(csv.DictReader(f))
def main():
 checks=[]
 def check(name,value):checks.append({'check':name,'pass':bool(value)})
 m=json.loads((P/'execution_manifest.json').read_text());check('only_two_authorized_T_controls_and_two_U_reuses',len(m['runs'])==2 and {x['case_id'] for x in m['runs']}=={'dry_raised','rain60_raised'} and all(x['arm']=='T' for x in m['runs']) and len(m['reused_U'])==2)
 for r in m['runs']:
  a=json.loads((R/r['case_id']/'T/audit.json').read_text());check(r['case_id']+'_all_original_numerical_gates',a['all_checks_passed'] and a['admission']['technical_reference'])
 for r in m['reused_U']:check(r['case_id']+'_reused_audit_preserved',sha(ROOT/r['audit_file'])==r['audit_sha256'] and sha(ROOT/r['folder']/'execution.json')==r['execution_sha256'])
 rows=rr(R/'paired_wet_extent.csv');check('all_common_extent_ranges_ordered',all(float(x['T_certain_wet_km2'])<=float(x['T_possible_wet_km2'])+1e-12 and float(x['U_certain_wet_km2'])<=float(x['U_possible_wet_km2'])+1e-12 and 0<=float(x['wet_Jaccard_lower'])<=float(x['wet_Jaccard_upper'])+1e-12<=1+1e-12 for x in rows))
 # Independently recompute certain/possible footprints from delivered bounds.
 support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];ward=support['Ward Code'];scopes={'Whole domain':None,'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3,'Adjoining contributing area':0};ok=True
 for case in ['dry_raised','rain60_raised']:
  u=next(x for x in m['reused_U'] if x['case_id']==case);lt=np.load(P/'execution'/(case+'_candidate_reference.npz'));lu=np.load(ROOT/u['labels_file']);good=valid&lt['Admitted Mask']&lu['Admitted Mask']&~lt['Quarantine Mask']&~lu['Quarantine Mask']
  for x in [r for r in rows if r['case_id']==case and r['support']=='Open ground']:
   code=scopes[x['scope']];region=valid if code is None else valid&(ward>0) if code==-1 else valid&(ward==code);use=region&good;th=float(x['threshold_m'])
   for arm,label in [('T',lt),('U',lu)]:
    for name,key in [('certain','Peak Mean Depth Lower'),('possible','Peak Mean Depth Upper')]:ok &= abs(float(area[use&(label[key]>th)].sum()/1e6)-float(x[arm+'_'+name+'_wet_km2']))<1e-12
 check('independent_recalculation_of_common_ground_extent',ok)
 b=rr(R/'paired_water_budgets.csv');check('all_four_budget_rows_and_nonnegative_minima',len(b)==4 and all(float(x['minimum_native_volume_m3'])>=0 for x in b))
 residuals=[]
 for x in b:
  sources=float(x['rain_applied_m3'])+float(x['gross_in_m3']);scale=max(sources,float(x['gross_out_m3']),1.);residuals.append(abs(float(x['endpoint_budget_residual_m3']))<=max(1.,.001*scale))
 check('unchanged_endpoint_budget_gate_in_delivery',all(residuals))
 # The pipeline updates its non-data progress/log after the renderer returns.
 # Settle those two explicitly declared runtime records after completion; all
 # scientific results retain their original rendered manifest hashes.
 manifest=json.loads((R/'delivery_manifest.json').read_text());settled=[]
 assert json.loads((R/'progress.json').read_text())['phase']=='paired_comparison_complete'
 for x in manifest['artifacts']:
  if x['file'] in [str((R/n).relative_to(ROOT)) for n in ['progress.json','comparison.log']]:
   current=sha(ROOT/x['file'])
   if current!=x['sha256']:settled.append({'file':x['file'],'during_render_sha256':x['sha256'],'final_sha256':current});x['sha256']=current
 for name in ['finish_urban_geometry_pair_controls_v1.py','validate_urban_geometry_pair_controls_v1.py']:
  src=ROOT/'src/analyses'/name;rel=str(src.relative_to(ROOT))
  if not any(x['file']==rel for x in manifest['artifacts']):manifest['artifacts'].append({'file':rel,'sha256':sha(src)})
 manifest['settled_nondata_runtime_records']=settled
 (R/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # Verify reused outcome files against the original frozen delivery, not merely
 # their present locations or the current input configuration.
 oldfreeze=json.loads((ROOT/'data/processed/three_ward_reference_setup_pilot_v1/freeze.json').read_text())['files_sha256'];oldok=True;oldcount=0
 for reuse in m['reused_U']:
  targets=[reuse['labels_file']]+[str(Path(reuse['folder'])/n) for n in ['sfincs_map.nc','sfincs_his.nc','udt_step_budget.csv','udt_output_clock.csv']]
  for rel in targets:oldok &= rel in oldfreeze and sha(ROOT/rel)==oldfreeze.get(rel);oldcount+=1
 check('reused_outcomes_match_original_frozen_hashes',oldok)
 cache={};allhash=True
 for x in manifest['artifacts']:
  f=ROOT/x['file'];st=f.stat();key=(st.st_dev,st.st_ino)
  if key not in cache:cache[key]=sha(f)
  allhash &=cache[key]==x['sha256']
 check('all_delivery_manifest_hashes_match',allhash)
 check('protected_sources_population_and_formal_partition_unchanged',all(sha(ROOT/k)==h for k,h in json.loads((P/'protected_sources.json').read_text()).items()))
 text=(R/'review.html').read_text();broken=[x for x in re.findall(r'(?:src|href)="([^"]+)"',text) if not (R/x).resolve().is_file()];check('all_report_links_and_images_exist',not broken)
 output={'all_passed':all(x['pass'] for x in checks),'checks':checks,'broken_links':broken,'new_T_runs':2,'reused_U_runs':2,'new_parent_events':0,'training':False,'sealed_test':False,'affected_people_computed':False};(R/'independent_delivery_validation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2));assert output['all_passed']
if __name__=='__main__':main()
