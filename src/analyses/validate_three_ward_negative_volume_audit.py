"""Independent delivery checks for existing-control negative-volume evidence."""
from pathlib import Path
import sys,json,csv,re,argparse
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from audit_three_ward_negative_volumes import OUT,REPORT,CASES,E2E,readnp,arr,structured
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_sfincs_cpu_build import sha,dump
from load_three_ward_reference_labels import load_reference

def main(freeze=False):
 out=ROOT/OUT;report=ROOT/REPORT;s=json.loads((report/'summary.json').read_text());prov=json.loads((report/'provenance.json').read_text());checks=[]
 def check(n,ok):checks.append({'check':n,'pass':bool(ok)});print(('PASS 'if ok else'FAIL ')+n,flush=True)
 check('all_137_or_more_spatial_temporal_and_source_audit_checks_pass',s['all_checks_passed']and len(s['checks'])>=137 and all(c['pass']for c in s['checks']))
 check('zero_new_solver_runs_training_or_58_case_execution',s['new_solver_runs']==s['training']==s['prospective_58_executed']==0 and not s['formal_reference_release'])
 # Hash each inode once: several old run inputs are shared hardlinks.
 cache={}
 def hash_cached(p):
  st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 check('every_bound_input_output_and_policy_unchanged',all(hash_cached(ROOT/p)==h for p,h in prov['source_sha256'].items()))
 golden=json.loads((ROOT/E2E/'freeze.json').read_text())['files_sha256']
 check('entire_previous_frozen_control_delivery_unchanged',all(hash_cached(ROOT/p)==h for p,h in golden.items()))
 pending=json.loads((ROOT/'data/processed/three_ward_minimum_scenarios_v1/run_manifest.json').read_text())['pending_runs']
 check('all_58_library_cases_remain_unrun',len(pending)==58 and all(r['status']=='unrun'and r['response_path']is None for r in pending))
 sup=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=sup['Eligible Ground Mask'];area=sup['Ground Area'];ward=sup['Ward Code']
 manifest=json.loads((ROOT/E2E/'run_manifest.json').read_text())['runs'];runs={r['case_id']:r for r in manifest}
 evidence=list(csv.DictReader((report/'negative_cell_evidence.csv').open()));coverage=list(csv.DictReader((report/'scope_coverage.csv').open()))
 check('local_unknown_times_are_never_filled_or_inferred',all(r['exact_first_crossing_s']==r['duration_s']==''for r in evidence)and all(r['first_observed_negative_time_s']==r['last_observed_negative_time_s']==''for r in evidence if r['snapshot_seen']=='False'))
 finebase=ROOT/'data/processed/three_ward_sfincs_scenarios';con=ROOT/'data/processed/three_ward_urban_constraints';npz={}
 with rasterio.open(finebase/'terrain_sampling_0_50m.tif')as dem,rasterio.open(con/'constraints_0_50m.tif')as cl,rasterio.open(con/'hydraulic_obstacle_0_50m.tif')as hyd:
  for case in CASES:
   d=readnp(out/(case+'_audit_masks.npz'));npz[case]=d;ever=d['Online Ever Negative'];sample=d['Sampled Ever Negative'];cs=s['cases'][case]
   folder=ROOT/runs[case]['folder'];ledger=structured(folder/'udt_step_budget.csv')
   check(case+'_online_and_sampled_counts_reproduced',int(ever.sum())==cs['online_ever_cells']and int(sample.sum())==cs['sampled_ever_cells']and not np.any(sample&~ever))
   check(case+'_true_first_global_crossing_from_every_step',ledger['actual_state_time'][np.flatnonzero(ledger['negative_native_count']>0)[0]]==cs['first_global_negative_time_s'])
   partition=sum(d[k].astype('i1')for k in ['Exact Label Mask','Interval Label Mask','Wide Interval Mask','Quarantine Mask'])
   check(case+'_candidate_classes_partition_and_direct_quarantine',np.array_equal(partition,valid.astype('i1'))and np.all(d['Quarantine Mask'][ever])and not np.any(d['Admitted Mask']&ever))
   check(case+'_evidence_table_matches_full_online_set',len([r for r in evidence if r['case']==case])==int(ever.sum())and all(ever[int(r['row_north']),int(r['col_east'])]for r in evidence if r['case']==case))
   check(case+'_ward_counts_reconcile_whole_domain',sum(int(r['online_negative_cells'])for r in coverage if r['case']==case and r['scope']in ['Chiyoda','Chuo','Minato','Contributing area'])==int(ever.sum()))
   # Independently recompute fine-ground depths and barrier contacts with direct NumPy pixels.
   # Deterministic strata include all wards, online-only flags, largest peaks and censoring.
   select=[]
   for cond in [valid&(ward==1),valid&(ward==2),valid&(ward==3),ever,d['Online Only Negative'],valid&(d['Peak Mean Depth Lower']>.3)]:
    ids=np.flatnonzero(cond)
    if len(ids):select.extend(ids[np.linspace(0,len(ids)-1,min(3,len(ids))).astype(int)].tolist())
   with Dataset(folder/'sfincs_map.nc')as nc:
    peak=np.flipud(arr(nc['zsmax'][0]));minimum=np.flipud(arr(nc['zb'][:]))
    ids=np.flatnonzero(valid&~np.isfinite(peak))
    if len(ids):select.extend(ids[:3].tolist())
    errors=[];bar_ok=True
    for ind in set(select):
     r,c=divmod(ind,770);win=Window(c*20,r*20,20,20);z=dem.read(1,window=win);cls=cl.read(1,window=win);hz=hyd.read(1,window=win);eligible=np.isfinite(z)&(cls!=1)
     eta=peak[r,c]if np.isfinite(peak[r,c])else minimum[r,c]+.01
     lower=float(np.maximum(eta-z[eligible].astype('f8'),0).mean())if np.isfinite(peak[r,c])else 0.
     upper=float(np.maximum(eta-z[eligible].astype('f8'),0).mean());errors.extend([abs(lower-d['Peak Mean Depth Lower'][r,c]),abs(upper-d['Peak Mean Depth Upper'][r,c])])
     b=hz[(cls==1)&np.isfinite(hz)];barrier=float(b.min())if len(b)else np.inf
     bar_ok &=bool(d['Assumed Barrier Contact'][r,c])==bool(np.isfinite(peak[r,c])and np.isfinite(barrier)and peak[r,c]>=barrier-.01)
    check(case+'_stratified_original_fine_pixel_label_reconstruction',bool(errors)and max(errors)<1e-10 and bar_ok)
   try:load_reference(case,root=ROOT)
   except PermissionError:denied=True
   else:denied=False
   check(case+'_existing_reference_loader_still_denies_release',denied)
 wet=list(csv.DictReader((report/'wet_area_exposure.csv').open()));patch=list(csv.DictReader((report/'principal_peak_regions.csv').open()))
 for case in CASES:
  d=npz[case]
  for th in [.05,.1,.3,.5]:
   sel=valid&(ward>0)&(d['Peak Mean Depth Lower']>th);r=next(r for r in wet if r['case']==case and r['scope']=='Three wards'and float(r['threshold_m'])==th)
   check(case+f'_wet_exposure_{th:g}m_reproducible',abs(area[sel&d['Online Ever Negative']].sum()/1e6-float(r['negative_certain_wet_km2']))<1e-12)
 check('hotspot_support_checks_not_reported_as_hydraulic_repair',all('not a hydraulic correction'in r['scope_of_connectivity_check']for r in patch)and not s['numerical_error_propagation_bounded'])
 before=(report/'previous-AnaSOP.md').read_text();after=(ROOT/'docs/AnaSOP.md').read_text();split=lambda t:re.split(r'(?=^## [1-8]\. )',t,flags=re.M)
 parts0,parts1=split(before),split(after);changed={int(re.match(r'## (\d)',a).group(1))for a,b in zip(parts0,parts1)if a!=b}
 check('AnaSOP_only_4_6_7_quality_notes_and_no_final_variables',changed=={4,6,7}and before.count('| yes |')==after.count('| yes |')and all(b.isascii()for b in parts1 if b.startswith(('## 4. ','## 6. ','## 7. '))))
 page=(report/'review.html').read_text();links=re.findall(r'(?:href|src)="([^"]+)"',page)
 check('review_figure_and_links_exist',all((report/p).is_file()for p in links))
 check('no_runtime_solver_or_training_imports_in_new_analysis',all('subprocess'not in (ROOT/'src/analyses'/p).read_text()for p in ['audit_three_ward_negative_volumes.py','review_three_ward_negative_volumes.py']))
 if freeze:
  assert all(c['pass']for c in checks)
  paths=[p for base in [out,report]for p in base.rglob('*')if p.is_file()and p.name not in ['freeze.json','delivery-validation.json']]
  paths +=[ROOT/'src/analyses'/p for p in ['audit_three_ward_negative_volumes.py','review_three_ward_negative_volumes.py','validate_three_ward_negative_volume_audit.py']]
  dump(out/'freeze.json',{'files_sha256':{str(p.relative_to(ROOT)):hash_cached(p)for p in sorted(paths)},'state':'existing_controls_negative_quality_diagnostic_only','formal_reference_release':False,'training_release':False,'new_solver_runs':0})
 if(out/'freeze.json').exists():
  f=json.loads((out/'freeze.json').read_text());check('all_new_frozen_delivery_hashes_match',all(hash_cached(ROOT/p)==h for p,h in f['files_sha256'].items()))
 result={'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'new_solver_runs':0,'training':0,'prospective_58_executed':0,'formal_reference_release':False};dump(report/'delivery-validation.json',result)
 print(json.dumps({'checks':len(checks),'all_passed':result['all_checks_passed']}))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');main(a.parse_args().freeze)
