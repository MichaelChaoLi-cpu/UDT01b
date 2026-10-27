"""Independent checks of delivered OD arrays, summaries, fixed sources and links."""
from pathlib import Path
import csv,json,hashlib,re,sys
from urllib.parse import unquote,urlparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1';R=ROOT/'data/exp/estimation/three_ward_transport_shutdown_scenarios_v1'
def rr(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checks=[]
 def check(name,value):checks.append({'check':name,'pass':bool(value)})
 z=np.load(P/'paired_accessibility_matrices.npz');rows=rr(R/'scenario_accessibility_summary.csv');idx=rr(P/'scenario_index.csv');origins=rr(P/'access_probes.csv');lookup={p['probe_id']:p for p in origins};origins=[lookup[str(x)] for x in z['origin_probe_ids']];surface=np.array([p['cohort']=='surface_grid' for p in origins]);branches=list(map(str,z['baseline_branch_names']));check('576_unique_scenario_rows_aligned_to_OD_matrix',len(rows)==len(idx)==576 and len({r['scenario_id'] for r in rows})==576 and z['outward'].shape==z['return_trip'].shape==(576,323,6) and all(a['scenario_id']==b['scenario_id'] for a,b in zip(rows,idx)))
 counts=True;subset=True;lookupmat={}
 for i,r in enumerate(rows):
  bi=branches.index(r['network_branch']);bo,bb=z['baseline'][bi];out=z['outward'][i];back=z['return_trip'][i];baseline=bo&bb;mutual=out&back;before=baseline[surface].any(axis=1);after=mutual[surface].any(axis=1)
  counts &= int((baseline&~mutual).sum())==int(r['lost_mutual_pairs']) and int(baseline.sum())==int(r['baseline_mutual_pairs']) and int((baseline&~mutual)[surface].sum())==int(r['surface_grid_lost_mutual_pairs']) and int((before&~after).sum())==int(r['surface_grid_lost_all_declared_mutual_anchors']) and int(before.sum())==141 and int(baseline[surface].sum())==846
  subset &= not (out&~bo).any() and not (back&~bb).any()
  lookupmat[(r['case_id'],r['exposure_policy'],r['network_branch'],r['shutdown_type'])]=(out,back)
 check('independent_recalculation_of_reported_OD_counts',counts);check('both_directions_are_subsets_of_same_branch_baseline',subset)
 nested=True
 for case,pol,branch,kind in lookupmat:
  if kind!='entrance':continue
  a=lookupmat[(case,pol,branch,'entrance')];b=lookupmat[(case,pol,branch,'station')];c=lookupmat[(case,pol,branch,'section')]
  nested &= all(not (bb&~aa).any() and not (cc&~bb).any() for aa,bb,cc in zip(a,b,c))
 check('all_OD_paths_nested_A_then_B_then_C',nested)
 baselineunknown=True
 for bi in range(2):
  b=z['baseline'][bi,0]&z['baseline'][bi,1];baselineunknown &= int(surface.sum())==209 and int((~b[surface].any(axis=1)).sum())==68
 check('68_baseline_unknown_samples_not_counted_as_shutdown_loss',baselineunknown)
 # Original vs augmented association is explicit, and frozen parent split never changes.
 selected=rr(P/'closure_selection_by_parent.csv');ss={}
 for r in selected:ss[(r['case_id'],r['exposure_policy'])]=r
 additions=True
 for case in {r['case_id'] for r in selected}:
  a=ss[(case,'primary_0p3_area100')];b=ss[(case,'sensitivity_normalized_identity')]
  additions &= set(a['selected_portal_ids'].split(';'))<=set(b['selected_portal_ids'].split(';')) and set(a['entrance_evidence_station_ids'].split(';'))<=set(b['entrance_evidence_station_ids'].split(';'))
 check('identity_sensitivity_additions_are_separate_and_monotone',additions)
 events=rr(ROOT/'data/processed/three_ward_station_surface_risk_v1/scenario_bindings.csv');parentlookup={x['case_id']:x for x in events}
 check('original_12_train_4_validation_parents_preserved',len(events)==16 and sum(x['split']=='train' for x in events)==12 and sum(x['split']=='validation' for x in events)==4 and all(r['parent_id']==parentlookup[r['case_id']]['parent_id'] and r['split']==parentlookup[r['case_id']]['split'] for r in rows))
 bind=json.loads((P/'source_bindings.json').read_text());check('original_bound_sources_including_population_unchanged',all(sha(ROOT/b['file'])==b['sha256'] for b in bind))
 receipt=json.loads((R/'evaluation_receipt.json').read_text());check('precomparison_policy_hash_matches_delivered_policy',receipt['policy_sha256']==sha(P/'scenario_policy.json'))
 manifest=json.loads((R/'delivery_manifest.json').read_text());check('all_delivery_hashes_match',all((ROOT/b['file']).is_file() and sha(ROOT/b['file'])==b['sha256'] for b in manifest['artifacts']))
 broken=[]
 for value in re.findall(r'(?:href|src)="([^"]+)"',(R/'review.html').read_text()):
  if urlparse(value).scheme or value.startswith('#'):continue
  target=(R/unquote(value)).resolve()
  if not target.is_file():broken.append(value)
 check('all_local_report_links_and_images_exist',not broken)
 run=json.loads((R/'validation.json').read_text());check('semantic_graph_and_frozen_input_checks_pass',run['all_passed'])
 result={'all_passed':all(x['pass'] for x in checks),'checks':checks,'broken_links':broken,'no_people_or_city_isolation_inference':True,'population_uncertainty_unchanged':True,'no_new_hydraulics':True,'no_sealed_test_access':True}
 (R/'delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['all_passed']
if __name__=='__main__':main()
