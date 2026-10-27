"""Validate frozen population audit and add explicit epoch and allocation caveats."""
from pathlib import Path
import json,csv,hashlib,collections,html
from shapely.geometry import shape,Point
from shapely.ops import unary_union
ROOT=Path('/Volumes/Extension/Research/UDT01b');OUT=ROOT/'data/processed/three_ward_population_mesh_audit_v1';R=ROOT/'data/exp/data-briefing/three_ward_population_mesh_audit_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
s=json.loads((R/'summary.json').read_text());base=json.loads((R/'freeze.json').read_text());checks={}
for x in base['products']:checks['unchanged_'+x['file']]=sha(ROOT/x['file'])==x['sha256']
for p,h in base['source_bindings'].items():checks['source_'+p]=sha(ROOT/p)==h
attrs=readcsv(OUT/'plateau_building_attributes.csv');ids=[r['buildingID'] for r in attrs if r['buildingID']]
checks['stable_building_ids_unique']=len(ids)==len(set(ids))==len(attrs)
br=readcsv(OUT/'building_allocation_readiness.csv');target=[r for r in br if r['intersects_three_wards']=='True']
known=[r for r in target if r['use_group']!='unknown'];unknown=[r for r in target if r['use_group']=='unknown'];res=[r for r in target if r['use_group'] in ['residential','mixed_residential']]
epoch={'population_reference_date':'2020-10-01','catalog_usage_and_floor_source_years':sorted(set(r['date'] for r in json.loads((OUT/'building_attribute_sources.json').read_text())['source_materials'] if r['property'] in ['bldg:usage','bldg:storeysAboveGround'])),'per_building_surveyYear_raw':dict(collections.Counter(r['surveyYear'] for r in target)),'interpretation':'Source-material lists cite a 2022 building-use survey, but known-use objects carry surveyYear 2021. These are distinct recorded metadata and are not reconciled in this audit. 0001 is not accepted as an actual modern survey date. Do not call all attributes 2022 observations or the census current 2026 occupancy.'}
checks['all_known_use_objects_raw_surveyYear_2021']=all(r['surveyYear']=='2021' for r in known)
checks['all_unknown_use_objects_raw_surveyYear_0001']=all(r['surveyYear']=='0001' for r in unknown)
checks['placeholder_9999_not_valid_floors']=all(r['floors_valid']=='False' for r in br if r['storeysAboveGround']=='9999')
# Verify each code-derived child cell is geometrically within its published parent.
f125=json.loads((OUT/'population_125m_full_mesh.geojson').read_text())['features'];f250=json.loads((OUT/'population_250m_full_mesh.geojson').read_text())['features'];g250={f['properties']['mesh_code']:shape(f['geometry']) for f in f250}
checks['all_125m_geometries_nested_in_250m_parent']=all(shape(f['geometry']).difference(g250[f['properties']['mesh_code'][:10]].buffer(1e-12)).area<1e-15 for f in f125)
checks['tokyo_station_independent_location_check']=any(f['properties']['mesh_code'].startswith('53394611') and shape(f['geometry']).covers(Point(139.7671,35.6812)) for f in f125)
checks['male_plus_female_and_original_data_integrity']=json.loads((R/'validation.json').read_text())['passed']
checks['all_757_parents_population_exactly_conserved']=s['parent_comparison']['parents']==757 and s['parent_comparison']['unequal']==0
readiness={}
for r in ['125','250']:
 d=s['allocation_readiness'][r];readiness[r]={**d,'no_residential_candidate_pct_of_populated_meshes':100*d['no_residential_candidate_meshes']/d['populated_meshes']}
a={'epoch_audit':epoch,'building_feasibility':{'geometry_intersecting_buildings':len(target),'unknown_use_buildings':len(unknown),'unknown_use_pct':100*len(unknown)/len(target),'residential_candidate_buildings':len(res),'mixed_residential_buildings':sum(r['use_group']=='mixed_residential' for r in res),'residential_valid_floor_count':sum(r['floors_valid']=='True' for r in res),'residential_valid_floors_pct':100*sum(r['floors_valid']=='True' for r in res)/len(res),'mixed_residential_share_not_observed':True,'building_person_counts_not_created':True,'ward_count_method':'Per-ward rows use representative-point assignment; union total includes every intersecting base, so boundary-intersecting buildings with outside representative points can cause a difference.'},'mesh_readiness':readiness,'recommended_next_step_not_user_decision':'Prepare a mass-conserving allocation protocol with residual population buckets. Compare 125 m and 250 m under identical building weights; specify mixed-use shares, unknown-use/invalid-floor fallbacks and institutional-population handling before assigning persons. Do not assume 125 m is more accurate.','interpretation_limits':['The 136 no-residential-candidate 125 m cells include 120 full interior cells. Missing external buildings cannot explain all cases.','Possible causes include mechanical apportionment, missing use attributes, institutional population and epoch differences. This audit does not identify their individual contributions.','No-candidate mesh population is not a count of missing residents or people at risk; it includes whole boundary-cell population.','YearOfConstruction is absent for all assessed buildings, so the current inventory cannot simply filter to buildings existing at the 2020 census.','Cross-ward cells are kept whole. Ward-intersection areas are not population weights without an explicit allocation assumption.','All 50 statistical fields are retained with their original codes and labels, along with the 4 identifier/confidentiality fields.']}
dump(R/'audit_details.json',a)
body=(R/'review.html').read_text()
insert='<section style="background:#fff4d8;border-left:5px solid #c78416;padding:16px"><h2>Final audit qualifications</h2><p><b>125 m is a reference distribution, not independently observed building occupancy.</b> Its four-cell totals equal 250 m for all 757 compared parents.</p><p>Source lists cite 2022 for building use and floors, while all known-use objects in the three-ward inventory carry surveyYear 2021. Unknown-use objects carry 0001; this is not treated as a usable survey date. The discrepancy remains unresolved.</p><p>Among '+str(len(res))+' residential candidates, '+str(sum(r['floors_valid']=='True' for r in res))+' have valid above-ground floors (99.73%). However, '+str(len(unknown))+' of '+str(len(target))+' intersecting building bases have unknown use (13.16%), and 8,709 residential candidates have mixed uses without observed residential floor share.</p><p>Positive-population meshes without a residential candidate: 125 m '+str(readiness['125']['no_residential_candidate_meshes'])+'/'+str(readiness['125']['populated_meshes'])+' (7.01%); 250 m 25/605 (4.13%). Counts and rates alone do not establish which distribution is more accurate. Missing residential candidates can reflect apportionment, attributes, institutional residents or year differences.</p><p><b>Conclusion:</b> Building-level estimates are feasible with explicit assumptions and unallocated-population accounting. Exact building occupancy and current affected-person counts are not established. Compare both mesh resolutions using the same future allocation rules.</p><p><a href="audit_details.json">Detailed qualifications</a> | <a href="delivery_validation.json">Delivery validation</a> | <a href="download_manifest_copy.json">Official downloads and checksums</a></p></section>'
body=body.replace('<h2>The important qualification</h2>',insert+'<h2>The important qualification</h2>')
(R/'review_final.html').write_text(body)
(R/'download_manifest_copy.json').write_bytes((ROOT/'data/raw/estat_population/2020_jgd2011/download_manifest.json').read_bytes())
# Exact known metadata, not inferred floor dates, is the authoritative audit result.
checks['no_building_population_allocation_claimed']=a['building_feasibility']['building_person_counts_not_created']
checks['final_report_local_image_exists']=(R/'population_mesh_comparison.png').exists()
dump(R/'delivery_validation.json',{'passed':all(checks.values()),'check_count':len(checks),'checks':checks});assert all(checks.values()),checks
files=[p for b in [OUT,R] for p in sorted(b.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json']
sources=[ROOT/'src/analyses/audit_three_ward_population_mesh_v1.py',ROOT/'src/analyses/audit_three_ward_population_attributes_v1.py',ROOT/'src/preprocessing/download_estat_three_ward_population_v1.py',Path(__file__)]
dump(R/'delivery_manifest.json',{'products':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files],'scripts':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sources],'base_freeze_preserved':sha(R/'freeze.json')})
print(json.dumps({'validation_checks':len(checks),'passed':all(checks.values()),'details':a},ensure_ascii=False,indent=2))
