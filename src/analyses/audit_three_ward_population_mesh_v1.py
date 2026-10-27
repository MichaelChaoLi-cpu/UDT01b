"""Extract census meshes and audit PLATEAU building allocation feasibility.
Uses existing local numpy, shapely, pyproj and matplotlib; no training or simulation.
Full mesh population is retained. Clipped geometry is a display footprint only.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/udt_population_mpl')
import csv,io,zipfile,json,math,hashlib,collections,html
import numpy as np
from shapely.geometry import shape,box,mapping
from shapely.ops import transform,unary_union
from shapely import STRtree
from pyproj import Transformer
ROOT=Path('/Volumes/Extension/Research/UDT01b')
RAW=ROOT/'data/raw/estat_population/2020_jgd2011'
OUT=ROOT/'data/processed/three_ward_population_mesh_audit_v1'
REPORT=ROOT/'data/exp/data-briefing/three_ward_population_mesh_audit_v1'
OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows,keys=None):
 keys=keys or list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
ana=sha(ROOT/'docs/AnaSOP.md')
xy=Transformer.from_crs(6668,6677,always_xy=True).transform
ll=Transformer.from_crs(6677,6668,always_xy=True).transform
boundary=ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson'
wj=json.loads(boundary.read_text());wards={f['properties']['ward']:transform(xy,shape(f['geometry'])) for f in wj['features']}
u=unary_union(list(wards.values()));bounds=unary_union([shape(f['geometry']) for f in wj['features']]).bounds

def encode(ri,ci,level):
 n=2**(level-3);lr,lc=ri//n,ci//n
 code=f'{lr//80:02}{lc//80:02}{lr%80//10}{lc%80//10}{lr%10}{lc%10}'
 for k in range(level-4,-1,-1):code+=str(1+2*((ri>>k)&1)+((ci>>k)&1))
 return code

def mesh(code):
 lat=int(code[:2])*2/3+int(code[4])/12+int(code[6])/120
 lon=int(code[2:4])+100+int(code[5])/8+int(code[7])/80
 dy,dx=1/120,1/80
 for q in code[8:]:
  dy/=2;dx/=2;v=int(q)-1;lat+=(v//2)*dy;lon+=(v%2)*dx
 return box(lon,lat,lon+dx,lat+dy)
assert encode(int(35.68*960),int((139.76-100)*640),6).startswith('5339')
selected={};geoms={};allrows={};fieldrows=[];counts={};features={};clipfeatures={};links=[]
for level,res,tid in [(6,125,'T001231'),(5,250,'T001142')]:
 n=2**(level-3);dy=1/(120*n);dx=1/(80*n)
 gs={}
 for ri in range(math.floor(bounds[1]/dy),math.floor(bounds[3]/dy)+1):
  for ci in range(math.floor((bounds[0]-100)/dx),math.floor((bounds[2]-100)/dx)+1):
   code=encode(ri,ci,level);g=transform(xy,mesh(code))
   if g.intersection(u).area>.01:gs[code]=g
 # Keep all four child records of any selected 250 m parent for a complete parent comparison.
 wanted=set(gs) if res==250 else {p+str(q) for p in {c[:10] for c in gs} for q in range(1,5)}
 rows={};source_total=0
 with zipfile.ZipFile(RAW/f'population_{res}m_5339.zip') as z:
  assert len(z.namelist())==1
  with z.open(z.namelist()[0]) as h:
   rr=csv.reader(io.TextIOWrapper(h,encoding='cp932'));header=next(rr);labels=next(rr)
   assert len(header)==54 and len(labels)==54
   for r in rr:
    source_total+=1
    if r[0] in wanted:rows[r[0]]=dict(zip(header,r))
 for code,label in zip(header,labels):fieldrows.append({'resolution_m':res,'field':code,'official_label':label.strip(),'role':'mesh_identifier' if code=='KEY_CODE' else 'confidentiality_control' if code in ['HTKSYORI','HTKSAKI','GASSAN'] else 'population_or_household_statistic','suppression_marker':'*','unknown_breakdown_marker':'Y'})
 def pop(code):return int(rows[code][tid+'001']) if code in rows else 0
 fs=[];clips=[];table=[]
 for code,g in sorted(gs.items()):
  r=rows.get(code);assert r is None or r[tid+'001'].isdigit()
  fractions={w:g.intersection(wg).area/g.area for w,wg in wards.items()}
  inside=g.intersection(u).area/g.area
  p={'mesh_code':code,'population':pop(code),'source_record_present':r is not None,'resident_date':'2020-10-01','reference_table':res==125,'inside_three_ward_area_fraction':inside,'boundary_crossing':inside<1-1e-6,'HTKSYORI':r['HTKSYORI'] if r else '', 'population_applies_to':'entire_mesh_not_clipped_polygon'}
  fs.append({'type':'Feature','properties':p,'geometry':mapping(transform(ll,g))})
  clips.append({'type':'Feature','properties':p,'geometry':mapping(transform(ll,g.intersection(u)))})
  table.append({**p,**{'area_fraction_'+w[:5]:v for w,v in fractions.items()}})
  for w,v in fractions.items():
   if v>1e-10:links.append({'resolution_m':res,'mesh_code':code,'ward':w,'intersection_area_m2':v*g.area,'mesh_area_fraction':v,'whole_mesh_population':pop(code),'ward_population':'not_identified_from_mesh_alone'})
 csvsave(OUT/f'population_{res}m_source_rows.csv',list(rows.values()),header)
 csvsave(OUT/f'population_{res}m_mesh_index.csv',table)
 dump(OUT/f'population_{res}m_full_mesh.geojson',{'type':'FeatureCollection','features':fs})
 dump(OUT/f'population_{res}m_clipped_display.geojson',{'type':'FeatureCollection','features':clips})
 selected[res]={r['mesh_code']:r for r in table};geoms[res]=gs;allrows[res]=rows;features[res]=fs;clipfeatures[res]=clips
 counts[res]={'source_archive_rows':source_total,'intersecting_meshes':len(gs),'source_populated_meshes':sum(c in rows for c in gs),'zero_population_meshes_without_source_record':sum(c not in rows for c in gs),'intersecting_full_mesh_population_not_ward_total':sum(pop(c) for c in gs),'boundary_crossing_meshes':sum(r['boundary_crossing'] for r in table),'interior_full_mesh_population':sum(pop(c) for c in gs if not selected[res][c]['boundary_crossing']),'confidentiality_flags':dict(collections.Counter(r['HTKSYORI'] for c,r in rows.items() if c in gs))}
 print('MESH',res,counts[res],flush=True)
csvsave(OUT/'mesh_ward_intersections.csv',links);csvsave(OUT/'field_dictionary.csv',fieldrows)
compare=[]
for code,r in selected[250].items():
 p125=sum(int(allrows[125].get(code+str(q),{}).get('T001231001','0')) for q in range(1,5))
 p250=r['population'];compare.append({'parent_250m':code,'population_250m':p250,'sum_four_125m_population':p125,'difference_125_minus_250':p125-p250,'boundary_crossing':r['boundary_crossing']})
csvsave(OUT/'125m_to_250m_population_comparison.csv',compare)
print('PARENT COMPARISON',len(compare),'unequal',sum(r['difference_125_minus_250']!=0 for r in compare),flush=True)
# Join the full attribute inventory to already extracted building base geometry.
attrfile=OUT/'plateau_building_attributes.csv'
with attrfile.open() as f:attrs={r['gml_id']:r for r in csv.DictReader(f)}
vector=ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson'
v=json.loads(vector.read_text());bf=[f for f in v['features'] if f['properties']['class']=='building'];del v
bgs=[];br=[];missing=[];pure={'411','412'};mixed={'413','414','415'}
def floorvalid(s):
 try:v=float(s);return math.isfinite(v) and v>0 and v.is_integer() and v!=9999
 except (ValueError,TypeError):return False
for f in bf:
 p=f['properties'];ident=p['gml_id'];a=attrs.get(ident)
 if a is None:missing.append(ident);continue
 g=shape(f['geometry']);inside=g.intersection(u).area
 point=g.representative_point();matches=[w for w,wg in wards.items() if wg.covers(point)]
 role='residential' if a['usage'] in pure else 'mixed_residential' if a['usage'] in mixed else 'unknown' if a['usage'] in ['','461'] else 'other'
 r={**a,'use_group':role,'floors_valid':floorvalid(a['storeysAboveGround']),'base_area_m2':g.area,'geometry_method':p['method'],'area_inside_three_wards_m2':inside,'ward_by_interior_point':matches[0] if matches else 'outside','intersects_three_wards':inside>.01,'crosses_ward_boundary':inside>.01 and inside<g.area-.01}
 br.append(r);bgs.append(g)
csvsave(OUT/'building_allocation_readiness.csv',br)
wardstats=[]
for w in list(wards)+['three_wards']:
 rr=[r for r in br if (r['intersects_three_wards'] if w=='three_wards' else r['ward_by_interior_point']==w)]
 res=[r for r in rr if r['use_group'] in ['residential','mixed_residential']]
 wardstats.append({'ward':w,'building_count':len(rr),'known_use_count':sum(r['use_group']!='unknown' for r in rr),'valid_floors_count':sum(r['floors_valid'] for r in rr),'residential_candidate_count':len(res),'pure_residential_count':sum(r['use_group']=='residential' for r in rr),'mixed_residential_count':sum(r['use_group']=='mixed_residential' for r in rr),'residential_valid_floors_count':sum(r['floors_valid'] for r in res),'residential_valid_floors_pct':100*sum(r['floors_valid'] for r in res)/len(res) if res else None,'usage_distribution':json.dumps(dict(collections.Counter(r['usage'] for r in rr))),'survey_year_distribution':json.dumps(dict(collections.Counter(r['surveyYear'] for r in rr))),'construction_year_present':sum(bool(r['yearOfConstruction']) for r in rr)})
csvsave(REPORT/'building_ward_summary.csv',wardstats)
# Readiness checks retain whole cells and outside-ward building candidates at the edge.
tree=STRtree(bgs);readiness=[]
for res,gs in geoms.items():
 for code,g in gs.items():
  inds=tree.query(g,predicate='intersects');candidates=[]
  for i in inds:
   area=g.intersection(bgs[i]).area
   if area>.01 and br[i]['use_group'] in ['residential','mixed_residential']:candidates.append((int(i),area))
  area=sum(a for i,a in candidates);validarea=sum(a for i,a in candidates if br[i]['floors_valid']);mixedarea=sum(a for i,a in candidates if br[i]['use_group']=='mixed_residential')
  pop=selected[res][code]['population'];readiness.append({'resolution_m':res,'mesh_code':code,'population':pop,'boundary_crossing':selected[res][code]['boundary_crossing'],'residential_candidates':len(candidates),'residential_base_area_m2':area,'residential_area_with_valid_floors_fraction':validarea/area if area else 0,'mixed_residential_area_fraction':mixedarea/area if area else 0,'has_no_residential_candidate':pop>0 and not candidates,'requires_floor_fallback':pop>0 and any(not br[i]['floors_valid'] for i,a in candidates),'residential_floor_proxy_m2_with_valid_floors_only':sum(a*float(br[i]['storeysAboveGround']) for i,a in candidates if br[i]['floors_valid']),'outside_ward_candidates':sum(not br[i]['intersects_three_wards'] for i,a in candidates),'not_a_population_allocation':True})
csvsave(OUT/'mesh_building_readiness.csv',readiness)
readysummary={}
for res in [125,250]:
 rr=[r for r in readiness if r['resolution_m']==res and r['population']>0]
 bad=[r for r in rr if r['has_no_residential_candidate']]
 readysummary[res]={'populated_meshes':len(rr),'no_residential_candidate_meshes':len(bad),'whole_mesh_population_in_no_candidate_cells_not_unallocated_people':sum(r['population'] for r in bad),'no_candidate_interior_meshes':sum(not r['boundary_crossing'] for r in bad),'meshes_requiring_floor_fallback':sum(r['requires_floor_fallback'] for r in rr)}
summary={'statistical_date':'2020-10-01','datum':'JGD2011 geographic EPSG:6668; area computations EPSG:6677','primary_mesh':'5339; all three ward bounds lie within this primary mesh','125m_table':'T001231, reference table released 2025-12-16','250m_table':'T001142, released 2024-03-14','fields_per_table':54,'statistical_fields_per_table':50,'mesh_generation':'JIS X 0410 subdivision; 125 m census data mechanically apportions basic enumeration units across matched meshes when units span cells. Not independently observed 125 m households or building counts.','cautions':'No source record means no population in this official census mesh table; absence of an age value marked * or Y is not zero. Total population is not suppressed. Other attributes may include sums transferred from suppressed meshes.','counts':counts,'parent_comparison':{'parents':len(compare),'unequal':sum(r['difference_125_minus_250']!=0 for r in compare),'net_difference':sum(r['difference_125_minus_250'] for r in compare),'sum_absolute_difference':sum(abs(r['difference_125_minus_250']) for r in compare),'max_absolute_difference':max(abs(r['difference_125_minus_250']) for r in compare)},'buildings':wardstats,'allocation_readiness':readysummary,'geometry_attribute_missing_ids':missing,'allocation_status':'Feasible for assumption-dependent estimates, not ready for unconditional exact building occupancy. No persons allocated in this audit.','boundary_policy':'Full cells retained with full population; intersection fractions are geometry only. Do not multiply population by ward area fraction and call it an official ward count. External buildings beyond the three downloaded packages may be incomplete.','proposed_allocation':'For each full mesh, P_bm=P_m*A_bm*F_b*r_b/sum(A_jm*F_j*r_j). A_bm is building-base intersection area; F is valid above-ground floors; r is residential share. Mixed-use r, unknown floors and absent residential candidates require explicit fallback/sensitivity rules. Keep a residual bucket, never silently relocate population.','remaining_gaps':['2020 census versus PLATEAU usage/floors source principally 2022; geometry source vintages vary. Per-object surveyYear distributions retained.','No verified dwelling counts, residential floor-by-floor use, actual occupancy or complete entrance network.','Missing and placeholder floors must not enter weights; 9999 is excluded as an unusable value, exact sentinel meaning needs specification confirmation.','Institutional residents can occur in facilities outside ordinary residential use codes; do not force these onto homes.','2020 census is a resident-stock baseline, not current 2026 population or daytime workers.'],'sources':['https://www.stat.go.jp/data/mesh/pdf/ryuijikou.pdf','https://www.stat.go.jp/data/mesh/pdf/gaiyo2.pdf','https://www.e-stat.go.jp/gis/statmap-search/data?datatype=1&statsId=T001231&downloadType=1','https://www.e-stat.go.jp/gis/statmap-search/data?datatype=1&statsId=T001142&downloadType=1'],'AnaSOP_sha256_unchanged':ana}
dump(REPORT/'summary.json',summary)
# Scientific figure: common spatial extent, identical density scale.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
fig,axs=plt.subplots(1,3,figsize=(15,6),layout='constrained');norm=Normalize(0,40000)
for ax,res in zip(axs[:2],[125,250]):
 polygons=[];values=[]
 for code,g in geoms[res].items():
  polygons.append(np.asarray(g.exterior.coords));values.append(selected[res][code]['population']/(g.area/1e6))
 pc=PolyCollection(polygons,array=np.array(values),cmap='YlOrRd',norm=norm,edgecolors='none');ax.add_collection(pc);ax.set_title(f'{res} m residents / km2 (full mesh)')
 fig.colorbar(pc,ax=ax,shrink=.65)
polygons=[np.asarray(geoms[250][r['parent_250m']].exterior.coords) for r in compare];v=[r['difference_125_minus_250'] for r in compare];mx=max(1,max(abs(a) for a in v))
pc=PolyCollection(polygons,array=np.array(v),cmap='coolwarm',norm=Normalize(-mx,mx),edgecolors='none');axs[2].add_collection(pc);axs[2].set_title('Sum 4 x 125 m minus 250 m (persons)');fig.colorbar(pc,ax=axs[2],shrink=.65)
for ax in axs:
 for w,g in wards.items():
  for po in ([g] if g.geom_type=='Polygon' else g.geoms):
   xx,yy=po.exterior.xy;ax.plot(xx,yy,color='#263445',lw=.65)
 ax.set_xlim(u.bounds[0]-100,u.bounds[2]+100);ax.set_ylim(u.bounds[1]-100,u.bounds[3]+100);ax.set_aspect('equal');ax.set_xlabel('JGD2011 IX easting (m)');ax.set_ylabel('Northing (m)')
fig.suptitle('2020-10-01 resident population | Chiyoda, Chuo, Minato\n125 m is a mechanically apportioned reference table; boundary cells retain whole-cell counts',fontsize=11)
fig.savefig(REPORT/'population_mesh_comparison.png',dpi=170);plt.close(fig)
def table(rows,keys):return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table>'
body='''<h1>Three-ward resident population and building readiness</h1><p>Official data downloaded. Statistical date: <b>1 October 2020</b>. Both resolutions use JGD2011. This audit does not allocate building occupants or estimate affected people.</p><h2>The important qualification</h2><p>The official 125 m product is a reference table. Basic enumeration units often span several 125 m cells and their counts are mechanically apportioned. Finer cells do not establish independently measured finer population.</p><img src="population_mesh_comparison.png" alt="Population mesh comparison"><h2>Mesh coverage</h2>'''
body+=table([{'resolution':r,**counts[r]} for r in [125,250]],['resolution','intersecting_meshes','source_populated_meshes','boundary_crossing_meshes','interior_full_mesh_population'])
body+='<p>Counts on boundary cells apply to full cells. The displayed clipped shape is not a separately measured ward population. No geometric prorating is presented as a census ward total.</p><h2>125 m versus 250 m</h2><pre>'+html.escape(json.dumps(summary['parent_comparison'],indent=2))+'</pre><h2>Building attributes</h2>'
body+=table(wardstats,['ward','building_count','residential_candidate_count','mixed_residential_count','residential_valid_floors_count','residential_valid_floors_pct'])
body+='<h2>Allocation readiness</h2>'+table([{'resolution':r,**readysummary[r]} for r in [125,250]],['resolution','populated_meshes','no_residential_candidate_meshes','no_candidate_interior_meshes','meshes_requiring_floor_fallback'])
body+='<p>Building candidates include ordinary and mixed residential uses. Candidate coverage is not a measure of population allocation accuracy. Source years, institution populations and incomplete outer-boundary building coverage remain limitations.</p><h2>How an eventual allocation can work</h2><p>'+html.escape(summary['proposed_allocation'])+'</p><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in summary['remaining_gaps'])+'</ul><p>Use 125 m as one sensitivity representation and retain 250 m as the paired comparator; do not declare either validated building truth.</p><h2>Official sources</h2><ul>'+''.join('<li><a href="'+html.escape(s)+'">'+html.escape(s)+'</a></li>' for s in summary['sources'])+'</ul><p><a href="summary.json">Machine-readable summary</a> | <a href="building_ward_summary.csv">Building summary</a></p>'
(REPORT/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Population mesh audit</title><style>body{font:16px system-ui;max-width:1350px;margin:35px auto;padding:0 20px;color:#203044;line-height:1.6}img{width:100%}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd5de;text-align:left}th{background:#eef3f7}pre{background:#f4f6f8;padding:15px}</style>'+body+'</html>')
checks={'original_AnaSOP_unchanged':sha(ROOT/'docs/AnaSOP.md')==ana,'no_missing_geometry_attribute_join':not missing,'mesh_codes_unique':all(len(d)==len(set(d)) for d in selected.values()),'125_to_250_parent_coverage':set(selected[250])=={c[:10] for c in selected[125]},'all_meshes_primary_5339':all(c.startswith('5339') for d in selected.values() for c in d),'both_tables_have_54_fields':len(fieldrows)==108,'sex_totals_match':all(int(r[t+'001'])==int(r[t+'002'])+int(r[t+'003']) for t,res in [('T001231',125),('T001142',250)] for r in allrows[res].values()),'all_clipped_meshes_intersect_wards':all(g.intersection(u).area>.01 for d in geoms.values() for g in d.values()),'clipping_partitions_ward_area':all(abs(unary_union([g.intersection(u) for g in geoms[r].values()]).area-u.area)<.1 for r in [125,250]),'no_population_allocation_performed':True}
for r in json.loads((RAW/'download_manifest.json').read_text()):checks['raw_sha_'+Path(r['file']).name]=sha(ROOT/r['file'])==r['sha256']
dump(REPORT/'validation.json',{'passed':all(checks.values()),'checks':checks});assert all(checks.values()),checks
files=[p for base in [OUT,REPORT] for p in sorted(base.rglob('*')) if p.is_file() and p.name!='freeze.json']
dump(REPORT/'freeze.json',{'products':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],'source_bindings':{str(p.relative_to(ROOT)):sha(p) for p in [boundary,vector,Path(__file__),ROOT/'src/analyses/audit_three_ward_population_attributes_v1.py']},'raw_manifest':str((RAW/'download_manifest.json').relative_to(ROOT))})
print('COMPLETE',json.dumps({'counts':counts,'comparison':summary['parent_comparison'],'buildings':wardstats,'readiness':readysummary,'checks':checks},ensure_ascii=False),flush=True)
