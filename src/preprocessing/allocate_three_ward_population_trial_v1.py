"""Trial paired census-to-building allocations with explicit uncertainty ledgers.
No training, hydraulic simulations, transport-isolation or affected-person counts.
Uses local standard-library CSV plus numpy, shapely, pyproj and matplotlib.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/udt_population_trial_mpl')
import csv,io,json,zipfile,hashlib,math,collections,html,time
import numpy as np
from shapely import STRtree
from shapely.geometry import shape,box,mapping
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2]
PREV=ROOT/'data/processed/three_ward_population_mesh_audit_v1'
OUT=ROOT/'data/processed/three_ward_population_allocation_trial_v1'
REPORT=ROOT/'data/exp/data-preprocessing/three_ward_population_allocation_trial_v1'
OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def writecsv(p,rows,keys=None):
 keys=keys or list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def tagbool(s):return str(s)=='True'
ana=sha(ROOT/'docs/AnaSOP.md')
previous_manifest=ROOT/'data/exp/data-briefing/three_ward_population_mesh_audit_v1/delivery_manifest.json'
previous=json.loads(previous_manifest.read_text())
assert all(sha(ROOT/r['file'])==r['sha256'] for r in previous['products'])
xy=Transformer.from_crs(6668,6677,always_xy=True).transform
wj=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())
wards={f['properties']['ward'][:5]:transform(xy,shape(f['geometry'])) for f in wj['features']};wardnames=list(wards);union=unary_union(list(wards.values()))
index={r['mesh_code']:r for r in readcsv(PREV/'population_250m_mesh_index.csv')}
parent_geoms={f['properties']['mesh_code']:transform(xy,shape(f['geometry'])) for f in json.loads((PREV/'population_250m_full_mesh.geojson').read_text())['features']}
source={res:{r['KEY_CODE']:r for r in readcsv(PREV/f'population_{res}m_source_rows.csv')} for res in [125,250]}
def mesh(code):
 lat=int(code[:2])*2/3+int(code[4])/12+int(code[6])/120;lon=int(code[2:4])+100+int(code[5])/8+int(code[7])/80;dy,dx=1/120,1/80
 for q in code[8:]:
  dy/=2;dx/=2;v=int(q)-1;lat+=(v//2)*dy;lon+=(v%2)*dx
 return transform(xy,box(lon,lat,lon+dx,lat+dy))
geoms={250:parent_geoms,125:{p+str(q):mesh(p+str(q)) for p in index for q in range(1,5)}}
def population(res,c):return int(source[res].get(c,{}).get(('T001231' if res==125 else 'T001142')+'001','0'))
assert all(sum(population(125,p+str(q)) for q in range(1,5))==population(250,p) for p in index)
attrs={r['gml_id']:r for r in readcsv(PREV/'building_allocation_readiness.csv')}
vectors=ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson'
v=json.loads(vectors.read_text());bs=[f for f in v['features'] if f['properties']['class']=='building'];del v
bgs=[];br=[]
for f in bs:
 a=attrs[f['properties']['gml_id']];g=shape(f['geometry']);r=dict(a)
 r['Valid Floors']=tagbool(a['floors_valid']);r['Use Group']=a['use_group'];r['Residential Share']=1.0 if a['use_group']=='residential' else .5 if a['use_group']=='mixed_residential' else 0.0
 r['Floors']=float(a['storeysAboveGround']) if r['Valid Floors'] else None
 r['Capacity per Base Area']=r['Residential Share']*r['Floors'] if r['Floors'] is not None else None
 bgs.append(g);br.append(r)
assert len({r['buildingID'] for r in br})==len(br)
shared_weights=[{'Building ID':r['buildingID'],'GML ID':r['gml_id'],'Use Group':r['Use Group'],'Usage Code':r['usage'],'Detailed Usage Code':r['detailedUsage'],'Valid Above Ground Floors':r['Valid Floors'],'Above Ground Floors':r['Floors'],'Assumed Residential Share':r['Residential Share'],'Capacity per Base Area':r['Capacity per Base Area'],'Mixed Share Is Observed':False if r['Use Group']=='mixed_residential' else '', 'Survey Year Raw':r['surveyYear']} for r in br]
writecsv(OUT/'shared_building_weights.csv',shared_weights)
protocol={'human_decision':'MILI-D-20261016-017','status':'Exploratory trial, not a validated building occupancy dataset','spatial_scope':'All 757 250 m cells intersecting the three wards, with all four 125 m companions per parent (3028 child cells). Of these, only full 250 m parents inside the union are eligible for trial allocation. Outer boundary parent cells and all their children are held separately at both resolutions. This common support avoids assigning incomplete outside building stock and prevents different boundary masks from driving the comparison.','same_physical_weights':'Above-ground floor count multiplied by base-area intersection with the census cell and a fixed building residential share. Building capacity per base area is identical at both resolutions. No height-derived floors, roof-edge area or floor-area conversion factor is substituted.','pure_residential':'Usage 411 and 412: residential share 1, provided above-ground floors are valid. This remains an estimated building distribution.','mixed_residential':'Usage 413, 414 and 415: residential share 0.5 is an explicit analyst-selected exploratory assumption, not observed. If any positive-weight mixed residential base occurs in a cell, the entire cell allocation is marked conditional. This conditional mass is reported separately from ordinary floor-weight estimates.','unknown_use':'Unknown or unrecognized use blocks the whole positive-population cell: all its population goes to a pending pool, because there is no identified way to reserve a defensible share for the unknown building. No silent redistribution to known homes.','missing_floors':'A residential or mixed residential base without valid positive integer floors, including 9999, blocks the whole positive-population cell. No zero-filling, height/3 proxy or median imputation.','possible_institutional_population':'Detailed uses 1131 (medical) and 1132 (welfare) block the cell as possible collective-residence uncertainty. Other nonresidential uses receive no ordinary housing weight. This is not proof that facilities host census residents; unrecognized institutional residents may remain a limitation.','no_candidates':'If there is no positive residential capacity, all population remains pending. Pending records retain all applicable reason flags.','conservation':'For every mesh: ordinary estimated + conditional estimated + pending = source total population. Fractional expected counts are retained; do not round each building to an integer. The final allocation row absorbs floating-point normalization residue only.','common_support_comparison':'Compare building distributions only for full interior 250 m parents whose positive-population cells are completely allocated at both resolutions. Include both ordinary and conditional estimates but identify their assumption dependence. Report changes in eligibility separately from redistribution on matched support.','ward_reporting':'Allocate mesh-building expected counts to ward fragments in proportion to the same building-base intersection areas; these are modeled ward totals on the trial support, not official whole-ward populations or entrance locations.','age_and_commuting':'Audit existing resident age fields and official resident employment/schooling fields at mesh scale. Do not allocate ages, infer destinations, construct daytime building population or calculate affected/stranded persons in this trial.','original_data':'All prior frozen products, raw census curves, model splits and masks are unchanged. No AnaSOP sections are overwritten in this exploratory trial.'}
dump(OUT/'allocation_protocol.json',protocol)
tree=STRtree(bgs);allocations={};ledgers={};partcache={};started=time.time()
for res in [125,250]:
 rows=[];ledger=[]
 for j,(code,g) in enumerate(sorted(geoms[res].items())):
  parent=code[:10];p=population(res,code);boundary=tagbool(index[parent]['boundary_crossing']);flags=[];pieces=[];nc=nm=nu=nf=ni=0
  if boundary:flags.append('common_parent_boundary_or_incomplete_external_stock')
  # Even held cells retain factual inventory counts; no person estimate is made for them.
  for k in tree.query(g,predicate='intersects'):
   k=int(k);part=g.intersection(bgs[k]);area=part.area
   if area<=.01:continue
   r=br[k];nc+=1
   if r['Use Group']=='unknown':nu+=1
   if r['detailedUsage'] in ['1131','1132']:ni+=1
   if r['Use Group'] in ['residential','mixed_residential']:
    if not r['Valid Floors']:nf+=1
    else:
     if r['Use Group']=='mixed_residential':nm+=1
     capacity=area*r['Capacity per Base Area'];pieces.append((k,part,area,capacity))
  if nu:flags.append('unknown_building_use')
  if nf:flags.append('residential_floor_count_missing_or_placeholder')
  if ni:flags.append('possible_institutional_residents')
  if not pieces:flags.append('no_positive_residential_capacity')
  if not p:status='zero_population'
  elif flags:status='pending'
  else:status='conditional_mixed_use' if nm else 'ordinary_floor_weight_estimate'
  allocated=0.;wgcounts={w:0. for w in wardnames}
  if p and status!='pending':
   total=math.fsum(x[3] for x in pieces);amounts=[p*x[3]/total for x in pieces];amounts[-1]=p-math.fsum(amounts[:-1])
   for (k,part,area,cap),amount in zip(pieces,amounts):
    area_by_ward=[part.intersection(wards[w]).area for w in wardnames];den=math.fsum(area_by_ward)
    assert abs(den-area)<max(.1,area*1e-7),(code,k,area,den)
    wp=[amount*a/den for a in area_by_ward];wp[-1]=amount-math.fsum(wp[:-1])
    for w,value in zip(wardnames,wp):wgcounts[w]+=value
    row={'Resolution':res,'Mesh Code':code,'Parent 250m Code':parent,'Building ID':br[k]['buildingID'],'GML ID':br[k]['gml_id'],'Allocation Class':status,'Building Use Group':br[k]['Use Group'],'Intersection Base Area':area,'Shared Capacity per Base Area':br[k]['Capacity per Base Area'],'Weight':cap,'Estimated Residents':amount,**{w+' Estimated Residents':value for w,value in zip(wardnames,wp)}}
    rows.append(row)
   allocated=math.fsum(amounts)
  ordinary=allocated if status=='ordinary_floor_weight_estimate' else 0.;conditional=allocated if status=='conditional_mixed_use' else 0.;pending=float(p)-allocated
  ledger.append({'Resolution':res,'Mesh Code':code,'Parent 250m Code':parent,'Source Population':p,'Common Parent Boundary':boundary,'Allocation Class':status,'Ordinary Estimated Residents':ordinary,'Conditional Estimated Residents':conditional,'Pending Residents':pending,'Budget Residual':p-ordinary-conditional-pending,'Reason Flags':'|'.join(flags),'Intersecting Buildings':nc,'Positive Capacity Candidates':len(pieces),'Unknown Use Candidates':nu,'Invalid Residential Floor Candidates':nf,'Mixed Residential Candidates':nm,'Possible Institutional Candidates':ni,**{w+' Estimated Residents':wgcounts[w] for w in wardnames}})
  if j%500==0:print('ALLOCATE',res,j,len(geoms[res]),'seconds',round(time.time()-started),flush=True)
 allocations[res]=rows;ledgers[res]=ledger
 writecsv(OUT/f'building_allocations_{res}m.csv',rows,['Resolution','Mesh Code','Parent 250m Code','Building ID','GML ID','Allocation Class','Building Use Group','Intersection Base Area','Shared Capacity per Base Area','Weight','Estimated Residents',*[w+' Estimated Residents' for w in wardnames]])
 writecsv(OUT/f'mesh_population_ledger_{res}m.csv',ledger)
 print('COMPLETE RESOLUTION',res,'rows',len(rows),'ordinary',sum(r['Ordinary Estimated Residents'] for r in ledger),'conditional',sum(r['Conditional Estimated Residents'] for r in ledger),'pending',sum(r['Pending Residents'] for r in ledger),flush=True)
# Reasons overlap. Preserve both multi-flag reasons and a mutually exclusive primary status ledger.
reasonrows=[]
for res,ledger in ledgers.items():
 for reason in sorted({s for r in ledger for s in r['Reason Flags'].split('|') if s}):
  rr=[r for r in ledger if r['Source Population']>0 and r['Allocation Class']=='pending' and reason in r['Reason Flags'].split('|')]
  reasonrows.append({'Resolution':res,'Reason':reason,'Positive Population Cells':len(rr),'Population With This Flag':sum(r['Source Population'] for r in rr),'Reasons Overlap':True})
writecsv(REPORT/'pending_reason_flags.csv',reasonrows)
# Matched complete parents remove coverage differences from the allocation comparison.
l250={r['Mesh Code']:r for r in ledgers[250]};l125={r['Mesh Code']:r for r in ledgers[125]}
matched=[c for c,r in l250.items() if r['Source Population']>0 and not r['Common Parent Boundary'] and r['Pending Residents']==0 and all(l125[c+str(q)]['Pending Residents']==0 for q in range(1,5))]
matchedset=set(matched);byparent={125:collections.defaultdict(lambda:collections.defaultdict(float)),250:collections.defaultdict(lambda:collections.defaultdict(float))}
for res in [125,250]:
 for r in allocations[res]:
  if r['Parent 250m Code'] in matchedset:byparent[res][r['Parent 250m Code']][r['Building ID']]+=r['Estimated Residents']
parentdiff=[];bd=[];totals={res:collections.defaultdict(float) for res in [125,250]};matchedward={res:{w:0. for w in wardnames} for res in [125,250]}
for p in matched:
 keys=set(byparent[125][p])|set(byparent[250][p]);l1=math.fsum(abs(byparent[125][p].get(k,0)-byparent[250][p].get(k,0)) for k in keys)
 parentdiff.append({'Parent 250m Code':p,'Source Population':population(250,p),'Half L1 Redistribution Mass':l1/2,'Redistribution Fraction':l1/2/population(250,p),'Building Count':len(keys)})
 for res in [125,250]:
  for k,value in byparent[res][p].items():totals[res][k]+=value
for res in [125,250]:
 for r in allocations[res]:
  if r['Parent 250m Code'] in matchedset:
   for w in wardnames:matchedward[res][w]+=r[w+' Estimated Residents']
for k in sorted(set(totals[125])|set(totals[250])):
 a=totals[125].get(k,0);b=totals[250].get(k,0);bd.append({'Building ID':k,'125m Estimated Residents':a,'250m Estimated Residents':b,'Difference 125m Minus 250m':a-b,'Absolute Difference':abs(a-b)})
writecsv(OUT/'matched_support_building_comparison.csv',bd,['Building ID','125m Estimated Residents','250m Estimated Residents','Difference 125m Minus 250m','Absolute Difference'])
writecsv(OUT/'matched_support_parent_comparison.csv',parentdiff,['Parent 250m Code','Source Population','Half L1 Redistribution Mass','Redistribution Fraction','Building Count'])
wardrows=[{'Ward':w,'125m Matched Support Estimated Residents':matchedward[125][w],'250m Matched Support Estimated Residents':matchedward[250][w],'Difference 125m Minus 250m':matchedward[125][w]-matchedward[250][w],'Not Official Ward Population':True} for w in wardnames]
writecsv(REPORT/'matched_support_ward_comparison.csv',wardrows)
summaryrows=[]
for res,ledger in ledgers.items():
 for scope in ['full_parent_envelope','common_interior_support','outer_parent_boundary_pending']:
  rr=[r for r in ledger if scope=='full_parent_envelope' or (not r['Common Parent Boundary'] if scope=='common_interior_support' else r['Common Parent Boundary'])]
  summaryrows.append({'Resolution':res,'Scope':scope,'Mesh Count':len(rr),'Source Population':sum(r['Source Population'] for r in rr),'Ordinary Estimated Residents':sum(r['Ordinary Estimated Residents'] for r in rr),'Conditional Estimated Residents':sum(r['Conditional Estimated Residents'] for r in rr),'Pending Residents':sum(r['Pending Residents'] for r in rr),'Allocated Population Fraction':sum(r['Ordinary Estimated Residents']+r['Conditional Estimated Residents'] for r in rr)/sum(r['Source Population'] for r in rr) if sum(r['Source Population'] for r in rr) else 0})
writecsv(REPORT/'population_accounting_summary.csv',summaryrows)
# Existing census age fields: cell-local availability only when no suppression/receiving flag is present.
ageaudits=[];agevalid=[]
for res,tid in [(125,'T001231'),(250,'T001142')]:
 rows=[]
 for code in geoms[res]:
  raw=source[res].get(code);p=population(res,code)
  if not p:continue
  flag=raw['HTKSYORI'];vals=[raw[tid+x] for x in ['004','010','019']];numeric=all(x.isdigit() for x in vals);valid=flag=='0' and numeric and sum(map(int,vals))<=p
  rows.append({'Resolution':res,'Mesh Code':code,'Source Population':p,'HTKSYORI':flag,'Age 0 To 14 Raw':vals[0],'Age 15 To 64 Raw':vals[1],'Age 65 And Over Raw':vals[2],'Cell Local Age Partition Usable':valid,'Unknown Age Residents':p-sum(map(int,vals)) if valid else '', 'Common Parent Boundary':tagbool(index[code[:10]]['boundary_crossing'])})
 agevalid.extend(rows);inside=[r for r in rows if not r['Common Parent Boundary']]
 ageaudits.append({'Resolution':res,'Interior Populated Meshes':len(inside),'Cell Local Age Partition Usable Meshes':sum(r['Cell Local Age Partition Usable'] for r in inside),'Age Flags Or Invalid Meshes':sum(not r['Cell Local Age Partition Usable'] for r in inside),'Resident Population In Usable Age Cells':sum(r['Source Population'] for r in inside if r['Cell Local Age Partition Usable']),'Note':'Usable means a non-suppressed local age partition, not proof of building-level ages. 65+,75+,85+,95+ overlap and must not be summed.'})
writecsv(OUT/'age_field_quality.csv',agevalid);writecsv(REPORT/'age_field_availability.csv',ageaudits)
# Resident commuting traits are NOT work/school destination stocks or mesh-to-mesh OD.
commute_fields=[];commute_quality=[];commute_availability=[]
for res,tid in [(125,'T001232'),(250,'T001145')]:
 with zipfile.ZipFile(OUT/'official_definitions'/f'{tid}_5339.zip') as z:
  with z.open(z.namelist()[0]) as h:
   rr=csv.reader(io.TextIOWrapper(h,encoding='cp932'));head=next(rr);labels=next(rr);cr={r[0]:dict(zip(head,r)) for r in rr if r[0] in geoms[res]}
 writecsv(OUT/f'resident_commuting_{res}m_source_rows.csv',list(cr.values()),head)
 for k,label in zip(head,labels):
  if k.startswith(tid):commute_fields.append({'Resolution':res,'Field':k,'Official Label':label.strip(),'Spatial Basis':'Residence mesh; not workplace or school destination population','Candidate Role':'resident commuting characteristic; not yet included in building allocation'})
 for code in geoms[res]:
  if population(res,code)<=0:continue
  r=cr.get(code);flag=r['HTKSYORI'] if r else 'missing_record';values={name:r.get(tid+suffix,'') if r else '' for name,suffix in [('Resident Workers Or Students 15 Plus','086'),('Resident Workers 15 Plus','087'),('Resident Students 15 Plus','088'),('Resident Rail Users 15 Plus Outside Home','090')]}
  good=flag=='0' and all(v.isdigit() for v in values.values())
  commute_quality.append({'Resolution':res,'Mesh Code':code,'HTKSYORI':flag,'Unmerged Local Selected Fields Usable':good,'Common Parent Boundary':tagbool(index[code[:10]]['boundary_crossing']),**values})
 rr=[r for r in commute_quality if r['Resolution']==res and not r['Common Parent Boundary']]
 commute_availability.append({'Resolution':res,'Interior Populated Meshes':len(rr),'Unmerged Local Selected Fields Usable Meshes':sum(r['Unmerged Local Selected Fields Usable'] for r in rr),'Raw Statistical Fields':len(head)-4,'Contains Destination Coordinates':False,'Contains Mesh To Mesh OD':False,'Can Directly Supply Daytime Population':False})
writecsv(OUT/'resident_commuting_field_dictionary.csv',commute_fields);writecsv(OUT/'resident_commuting_field_quality.csv',commute_quality);writecsv(REPORT/'resident_commuting_availability.csv',commute_availability)
absdiff=np.array([r['Absolute Difference'] for r in bd]);mpop=sum(population(250,p) for p in matched);globalhalf=float(absdiff.sum()/2)
summary={'status':'Exploratory paired allocation complete; no affected-person calculation','protocol':protocol,'accounting':summaryrows,'matched_support':{'parents':len(matched),'source_population':mpop,'building_count':len(bd),'building_total_half_L1_redistribution_mass':globalhalf,'fraction_of_matched_population':globalhalf/mpop if mpop else None,'sum_per_parent_half_L1_mass':sum(r['Half L1 Redistribution Mass'] for r in parentdiff),'median_building_absolute_difference':float(np.median(absdiff)) if len(absdiff) else None,'p90_building_absolute_difference':float(np.quantile(absdiff,.9)) if len(absdiff) else None,'max_building_absolute_difference':float(absdiff.max()) if len(absdiff) else None,'metric_is_accuracy':False,'warning':'Matched complete support is a data-quality-selected subset. It is not representative of all three wards and cannot validate either mesh distribution.'},'ward_comparison':wardrows,'age_fields':ageaudits,'resident_commuting_fields':commute_availability,'limitations':['Both population products describe 2020-10-01 residents; PLATEAU source lists cite 2022 while known building objects carry surveyYear 2021.','0.5 mixed-use residential share is hypothetical. All populations in such cells are conditional estimates; no measured mixed floor shares exist.','Whole-cell withholding for unknown use or missing floors avoids arbitrary individual allocation but can reduce coverage sharply, especially at 250 m. Coverage differences are distinct from spatial redistribution.','Population conservation does not establish correct building occupancy. Building floor area is an approximation, not dwelling counts or floor-specific use.','The common outer-parent envelope includes residents outside the wards; outer parent cells are entirely held. Neither its 547184 count nor the interior support count is an official three-ward total.','No age-specific building residents, actual daytime stocks, commute OD, station demand or stranded people are constructed.','Travel mode fields are resident traits. They do not identify Tokyo Metro versus JR, destination stations, routes, time-of-day demand or current service use.'],'AnaSOP_sha256_unchanged':ana,'prior_delivery_manifest_sha256':sha(previous_manifest)}
dump(REPORT/'summary.json',summary)
# Whole-city overview of allocation coverage and within-parent redistribution.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained');parentlist=sorted(parent_geoms);polys=[np.array(parent_geoms[p].exterior.coords) for p in parentlist]
frac125={p:sum(l125[p+str(q)]['Ordinary Estimated Residents']+l125[p+str(q)]['Conditional Estimated Residents'] for q in range(1,5))/population(250,p) if population(250,p)>0 else np.nan for p in parentlist}
frac250={p:(l250[p]['Ordinary Estimated Residents']+l250[p]['Conditional Estimated Residents'])/population(250,p) if population(250,p)>0 else np.nan for p in parentlist};div={r['Parent 250m Code']:r['Redistribution Fraction'] for r in parentdiff}
for ax,vals,title in zip(axes,[frac125,frac250,div],['125 m allocation coverage','250 m allocation coverage','Redistribution on matched parents']):
 data=[vals.get(p,np.nan) if not l250[p]['Common Parent Boundary'] else np.nan for p in parentlist]
 pc=PolyCollection(polys,array=np.ma.masked_invalid(data),cmap='viridis',norm=Normalize(0,1),edgecolors='none');pc.cmap.set_bad('#e0e0e0');ax.add_collection(pc);fig.colorbar(pc,ax=ax,shrink=.68,label='Fraction of parent population')
 for wg in wards.values():
  for po in ([wg] if wg.geom_type=='Polygon' else wg.geoms):
   x,y=po.exterior.xy;ax.plot(x,y,color='#273747',lw=.65)
 ax.set_title(title);ax.set_xlim(union.bounds[0]-100,union.bounds[2]+100);ax.set_ylim(union.bounds[1]-100,union.bounds[3]+100);ax.set_aspect('equal');ax.set_xlabel('JGD2011 IX easting (m)');ax.set_ylabel('Northing (m)')
fig.suptitle('Paired 2020 resident allocation | Chiyoda, Chuo, Minato\nGray: withheld boundary, zero population, or outside matched support. Coverage includes conditional mixed-use estimates.',fontsize=11)
fig.savefig(REPORT/'allocation_overview.png',dpi=170);plt.close(fig)
def table(rows,keys):return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:,.3f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
body='<h1>Paired resident population allocation trial</h1><p><b>Purpose:</b> compare building allocation from 125 m and 250 m census meshes with identical physical weights. No stranded or affected persons, daytime building occupants or transport disruptions are calculated.</p><p>Population reference date: 2020-10-01. JGD2011. Source inputs and the previous frozen audit are preserved.</p><img src="allocation_overview.png" alt="Three-ward allocation coverage and redistribution overview"><h2>Population conservation and coverage</h2>'
body+='<p>The common geographic comparison support contains '+str(sum(not r['Common Parent Boundary'] for r in ledgers[250]))+' complete 250 m parents. All four 125 m child cells are included per parent. Boundary parent populations are held at both resolutions; they are not prorated into the wards. Counts are trial-support population, not official complete ward totals.</p>'
body+=table([r for r in summaryrows if r['Scope']=='common_interior_support'],['Resolution','Source Population','Ordinary Estimated Residents','Conditional Estimated Residents','Pending Residents','Allocated Population Fraction'])
body+='<h2>Rules shared across resolutions</h2><ul>'+''.join('<li><b>'+html.escape(k)+':</b> '+html.escape(protocol[k])+'</li>' for k in ['same_physical_weights','pure_residential','mixed_residential','unknown_use','missing_floors','possible_institutional_population','no_candidates','conservation'])+'</ul><p>Conditional estimates are separately accounted as unresolved assumptions, even though a trial building distribution is supplied. Pending mesh populations receive no building estimate.</p><h2>Difference after matching complete support</h2><pre>'+html.escape(json.dumps(summary['matched_support'],indent=2))+'</pre>'
body+=table(wardrows,['Ward','125m Matched Support Estimated Residents','250m Matched Support Estimated Residents','Difference 125m Minus 250m'])
body+='<p>Ward amounts use building-base fragments inside the matched trial support. Neither set is validated building truth, and complete-case selection limits generalization.</p><h2>Age information</h2>'+table(ageaudits,['Resolution','Interior Populated Meshes','Cell Local Age Partition Usable Meshes','Age Flags Or Invalid Meshes'])
body+='<p>Population/household tables already contain broad age fields. The disjoint 0-14, 15-64 and 65+ partition is inspected, retaining unknown-age residents. 65+,75+,85+,95+ are nested, not mutually exclusive. Suppressed cells and receiving cells are not accepted as standalone local age composition. No ages are allocated to buildings.</p><h2>Employment and schooling: what these tables actually contain</h2>'+table(commute_availability,['Resolution','Raw Statistical Fields','Interior Populated Meshes','Unmerged Local Selected Fields Usable Meshes','Can Directly Supply Daytime Population'])
body+='<p>The reviewed 125 m T001232 and 250 m T001145 tables count workers/students aged 15+ who <b>reside locally</b>, plus resident travel-mode traits. This can characterize residential dependence on rail. It does not measure workers or students arriving in that cell, daytime stock, mesh-to-mesh OD, destination stations or Metro/JR-specific use. Daytime exposure requires a separately audited workplace/school destination dataset.</p><h2>Limits and next use</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limitations'])+'</ul><p>Next candidate step: agree whether the loss of coverage from conservative whole-cell withholding is acceptable, or define an explicitly provisional rule for unknown buildings and institutional residents. More detailed age and commuting data should be used at compatible mesh/privacy-group support before any building-specific claims.</p><p><a href="summary.json">Summary JSON</a> | <a href="population_accounting_summary.csv">Population budgets</a> | <a href="pending_reason_flags.csv">Overlapping pending reasons</a> | <a href="validation.json">Validation</a></p><p>Official definitions: <a href="https://www.e-stat.go.jp/gis/statmap-search/data?datatype=1&amp;statsId=T001232&amp;downloadType=1">125 m resident employment/schooling</a>; <a href="https://www.e-stat.go.jp/help/data-definition-information/downloaddata/T001145.pdf">250 m definition</a>; <a href="https://www.stat.go.jp/data/mesh/pdf/ryuijikou.pdf">Confidentiality and 125 m reference-table cautions</a>.</p>'
(REPORT/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired population allocation trial</title><style>body{font:16px system-ui;max-width:1350px;margin:35px auto;padding:0 20px;color:#203044;line-height:1.6}img{width:100%}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd5de;text-align:left}th{background:#eef3f7}pre{background:#f4f6f8;padding:15px;overflow:auto}</style>'+body+'</html>')
weight_lookup={r['Building ID']:r['Capacity per Base Area'] for r in shared_weights}
checks={'prior_products_unchanged':all(sha(ROOT/r['file'])==r['sha256'] for r in previous['products']),'AnaSOP_unchanged':sha(ROOT/'docs/AnaSOP.md')==ana,'same_envelope_population':sum(r['Source Population'] for r in ledgers[125])==sum(r['Source Population'] for r in ledgers[250]),'every_mesh_conserves_population':all(abs(r['Budget Residual'])<1e-8 for rr in ledgers.values() for r in rr),'every_positive_mesh_has_exactly_one_class':all(sum(r[k]>0 for k in ['Ordinary Estimated Residents','Conditional Estimated Residents','Pending Residents'])==1 for rr in ledgers.values() for r in rr if r['Source Population']>0),'all_boundary_population_held':all(r['Pending Residents']==r['Source Population'] for rr in ledgers.values() for r in rr if r['Common Parent Boundary']),'all_unidentified_capacity_population_held':all(r['Pending Residents']==r['Source Population'] for rr in ledgers.values() for r in rr if r['Source Population']>0 and (r['Unknown Use Candidates'] or r['Invalid Residential Floor Candidates'] or r['Possible Institutional Candidates'])),'same_shared_capacity_across_both_resolutions':all(r['Shared Capacity per Base Area']==weight_lookup[r['Building ID']] for rr in allocations.values() for r in rr),'all_weights_positive':all(r['Weight']>0 for rr in allocations.values() for r in rr),'matched_total_population_conserved':all(abs(sum(totals[res].values())-mpop)<1e-6 for res in [125,250]),'matched_ward_totals_conserved':all(abs(sum(matchedward[res].values())-mpop)<1e-6 for res in [125,250]),'no_affected_population_output':True,'age_and_commuting_remain_mesh_level':True}
# Per-mesh sparse rows independently reconcile to the ledger.
for res in [125,250]:
 sums=collections.defaultdict(float)
 for r in allocations[res]:sums[r['Mesh Code']]+=r['Estimated Residents']
 checks[f'sparse_allocations_match_mesh_ledger_{res}']=all(abs(sums[r['Mesh Code']]-r['Ordinary Estimated Residents']-r['Conditional Estimated Residents'])<1e-7 for r in ledgers[res])
dump(REPORT/'validation.json',{'passed':all(checks.values()),'checks':checks});assert all(checks.values()),checks
files=[p for base in [OUT,REPORT] for p in sorted(base.rglob('*')) if p.is_file() and p.name!='freeze.json']
dump(REPORT/'freeze.json',{'products':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files],'source_script':{'file':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__))},'source_audit_delivery_manifest_sha256':sha(previous_manifest)})
print('COMPLETE',json.dumps({'accounting':summaryrows,'matched':summary['matched_support'],'age':ageaudits,'commuting':commute_availability,'checks':checks},ensure_ascii=False),flush=True)
