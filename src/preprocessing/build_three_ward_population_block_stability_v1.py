from pathlib import Path
import zipfile,xml.etree.ElementTree as E,collections
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform,unary_union
from shapely import make_valid,STRtree
from pyproj import Transformer
G='{http://www.opengis.net/gml}';xy=Transformer.from_crs(6668,6677,always_xy=True).transform
root=Path(__file__).resolve().parents[2]
def load(level):
 groups={}
 for ward in ['13101','13102','13103']:
  z=zipfile.ZipFile(root/f'data/processed/three_ward_population_block_stability_v1/official_boundaries/{level}_{ward}_2020_jgd2011_gml.zip');doc=E.fromstring(z.read(next(n for n in z.namelist() if n.endswith('.gml'))))
  for member in doc.findall(G+'featureMember'):
   f=member[0];r={e.tag.split('}')[-1]:e.text or '' for e in f if not e.tag.startswith(G)};key=ward+r['K_AREA'] if level=='B' else ward+r['KIHON1']+r['KIHON2'];polys=[]
   for patch in f.findall('.//'+G+'PolygonPatch'):
    def ring(x):
     a=np.array([float(v) for v in x.find('.//'+G+'posList').text.split()]).reshape(-1,2);return np.column_stack(xy(a[:,1],a[:,0]))
    p=Polygon(ring(patch.find(G+'exterior')),[ring(h) for h in patch.findall(G+'interior')]);polys.append(make_valid(p))
   if key not in groups:groups[key]={'parts':[],'pop':0,'ward':ward,'name':r['S_NAME'],'town':ward+r['KIHON1']+r['KIHON2'],'records':0}
   d=groups[key];d['parts']+=polys;d['pop']+=int(r['JINKO']);d['records']+=1
 for k,d in groups.items():d['geometry']=unary_union(d.pop('parts'))
 return groups
# The boundary reader above groups census survey records by the nine-digit basic
# unit code (ward + K_AREA), and preserves all multipart geometry and population.
if __name__=='__main__':
 import csv,json,hashlib,itertools,time
 from shapely.geometry import box,shape,mapping
 ROOT=Path(__file__).resolve().parents[2];root=ROOT
 PREV=ROOT/'data/processed/three_ward_population_tiered_intervals_v1'
 OUT=ROOT/'data/processed/three_ward_population_block_stability_v1'
 REPORT=ROOT/'data/exp/data-preprocessing/three_ward_population_block_stability_v1'
 REPORT.mkdir(parents=True,exist_ok=True)
 def readcsv(p):
  with p.open() as f:return list(csv.DictReader(f))
 def writecsv(p,rows):
  with p.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  return h.hexdigest()
 frozen_path=ROOT/'data/exp/data-preprocessing/three_ward_population_tiered_intervals_v1/delivery_manifest.json'
 frozen=json.loads(frozen_path.read_text());frozen_items=frozen['products']+frozen['scripts']
 assert all(sha(ROOT/r['file'])==r['sha256'] for r in frozen_items)
 protocol={
  'human_decision':'MILI-D-20261016-019',
  'primary_unit':'2020 census basic unit, ward + nine-digit K_AREA; union all survey-record polygons; sum JINKO across all source records. Towns are ward + first six K_AREA digits.',
  'dates':'Census 2020-10-01; JGD2011 boundary catalog release 2023-01-04. These are historical residents, not current occupancy or daytime population.',
  'boundary_meaning':'Census basic units generally approximate street blocks. Neither basic units nor towns are road-network isolation components.',
  'scope':'Original common 541 interior 250 m parents, 419921 residents; all 216 boundary parents remain held. No geographic prorating of boundary population.',
  'geometry':'GML EPSG:6668 latitude/longitude converted to EPSG:6677 metres. Preserve holes/multipart features. Same building fragments as frozen allocation.',
  'group_population':'Optimize the normalized sum of capacity-weighted building/mesh fractions inside each block, including potential unknown/institutional candidates only as an aggregate conditional location scenario. No individual occupancy is assigned to unknown buildings.',
  'unknown_pool':'Frozen mesh-level unlocated pool remains unlocated at building scale. Its possible block location is bounded by candidate geometry under the same q95/q99 capacity assumptions. Census-only no-capacity pools remain explicitly unresolved; no area-weighted point allocation.',
  'bounds':'For coefficients a_i, optimize p*sum(a_i*w_i)/sum(w_i), w_i in [L_i,U_i], over positive total capacity. Sort a_i; extrema occur at monotone endpoint assignments. Sum cell group extrema for conservative outer bounds. Never sum individual building upper bounds; block endpoints are not a jointly realizable map.',
  'complete_coverage':'Uncovered block area <= max(1 square metre, 1e-6 times area). Incomplete blocks are withheld as whole-block allocation inputs.',
  'screening_not_validation':True,
  'screen_thresholds':{
   'stable_conditional':{'point_spread':'<= max(10 people, 10% of four-scenario mean)','envelope_width':'<= max(25 people, 25% of mean)','census_reference_difference':'<= max(25 people, 25% of official block population)','unresolved_census_only_potential':'<= max(5 people, 5% of mean)'},
   'interval_only':{'point_spread':'<= max(25 people, 25% of mean)','envelope_width':'<= max(50 people, 100% of mean)','census_reference_difference':'<= max(50 people, 50% of official block population)','unresolved_census_only_potential':'<= max(10 people, 25% of mean)'},
   'otherwise':'withhold spatial allocation; use independently supplied census block total only as a 2020 aggregate baseline where appropriate. Thresholds are analyst screening choices, not official standards or human-selected parameters.'},
  'reference_check':'Official boundary JINKO is a same-census aggregate check, not independent survey validation. Existing allocations are neither rescaled nor fitted to it. Official counts can supply a historical block-total baseline, but do not validate within-block locations.',
  'interpretation':'q95/q99 are conditional capacity ceilings, not confidence intervals. Stability across resolution is not occupancy accuracy. No stranded counts, transport disruption, training or hydraulic simulation.'}
 dump(OUT/'protocol.json',protocol)
 blocks=load('B');towns=load('A');bkeys=sorted(blocks);tkeys=sorted(towns)
 for ward in ['13101','13102','13103']:
  assert sum(v['pop'] for v in blocks.values() if v['ward']==ward)==sum(v['pop'] for v in towns.values() if v['ward']==ward)
 geom=[blocks[k]['geometry'] for k in bkeys];btree=STRtree(geom);bidx={k:i for i,k in enumerate(bkeys)}
 # No positive-area block overlap is allowed; edges alone do not double count.
 overlap=[]
 for i,g in enumerate(geom):
  for j in btree.query(g,predicate='intersects'):
   if j>i and (a:=g.intersection(geom[j]).area)>.01:overlap.append((i,int(j),a))
 assert not overlap,overlap[:10]
 # Basic-unit parent codes are checked against official town polygons and totals.
 derived_towns={k:unary_union([v['geometry'] for v in blocks.values() if v['town']==k]) for k in tkeys}
 assert all(v['town'] in towns for v in blocks.values())
 town_boundary_checks=[]
 for k in tkeys:
  dg=derived_towns[k];ag=towns[k]['geometry'];town_boundary_checks.append({'town_id':k,'basic_unit_population':sum(v['pop'] for v in blocks.values() if v['town']==k),'official_town_population':towns[k]['pop'],'symmetric_difference_m2':dg.symmetric_difference(ag).area})
 writecsv(REPORT/'town_boundary_checks.csv',town_boundary_checks)
 assert all(r['basic_unit_population']==r['official_town_population'] for r in town_boundary_checks)
 cells=readcsv(PREV/'cell_index.csv');nc=len(cells)
 for c in cells:
  for k in ['Cell Index','resolution','population']:c[k]=int(c[k])
  c['boundary']=c['boundary']=='True'
 def mesh(code):
  lat=int(code[:2])*2/3+int(code[4])/12+int(code[6])/120;lon=int(code[2:4])+100+int(code[5])/8+int(code[7])/80;dy,dx=1/120,1/80
  for q in code[8:]:
   dy/=2;dx/=2;v=int(q)-1;lat+=(v//2)*dy;lon+=(v%2)*dx
  return transform(xy,box(lon,lat,lon+dx,lat+dy))
 cg=[mesh(c['code']) for c in cells]
 core=unary_union([cg[i] for i,c in enumerate(cells) if c['resolution']==250 and not c['boundary']])
 caps=np.load(PREV/'physical_capacity_intervals.npz');support=np.load(PREV/'mesh_building_support.npz')
 ci=support['cell_index'];bi=support['building_index'];area=support['intersection_area'];K=caps['known_residential'];L=caps['lower'];U=caps['upper'];C=caps['centre']
 v=json.loads((ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson').read_text());bg=[shape(f['geometry']) for f in v['features'] if f['properties']['class']=='building'];del v
 assert len(bg)==L.shape[1]
 # Each saved candidate fragment is intersected with disjoint official basic units.
 fragments=[];partgroups=[];gapfractions=[];max_area_error=0.;max_fraction_error=0.
 for n,(cc,bb,aa) in enumerate(zip(ci,bi,area)):
  part=cg[int(cc)].intersection(bg[int(bb)]);max_area_error=max(max_area_error,abs(part.area-aa))
  ids=btree.query(part,predicate='intersects');row={}
  for j in ids:
   a=part.intersection(geom[int(j)]).area
   if a>1e-8:row[int(j)]=a/aa
  s=sum(row.values());max_fraction_error=max(max_fraction_error,s-1)
  assert s<=1+1e-7,(n,s)
  if s>1:row={k:v/s for k,v in row.items()};s=1.
  gapfractions.append(max(0.,1-s));partgroups.append(row)
  if n%15000==0:print('FRAGMENT SUPPORT',n,'/',len(ci),flush=True)
 assert max_area_error<1e-5
 group_parts=collections.defaultdict(list)
 for n,row in enumerate(partgroups):
  for j,f in row.items():group_parts[j].append((n,f))
 np.savez_compressed(OUT/'fragment_block_support.npz',part_index=np.array([n for j in sorted(group_parts) for n,f in group_parts[j]],np.int32),block_index=np.array([j for j in sorted(group_parts) for n,f in group_parts[j]],np.int32),fraction=np.array([f for j in sorted(group_parts) for n,f in group_parts[j]]),gap_fraction=np.array(gapfractions),block_ids=np.array(bkeys))
 groups=[np.flatnonzero(ci==i) for i in range(nc)]
 def extrema(p,l,u,a):
  # Any optimum has U below a threshold and L above (min), reversed for max.
  ix=np.argsort(a);aa=a[ix];ll=l[ix];uu=u[ix];d=uu-ll
  den_min=ll.sum()+np.r_[0.,np.cumsum(d)];num_min=(aa*ll).sum()+np.r_[0.,np.cumsum(aa*d)]
  den_max=uu.sum()-np.r_[0.,np.cumsum(d)];num_max=(aa*uu).sum()-np.r_[0.,np.cumsum(aa*d)]
  vmin=num_min[den_min>1e-10]/den_min[den_min>1e-10];vmax=num_max[den_max>1e-10]/den_max[den_max>1e-10]
  lo=float(vmin.min()) if len(vmin) else 0.;hi=float(vmax.max()) if len(vmax) else 0.
  return p*max(0.,lo),p*min(1.,hi)
 # Small exact vertex checks include zero lower weights and fractional polygons.
 rng=np.random.default_rng(20261016);verr=0.
 for n in range(1,9):
  for trial in range(12):
   l=rng.uniform(0,2,n);l[rng.random(n)<.4]=0;u=l+rng.uniform(.1,3,n);a=rng.uniform(0,1,n);lo,hi=extrema(100,l,u,a);vv=[]
   for bits in itertools.product([0,1],repeat=n):
    w=np.where(bits,u,l)
    if w.sum()>1e-10:vv.append(100*np.dot(a,w)/w.sum())
   verr=max(verr,abs(lo-min(vv)),abs(hi-max(vv)))
 assert verr<1e-7
 allrows=[];accounting=[];samples=[];block_to_town=np.array([tkeys.index(blocks[k]['town']) for k in bkeys]);levels=[('block',bkeys,blocks,geom),('town',tkeys,towns,[derived_towns[k] for k in tkeys])]
 output_arrays={};residual_rows=[]
 for level,keys,meta,gg in levels:
  ng=len(keys);outputs=np.zeros((2,2,ng,6)) # total point, known point, candidate-reserve point, lower, upper, census-only potential
  for ri,res in enumerate([125,250]):
   for qi,q in enumerate(['q95','q99']):
    arr=outputs[ri,qi];gap=0.;nocap=0.;boundary=0.;inputpop=0.;prior_known=0.
    for cc,c in enumerate(cells):
     if c['resolution']!=res:continue
     p=c['population'];ix=groups[cc]
     if c['boundary']:boundary+=p;continue
     inputpop+=p
     if not p:continue
     bb=bi[ix];aa=area[ix];l=aa*L[qi,bb];u=aa*U[qi,bb];w=aa*C[qi,bb]
     if not len(ix) or u.sum()<=0:
      nocap+=p
      js=btree.query(cg[cc],predicate='intersects');targets={int(j) if level=='block' else int(block_to_town[j]) for j in js if cg[cc].intersection(geom[int(j)]).area>.01}
      for j in targets:arr[j,4]+=p;arr[j,5]+=p
      if qi==0 and level=='block':residual_rows.append({'resolution':res,'mesh_code':c['code'],'unlocated_population':p,'possible_basic_units':'|'.join(keys[j] for j in sorted(targets)),'area_prorating':False})
      continue
     point=p*w/w.sum();prior_known+=float(point[K[bb]].sum());gap+=float(np.dot(point,np.array(gapfractions)[ix]))
     targets={}
     for jj,part in enumerate(ix):
      for bj,f in partgroups[part].items():
       j=bj if level=='block' else int(block_to_town[bj])
       if j not in targets:targets[j]=np.zeros(len(ix))
       targets[j][jj]+=f
     for j,a in targets.items():
      lo,hi=extrema(p,l,u,a);known=float(np.dot(point[K[bb]],a[K[bb]]));reserve=float(np.dot(point[~K[bb]],a[~K[bb]]));pt=known+reserve
      assert lo-1e-5<=pt<=hi+1e-5,(cc,j,lo,pt,hi)
      arr[j,0]+=pt;arr[j,1]+=known;arr[j,2]+=reserve;arr[j,3]+=lo;arr[j,4]+=hi
      if level=='block' and len(samples)<40 and 1<len(ix)<=10 and len(set(a))>1:samples.append((p,l.copy(),u.copy(),a.copy(),lo,hi))
    residual=inputpop-arr[:,0].sum()-gap-nocap
    assert abs(residual)<1e-6,(level,res,q,residual)
    accounting.append({'level':level,'resolution':res,'capacity_ceiling':q,'interior_population':inputpop,'block_or_town_candidate_scenario_population':float(arr[:,0].sum()),'known_residential_scenario_population':float(arr[:,1].sum()),'candidate_pool_conditionally_localized_at_aggregate_scale':float(arr[:,2].sum()),'geometry_gap_pool':gap,'census_only_unlocated_pool':nocap,'boundary_held_population':boundary,'conservation_residual':residual,'frozen_known_residential_total':prior_known})
    print('AGGREGATED',level,res,q,flush=True)
  output_arrays[level]=outputs
  coverage=np.array([1.-g.difference(core).area/g.area if g.area else 0. for g in gg]);complete=np.array([g.difference(core).area<=max(1.,1e-6*g.area) for g in gg])
  for j,key in enumerate(keys):
   a=outputs[:,:,j];points=a[:,:,0];mean=float(points.mean());spread=float(np.ptp(points));lower=float(a[:,:,3].min());upper=float(a[:,:,4].max());width=upper-lower;potential=float(a[:,:,5].max());ref=meta[key]['pop'];rdiff=abs(mean-ref)
   if not complete[j]:status='withhold_partial_coverage'
   elif max(upper,ref)<=1e-6:status='zero_resident_reference'
   elif spread<=max(10,.10*mean) and width<=max(25,.25*mean) and rdiff<=max(25,.25*ref) and potential<=max(5,.05*mean):status='stable_conditional'
   elif spread<=max(25,.25*mean) and width<=max(50,mean) and rdiff<=max(50,.5*ref) and potential<=max(10,.25*mean):status='interval_only'
   else:status='withhold_spatial_allocation'
   flags=[]
   if not complete[j]:flags.append('incomplete_original_common_support')
   if spread>max(10,.1*mean):flags.append('resolution_or_capacity_sensitivity')
   if width>max(25,.25*mean):flags.append('wide_conditional_envelope')
   if rdiff>max(25,.25*ref):flags.append('same_census_block_total_disagreement')
   if potential>max(5,.05*mean):flags.append('census_only_unlocated_pool')
   row={'level':level,'unit_id':key,'ward':meta[key]['ward'],'name':meta[key]['name'],'town_id':meta[key]['town'],'area_m2':gg[j].area,'common_support_area_fraction':coverage[j],'complete_original_support':bool(complete[j]),'official_2020_residents':ref,'point_mean':mean,'point_min':float(points.min()),'point_max':float(points.max()),'point_spread':spread,'point_relative_spread':spread/mean if mean>0 else None,'envelope_lower':lower,'envelope_upper':upper,'envelope_width':width,'envelope_relative_width':width/mean if mean>0 else None,'census_reference_absolute_difference':rdiff,'census_reference_relative_difference':rdiff/ref if ref else None,'census_only_unlocated_potential':potential,'allocation_input_class':status,'quality_flags':'|'.join(flags),'official_count_role':'2020 aggregate baseline; no within-unit occupancy validation'}
   for ri,res in enumerate([125,250]):
    for qi,q in enumerate(['q95','q99']):
     for col,idx in [('point',0),('known_residential',1),('candidate_reserve',2),('lower',3),('upper',4)]:row[f'{res}m_{q}_{col}']=float(a[ri,qi,idx])
   allrows.append(row)
  np.savez_compressed(OUT/f'{level}_population_intervals.npz',values=outputs,unit_ids=np.array(keys),resolutions=np.array([125,250]),ceilings=np.array(['q95','q99']),columns=np.array(['point','known_residential','candidate_reserve','lower','upper','census_only_potential']))
  rows=[r for r in allrows if r['level']==level];writecsv(OUT/f'{level}_population_stability.csv',rows)
  features=[{'type':'Feature','properties':r,'geometry':mapping(gg[j])} for j,r in enumerate(rows)]
  dump(OUT/f'{level}_population_stability_6677.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':features})
 writecsv(REPORT/'population_accounting.csv',accounting)
 if residual_rows:writecsv(OUT/'census_only_unlocated_pools.csv',residual_rows)
 # Recheck actual group extrema against full vertex enumeration on small cells.
 actual_vertex_error=0.
 for p,l,u,a,lo,hi in samples:
  vv=[]
  for bits in itertools.product([0,1],repeat=len(l)):
   w=np.where(bits,u,l)
   if w.sum()>1e-10:vv.append(p*np.dot(a,w)/w.sum())
  actual_vertex_error=max(actual_vertex_error,abs(lo-min(vv)),abs(hi-max(vv)))
 checks={'frozen_delivery_unchanged':all(sha(ROOT/r['file'])==r['sha256'] for r in frozen_items),'basic_units_do_not_overlap':not overlap,'town_and_basic_unit_census_totals_match':all(r['basic_unit_population']==r['official_town_population'] for r in town_boundary_checks),'candidate_fragment_area_matches_frozen_support':max_area_error<1e-5,'block_fractions_do_not_double_count':max_fraction_error<1e-7,'all_four_point_budgets_conserve':all(abs(a['conservation_residual'])<1e-6 for a in accounting),'all_group_points_inside_intervals':all(r['envelope_lower']-1e-5<=r['point_min']<=r['point_max']<=r['envelope_upper']+1e-5 for r in allrows),'synthetic_extrema_match_vertex_enumeration':verr<1e-7,'actual_extrema_match_vertex_enumeration':actual_vertex_error<1e-7,'both_resolutions_same_original_population':all(a['interior_population']==419921 and a['boundary_held_population']==127263 for a in accounting),'no_incomplete_block_admitted':all(r['complete_original_support'] for r in allrows if r['allocation_input_class'] in ['stable_conditional','interval_only']),'no_affected_population_or_training':True}
 for q in ['q95','q99']:
  old=readcsv(PREV/(q+'_mesh_population_intervals.csv'))
  for res in [125,250]:
   expected=sum(float(r['Reserve Scenario Located Residents']) for r in old if int(r['Resolution'])==res and r['Boundary Hold']=='False')
   actual=next(a['frozen_known_residential_total']+0. for a in accounting if a['level']=='block' and a['resolution']==res and a['capacity_ceiling']==q)
   checks[f'frozen_known_population_matches_{res}_{q}']=abs(actual-expected)<1e-6
   mapped=next(a['known_residential_scenario_population'] for a in accounting if a['level']=='block' and a['resolution']==res and a['capacity_ceiling']==q)
   gap=next(a['geometry_gap_pool'] for a in accounting if a['level']=='block' and a['resolution']==res and a['capacity_ceiling']==q)
   checks[f'known_geometry_gap_explicit_{res}_{q}']=-1e-6<=expected-mapped<=gap+1e-6
 checks={k:bool(v) for k,v in checks.items()}
 assert all(checks.values()),checks
 dump(REPORT/'validation.json',{'checks':checks,'max_fragment_area_error_m2':max_area_error,'max_fraction_excess':max_fraction_error,'synthetic_vertex_cases':96,'max_synthetic_vertex_error':verr,'actual_vertex_cases':len(samples),'max_actual_vertex_error':actual_vertex_error})
 dump(REPORT/'aggregation_summary.json',{'basic_units':len(blocks),'towns':len(towns),'official_full_ward_population':sum(v['pop'] for v in blocks.values()),'common_interior_population':419921,'boundary_held_population':127263,'known_residential_central_values_unchanged':[{k:a[k] for k in ['resolution','capacity_ceiling','known_residential_scenario_population']} for a in accounting if a['level']=='block'],'max_town_boundary_symmetric_difference_m2':max(r['symmetric_difference_m2'] for r in town_boundary_checks)})
 print('DONE',json.dumps(checks),flush=True)
