"""Diagnose whole-cell withholding and produce tiered, assumption-bounded allocations.
All intervals are conditional capacity envelopes, not confidence intervals for true people.
No machine-learning training, hydraulic runs, age allocation or affected-person calculation.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/udt_tiered_population_mpl')
import csv,json,math,hashlib,collections,itertools,time
import numpy as np
from shapely import STRtree
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'data/processed/three_ward_population_mesh_audit_v1';PREV=ROOT/'data/processed/three_ward_population_allocation_trial_v1'
OUT=ROOT/'data/processed/three_ward_population_tiered_intervals_v1';REPORT=ROOT/'data/exp/data-preprocessing/three_ward_population_tiered_intervals_v1'
OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def writecsv(p,rows,keys=None):
 keys=keys or list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def validheight(s):
 try:h=float(s);return h if math.isfinite(h) and 0<h<9999 else None
 except (TypeError,ValueError):return None
prior_manifest=ROOT/'data/exp/data-preprocessing/three_ward_population_allocation_trial_v1/delivery_manifest.json';prior=json.loads(prior_manifest.read_text());assert all(sha(ROOT/r['file'])==r['sha256'] for r in prior['products']+prior['scripts']);ana=sha(ROOT/'docs/AnaSOP.md')
attrs=readcsv(AUDIT/'building_allocation_readiness.csv');byid={r['gml_id']:r for r in attrs}
# Held-out observed floor values check only the proxy's internal applicability, not unknown buildings.
observed=[r for r in attrs if r['intersects_three_wards']=='True' and r['use_group'] in ['residential','mixed_residential'] and r['floors_valid']=='True']
def is_holdout(r):return int(hashlib.sha256(r['buildingID'].encode()).hexdigest()[:8],16)%5==0
reference=[r for r in observed if not is_holdout(r)];holdout=[r for r in observed if is_holdout(r)]
profiles={}
for city in ['13101','13102','13103','all']:
 for family in ['residential','mixed_residential','all_residential']:
  rr=[r for r in reference if (city=='all' or r['city']==city) and (family=='all_residential' or r['use_group']==family)]
  if not rr:continue
  f=np.array([float(r['storeysAboveGround']) for r in rr]);rh=[float(r['storeysAboveGround'])/validheight(r['measuredHeight']) for r in rr if validheight(r['measuredHeight']) is not None]
  profiles[(city,family)]={'city':city,'family':family,'floor_n':len(f),'height_ratio_n':len(rh),'floor_median':float(np.median(f)),'floor_q95':float(np.quantile(f,.95)),'floor_q99':float(np.quantile(f,.99)),'ratio_median':float(np.median(rh)) if rh else None,'ratio_q95':float(np.quantile(rh,.95)) if rh else None,'ratio_q99':float(np.quantile(rh,.99)) if rh else None,'reference_height_min':min(validheight(r['measuredHeight']) for r in rr if validheight(r['measuredHeight']) is not None) if rh else None,'reference_height_max':max(validheight(r['measuredHeight']) for r in rr if validheight(r['measuredHeight']) is not None) if rh else None}
def profile(r):
 fam=r['use_group'] if r['use_group'] in ['residential','mixed_residential'] else 'all_residential'
 for key in [(r['city'],fam),(r['city'],'all_residential'),('all',fam),('all','all_residential')]:
  if key in profiles and profiles[key]['floor_n']>=100:return profiles[key]
 raise ValueError('No adequate floor reference stratum')
def inferred_floor(r,q):
 p=profile(r);h=validheight(r['measuredHeight']);useheight=h is not None and p['height_ratio_n']>=100
 median=max(1.,h*p['ratio_median'] if useheight else p['floor_median']);upper=max(1.,math.ceil(h*p['ratio_'+q] if useheight else p['floor_'+q]));median=min(median,upper)
 return median,float(upper),'height_ratio' if useheight else 'ward_use_floor_distribution',p,useheight and not(p['reference_height_min']<=h<=p['reference_height_max'])
writecsv(OUT/'floor_reference_profiles.csv',list(profiles.values()))
calrows=[]
for q in ['q95','q99']:
 for city in ['13101','13102','13103','all']:
  rr=[r for r in holdout if city=='all' or r['city']==city];checks=[]
  for r in rr:
   med,hi,method,p,extra=inferred_floor(r,q);f=float(r['storeysAboveGround']);checks.append((f<=hi+1e-9,abs(med-f),hi-1,method,extra))
  calrows.append({'Ceiling Scenario':q,'Ward':city,'Holdout Buildings':len(rr),'Observed Floor Upper Coverage':sum(x[0] for x in checks)/len(rr) if rr else None,'Median Proxy Absolute Floor Error':float(np.median([x[1] for x in checks])) if rr else None,'Mean Floor Range Width':float(np.mean([x[2] for x in checks])) if rr else None,'Height Proxy Count':sum(x[3]=='height_ratio' for x in checks),'Extrapolated Height Count':sum(x[4] for x in checks),'Not External Validation':True})
writecsv(REPORT/'heldout_floor_proxy_checks.csv',calrows)
xy=Transformer.from_crs(6668,6677,always_xy=True).transform
wj=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text());wards={f['properties']['ward'][:5]:transform(xy,shape(f['geometry'])) for f in wj['features']};union=unary_union(list(wards.values()))
vectors=ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson';vj=json.loads(vectors.read_text());bf=[f for f in vj['features'] if f['properties']['class']=='building'];del vj
bg=[shape(f['geometry']) for f in bf];br=[byid[f['properties']['gml_id']] for f in bf];tree=STRtree(bg);nb=len(br);qs=['q95','q99']
L=np.zeros((2,nb));U=np.zeros_like(L);C=np.zeros_like(L);K=np.zeros(nb,bool);tiers=[];capacityrows=[]
for i,r in enumerate(br):
 institutional=r['detailedUsage'] in ['1131','1132'];known=r['use_group'] in ['residential','mixed_residential'] and not institutional;unlocated=r['use_group']=='unknown' or institutional;valid=r['floors_valid']=='True';K[i]=known
 tier=('observed_floor_'+r['use_group'] if valid else 'inferred_floor_'+r['use_group']) if known else 'unlocated_institutional' if institutional else 'unlocated_unknown_use' if unlocated else 'nonresidential_excluded'
 tiers.append(tier)
 for qi,q in enumerate(qs):
  med=hi=0.;method='not_used';prof=None;extrap=False
  if known or unlocated:
   if valid:med=hi=float(r['storeysAboveGround']);method='observed_floor'
   else:med,hi,method,prof,extrap=inferred_floor(r,q)
   U[qi,i]=hi
   if known:
    pure=r['use_group']=='residential';L[qi,i]=(med if valid else 1.) if pure else 0.;C[qi,i]=med*(1 if pure else .5)
   else:C[qi,i]=hi # conservative unresolved-capacity reserve; no person is assigned to unknown building.
  capacityrows.append({'Ceiling Scenario':q,'Building Index':i,'Building ID':r['buildingID'],'GML ID':r['gml_id'],'Tier':tier,'Known Residential Candidate':known,'Floor Proxy Method':method,'Floor Proxy Median':med,'Floor Proxy Upper':hi,'Capacity Lower per Base Area':L[qi,i],'Capacity Centre per Base Area':C[qi,i],'Capacity Upper per Base Area':U[qi,i],'Reference Ward':prof['city'] if prof else '', 'Reference Family':prof['family'] if prof else '', 'Reference N':prof['floor_n'] if prof else '', 'Height Extrapolation':bool(extrap),'Capacity Is Observed':known and valid and r['use_group']=='residential','Interval Is Confidence Interval':False})
writecsv(OUT/'building_capacity_intervals.csv',capacityrows)
assert np.all(L<=C+1e-12) and np.all(C<=U+1e-12) and np.all(U[0]<=U[1]+1e-12) and np.allclose(C[0,K],C[1,K])
np.savez_compressed(OUT/'physical_capacity_intervals.npz',lower=L,upper=U,centre=C,known_residential=K,building_ids=np.array([r['buildingID'] for r in br]))
ledger={res:readcsv(PREV/f'mesh_population_ledger_{res}m.csv') for res in [125,250]}
def mesh(code):
 lat=int(code[:2])*2/3+int(code[4])/12+int(code[6])/120;lon=int(code[2:4])+100+int(code[5])/8+int(code[7])/80;dy,dx=1/120,1/80
 for q in code[8:]:dy/=2;dx/=2;v=int(q)-1;lat+=(v//2)*dy;lon+=(v%2)*dx
 return transform(xy,box(lon,lat,lon+dx,lat+dy))
# Long sparse geometry support is reused for both capacity ceilings.
cellmeta=[];partcell=[];partbuilding=[];partarea=[];partward=[];diag=[]
for res in [125,250]:
 for old in ledger[res]:
  ci=len(cellmeta);code=old['Mesh Code'];p=int(old['Source Population']);boundary=old['Common Parent Boundary']=='True';cellmeta.append({'resolution':res,'code':code,'parent':code[:10],'population':p,'boundary':boundary,'old_pending':float(old['Pending Residents']),'old_flags':old['Reason Flags']})
  if boundary or not p:continue
  g=mesh(code);totalarea=unknownarea=missingarea=resarea=0.;counter=collections.Counter()
  for bi in tree.query(g,predicate='intersects'):
   bi=int(bi);part=g.intersection(bg[bi]);a=part.area
   if a<=.01:continue
   r=br[bi];totalarea+=a
   if r['use_group']=='unknown':unknownarea+=a;counter['unknown_buildings']+=1
   if r['use_group'] in ['residential','mixed_residential']:
    resarea+=a
    if r['floors_valid']!='True':missingarea+=a;counter['missing_residential_floors']+=1
   if not U[:,bi].any():continue
   wa=[part.intersection(wg).area for wg in wards.values()];den=sum(wa);assert abs(den-a)<max(.1,a*1e-7)
   partcell.append(ci);partbuilding.append(bi);partarea.append(a);partward.append(np.array(wa)/den)
  diag.append({'Resolution':res,'Mesh Code':code,'Source Population':p,'All Building Base Area':totalarea,'Unknown Use Base Area':unknownarea,'Unknown Use Base Area Fraction':unknownarea/totalarea if totalarea else 0.,'Residential Base Area':resarea,'Missing Residential Floor Base Area':missingarea,'Missing Residential Floor Base Area Fraction':missingarea/resarea if resarea else 0.,**dict(counter)})
 print('GEOMETRY READY',res,flush=True)
partcell=np.array(partcell,np.int32);partbuilding=np.array(partbuilding,np.int32);partarea=np.array(partarea);partward=np.array(partward);nc=len(cellmeta)
np.savez_compressed(OUT/'mesh_building_support.npz',cell_index=partcell,building_index=partbuilding,intersection_area=partarea,ward_fraction=partward)
writecsv(OUT/'cell_index.csv',[{'Cell Index':i,**r} for i,r in enumerate(cellmeta)])
writecsv(OUT/'missing_attribute_footprint_diagnosis.csv',diag)
groups=[np.flatnonzero(partcell==i) for i in range(nc)]
# Closed-form marginal bounds over a positive-sum box of capacity weights.
def marginal_bounds(p,l,u):
 su=float(u.sum());sl=float(l.sum())
 if su<=0:return np.zeros(len(l)),np.zeros(len(l))
 dl=l+su-u;du=u+sl-l
 lo=np.divide(p*l,dl,out=np.zeros_like(l),where=dl>1e-12);hi=np.divide(p*u,du,out=np.zeros_like(u),where=du>1e-12)
 lo=np.where((dl<=1e-12)&(u>0),p,lo)
 return np.clip(lo,0,p),np.clip(hi,0,p)
def group_bounds(p,lk,uk,lr,ur):
 if uk<=0:return 0.,0.
 lo=p*lk/(lk+ur) if lk+ur>0 else p
 hi=p*uk/(uk+lr) if uk+lr>0 else p
 return max(0.,min(float(p),lo)),max(0.,min(float(p),hi))
allcell={};allbuild={};summaries=[];maxsampleviolation=0.;rng=np.random.default_rng(20261016)
random_fractions=rng.random((4,nb)) # same building parameters reused across both resolutions.
for qi,q in enumerate(qs):
 cellrows=[];brows=[]
 for ci,m in enumerate(cellmeta):
  p=m['population'];ix=groups[ci];bi=partbuilding[ix];aa=partarea[ix];known=K[bi];l=aa*L[qi,bi];u=aa*U[qi,bi];c=aa*C[qi,bi]
  if m['boundary'] or not p or not len(ix) or not u.any():
   nominal=np.zeros(len(ix));lo=np.zeros(len(ix));hi=np.zeros(len(ix));kl=ku=0.;placed=0.;reserve=float(p);loclo=0.
  else:
   nominal=p*c/c.sum();nominal[-1]=p-math.fsum(nominal[:-1]);lo,hi=marginal_bounds(p,l,u)
   kl,ku=group_bounds(p,float(l[known].sum()),float(u[known].sum()),float(l[~known].sum()),float(u[~known].sum()))
   placed=float(nominal[known].sum());reserve=float(p)-placed;loclo=float(lo[known].sum())
   for f in random_fractions:
    w=l+(u-l)*f[bi];s=w.sum()
    if s>0:
     pred=p*w/s;violation=max(float(np.max(lo-pred,initial=0)),float(np.max(pred-hi,initial=0)),kl-float(pred[known].sum()),float(pred[known].sum())-ku)
     maxsampleviolation=max(maxsampleviolation,violation)
  for j,k in enumerate(bi):
   if not known[j]:continue
   wp=nominal[j]*partward[ix[j]]
   brows.append({'Ceiling Scenario':q,'Resolution':m['resolution'],'Cell Index':ci,'Mesh Code':m['code'],'Parent 250m Code':m['parent'],'Building ID':br[k]['buildingID'],'Building Index':int(k),'Tier':tiers[k],'Reserve Scenario Estimated Residents':float(nominal[j]),'Scenario Marginal Lower':float(lo[j]),'Scenario Marginal Upper':float(hi[j]),'Interval Width':float(hi[j]-lo[j]),**{w+' Reserve Scenario Residents':float(x) for w,x in zip(wards,wp)},'Not Observed Occupancy':True})
  cellrows.append({'Ceiling Scenario':q,'Resolution':m['resolution'],'Cell Index':ci,'Mesh Code':m['code'],'Parent 250m Code':m['parent'],'Source Population':p,'Boundary Hold':m['boundary'],'Previous Strict Pending':m['old_pending'],'Reserve Scenario Located Residents':placed,'Reserve Scenario Unlocated Residents':reserve,'Known Candidate Group Lower':kl,'Known Candidate Group Upper':ku,'Unlocated Candidate Group Lower':float(p)-ku,'Unlocated Candidate Group Upper':float(p)-kl,'Group Interval Width':ku-kl,'Individual Lower Located Mass':loclo,'Remaining Location Uncertainty Pool':float(p)-loclo,'Reserve Scenario Budget Residual':p-placed-reserve,'Lower Mass Budget Residual':p-loclo-(float(p)-loclo),'Boundary Or No Capacity':m['boundary'] or not len(ix) or not u.any(),'Unrestricted Unknown Mass Bound':'[0,P] where unknown/institutional capacity exists; no informative bound without capacity assumptions' if len(ix) and any(~known) else 'conditional ordinary-housing model','Intervals Jointly Additive':False})
 allcell[q]=cellrows;allbuild[q]=brows
 writecsv(OUT/(q+'_mesh_population_intervals.csv'),cellrows);writecsv(OUT/(q+'_building_population_intervals.csv'),brows)
 for res in [125,250]:
  rr=[r for r in cellrows if r['Resolution']==res and not r['Boundary Hold']];total=sum(r['Source Population'] for r in rr)
  summaries.append({'Ceiling Scenario':q,'Resolution':res,'Common Interior Population':total,'Reserve Scenario Located Residents':sum(r['Reserve Scenario Located Residents'] for r in rr),'Reserve Scenario Unlocated Residents':sum(r['Reserve Scenario Unlocated Residents'] for r in rr),'Reserve Scenario Located Fraction':sum(r['Reserve Scenario Located Residents'] for r in rr)/total,'Known Candidate Group Lower':sum(r['Known Candidate Group Lower'] for r in rr),'Known Candidate Group Upper':sum(r['Known Candidate Group Upper'] for r in rr),'Group Interval Width':sum(r['Group Interval Width'] for r in rr),'Known Candidate Group Lower Fraction':sum(r['Known Candidate Group Lower'] for r in rr)/total,'Individual Lower Located Mass':sum(r['Individual Lower Located Mass'] for r in rr),'Remaining Location Uncertainty Pool':sum(r['Remaining Location Uncertainty Pool'] for r in rr),'Nontrivial Group Bounds Population':sum(r['Source Population'] for r in rr if r['Known Candidate Group Lower']>0),'No Known Candidate Population':sum(r['Source Population'] for r in rr if r['Known Candidate Group Upper']==0),'Uncertain Ranges Are Assumption Based':True})
 print('INTERVALS READY',q,flush=True)
writecsv(REPORT/'interval_accounting_summary.csv',summaries)
# Ablations of old binary flags explain amplification without attributing overlapping causes twice.
flags=['unknown_building_use','residential_floor_count_missing_or_placeholder','possible_institutional_residents'];ablation=[]
for res in [125,250]:
 rr=[r for r in ledger[res] if r['Common Parent Boundary']=='False' and int(r['Source Population'])>0]
 for bits in itertools.product([False,True],repeat=3):
  enabled=[f for f,b in zip(flags,bits) if b];pending=sum(int(r['Source Population']) for r in rr if 'no_positive_residential_capacity' in r['Reason Flags'] or any(f in r['Reason Flags'] for f in enabled))
  ablation.append({'Resolution':res,'Block Unknown Use':bits[0],'Block Missing Residential Floors':bits[1],'Block Institutional Candidates':bits[2],'Pending Population':pending,'Candidate Conditioned Coverage Fraction':1-pending/sum(int(r['Source Population']) for r in rr),'Not A New Admitted Population Layer':True})
writecsv(REPORT/'whole_cell_rule_ablation.csv',ablation)
diagnosis=[]
for res in [125,250]:
 dd=[d for d in diag if d['Resolution']==res];total=sum(d['Source Population'] for d in dd)
 for flag,col,ncol in [('unknown_use','Unknown Use Base Area Fraction','unknown_buildings'),('missing_residential_floors','Missing Residential Floor Base Area Fraction','missing_residential_floors')]:
  rr=[d for d in dd if d.get(ncol,0)>0];diagnosis.append({'Resolution':res,'Gap':flag,'Touched Populated Cells':len(rr),'Population In Touched Cells':sum(d['Source Population'] for d in rr),'Population Fraction Touched':sum(d['Source Population'] for d in rr)/total,'Median Gap Base Area Fraction In Touched Cells':float(np.median([d[col] for d in rr])) if rr else 0.,'Population Weighted Gap Base Area Fraction In Touched Cells':sum(d['Source Population']*d[col] for d in rr)/sum(d['Source Population'] for d in rr) if rr else 0.,'Not Population Share In Missing Buildings':True})
writecsv(REPORT/'missingness_amplification_summary.csv',diagnosis)
# Compare identical physical-capacity rules on complete candidate support at both resolutions.
comparisons=[];wardcomp=[];parentdiffs={};common={}
for q in qs:
 look={res:{r['Mesh Code']:r for r in allcell[q] if r['Resolution']==res} for res in [125,250]}
 matched=[p for p,r in look[250].items() if not r['Boundary Hold'] and r['Source Population']>0 and r['Known Candidate Group Upper']>0 and all(look[125][p+str(j)]['Source Population']==0 or look[125][p+str(j)]['Known Candidate Group Upper']>0 for j in range(1,5))];ms=set(matched);common[q]=matched
 d={res:collections.defaultdict(lambda:collections.defaultdict(float)) for res in [125,250]};bt={res:collections.defaultdict(float) for res in [125,250]};wg={res:{w:0. for w in wards} for res in [125,250]}
 for r in allbuild[q]:
  if r['Parent 250m Code'] in ms:
   res=r['Resolution'];p=r['Parent 250m Code'];bid=r['Building ID'];d[res][p][bid]+=r['Reserve Scenario Estimated Residents'];bt[res][bid]+=r['Reserve Scenario Estimated Residents']
   for w in wards:wg[res][w]+=r[w+' Reserve Scenario Residents']
 for res in [125,250]:
  for r in allcell[q]:
   if r['Resolution']==res and r['Parent 250m Code'] in ms:d[res][r['Parent 250m Code']]['UNLOCATED']+=r['Reserve Scenario Unlocated Residents']
 pd=[]
 for p in matched:
  keys=set(d[125][p])|set(d[250][p]);half=math.fsum(abs(d[125][p].get(k,0)-d[250][p].get(k,0)) for k in keys)/2
  pd.append({'Ceiling Scenario':q,'Parent 250m Code':p,'Source Population':look[250][p]['Source Population'],'Joint Half L1 Including Unlocated Pool':half,'Joint Redistribution Fraction':half/look[250][p]['Source Population'],'Unlocated 125m':d[125][p]['UNLOCATED'],'Unlocated 250m':d[250][p]['UNLOCATED']})
 pop=sum(look[250][p]['Source Population'] for p in matched);globalhalf=(sum(abs(bt[125].get(k,0)-bt[250].get(k,0)) for k in set(bt[125])|set(bt[250]))+sum(abs(d[125][p]['UNLOCATED']-d[250][p]['UNLOCATED']) for p in matched))/2
 comparisons.append({'Ceiling Scenario':q,'Matched Parents':len(matched),'Matched Population':pop,'Joint Half L1 Including Unlocated Pools':globalhalf,'Joint Redistribution Fraction':globalhalf/pop if pop else None,'Known Buildings 125m':sum(bt[125].values()),'Known Buildings 250m':sum(bt[250].values()),'Unlocated 125m':sum(d[125][p]['UNLOCATED'] for p in matched),'Unlocated 250m':sum(d[250][p]['UNLOCATED'] for p in matched),'Is Accuracy Metric':False})
 parentdiffs[q]=pd
 for w in wards:wardcomp.append({'Ceiling Scenario':q,'Ward':w,'125m Reserve Scenario Located':wg[125][w],'250m Reserve Scenario Located':wg[250][w],'Difference 125m Minus 250m':wg[125][w]-wg[250][w],'Not Full Ward Population':True})
writecsv(REPORT/'matched_resolution_comparison.csv',comparisons);writecsv(REPORT/'matched_ward_located_comparison.csv',wardcomp);writecsv(OUT/'matched_parent_differences.csv',parentdiffs['q95']+parentdiffs['q99'])
# Conditional-model limitations are part of the data contract.
protocol={'human_decision':'MILI-D-20261016-018','scope':'Same 541 interior 250 m parents and 2164 children, 419921 census residents; all 216 outer parent cells and all their children remain boundary-held. Reporting is at the three-ward scale on this explicitly partial common support.','tiers':{'observed_pure':'Base area x observed valid floors; fixed capacity within this proxy model.','observed_mixed':'Residential share in [0,1], centre 0.5; no unobserved share is treated as known.','missing_floor_known_residential':'Above-ground floors in [1, upper scenario]; reference median supplies the nominal floor. Valid positive measured height uses empirical floor/height ratios, otherwise ward/use floor distribution.','unknown_or_institutional':'Do not assign persons to an unknown building. Potential capacity in [0, floor upper x base area] contributes to a separate unresolved mesh pool; nominal reserve uses its upper capacity.','no_candidate':'All source population remains unlocated if no known residential candidate has positive upper capacity.','outside_common_support':'Boundary population remains wholly held. No geometric population prorating.'},'stratification':'Ward x residential/mixed family, then ward all residential, then all-ward same family, then all-ward residential. At least 100 reference observations per profile; height proxy also needs 100 height/floor pairs.','reference_holdout':'Deterministic building-ID SHA256 modulo 5: 80% reference, 20% held-out observed floors. This is an internal proxy check, not ML training and not validation on the truly missing buildings; original research train/validation/test partitions are untouched.','ceiling_scenarios':'q95 and q99 are empirical capacity ceilings, with lower above-ground floor bound 1. They are NOT confidence/credible intervals. Applying a finite ceiling to missing buildings is a sensitivity assumption. Valid recorded floors remain unchanged. Positive height can be outside the reference height range and is flagged as extrapolation.','point_scenario':'Known pure shares 1; mixed shares 0.5; missing floors reference median; unknown/institutional capacity set to its upper scenario and retained in the unlocated pool. This is a conservative-reserve model scenario, not an occupancy estimate with proven accuracy.','marginal_formula':'For normalized population p*w_i/sum(w), w_i in [l_i,u_i]: lower=p*l_i/(l_i+sum_{j!=i}u_j), upper=p*u_i/(u_i+sum_{j!=i}l_j). Positive-sum degeneracies are handled explicitly. Marginal endpoints cannot be added to construct a joint map.','group_formula':'Known group [p*L_K/(L_K+U_R), p*U_K/(U_K+L_R)]. Known plus unlocated is complementary per cell. Building lower bounds plus a remaining location-uncertainty pool also conserve p.','unrestricted_identification':'Without a finite missing-capacity assumption, unknown/institutional occupants may occupy anywhere from 0 to the full mesh population. Apparent coverage gains below are conditional on the stated ceilings.','cross_mesh_dependence':'Nominal building capacity rules are identical at both resolutions and consistent across fragments. Cellwise interval sums are conservative outer envelopes; shared building parameters across cells imply endpoint maps need not be jointly attainable.','interpretation':'Population conservation and internal proxy coverage do not validate building occupancy, true residential share or 2020 compatibility of 2021/2022 building attributes. No affected/stranded population is computed.'}
dump(OUT/'protocol.json',protocol)
# Reconstructive validation on actual normalized weights.
checks={'prior_delivery_unchanged':all(sha(ROOT/r['file'])==r['sha256'] for r in prior['products']+prior['scripts']),'AnaSOP_unchanged':sha(ROOT/'docs/AnaSOP.md')==ana,'capacity_centre_inside_bounds':bool(np.all(L<=C+1e-12)&np.all(C<=U+1e-12)),'q99_capacity_upper_not_smaller':bool(np.all(U[0]<=U[1]+1e-12)),'known_central_weights_same_ceiling_scenarios':bool(np.allclose(C[0,K],C[1,K])),'every_point_budget_conserves_population':all(abs(r['Reserve Scenario Budget Residual'])<1e-7 for rr in allcell.values() for r in rr),'every_partial_lower_budget_conserves_population':all(abs(r['Lower Mass Budget Residual'])<1e-7 for rr in allcell.values() for r in rr),'point_inside_group_interval':all(r['Known Candidate Group Lower']-1e-7<=r['Reserve Scenario Located Residents']<=r['Known Candidate Group Upper']+1e-7 for rr in allcell.values() for r in rr),'bounds_nonnegative_and_ordered':all(0<=r['Known Candidate Group Lower']<=r['Known Candidate Group Upper']+1e-7<=r['Source Population']+1e-7 for rr in allcell.values() for r in rr),'q99_group_lower_not_greater':all(a['Known Candidate Group Lower']>=b['Known Candidate Group Lower']-1e-7 for a,b in zip(allcell['q95'],allcell['q99'])),'sampled_physical_building_parameters_enclosed':maxsampleviolation<1e-6,'all_boundary_population_still_unlocated':all(r['Reserve Scenario Unlocated Residents']==r['Source Population'] for rr in allcell.values() for r in rr if r['Boundary Hold']),'no_population_assigned_to_unknown_or_institutional_buildings':all(K[r['Building Index']] for rr in allbuild.values() for r in rr),'no_affected_population':True}
for q in qs:
 sums=collections.defaultdict(float);lows=collections.defaultdict(float)
 for r in allbuild[q]:sums[r['Cell Index']]+=r['Reserve Scenario Estimated Residents'];lows[r['Cell Index']]+=r['Scenario Marginal Lower']
 checks[q+'_building_rows_match_mesh_budget']=all(abs(sums[r['Cell Index']]-r['Reserve Scenario Located Residents'])<1e-6 and abs(lows[r['Cell Index']]-r['Individual Lower Located Mass'])<1e-6 for r in allcell[q])
 checks[q+'_point_inside_individual_interval']=all(r['Scenario Marginal Lower']-1e-7<=r['Reserve Scenario Estimated Residents']<=r['Scenario Marginal Upper']+1e-7 for r in allbuild[q])
dump(REPORT/'validation.json',{'passed':all(checks.values()),'checks':checks,'max_sampled_interval_violation':maxsampleviolation});assert all(checks.values()),checks
summary={'diagnosis':diagnosis,'interval_accounting':summaries,'resolution_comparison':comparisons,'heldout_proxy_checks':calrows,'protocol':protocol,'reference_building_count':len(reference),'holdout_building_count':len(holdout),'recorded_unknown_use_gap':{'intersecting_unknown_use_buildings':sum(r['intersects_three_wards']=='True' and r['use_group']=='unknown' for r in attrs),'all_unknown_use_have_placeholder_floors_and_detail':all(r['storeysAboveGround']=='9999' and r['detailedUsage']=='0' for r in attrs if r['intersects_three_wards']=='True' and r['use_group']=='unknown')},'previous_delivery_sha256':sha(prior_manifest)}
dump(REPORT/'summary.json',summary)
print('COMPLETE',json.dumps({'diagnosis':diagnosis,'intervals':summaries,'comparison':comparisons,'heldout':calrows,'checks':checks},ensure_ascii=False),flush=True)
