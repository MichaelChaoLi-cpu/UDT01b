"""Separate candidate-conditioned spatial diagnostic; never replaces strict ledger.
No building counts are promoted to validated occupancy or affected population.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/udt_population_trial_mpl')
import json,csv,math,hashlib,collections,html
import numpy as np
from shapely import STRtree
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'data/processed/three_ward_population_mesh_audit_v1';BASE=ROOT/'data/processed/three_ward_population_allocation_trial_v1';OUT=ROOT/'data/processed/three_ward_population_allocation_conditional_diagnostic_v1';REPORT=ROOT/'data/exp/data-preprocessing/three_ward_population_allocation_trial_v1';OUT.mkdir(parents=True,exist_ok=True)
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
basefreeze=json.loads((REPORT/'freeze.json').read_text());assert all(sha(ROOT/r['file'])==r['sha256'] for r in basefreeze['products']);base_script_sha=sha(ROOT/'src/preprocessing/allocate_three_ward_population_trial_v1.py')
xy=Transformer.from_crs(6668,6677,always_xy=True).transform
wj=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text());wards={f['properties']['ward'][:5]:transform(xy,shape(f['geometry'])) for f in wj['features']};union=unary_union(list(wards.values()));wardnames=list(wards)
weights={r['GML ID']:r for r in readcsv(BASE/'shared_building_weights.csv')};v=json.loads((ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson').read_text());bf=[f for f in v['features'] if f['properties']['class']=='building'];del v
gs=[shape(f['geometry']) for f in bf];br=[weights[f['properties']['gml_id']] for f in bf];tree=STRtree(gs)
def mesh(code):
 lat=int(code[:2])*2/3+int(code[4])/12+int(code[6])/120;lon=int(code[2:4])+100+int(code[5])/8+int(code[7])/80;dy,dx=1/120,1/80
 for q in code[8:]:dy/=2;dx/=2;v=int(q)-1;lat+=(v//2)*dy;lon+=(v%2)*dx
 return transform(xy,box(lon,lat,lon+dx,lat+dy))
ledger={res:readcsv(BASE/f'mesh_population_ledger_{res}m.csv') for res in [125,250]};parentlist=[r['Mesh Code'] for r in ledger[250]];pg={p:mesh(p) for p in parentlist};outputs={};newledger={};allward={};rowsummary=[]
for res in [125,250]:
 rows=[];nl=[];wg={w:0. for w in wardnames}
 for entry in ledger[res]:
  code=entry['Mesh Code'];p=int(entry['Source Population']);boundary=entry['Common Parent Boundary']=='True';pieces=[]
  if p and not boundary:
   g=pg[code] if res==250 else mesh(code)
   for i in tree.query(g,predicate='intersects'):
    i=int(i);r=br[i]
    if r['Use Group'] not in ['residential','mixed_residential'] or r['Valid Above Ground Floors']!='True':continue
    part=g.intersection(gs[i]);area=part.area
    if area<=.01:continue
    cap=float(r['Capacity per Base Area']);pieces.append((i,part,area,area*cap,cap))
  has=bool(pieces);pending=0. if has else float(p);condsum=0.
  if has:
   total=math.fsum(x[3] for x in pieces);amounts=[p*x[3]/total for x in pieces];amounts[-1]=p-math.fsum(amounts[:-1]);condsum=math.fsum(amounts)
   for (i,part,area,weight,cap),amount in zip(pieces,amounts):
    wa=[part.intersection(wards[w]).area for w in wardnames];den=math.fsum(wa);assert abs(den-area)<max(.1,area*1e-7);wp=[amount*a/den for a in wa];wp[-1]=amount-math.fsum(wp[:-1])
    for w,a in zip(wardnames,wp):wg[w]+=a
    rows.append({'Resolution':res,'Mesh Code':code,'Parent 250m Code':code[:10],'Building ID':br[i]['Building ID'],'GML ID':br[i]['GML ID'],'Candidate Conditioned Residents':amount,'Shared Capacity per Base Area':cap,'Weight':weight,'Strict Ledger Class':entry['Allocation Class'],'Diagnostic Only':True,'Admissible As Reliable Population':False,**{w+' Candidate Conditioned Residents':a for w,a in zip(wardnames,wp)}})
  ambiguous=entry['Allocation Class']=='pending' and not boundary and p>0
  nl.append({'Resolution':res,'Mesh Code':code,'Parent 250m Code':code[:10],'Source Population':p,'Common Parent Boundary':boundary,'Candidate Conditioned Residents':condsum,'Diagnostic Unplaced Residents':pending,'Budget Residual':p-condsum-pending,'Strict Pending Residents':float(entry['Pending Residents']),'Strict Reason Flags':entry['Reason Flags'],'Excluded Capacity Occupants Lower Bound':0 if ambiguous else '', 'Excluded Capacity Occupants Upper Bound':p if ambiguous else '', 'Excluded Capacity Bounds Are Informative':False if ambiguous else '', 'Diagnostic Only':True})
 outputs[res]=rows;newledger[res]=nl;allward[res]=wg
 writecsv(OUT/f'candidate_conditioned_buildings_{res}m.csv',rows,['Resolution','Mesh Code','Parent 250m Code','Building ID','GML ID','Candidate Conditioned Residents','Shared Capacity per Base Area','Weight','Strict Ledger Class','Diagnostic Only','Admissible As Reliable Population',*[w+' Candidate Conditioned Residents' for w in wardnames]])
 writecsv(OUT/f'candidate_conditioned_mesh_ledger_{res}m.csv',nl)
 interior=[r for r in nl if not r['Common Parent Boundary']]
 rowsummary.append({'Resolution':res,'Common Interior Source Population':sum(r['Source Population'] for r in interior),'Candidate Conditioned Residents':sum(r['Candidate Conditioned Residents'] for r in interior),'Diagnostic Unplaced Residents':sum(r['Diagnostic Unplaced Residents'] for r in interior),'Candidate Coverage Fraction':sum(r['Candidate Conditioned Residents'] for r in interior)/sum(r['Source Population'] for r in interior),'Still Strict Pending Residents':sum(r['Strict Pending Residents'] for r in interior),'Diagnostic Building Rows':len(rows)})
 print('DIAGNOSTIC',res,rowsummary[-1],flush=True)
lookup={res:{r['Mesh Code']:r for r in newledger[res]} for res in [125,250]}
matched=[p for p in parentlist if not lookup[250][p]['Common Parent Boundary'] and lookup[250][p]['Source Population']>0 and lookup[250][p]['Diagnostic Unplaced Residents']==0 and all(lookup[125][p+str(q)]['Diagnostic Unplaced Residents']==0 for q in range(1,5))];matchedset=set(matched)
byparent={res:collections.defaultdict(lambda:collections.defaultdict(float)) for res in [125,250]};btotal={res:collections.defaultdict(float) for res in [125,250]};wardmatched={res:{w:0. for w in wardnames} for res in [125,250]}
for res in [125,250]:
 for r in outputs[res]:
  if r['Parent 250m Code'] in matchedset:
   byparent[res][r['Parent 250m Code']][r['Building ID']]+=r['Candidate Conditioned Residents'];btotal[res][r['Building ID']]+=r['Candidate Conditioned Residents']
   for w in wardnames:wardmatched[res][w]+=r[w+' Candidate Conditioned Residents']
pd=[]
for p in matched:
 keys=set(byparent[125][p])|set(byparent[250][p]);half=math.fsum(abs(byparent[125][p].get(k,0)-byparent[250][p].get(k,0)) for k in keys)/2
 pd.append({'Parent 250m Code':p,'Source Population':lookup[250][p]['Source Population'],'Half L1 Redistribution Mass':half,'Redistribution Fraction':half/lookup[250][p]['Source Population'],'Diagnostic Only':True})
bd=[{'Building ID':k,'125m Candidate Conditioned Residents':btotal[125].get(k,0),'250m Candidate Conditioned Residents':btotal[250].get(k,0),'Difference 125m Minus 250m':btotal[125].get(k,0)-btotal[250].get(k,0)} for k in sorted(set(btotal[125])|set(btotal[250]))]
writecsv(OUT/'matched_candidate_building_comparison.csv',bd);writecsv(OUT/'matched_candidate_parent_comparison.csv',pd)
wardrows=[{'Ward':w,'125m Candidate Conditioned Residents':wardmatched[125][w],'250m Candidate Conditioned Residents':wardmatched[250][w],'Difference 125m Minus 250m':wardmatched[125][w]-wardmatched[250][w],'Matched Support Only':True,'Diagnostic Only':True} for w in wardnames];writecsv(REPORT/'candidate_conditioned_ward_comparison.csv',wardrows)
mpop=sum(lookup[250][p]['Source Population'] for p in matched);half=math.fsum(abs(r['Difference 125m Minus 250m']) for r in bd)/2
summary={'purpose':'Candidate-conditioned spatial diagnostic, separate from the conservative population admission ledger','assumption':'All population in a diagnostic cell is conditionally located among known residential or mixed-residential candidates with valid floors. Unknown-use and invalid-floor buildings are omitted from this diagnostic capacity only. Their true occupants remain unidentified, with only trivial [0, mesh population] bounds; they are not measured as zero.','shared_weights':'Exactly the same base-intersection area times valid floor count times residential share (pure 1, mixed assumed 0.5) as the strict trial.','not_promoted':'The diagnostic does not release, replace or subtract from the strict pending population, and is not an admissible affected-person inventory. Diagnostic values and strict pending populations are alternative representations and must not be added.','coverage':rowsummary,'matched_support':{'parent_count':len(matched),'source_population':mpop,'building_count':len(bd),'half_L1_building_population_redistribution':half,'fraction_of_matched_population':half/mpop if mpop else None,'is_accuracy_metric':False},'ward_comparison':wardrows,'scientific_limit':'This broader comparison shows spatial sensitivity conditional on known candidates. It cannot establish that unknown buildings are unoccupied, that mixed share 0.5 is correct, or that either resolution is more accurate.'}
dump(REPORT/'candidate_conditioned_summary.json',summary);dump(OUT/'diagnostic_protocol.json',summary)
# City-wide diagnostic map; no building-by-building inspection.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained');polys=[np.array(pg[p].exterior.coords) for p in parentlist];div={r['Parent 250m Code']:r['Redistribution Fraction'] for r in pd}
fractions={125:{p:sum(lookup[125][p+str(q)]['Candidate Conditioned Residents'] for q in range(1,5))/lookup[250][p]['Source Population'] if lookup[250][p]['Source Population'] else np.nan for p in parentlist},250:{p:lookup[250][p]['Candidate Conditioned Residents']/lookup[250][p]['Source Population'] if lookup[250][p]['Source Population'] else np.nan for p in parentlist}}
for ax,vals,title in zip(axes,[fractions[125],fractions[250],div],['125 m candidate coverage','250 m candidate coverage','Matched redistribution fraction']):
 data=[vals.get(p,np.nan) if not lookup[250][p]['Common Parent Boundary'] else np.nan for p in parentlist];pc=PolyCollection(polys,array=np.ma.masked_invalid(data),cmap='viridis',norm=Normalize(0,1),edgecolors='none');pc.cmap.set_bad('#dedede');ax.add_collection(pc);fig.colorbar(pc,ax=ax,shrink=.68,label='Fraction of parent population')
 for wg in wards.values():
  for po in ([wg] if wg.geom_type=='Polygon' else wg.geoms):x,y=po.exterior.xy;ax.plot(x,y,color='#273747',lw=.65)
 ax.set_xlim(union.bounds[0]-100,union.bounds[2]+100);ax.set_ylim(union.bounds[1]-100,union.bounds[3]+100);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('JGD2011 IX easting (m)');ax.set_ylabel('Northing (m)')
fig.suptitle('Candidate-conditioned diagnostic only | Does not release strict pending population\nSame building weights; mixed residential share assumed 0.5. Gray: boundary, zero population or unmatched support.',fontsize=11);fig.savefig(REPORT/'candidate_conditioned_overview.png',dpi=170);plt.close(fig)
# An explicit supplement captures additional definitions without rewriting the frozen strict report.
metadata={'five_year_age_tables':{'125m':'T001233','250m':'T001196','fields':'0-4,5-9,...,90-94,95+ by sex; mean and median age','status':'Official definitions downloaded and reviewed; full five-year age data not downloaded or allocated.'},'detailed_resident_commuting_tables':{'125m':'T001236','250m':'T001199','fields':'Locally resident workers/students aged 15+; work at home; work/school in home municipality, other wards/municipalities within prefecture, or another prefecture; resident travel modes and education.','status':'Official definitions downloaded and reviewed; detailed numeric archives not downloaded.','spatial_basis':'Residence, with broad destination categories. Not a destination-mesh population or OD matrix.','potential_use':'Later characterize children/older residents and resident dependence on out-of-area employment, schools and rail; preserve confidentiality-group support.','limits':'No particular destination municipality/station pair, route, Metro-versus-JR split or dynamic daytime presence. Broad destination totals include unknown destination categories in their totals.'},'age_local_quality':'HTKSYORI 1 receiving cells and 2 suppressed cells are not accepted as independent local age distributions. No suppressed value is filled with zero and no sum of overlapping senior age groups is formed.'}
dump(REPORT/'additional_age_commuting_definitions.json',metadata)
def table(rows,keys):return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:,.3f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
strict=json.loads((REPORT/'summary.json').read_text());strictrows=[r for r in strict['accounting'] if r['Scope']=='common_interior_support']
body='<h1>Three-ward population allocation comparison</h1><p><b>Completed:</b> same-weight 125 m and 250 m allocation trials, separate uncertainty accounting, per-cell population conservation, and age/commuting definition audit. <b>No affected or stranded people are calculated.</b></p><p>Both resolutions use 541 complete interior 250 m parents (419,921 census residents) as common support. This is a comparison subset, not the full three-ward population. An additional 127,263 people in intersecting outer parent cells are held separately; that envelope includes outside-ward residents.</p><h2>Authoritative conservative ledger</h2><p>Base-intersection area x valid above-ground floors x residential share. Pure residential share is 1; mixed residential share 0.5 is a stated trial assumption. Every mixed-use cell allocation is conditional. Cells with unknown building uses, invalid residential floors or possible medical/welfare institutional residents remain wholly pending; no residential candidate also means pending. No height-based floors, median floors or unknown-use population is invented.</p>'
body+=table(strictrows,['Resolution','Source Population','Ordinary Estimated Residents','Conditional Estimated Residents','Pending Residents'])
body+='<p>The strict complete-case comparison contains only two parent cells, with 1,036 residents. It cannot establish three-ward stability. Whole-cell withholding propagates incomplete building attributes more strongly at 250 m; the coverage gap is not proof that 125 m population is more accurate.</p><p><a href="review.html">Preserved strict report and ledgers</a></p><h2>Separate candidate-conditioned diagnostic</h2><p>This diagnostic places population among identified, valid-floor residential candidates <b>under that condition only</b>. Unknown and missing-floor buildings are not assumed unoccupied; their occupants remain unidentified. All original strict pending populations remain pending in the main ledger. These two representations must not be added.</p><img src="candidate_conditioned_overview.png" alt="Three-ward candidate-conditioned diagnostic">'
body+=table(rowsummary,['Resolution','Common Interior Source Population','Candidate Conditioned Residents','Diagnostic Unplaced Residents','Candidate Coverage Fraction'])
body+='<h3>Matched spatial comparison</h3><pre>'+html.escape(json.dumps(summary['matched_support'],indent=2))+'</pre>'+table(wardrows,['Ward','125m Candidate Conditioned Residents','250m Candidate Conditioned Residents','Difference 125m Minus 250m'])
body+='<p>Half-L1 is half the sum of absolute changes in modeled building counts on matched support. It describes redistribution of estimated population, not observed people moving, forecast error, accuracy improvement or flood consequences. Exact totals on shared 250 m parents are imposed by the census constraints and are not an independent validation.</p><h2>Age, work and school data</h2><p>The existing population tables contain 0-14,15-64,65+ and nested senior groups. Separately, T001233 (125 m) and T001196 (250 m) provide five-year ages, sex, mean and median age. Their definitions have been reviewed; detailed age values are not yet downloaded or allocated.</p><p>The downloaded T001232/T001145 numerical tables describe resident employment/schooling and resident rail-use traits. The more detailed T001236/T001199 definitions add working at home, work/school within the home municipality, elsewhere in the prefecture and in another prefecture. They are still <b>resident-based broad destination categories</b>, not incoming daytime population or a mesh-to-mesh OD matrix.</p>'
body+=table(strict['age_fields'],['Resolution','Interior Populated Meshes','Cell Local Age Partition Usable Meshes','Age Flags Or Invalid Meshes'])
body+='<p>These age-quality figures concern the existing broad-age population table, not the newly reviewed five-year-age table. Confidentiality receiving/suppressed cells cannot be treated as independent local age compositions. The age or commute profile of an individual building remains unobserved.</p><p>For the transport study, these fields can later describe older/younger resident exposure, resident rail dependence and out-of-area commute dependence. Actual daytime exposure requires audited work/school destination stocks; station demand and return-home trips additionally require compatible destination/OD and time-of-day assumptions.</p><h2>Delivery</h2><p><a href="population_accounting_summary.csv">Strict budgets</a> | <a href="candidate_conditioned_summary.json">Diagnostic summary</a> | <a href="additional_age_commuting_definitions.json">Additional field definitions</a> | <a href="delivery_validation.json">Final validation</a></p><p>Official sources: <a href="https://www.e-stat.go.jp/help/data-definition-information/downloaddata/T001196.pdf">Five-year age definition</a>; <a href="https://www.e-stat.go.jp/help/data-definition-information/downloaddata/T001199.pdf">Detailed resident commuting definition</a>; <a href="https://www.stat.go.jp/data/mesh/pdf/ryuijikou.pdf">Mesh and confidentiality cautions</a>.</p>'
(REPORT/'review_complete.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Population allocation comparison</title><style>body{font:16px system-ui;max-width:1350px;margin:35px auto;padding:0 20px;color:#203044;line-height:1.6}img{width:100%}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd5de;text-align:left}th{background:#eef3f7}pre{background:#f4f6f8;padding:15px;overflow:auto}</style>'+body+'</html>')
wl={r['Building ID']:float(r['Capacity per Base Area']) for r in weights.values() if r['Capacity per Base Area']}
checks={'strict_products_unchanged':all(sha(ROOT/r['file'])==r['sha256'] for r in basefreeze['products']),'strict_script_unchanged':sha(ROOT/'src/preprocessing/allocate_three_ward_population_trial_v1.py')==base_script_sha,'strict_pending_population_not_released':all(float(old['Pending Residents'])==new['Strict Pending Residents'] for res in [125,250] for old,new in zip(ledger[res],newledger[res])),'same_physical_weight_capacity':all(r['Shared Capacity per Base Area']==wl[r['Building ID']] for rr in outputs.values() for r in rr),'all_diagnostic_counts_nonnegative':all(r['Candidate Conditioned Residents']>=0 for rr in outputs.values() for r in rr),'every_diagnostic_cell_conserves_population':all(abs(r['Budget Residual'])<1e-8 for rr in newledger.values() for r in rr),'matched_population_equal':all(abs(sum(btotal[res].values())-mpop)<1e-6 for res in [125,250]),'matched_ward_population_equal':all(abs(sum(wardmatched[res].values())-mpop)<1e-6 for res in [125,250]),'no_affected_or_stranded_persons':True,'diagnostic_does_not_claim_reliability':all(r['Admissible As Reliable Population']==False for rr in outputs.values() for r in rr)}
for res in [125,250]:
 sums=collections.defaultdict(float)
 for r in outputs[res]:sums[r['Mesh Code']]+=r['Candidate Conditioned Residents']
 checks[f'diagnostic_rows_reconcile_{res}']=all(abs(sums[r['Mesh Code']]-r['Candidate Conditioned Residents'])<1e-7 for r in newledger[res])
for p in (BASE/'official_definitions').glob('*_source.json'):
 d=json.loads(p.read_text());name=p.name.replace('_source.json','.zip' if '_5339_' in p.name else '.pdf');checks['official_sha_'+name]=sha(p.with_name(name))==d['sha256']
dump(REPORT/'delivery_validation.json',{'passed':all(checks.values()),'checks':checks,'strict_validation_preserved':json.loads((REPORT/'validation.json').read_text())});assert all(checks.values()),checks
files=[p for b in [BASE,OUT,REPORT] for p in sorted(b.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json']
dump(REPORT/'delivery_manifest.json',{'products':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files],'scripts':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [ROOT/'src/preprocessing/allocate_three_ward_population_trial_v1.py',Path(__file__)]],'preserved_strict_freeze_sha256':sha(REPORT/'freeze.json')})
print('COMPLETE',json.dumps({'diagnostic':summary,'validation':checks},ensure_ascii=False),flush=True)
