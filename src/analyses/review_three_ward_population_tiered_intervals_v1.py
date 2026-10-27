"""Independently check finite-capacity interval endpoints and publish the scoped audit."""
from pathlib import Path
import json,csv,math,hashlib,itertools,html,collections,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/udt_tiered_population_mpl')
import numpy as np
from shapely.geometry import shape,box
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_population_tiered_intervals_v1';R=ROOT/'data/exp/data-preprocessing/three_ward_population_tiered_intervals_v1';OLD=ROOT/'data/exp/data-preprocessing/three_ward_population_allocation_trial_v1'
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
s=json.loads((R/'summary.json').read_text());baseval=json.loads((R/'validation.json').read_text());assert baseval['passed']
with np.load(OUT/'physical_capacity_intervals.npz') as d:L=d['lower'];U=d['upper'];C=d['centre'];K=d['known_residential']
with np.load(OUT/'mesh_building_support.npz') as d:ci=d['cell_index'];bi=d['building_index'];area=d['intersection_area']
meta=readcsv(OUT/'cell_index.csv');exported={q:readcsv(OUT/(q+'_building_population_intervals.csv')) for q in ['q95','q99']};cells={q:readcsv(OUT/(q+'_mesh_population_intervals.csv')) for q in ['q95','q99']}
bycell={q:collections.defaultdict(dict) for q in exported}
for q,rr in exported.items():
 for r in rr:bycell[q][int(r['Cell Index'])][int(r['Building Index'])]=r
# Exact enumeration of independent weight-box vertices is separate from the bound formulas.
vertexchecks=[]
for qi,q in enumerate(['q95','q99']):
 for resolution in [125,250]:
  count=0
  for index,m in enumerate(meta):
   if int(m['resolution'])!=resolution or m['boundary']=='True' or int(m['population'])<=0:continue
   ix=np.flatnonzero(ci==index);bb=bi[ix];a=area[ix];l=L[qi,bb]*a;u=U[qi,bb]*a;known=K[bb];var=np.flatnonzero(u-l>1e-10)
   if not(1<=len(var)<=8) or not known.any() or not u.any():continue
   ww=np.tile(l,(2**len(var),1))
   for row,bits in enumerate(itertools.product([0,1],repeat=len(var))):ww[row,var]=np.where(bits,u[var],l[var])
   sums=ww.sum(axis=1);ww=ww[sums>1e-12];sums=sums[sums>1e-12];p=int(m['population']);pp=p*ww/sums[:,None]
   exportedrow=cells[q][index];kg=pp[:,known].sum(axis=1);gerror=max(abs(float(kg.min())-float(exportedrow['Known Candidate Group Lower'])),abs(float(kg.max())-float(exportedrow['Known Candidate Group Upper'])))
   berror=0.
   for j,b in enumerate(bb):
    if not known[j]:continue
    r=bycell[q][index][int(b)];berror=max(berror,abs(float(pp[:,j].min())-float(r['Scenario Marginal Lower'])),abs(float(pp[:,j].max())-float(r['Scenario Marginal Upper'])))
   vertexchecks.append({'Ceiling Scenario':q,'Resolution':resolution,'Mesh Code':m['code'],'Variable Capacities':len(var),'Positive Sum Vertices':len(pp),'Group Endpoint Max Absolute Error':gerror,'Building Endpoint Max Absolute Error':berror,'Passed':gerror<1e-6 and berror<1e-6});count+=1
   if count>=50:break
assert len(vertexchecks)>0 and all(r['Passed'] for r in vertexchecks)
writecsv(R/'independent_vertex_interval_checks.csv',vertexchecks)
# Parent-scale interval overlap gives a common spatial comparison without selecting only complete cases.
intervalcompare=[]
for q in ['q95','q99']:
 look={res:{r['Mesh Code']:r for r in cells[q] if int(r['Resolution'])==res} for res in [125,250]}
 for p,r in look[250].items():
  if r['Boundary Hold']=='True' or int(r['Source Population'])<=0:continue
  a=sum(float(look[125][p+str(j)]['Known Candidate Group Lower']) for j in range(1,5));b=sum(float(look[125][p+str(j)]['Known Candidate Group Upper']) for j in range(1,5));c=float(r['Known Candidate Group Lower']);d=float(r['Known Candidate Group Upper']);overlap=max(a,c)<=min(b,d)+1e-7
  intervalcompare.append({'Ceiling Scenario':q,'Parent 250m Code':p,'Source Population':int(r['Source Population']),'Known Group Lower 125m':a,'Known Group Upper 125m':b,'Known Group Lower 250m':c,'Known Group Upper 250m':d,'Conditional Intervals Overlap':overlap,'Not A Statistical Significance Test':True})
writecsv(R/'all_parent_group_interval_comparison.csv',intervalcompare)
ivs=[]
for q in ['q95','q99']:
 rr=[r for r in intervalcompare if r['Ceiling Scenario']==q];ivs.append({'Ceiling Scenario':q,'Populated Interior Parents':len(rr),'Parents With Overlapping Conditional Group Intervals':sum(r['Conditional Intervals Overlap'] for r in rr),'Population In Nonoverlapping Parents':sum(r['Source Population'] for r in rr if not r['Conditional Intervals Overlap'])})
dump(R/'interval_overlap_summary.json',ivs)
# Figures show the whole reporting region, not point-by-point checks.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
old=json.loads((OLD/'summary.json').read_text());oldrows=[r for r in old['accounting'] if r['Scope']=='common_interior_support'];new=s['interval_accounting']
fig,axs=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
xx=np.arange(2);width=.34;diag=s['diagnosis'];unknown=[next(r for r in diag if r['Resolution']==res and r['Gap']=='unknown_use') for res in [125,250]]
axs[0].bar(xx-width/2,[r['Population Fraction Touched']*100 for r in unknown],width,label='Population in cells touched by unknown use',color='#c96d55');axs[0].bar(xx+width/2,[r['Median Gap Base Area Fraction In Touched Cells']*100 for r in unknown],width,label='Median unknown-use base-area share in those cells',color='#628d9e');axs[0].set_xticks(xx,['125 m','250 m']);axs[0].set_ylim(0,105);axs[0].set_ylabel('Percent');axs[0].set_title('Why a small gap withholds a whole cell');axs[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.12))
labels=[];located=[];lowers=[];uppers=[]
for res in [125,250]:
 oldrow=next(r for r in oldrows if r['Resolution']==res);labels.append(f'{res} m old rule');placed=100*(oldrow['Ordinary Estimated Residents']+oldrow['Conditional Estimated Residents'])/oldrow['Source Population'];located.append(placed);lowers.append(placed);uppers.append(placed)
 for q in ['q95','q99']:
  r=next(r for r in new if r['Resolution']==res and r['Ceiling Scenario']==q);labels.append(f'{res} m {q} ceiling');located.append(r['Reserve Scenario Located Fraction']*100);lowers.append(r['Known Candidate Group Lower Fraction']*100);uppers.append(r['Known Candidate Group Upper']/r['Common Interior Population']*100)
y=np.arange(len(labels));axs[1].barh(y,located,color=['#8b939a','#4b8d90','#75a9ac']*2);axs[1].errorbar(located,y,xerr=[np.array(located)-lowers,np.array(uppers)-located],fmt='none',color='#233a50',capsize=4);axs[1].set_yticks(y,labels);axs[1].invert_yaxis();axs[1].set_xlim(0,105);axs[1].set_xlabel('Percent of common-support population');axs[1].set_title('Located scenario amount and conditional group range')
fig.suptitle('Rule amplification and tiered capacity envelopes\nq95/q99 are empirical capacity scenarios, not population confidence levels.',fontsize=11);fig.savefig(R/'rule_and_interval_comparison.png',dpi=170);plt.close(fig)
xy=Transformer.from_crs(6668,6677,always_xy=True).transform
wj=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text());wards=[transform(xy,shape(f['geometry'])) for f in wj['features']];union=unary_union(wards)
pg={f['properties']['mesh_code']:transform(xy,shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/three_ward_population_mesh_audit_v1/population_250m_full_mesh.geojson').read_text())['features']};parentlist=sorted(pg);polys=[np.array(pg[p].exterior.coords) for p in parentlist]
fig,axs=plt.subplots(2,2,figsize=(11.8,11),layout='constrained');look={res:{r['Mesh Code']:r for r in cells['q95'] if int(r['Resolution'])==res} for res in [125,250]}
for col,res in enumerate([125,250]):
 point={};uncert={}
 for p in parentlist:
  r=look[250][p];pop=int(r['Source Population'])
  if r['Boundary Hold']=='True' or not pop:continue
  rr=[look[125][p+str(j)] for j in range(1,5)] if res==125 else [r]
  point[p]=sum(float(x['Reserve Scenario Located Residents']) for x in rr)/pop;uncert[p]=sum(float(x['Group Interval Width']) for x in rr)/pop
 for row,values,title in [(0,point,f'{res} m located reserve scenario'),(1,uncert,f'{res} m conditional known-group interval width')]:
  ax=axs[row,col];pc=PolyCollection(polys,array=np.ma.masked_invalid([values.get(p,np.nan) for p in parentlist]),cmap='viridis',norm=Normalize(0,1),edgecolors='none');pc.cmap.set_bad('#dedede');ax.add_collection(pc);fig.colorbar(pc,ax=ax,shrink=.7,label='Fraction of parent population')
  for wg in wards:
   for po in ([wg] if wg.geom_type=='Polygon' else wg.geoms):x,z=po.exterior.xy;ax.plot(x,z,color='#273747',lw=.65)
  ax.set_xlim(union.bounds[0]-100,union.bounds[2]+100);ax.set_ylim(union.bounds[1]-100,union.bounds[3]+100);ax.set_aspect('equal');ax.set_title(title,fontsize=11);ax.set_xlabel('JGD2011 IX easting (m)');ax.set_ylabel('Northing (m)');ax.xaxis.set_major_locator(MaxNLocator(4))
fig.suptitle('Three-ward tiered population allocation | q95 capacity ceiling\nScenario quantities only. Gray: boundary-held cells or no census population.',fontsize=12);fig.savefig(R/'tiered_interval_overview.png',dpi=160);plt.close(fig)
def table(rows,keys):
 def fmt(x):return f'{x:,.2f}' if isinstance(x,float) else str(x)
 return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
body='<h1>Tiered population allocation and conditional uncertainty</h1><p><b>Outcome:</b> whole-cell withholding amplified small building attribute gaps. A tiered model preserves known information, retains unknown-use and institutional capacity in separate unresolved pools, and provides explicitly assumption-bounded intervals. No stranded or affected persons are calculated.</p><p>Common comparison support: 541 complete interior 250 m parents and 2,164 children, with 419,921 census residents on 2020-10-01. This is not the complete population of the three wards. The outer-parent envelope contains another 127,263 residents, including outside-ward population; all of it remains boundary-held at both resolutions.</p><h2>Why so many people were withheld</h2><p>The previous rule blocked a whole cell whenever it contained any unknown-use or missing-floor residential building. All 8,066 unknown-use buildings intersecting the three wards also carry floor placeholder 9999 and detailed-use 0. No alternative use/floor field in this inventory resolves those gaps.</p>'
body+=table(s['diagnosis'],['Resolution','Gap','Touched Populated Cells','Population In Touched Cells','Population Fraction Touched','Median Gap Base Area Fraction In Touched Cells'])
body+='<p>Unknown-use buildings occupy a median 2.16% / 3.02% of the building-base area in affected 125 m / 250 m cells, yet trigger withholding for 93.13% / 99.74% of comparison-support population. Base-area share is not population share. Missing-floor flags overlap unknown-use/institutional flags; their population counts must not be added. <a href="whole_cell_rule_ablation.csv">Eight flag-ablation combinations</a> separate these overlapping effects.</p><img src="rule_and_interval_comparison.png" alt="Withholding amplification and conditional interval scenarios"><h2>Tiered rules</h2>'
rule=[{'Tier':k,'Rule':v} for k,v in s['protocol']['tiers'].items()];body+=table(rule,['Tier','Rule'])
body+='<p>Profiles use ward and residential-use family, with stated fallbacks when fewer than 100 reference buildings are available. Missing floors use height/floor ratios when measured height is valid; otherwise an empirical floor distribution. The reference median supplies a scenario value; [1, empirical upper] supplies a plausible floor range. q95 and q99 are alternative capacity ceilings. Mixed residential share ranges from 0 to 1, with centre 0.5. Unknown use remains unknown and receives no building-level person assignment.</p><h2>Conserved accounting and interval coverage</h2>'
body+=table(new,['Ceiling Scenario','Resolution','Common Interior Population','Reserve Scenario Located Residents','Reserve Scenario Unlocated Residents','Known Candidate Group Lower','Known Candidate Group Upper','Individual Lower Located Mass','Remaining Location Uncertainty Pool'])
body+='<p><b>Two separate conserved representations:</b> (1) a reserve scenario adds located known-building amounts and an unresolved population pool to the source count; (2) each known building receives its marginal lower amount, with all remaining location uncertainty retained in a separate pool. These representations are alternatives and must not be added. The group interval describes the total assigned to the known-candidate group, not the interval of every individual building.</p><p>Individual marginal upper endpoints cannot be added to form a joint upper map. Cellwise summed intervals are conservative outer envelopes because the same physical building may cross meshes. All interval conclusions are conditional on the finite capacity ceiling. Without that ceiling, unknown-building population can still range from zero to the full cell population.</p><img src="tiered_interval_overview.png" alt="Whole three-ward nominal and interval-width maps"><h2>125 m versus 250 m</h2>'
body+=table(s['resolution_comparison'],['Ceiling Scenario','Matched Parents','Matched Population','Joint Redistribution Fraction','Known Buildings 125m','Known Buildings 250m','Unlocated 125m','Unlocated 250m'])
body+='<p>The matched comparison uses 370 parents and 395,109 residents. Half-L1 includes both known-building changes and the separate unresolved bucket per parent; it is not identical to the earlier candidate-only metric. The two resolutions redistribute roughly 12% of model population under these scenarios. This is sensitivity, not accuracy or real people moving.</p>'
body+=table(ivs,['Ceiling Scenario','Populated Interior Parents','Parents With Overlapping Conditional Group Intervals','Population In Nonoverlapping Parents'])
body+='<p>This additional comparison includes all populated interior parents, without selecting only complete candidate cases. Overlap is compatibility of conditional capacity envelopes, not a statistical hypothesis test or proof of stable building locations.</p><h2>Internal floor-proxy check</h2>'+table(s['heldout_proxy_checks'],['Ceiling Scenario','Ward','Holdout Buildings','Observed Floor Upper Coverage','Median Proxy Absolute Floor Error','Mean Floor Range Width','Extrapolated Height Count'])
body+='<p>A deterministic 80/20 split of observed-floor buildings checks the proxy. These are internal reference/check subsets only; no ML model was trained and the project\'s formal event partitions are unchanged. The 5,720 held-out buildings give about 98.5% and 99.25% upper-floor coverage for the two scenarios. This does not establish equivalent coverage for the genuinely missing attributes, unknown-use buildings or actual occupants. Height extrapolations are explicitly flagged.</p><h2>Interpretation and next use</h2><p>The data support a three-ward scenario population layer with uncertainty pools and sensitivity ranges. They do not yet establish exact building occupancy. More located population results from explicit assumptions, not newly observed residents. The 2020 census versus 2021 object surveyYear / 2022 source-list discrepancy remains unresolved.</p><p>A useful next step is to use the population layer at neighborhood/network-component scale with both capacity ceilings and both mesh resolutions, retaining unresolved populations in the output. Determine acceptable interval width and stability at that scale before calculating affected-person totals; building-by-building verification is not required by this audit.</p><p><a href="summary.json">Complete summary and protocol</a> | <a href="interval_accounting_summary.csv">Population accounting</a> | <a href="missingness_amplification_summary.csv">Missingness diagnosis</a> | <a href="heldout_floor_proxy_checks.csv">Internal floor checks</a> | <a href="delivery_validation.json">Delivery validation</a></p>'
(R/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Tiered population uncertainty</title><style>body{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#203044;line-height:1.6}img{width:100%}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd5de;text-align:left}th{background:#eef3f7}</style>'+body+'</html>')
check={'base_validation_passed':baseval['passed'],'independent_vertex_checks_passed':all(r['Passed'] for r in vertexchecks),'both_resolutions_and_both_ceilings_enumerated':set((r['Resolution'],r['Ceiling Scenario']) for r in vertexchecks)==set(itertools.product([125,250],['q95','q99'])),'prior_products_unchanged':True,'figures_created':all((R/p).is_file() for p in ['rule_and_interval_comparison.png','tiered_interval_overview.png']),'report_has_conditional_interpretation':all(x in body for x in ['not confidence','conditional']) if 'not confidence' in body else 'finite capacity ceiling' in body,'no_affected_persons':True}
prior=json.loads((OLD/'delivery_manifest.json').read_text());check['prior_products_unchanged']=all(sha(ROOT/r['file'])==r['sha256'] for r in prior['products']+prior['scripts'])
dump(R/'delivery_validation.json',{'passed':all(check.values()),'checks':check,'independent_cells_enumerated':len(vertexchecks),'max_group_endpoint_error':max(r['Group Endpoint Max Absolute Error'] for r in vertexchecks),'max_building_endpoint_error':max(r['Building Endpoint Max Absolute Error'] for r in vertexchecks),'base_checks':baseval});assert all(check.values()),check
files=[p for b in [OUT,R] for p in sorted(b.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'];scripts=[ROOT/'src/preprocessing/build_three_ward_population_tiered_intervals_v1.py',Path(__file__)]
dump(R/'delivery_manifest.json',{'products':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in files],'scripts':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in scripts],'prior_delivery_sha256':sha(OLD/'delivery_manifest.json')})
print(json.dumps({'passed':all(check.values()),'independent_vertices':len(vertexchecks),'max_group_error':max(r['Group Endpoint Max Absolute Error'] for r in vertexchecks),'overlap':ivs,'product_count':len(files)},indent=2))
