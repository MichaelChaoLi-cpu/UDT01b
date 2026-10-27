"""Review conserved 125/250 m population scenarios on common census blocks.
No new population allocation assumptions, network isolation or affected counts.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/udt_block_population_mpl')
import csv,json,hashlib,html,collections
import numpy as np
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/three_ward_population_block_stability_v1'
R=ROOT/'data/exp/data-preprocessing/three_ward_population_block_stability_v1'
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
validation=json.loads((R/'validation.json').read_text());assert all(validation['checks'].values())
base=json.loads((R/'aggregation_summary.json').read_text());account=readcsv(R/'population_accounting.csv')
wardnames={'13101':'Chiyoda','13102':'Chuo','13103':'Minato'}
allrows=[];summaries=[];comparisons=[];reference_checks=[]
for level in ['block','town']:
 rows=readcsv(OUT/f'{level}_population_stability.csv')
 for r in rows:
  r['complete_original_support']=r['complete_original_support']=='True'
  for k,v in list(r.items()):
   if k not in ['level','unit_id','ward','name','town_id','allocation_input_class','quality_flags','official_count_role','complete_original_support']:
    r[k]=float(v) if v else None
 for r in rows:
  r['stability_screen_class']=r['allocation_input_class']
  r['census_reference_inside_conditional_envelope']=r['envelope_lower']-1e-6<=r['official_2020_residents']<=r['envelope_upper']+1e-6
  if not r['census_reference_inside_conditional_envelope']:
   r['quality_flags']=(r['quality_flags']+'|same_census_reference_outside_envelope').strip('|')
   if r['allocation_input_class'] in ['stable_conditional','interval_only']:r['allocation_input_class']='withhold_reference_incompatible'
  if r['complete_original_support'] and r['official_2020_residents']==0:
   r['allocation_input_class']='zero_resident_reference'
   if r['point_max']>1e-6:r['quality_flags']=(r['quality_flags']+'|model_positive_population_in_official_zero_unit').strip('|')
 allrows+=rows
 for ward in ['all','13101','13102','13103']:
  rr=[r for r in rows if ward=='all' or r['ward']==ward]
  for status in ['stable_conditional','interval_only','withhold_spatial_allocation','withhold_reference_incompatible','withhold_partial_coverage','zero_resident_reference']:
   ss=[r for r in rr if r['allocation_input_class']==status]
   summaries.append({'level':level,'ward':wardnames.get(ward,'Three wards'),'allocation_input_class':status,'units':len(ss),'units_with_positive_official_residents':sum(r['official_2020_residents']>0 for r in ss),'official_2020_residents_in_units':sum(r['official_2020_residents'] for r in ss),'mean_scenario_population_in_units':sum(r['point_mean'] for r in ss),'median_point_relative_spread':float(np.median([r['point_relative_spread'] for r in ss if r['point_relative_spread'] is not None])) if any(r['point_relative_spread'] is not None for r in ss) else None})
  for q in ['q95','q99']:
   for scope in ['all_common_support_fragments','fully_covered_units']:
    ss=[r for r in rr if scope=='all_common_support_fragments' or r['complete_original_support']]
    x=np.array([r[f'125m_{q}_point'] for r in ss]);y=np.array([r[f'250m_{q}_point'] for r in ss]);den=float((x.sum()+y.sum())/2);half=float(np.abs(x-y).sum()/2)
    comparisons.append({'level':level,'ward':wardnames.get(ward,'Three wards'),'scope':scope,'capacity_ceiling':q,'units':len(ss),'scenario_population_125m':float(x.sum()),'scenario_population_250m':float(y.sum()),'half_l1_movement_on_units':half,'half_l1_over_mean_supported_population':half/den if den else None,'spearman_rank_correlation':None,'not_an_accuracy_metric':True})
    from scipy.stats import spearmanr
    if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0:comparisons[-1]['spearman_rank_correlation']=float(spearmanr(x,y).statistic)
  full=[r for r in rr if r['complete_original_support']];pop=sum(r['official_2020_residents'] for r in full)
  reference_checks.append({'level':level,'ward':wardnames.get(ward,'Three wards'),'fully_covered_units':len(full),'official_population':pop,'mean_allocated_scenario_population':sum(r['point_mean'] for r in full),'absolute_unit_error_sum':sum(r['census_reference_absolute_difference'] for r in full),'absolute_unit_error_over_reference_population':sum(r['census_reference_absolute_difference'] for r in full)/pop if pop else None,'same_census_not_independent_validation':True})
writecsv(R/'admission_summary.csv',summaries);writecsv(R/'resolution_stability_metrics.csv',comparisons);writecsv(R/'same_census_reference_checks.csv',reference_checks)
# Identify coarse reporting areas with enough stable block-level population support.
# Counts are a quality inventory; do not label towns as transport islands.
town_inventory=[]
for t in [r for r in allrows if r['level']=='town']:
 bb=[r for r in allrows if r['level']=='block' and r['town_id']==t['unit_id']];pop=t['official_2020_residents'];stable=sum(r['official_2020_residents'] for r in bb if r['allocation_input_class']=='stable_conditional');conditional=sum(r['official_2020_residents'] for r in bb if r['allocation_input_class'] in ['stable_conditional','interval_only'])
 town_inventory.append({'town_id':t['unit_id'],'ward':t['ward'],'town_name':t['name'],'official_2020_residents':pop,'basic_units':len(bb),'stable_conditional_basic_units':sum(r['allocation_input_class']=='stable_conditional' for r in bb),'interval_only_basic_units':sum(r['allocation_input_class']=='interval_only' for r in bb),'official_residents_in_stable_basic_units':stable,'official_residents_in_stable_or_interval_basic_units':conditional,'stable_or_interval_population_fraction':conditional/pop if pop else None,'town_aggregate_input_class':t['allocation_input_class'],'not_a_transport_island':True})
writecsv(OUT/'town_population_input_inventory.csv',town_inventory)
writecsv(OUT/'population_input_admission_list.csv',[{k:r[k] for k in ['unit_id','ward','name','town_id','official_2020_residents','point_min','point_max','envelope_lower','envelope_upper','allocation_input_class','stability_screen_class','census_reference_inside_conditional_envelope','quality_flags','complete_original_support']} for r in allrows if r['level']=='block'])
# Reference-only totals stay separately identified from model allocation scenarios.
writecsv(OUT/'official_2020_block_count_baseline.csv',[{'block_id':r['unit_id'],'ward':r['ward'],'town_id':r['town_id'],'name':r['name'],'official_2020_residents':r['official_2020_residents'],'count_role':'historical census block total; not building occupancy','within_block_allocation_class':r['allocation_input_class']} for r in allrows if r['level']=='block'])
# Three-ward maps use a common projected extent, no point-specific investigation.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize,ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
features=json.loads((OUT/'block_population_stability_6677.geojson').read_text())['features'];rr=[r for r in allrows if r['level']=='block'];geoms=[shape(f['geometry']) for f in features]
dump(OUT/'population_input_admission_6677.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':[{'type':'Feature','geometry':f['geometry'],'properties':r} for f,r in zip(features,rr)]})
verts=[];owner=[]
for i,g in enumerate(geoms):
 for p in ([g] if g.geom_type=='Polygon' else g.geoms):
  if p.geom_type=='Polygon':verts.append(np.asarray(p.exterior.coords));owner.append(i)
owner=np.array(owner)
from shapely.ops import unary_union
wardgeom={w:unary_union([g for g,r in zip(geoms,rr) if r['ward']==w]) for w in wardnames}
extent=unary_union(geoms).bounds
fig,axs=plt.subplots(2,2,figsize=(13,12),layout='constrained')
statuslist=['stable_conditional','interval_only','withhold_spatial_allocation','withhold_reference_incompatible','withhold_partial_coverage','zero_resident_reference'];colors=['#2a9980','#e4b45a','#bd6259','#8052a2','#d1d3d5','#eff2f3']
for i,(ax,title,col) in enumerate(zip(axs.flat,['125 m source | q95 capacity scenario','250 m source | q95 capacity scenario','Spread across both resolutions and ceilings','Population spatial-input screening'],['125m_q95_point','250m_q95_point','point_relative_spread','allocation_input_class'])):
 if i<2:
  vals=np.array([r[col] for r in rr]);vals=np.log10(1+vals);norm=Normalize(0,np.log10(2001));cmap=plt.get_cmap('viridis');label='log10(1 + conditional residents)'
 elif i==2:
  vals=np.array([r[col] if r[col] is not None else np.nan for r in rr]);norm=Normalize(0,.5);cmap=plt.get_cmap('magma');label='(maximum - minimum) / scenario mean'
 else:
  vals=np.array([statuslist.index(r[col]) for r in rr]);norm=BoundaryNorm(np.arange(-.5,6.5),6);cmap=ListedColormap(colors)
 if i<3:
  vals=np.array(vals,dtype=float);vals[[not r['complete_original_support'] for r in rr]]=np.nan;cmap=cmap.copy();cmap.set_bad('#d1d3d5')
 pc=PolyCollection(verts,array=np.ma.masked_invalid(vals[owner]),cmap=cmap,norm=norm,edgecolors='none');ax.add_collection(pc)
 for w,g in wardgeom.items():
  for p in ([g] if g.geom_type=='Polygon' else g.geoms):
   x,y=p.exterior.xy;ax.plot(x,y,color='#293a45',lw=.5)
  anchor=g.representative_point();ax.text(anchor.x,anchor.y,wardnames[w],fontsize=8,color='black',bbox={'facecolor':'white','alpha':.8,'pad':1,'edgecolor':'none'})
 ax.set_xlim(extent[0]-100,extent[2]+100);ax.set_ylim(extent[1]-100,extent[3]+100);ax.set_aspect('equal');ax.set_title(title,fontsize=11);ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5));ax.tick_params(labelsize=8);ax.set_xlabel('JGD2011 IX easting (m)');ax.set_ylabel('Northing (m)')
 if i<3:fig.colorbar(pc,ax=ax,shrink=.7,label=label,extend='max' if i==2 else 'neither')
 else:ax.legend(handles=[Patch(color=c,label=s.replace('_',' ')) for c,s in zip(colors,statuslist)],loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=7)
fig.suptitle('Historical resident population on census basic units | Three wards\nConditional spatial allocation, not observed occupancy. Gray: incomplete original support.',fontsize=12)
fig.savefig(R/'block_population_stability_overview.png',dpi=160);plt.close(fig)
# Scatter plots expose agreement between scenarios and disagreement with census.
fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained');full=[r for r in rr if r['complete_original_support']]
for w,c in zip(wardnames,['#448caa','#d99048','#638f67']):
 ss=[r for r in full if r['ward']==w]
 axs[0].scatter([r['125m_q95_point'] for r in ss],[r['250m_q95_point'] for r in ss],s=6,alpha=.35,color=c,label=wardnames[w])
 axs[1].scatter([r['official_2020_residents'] for r in ss],[r['point_mean'] for r in ss],s=6,alpha=.35,color=c)
 axs[2].scatter([r['point_mean'] for r in ss],[r['envelope_width'] for r in ss],s=6,alpha=.35,color=c)
for ax in axs:ax.set_xscale('symlog',linthresh=10);ax.set_yscale('symlog',linthresh=10);ax.grid(alpha=.15)
for ax in axs[:2]:ax.plot([0,10000],[0,10000],color='#333',lw=.7,ls='--')
axs[0].set(xlabel='125 m q95 conditional residents',ylabel='250 m q95 conditional residents',title='Resolution agreement');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Official 2020 basic-unit residents',ylabel='Mean conditional allocation',title='Same-census aggregate reference')
axs[2].set(xlabel='Mean conditional allocation',ylabel='Outer envelope width (residents)',title='Conditional uncertainty after aggregation')
fig.suptitle('Complete common-support basic units only | Agreement does not establish accuracy',fontsize=11);fig.savefig(R/'block_population_reference_comparison.png',dpi=160);plt.close(fig)
admission_policy={
 'preliminary_stability_protocol':str((OUT/'protocol.json').relative_to(ROOT)),
 'additional_reference_guard':'A unit passing the preliminary stability screens is still withheld if its same-census reference count lies outside the recomputed four-scenario conditional envelope (numerical tolerance 1e-6 person). The existing scenarios and intervals are unchanged. This conservative compatibility gate is separate from the provisional magnitude/spread thresholds.',
 'why':'Agreement between two mechanically related census mesh products can conceal shared spatial-allocation bias. Same-census block counts supply an aggregate reference, not independent occupancy validation.',
 'zero_reference_guard':'Fully covered units with official zero residents are supplied only as a zero historical census baseline; positive model scenarios there are flagged and are not population inputs.',
 'authoritative_admission_outputs':['population_input_admission_list.csv','population_input_admission_6677.geojson','town_population_input_inventory.csv'],
 'preliminary_products':'block_population_stability.csv / town_population_stability.csv and their GeoJSONs contain the preliminary stability class before the additional census-reference guard. Use the final admission list for downstream selection.'}
dump(OUT/'admission_policy.json',admission_policy)
ward_inventory=[]
for w in wardnames.values():
 ss=[r for r in summaries if r['level']=='block' and r['ward']==w];st=next(r for r in ss if r['allocation_input_class']=='stable_conditional');iv=next(r for r in ss if r['allocation_input_class']=='interval_only')
 ward_inventory.append({'ward':w,'stable_basic_units':st['units'],'official_residents_in_stable_units':st['official_2020_residents_in_units'],'interval_basic_units':iv['units'],'official_residents_in_interval_units':iv['official_2020_residents_in_units']})
writecsv(R/'ward_population_input_summary.csv',ward_inventory)
summary={**base,'admission_summary':summaries,'resolution_stability':comparisons,'same_census_reference_checks':reference_checks,'interpretation':'Official 2020 counts provide a separate aggregate baseline. Allocation admission is conditional on capacity ceilings and coverage; no transport isolation or stranded population has been calculated.'}
dump(R/'summary.json',summary)
def table(rows,keys):
 def fmt(x):
  if x is None:return '—'
  return f'{x:,.3f}' if isinstance(x,float) else str(x)
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table></div>'
alls=[r for r in summaries if r['level']=='block' and r['ward']=='Three wards'];cmp=[r for r in comparisons if r['ward']=='Three wards' and r['scope']=='all_common_support_fragments'];refs=[r for r in reference_checks if r['ward']=='Three wards']
body='''<h1>Population stability at street-block scale</h1>
<p class="lead">Two mesh resolutions and two capacity ceilings have been aggregated onto the same official 2020 census basic units. Point scenarios conserve population; group bounds are recomputed rather than summed from building bounds. This is a historical nighttime population input audit, not a stranded-person calculation.</p>
<h2>What is available</h2><p>4,746 census basic units and 332 town-level units cover Chiyoda, Chuo and Minato. Official same-census boundary counts sum to 496,345 residents: Chiyoda 66,680; Chuo 169,179; Minato 260,486. These totals provide a separate 2020 block-population baseline; they do not establish who occupies each building today.</p>
<p>The frozen allocation comparison remains on 541 complete interior 250 m parents, representing 419,921 residents. Another 127,263 residents in the outer parent envelope remain held, including population outside the three wards. This envelope must never be added to the official ward totals. Partial-support units are withheld as whole-block spatial inputs.</p>
<img src="block_population_stability_overview.png" alt="Three-ward maps of two population scenarios, sensitivity and screening classes">
<h2>Which units can enter later analysis</h2><p><b>Stable conditional:</b> candidate spatial inputs for a scenario-based transport analysis, retaining the conditional envelope. <b>Interval only:</b> carry lower/upper scenarios, do not use a single occupancy map. <b>Withhold spatial allocation:</b> use an official aggregate count if a block-total question is sufficient; the building distribution remains unresolved. <b>Reference incompatible:</b> preliminary stability screens passed, but the official same-census count lies outside the conditional envelope; withhold the spatial allocation. <b>Partial coverage:</b> complete the original geographic support before using a whole-block allocation. <b>Zero resident reference:</b> use the official zero historical baseline; positive model scenarios in these units are flagged and are not population inputs. No class is a safety certification.</p>'''
body+=table(alls,['allocation_input_class','units','units_with_positive_official_residents','official_2020_residents_in_units','mean_scenario_population_in_units'])
body+=table(ward_inventory,['ward','stable_basic_units','official_residents_in_stable_units','interval_basic_units','official_residents_in_interval_units'])
body+='<p>The final admission list includes the reference compatibility guard. The scenario/stability CSVs retain the earlier preliminary class, so downstream selection must use the final admission list. Official residents in each class and model-scenario amounts have different roles. Class totals above are not affected or stranded counts.</p><h2>Scale stability</h2>'+table(cmp,['level','capacity_ceiling','half_l1_movement_on_units','half_l1_over_mean_supported_population','spearman_rank_correlation'])
body+='''<p>The half-L1 measure is half the sum of absolute differences across the displayed units. It summarizes spatial redistribution, not population prediction error. Town-level agreement can conceal block-level shifts. The remaining no-capacity and geometry-gap pools are exported separately and included in population conservation checks.</p>
<img src="block_population_reference_comparison.png" alt="Resolution agreement, census comparison and conditional width">
<h2>Does agreement imply correctness?</h2><p>No. The same census produced both mesh totals. The boundary product supplies basic-unit JINKO totals as an additional aggregate reference from that census. Disagreement is a spatial-allocation warning; agreement is not independent validation. Allocations were not refitted or rescaled to these counts.</p>'''+table(refs,['level','fully_covered_units','official_population','mean_allocated_scenario_population','absolute_unit_error_over_reference_population'])
body+='''<h2>How uncertainty was aggregated</h2><p>Each candidate building/mesh fragment contributes the fraction of its base geometry inside the reporting unit. For each mesh we optimize the normalized capacity-weighted group total over the original lower and upper weights. This lets uncertainty within one block cancel when all possible locations remain inside that block. Shared building parameters across meshes mean sums of cell extrema are conservative outer envelopes; lower and upper endpoint maps are not jointly realizable population maps.</p>
<p>Unknown-use and institutional candidates remain unallocated to individual buildings. Their possible locations can nevertheless be bounded at block scale, conditional on the same candidate geometry and finite capacity ceilings. Such block-scale localization is not an observed occupancy assignment. If no positive candidate capacity exists, the census-only pool remains unresolved; intersecting-unit upper bounds include its potential contribution without inventing an area-weighted point estimate.</p>
<p>q95/q99 denote empirical floor-capacity ceiling scenarios, not population confidence levels. Mixed-use share, missing floors, unknown use, the 2020 census versus later building inventory, and boundary alignment remain material assumptions.</p>
<h2>Provisional screening rules</h2><p>Stable conditional units require complete original support; a four-scenario spread no larger than max(10 people, 10% of the mean); envelope width no larger than max(25 people, 25%); same-census reference difference no larger than max(25 people, 25% of the reference); and census-only unresolved potential no larger than max(5 people, 5%). A separate conservative reference guard then withholds any otherwise eligible unit whose same-census count lies outside its outer conditional envelope. This guard was added after detecting jointly stable but reference-incompatible allocations; no weights or envelopes were changed. Interval-only units use the looser documented screens. These are analyst screening choices, not official thresholds or proof of accuracy. Continuous diagnostics remain in the tables.</p>
<h2>Next connection to transport analysis</h2><p>Use the separately identified official 2020 basic-unit totals as a historical nighttime aggregate baseline. Where within-block spatial placement is needed, use the stable conditional allocation or explicit interval scenarios; keep withheld units as aggregate unresolved pools. Overlay a routable road and station-access network before deriving isolation components. A census boundary is not itself an island boundary. Daytime workers, students, visitors, station entrances and service interruption rules remain separate inputs. No population has yet been counted as isolated.</p>
<h2>Files and verification</h2><ul>
<li><a href="../../../processed/three_ward_population_block_stability_v1/population_input_admission_list.csv">Basic-unit allocation input admission list</a></li>
<li><a href="../../../processed/three_ward_population_block_stability_v1/town_population_input_inventory.csv">Town inventory of stable and interval-only basic units</a></li>
<li><a href="../../../processed/three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv">Separate official 2020 block-count baseline</a></li>
<li><a href="../../../processed/three_ward_population_block_stability_v1/block_population_stability.csv">All block scenarios and intervals (preliminary stability class)</a></li>
<li><a href="../../../processed/three_ward_population_block_stability_v1/town_population_stability.csv">Town scenarios and directly recomputed intervals</a></li>
<li><a href="../../../processed/three_ward_population_block_stability_v1/protocol.json">Aggregation rules and limitations</a>; <a href="../../../processed/three_ward_population_block_stability_v1/admission_policy.json">final admission guard</a></li>
<li><a href="validation.json">Numerical checks</a>, <a href="population_accounting.csv">conservation ledger</a>, <a href="resolution_stability_metrics.csv">stability metrics</a>, <a href="delivery_manifest.json">source and product version bindings</a></li></ul>
<p>Official sources: <a href="https://www.e-stat.go.jp/help/data-definition-information/about-recorded-data">Basic-unit definitions</a>; <a href="https://www.e-stat.go.jp/help/data-definition-information/download">Boundary cautions</a>. Download URLs and hashes are retained with the six official GML archives. All prior frozen allocations, raw data and research splits remain unchanged. No training, hydraulic simulation or affected-person calculation.</p>
'''
page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-ward population block stability</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#243747;background:#f7f9fa}h1{font-size:30px;line-height:1.2}h2{margin-top:34px;font-size:22px}.lead{font-size:18px}img{display:block;width:100%;background:white;border:1px solid #ddd;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{padding:9px;border:1px solid #dce2e7;text-align:left}th{background:#e9f0f3}a{color:#176b83}</style>'+body+'</html>'
(R/'review.html').write_text(page)
# Numerical cross-level and nested-envelope checks do not alter any allocation.
blockrows=[r for r in allrows if r['level']=='block'];townrows=[r for r in allrows if r['level']=='town'];crosslevel_error=0.;town_interval_tightening=True
for t in townrows:
 bb=[r for r in blockrows if r['town_id']==t['unit_id']]
 for res in [125,250]:
  for q in ['q95','q99']:
   stem=f'{res}m_{q}_'
   for c in ['point','known_residential','candidate_reserve']:crosslevel_error=max(crosslevel_error,abs(t[stem+c]-sum(r[stem+c] for r in bb)))
   town_interval_tightening &= t[stem+'lower']+1e-6>=sum(r[stem+'lower'] for r in bb) and t[stem+'upper']<=sum(r[stem+'upper'] for r in bb)+1e-6
checks_extra={'source_archives_match_download_hashes':all(sha(ROOT/r['file'])==r['sha256'] for r in json.loads((OUT/'official_boundaries/download_manifest.json').read_text())),'block_points_sum_to_town_points':crosslevel_error<1e-6,'direct_town_intervals_no_wider_than_summed_block_intervals':town_interval_tightening,'q99_envelopes_contain_q95_envelopes':all(r[f'{res}m_q99_lower']<=r[f'{res}m_q95_lower']+1e-6 and r[f'{res}m_q99_upper']+1e-6>=r[f'{res}m_q95_upper'] for r in allrows for res in [125,250]),'official_block_baseline_conserves_ward_census_total':sum(r['official_2020_residents'] for r in blockrows)==496345}
assert all(checks_extra.values()),checks_extra
dump(R/'cross_scale_validation.json',{'checks':checks_extra,'max_block_to_town_point_error':crosslevel_error})
# HTML links are relative to data/exp/data-preprocessing/<run>; three .. reach data.
from urllib.parse import unquote
import re
links=re.findall(r'(?:href|src)="([^"]+)"',page);bad=[x for x in links if not x.startswith(('https:','http:','#')) and x!='delivery_manifest.json' and not (R/unquote(x)).resolve().exists()];assert not bad,bad
files=sorted(p for directory in [OUT,R] for p in directory.rglob('*') if p.is_file() and p.name not in ['delivery_manifest.json','delivery_validation.json'])
scripts=[ROOT/'src/preprocessing/build_three_ward_population_block_stability_v1.py',Path(__file__).resolve()]
dump(R/'delivery_manifest.json',{'human_decision':'MILI-D-20261016-019','products':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files],'scripts':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in scripts],'prior_delivery_manifest_sha256':sha(ROOT/'data/exp/data-preprocessing/three_ward_population_tiered_intervals_v1/delivery_manifest.json')})
dump(R/'delivery_validation.json',{'local_report_links_exist':not bad,'report_local_links':sum(not x.startswith(('http:','https:','#')) for x in links),'all_numerical_checks_pass':all(validation['checks'].values()) and all(checks_extra.values()),'all_zero_reference_units_have_zero_official_count':all(r['official_2020_residents']==0 for r in allrows if r['allocation_input_class']=='zero_resident_reference'),'all_admitted_units_reference_compatible':all(r['census_reference_inside_conditional_envelope'] for r in allrows if r['allocation_input_class'] in ['stable_conditional','interval_only']),'no_training_or_affected_count':True})
print(json.dumps({'admission':alls,'comparison':cmp,'reference':refs},indent=2))
