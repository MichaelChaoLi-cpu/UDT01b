#!/usr/bin/env python3
"""Resident Access Loss and Restoration.
AnaSOP 5: matched frozen receiving-allowance cases and fixed August anchor.
AnaSOP 6: absolute interval integration and direct candidate-paired changes.
AnaSOP 7-8: three-parent loss intervals and higher-vs-lower allowance maps.
No model, routing, training, population allocation or new scenario is run.
"""
from pathlib import Path
import os,json,hashlib,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch,Patch
from matplotlib.path import Path as MPath
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_final_evidence_v1'
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
CONTRACT=ROOT/'data/exp/research-design/three_ward_final_evidence_contract_v1'
SLUG='resident_access_loss_and_restoration'
OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')
EVENTS=['E20240731','E20240830','E20250911']
ARM={'none':('No action','#252b30'),'Q2p5':('Q2.5: lower allowance','#4477aa'),'Q5':('Q5: higher allowance','#228833')}
MAP={'All-candidate restoration':('#228833',None),'Possible restoration':('#a5d4b4',None),'All-candidate new loss':('#bb3333',None),'Possible new loss':('#edb77d',None),'Mixed changes':('#9966aa',None),'No access change':('#e5ebef',None),'Unresolved attachment':('#9da9b2','////'),'No recorded residents':('#ffffff',None)}
BIND={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return h.hexdigest()
def bind(p):p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def date(e):return e[1:5]+'-'+e[5:7]+'-'+e[7:9]
def polygon_path(g):
 vertices=[];codes=[]
 for poly in ([g]if g.geom_type=='Polygon'else list(g.geoms)):
  poly=orient(poly,sign=1.)
  for ring in [poly.exterior]+list(poly.interiors):
   xy=np.asarray(ring.coords);vertices.extend(xy);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(xy)-2)+[MPath.CLOSEPOLY])
 return MPath(np.asarray(vertices),codes)
def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True);bind(Path(__file__))
 assert read(CONTRACT/'validation.json')['passed']
 assert read(ROOT/'data/exp/data-preprocessing/three_ward_final_evidence_v1/validation.json')['passed']
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();assert '| Resident Access Loss and Restoration |'in doc
 sourcebindings=read(ROOT/'data/exp/data-preprocessing/three_ward_final_evidence_v1/source_bindings.json')
 for n,h in sourcebindings.items():assert sha(ROOT/n)==h,('Source changed',n)
 data=pd.read_parquet(bind(P/'resident_loss.parquet'));pairs=pd.read_parquet(bind(P/'paired_resident_effects.parquet'));maps=pd.read_parquet(bind(P/'map_classifications.parquet'))
 assert len(data)==18 and len(pairs)==18 and len(maps)==9492
 assert not data.duplicated(['Rainfall Parent','Receiving Water Stage','Measure Configuration']).any()
 assert data['Unresolved Resident Stock'].eq(41308).all()
 assert (data['Resident Access Loss Lower']<=data['Resident Access Loss Upper']).all()
 # Every original basic-unit reporting geometry is retained, with no centroid allocation.
 geofile=ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson'
 raw=read(geofile);features=raw['features'];assert len(features)==4746
 togeo=Transformer.from_crs(6677,4326,always_xy=True)
 geoms={str(x['properties']['unit_id']):transform(togeo.transform,shape(x['geometry']))for x in features}
 assert len(geoms)==4746 and set(geoms)==set(maps['Block ID'])
 wards=read(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/wards_6677.geojson')['features'];wardgeoms=[transform(togeo.transform,shape(x['geometry']))for x in wards]
 bounds=np.array([g.bounds for g in wardgeoms]);xmin,ymin=bounds[:,:2].min(0);xmax,ymax=bounds[:,2:].max(0)
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.05)
 plt.rcParams.update({'axes.linewidth':.8,'savefig.facecolor':'white','hatch.linewidth':.35})
 fig=plt.figure(figsize=(14,12));xs=[.09,.585];width=.365
 axes=[fig.add_axes([x,.625,width,.285])for x in xs]+[fig.add_axes([x,.16,width,.385])for x in xs]
 extent=[];interval_count=0
 for ax,stage,letter in zip(axes[:2],['H','L'],'ab'):
  for ei,event in enumerate(EVENTS):
   center=2-ei
   for arm,offset in zip(ARM,[.23,0,-.23]):
    z=data[data['Rainfall Parent'].eq(event)&data['Receiving Water Stage'].eq(stage)&data['Measure Configuration'].eq(arm)];assert len(z)==1;z=z.iloc[0]
    lo,hi=z['Resident Access Loss Lower']/1000,z['Resident Access Loss Upper']/1000;y=center+offset;color=ARM[arm][1]
    line=ax.hlines(y,lo,hi,color=color,lw=2.3,zorder=3);assert np.allclose(line.get_segments()[0],[[lo,y],[hi,y]])
    ax.plot([lo,hi],[y,y],linestyle='none',marker='|',markersize=8,color=color,markeredgewidth=1.4,zorder=4);interval_count+=1
   if ei<2:ax.axhline(center-.5,color='#e4e9ed',lw=.6,zorder=0)
  ax.set(xlim=(0,1050),ylim=(-.55,2.55),xlabel='Resident access-loss exposure (thousand resident-hours)')
  ax.set_xticks([0,250,500,750,1000]);ax.set_yticks([2,1,0],[date(e)for e in EVENTS]);ax.grid(axis='x',color='#d9e0e5',lw=.65);ax.set_axisbelow(True);ax.tick_params(labelsize=9)
  ax.text(0,1.05,f'{letter}: '+('High stage'if stage=='H'else'Low stage')+', resident access loss',transform=ax.transAxes,fontsize=11.2,fontweight='bold',va='bottom')
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
 mapstats=[];seen=set();aspect=1/np.cos(np.deg2rad((ymin+ymax)/2));dx=(xmax-xmin)*1.08;dy=(ymax-ymin)*1.08
 boxratio=(.385*12)/(width*14)
 dx=max(dx,dy*aspect/boxratio);dy=boxratio*dx/aspect
 cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
 for ax,stage,letter in zip(axes[2:],['H','L'],'cd'):
  subset=maps[maps['Receiving Water Stage'].eq(stage)].set_index('Block ID');assert len(subset)==4746 and subset['Resident Population'].sum()==496345
  for name,(color,hatch)in MAP.items():
   ids=subset.index[subset.Class.eq(name)].tolist();seen.add(name)if ids else None
   if ids:
    patch=[PathPatch(polygon_path(geoms[i]))for i in ids]
    collection=PatchCollection(patch,facecolor=color,edgecolor='#b6c0c8'if name in ['No access change','No recorded residents']else '#60717e'if hatch else color,linewidth=.13,hatch=hatch,zorder=2);ax.add_collection(collection)
   mapstats.append({'Receiving Water Stage':stage,'Class':name,'Basic Units':len(ids),'Resident Stock in Reporting Units':int(subset.loc[ids,'Resident Population'].sum())if ids else 0,'Meaning':'Stock in reporting polygons; possible restoration is not a count of people restored'})
  for g in wardgeoms:
   polys=[g]if g.geom_type=='Polygon'else list(g.geoms)
   for poly in polys:
    xy=np.asarray(poly.exterior.coords);ax.plot(xy[:,0],xy[:,1],color='#445762',lw=.8,zorder=5)
  ax.set(xlim=(cx-dx/2,cx+dx/2),ylim=(cy-dy/2,cy+dy/2),xlabel='Longitude',ylabel='Latitude');ax.set_aspect(aspect,adjustable='box');ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f} E'));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f} N'));ax.tick_params(labelsize=8.5)
  ax.grid(color='#8d9da9',lw=.45,alpha=.5,zorder=6)
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
  ax.text(0,1.05,f'{letter}: '+('High stage'if stage=='H'else'Low stage')+', Q5 vs Q2.5',transform=ax.transAxes,fontsize=11.2,fontweight='bold',va='bottom')
 handles=[Line2D([0],[0],lw=2.3,color=c,label=label)for label,c in ARM.values()]
 handles += [Patch(facecolor=c,edgecolor='#60707c',hatch=h,label=name)for name,(c,h)in MAP.items()if name in seen]
 legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.025),ncol=4,frameon=False,fontsize=9,columnspacing=1.8,handlelength=2)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();lb=legend.get_window_extent(renderer)
 assert lb.y1<min(ax.get_tightbbox(renderer).y0 for ax in axes[2:]),'Legend overlaps map axes'
 alignment=[]
 for upper,lower in [(axes[0],axes[2]),(axes[1],axes[3])]:
  assert np.allclose([upper.get_position().x0,upper.get_position().x1],[lower.get_position().x0,lower.get_position().x1],atol=1e-8);alignment.append(True)
 for ax in axes:
  box=ax.get_tightbbox(renderer);assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
 assert interval_count==18 and all(ax.get_title()==''for ax in axes)
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135);plt.close(fig)
 with Image.open(OUT)as im:dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<=x<=451 for x in dpi)
 data.to_csv(R/(SLUG+'_plot_data.csv'),index=False);pairs.to_csv(R/(SLUG+'_paired_effects.csv'),index=False);pd.DataFrame(mapstats).to_csv(R/(SLUG+'_map_summary.csv'),index=False)
 # Confirm no bound data or qualified source was changed by visualization.
 assert all(sha(ROOT/n)==h for n,h in BIND.items())
 validation={'passed':True,'decision':'MILI-D-20261024-002','intervals':18,'map_features_per_panel':4746,'population_total':496345,'unresolved_stock':41308,'anchor_frame':97,'anchor_elapsed_hours':97/12,'anchor_source_candidate_pairs_verified':True,'H_L_anchor_classifications_identical':bool(maps[maps['Receiving Water Stage'].eq('H')].set_index('Block ID').Class.sort_index().equals(maps[maps['Receiving Water Stage'].eq('L')].set_index('Block ID').Class.sort_index())),'main_depth':'lower','candidate_radius_m':10,'dimensions':dimensions,'dpi':dpi,'columns_aligned':all(alignment),'four_sided_map_frames':True,'longitude_latitude_graticules':True,'legend_below_panels':True,'interval_midpoints':False,'absolute_intervals_not_subtracted_for_effects':True,'new_simulations':0,'new_routing':0,'sources_unchanged':True,'output_sha256':sha(OUT),'awaiting_human_review':True}
 dump(R/(SLUG+'_validation.json'),validation);dump(R/(SLUG+'_source_bindings.json'),BIND)
 caption='''Panels a and b compare nine-hour resident vehicle-access-loss exposure under no action, S1R with total assumed receiving allowance 2.5 cubic meters per second (Q2.5), and S1R with allowance 5.0 (Q5), across three inspected historical rainfall parents. Each horizontal interval retains the original all-candidate/any-candidate bounds; endpoints are marked and no midpoint is estimated. Units are thousand resident-hours, not unique affected or stranded people. Smaller values mean less modeled access-loss exposure. The displayed absolute bounds cannot be subtracted to obtain a paired treatment-benefit interval. Direct candidate-paired net, restoration and new-loss intervals are supplied separately in the companion ledger. Panels c and d map the additional access changes under Q5 relative to Q2.5 in August at the original frame-97 anchor, approximately 8 hours 5 minutes after model start, under high and low stage assumptions. Both maps retain identical official reporting-unit footprints, candidate sets and time. At this frozen anchor, access-change classifications are identical across the two stage assumptions; this does not imply identical full-window loss histories. All-candidate restoration means every eligible candidate regains bidirectional vehicle access; possible restoration means at least one does. An unchanged unit may retain access or remain disconnected; it does not mean unaffected by rainfall. Unresolved attachment is distinct from known access, and zero-resident reporting units remain visible. Stock in a potentially restored polygon is not a confirmed count of people restored. Official 2020 nighttime population is 496,345; 41,308 residents remain unresolved under the frozen 10 m candidate rule. The primary lower road-depth branch uses the unchanged 0.30 m closure rule; unknown road exposure remains conditionally available, not verified safe. Access is bidirectional vehicle connectivity to the declared external portal set, not multimodal isolation, actual occupancy or urban-function operation. The same 319 hypothetical native intake locations, weighted allocator, unit ceiling and other eligibility rules are held fixed. Only the five receiving-pool allowances differ; neither allowance is observed pump equipment or surveyed downstream capacity. Historical radar forcing, dry starts and assumed stages provide conditional scenarios without independent flood or congestion validation. Added receiving-river storage and downstream impacts remain unmodeled.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 methods='''# Resident Access Loss and Restoration: reproducible scope\n\nThe figure uses the user-approved final evidence contract and existing qualified exports. Sections 5-7 of AnaSOP bind identical candidate locations, population stocks, clocks, road-depth branch and qualified support. Absolute resident-access intervals integrate the original all/any flags. The companion treatment intervals are inherited direct candidate pairs, never differences of absolute endpoints. The consolidated preprocessing script independently verifies all twelve original 28+8 hydraulic gates, road-state integrals, population total, unresolved denominator and both maps against the original anchor paired-stock summaries.\n\nThe map generator transforms official EPSG:6677 reporting geometries to longitude/latitude solely for display. Native cached bidirectional access masks define changes at the original 10 m candidates. No new routing or centroid assignment is performed. The panel footprint covers the whole three wards, with geographic proportions, graticules, complete borders, aligned columns, a bottom legend and English annotations. A 450 dpi PNG and smaller review preview are exported. Maps show a frozen diagnostic time, while interval panels cover the full nine-hour window; these are explicitly different temporal summaries.\n\nThe generated image is pending human review. No subsequent figure is automatically generated or marked complete.\n'''
 (R/(SLUG+'_methods.md')).write_text(methods)
 def table(d):return '<div class="scroll">'+d.to_html(index=False,float_format=lambda x:f'{x:,.3f}',escape=True)+'</div>'
 body='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Resident Access Loss and Restoration</title><style>body{font:16px/1.75 system-ui;background:#f6f9fb;color:#24404f;max-width:1350px;margin:28px auto;padding:0 24px}img{width:100%;height:auto;background:white}aside{background:#e3eef3;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:8px;border:1px solid #ccd7df;white-space:nowrap}a{color:#176a96}</style></head><body><h1>Resident Access Loss and Restoration</h1><aside>Pending visual review. English annotations, 450 dpi. Upper panels show remaining loss intervals; lower panels show additional Q5-versus-Q2.5 access changes at the fixed August anchor. No midpoint estimate or actual stranded-person count is supplied.</aside>'''
 body+=f'<img src="../../../results/figures/Figure_{SLUG}.png?v={validation["output_sha256"][:12]}" alt="Two resident loss interval panels and two whole-three-ward access restoration maps">'
 body+='<h2>How to read the figure</h2><p>Upper panels: lower is better. The line ends show the original candidate uncertainty bounds. A long interval reflects uncertain network attachment, not statistical sampling error. Lower panels: green indicates additional restoration under the higher receiving allowance, not every area protected by either measure. Gray hatching is unresolved attachment. Unchanged access can include roads or residents that remain disconnected.</p><h2>Caption</h2><p>'+html.escape(caption)+'</p>'
 body+='<h2>Absolute resident loss (resident-hours)</h2>'+table(data)+'<h2>Directly paired benefits and new losses (resident-hours)</h2>'+table(pairs)+'<h2>Map classification and reporting-unit stocks</h2>'+table(pd.DataFrame(mapstats))
 body+=f'<p><a href="../../../results/figures/Figure_{SLUG}.png">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Validation</a> · <a href="{SLUG}_methods.md">Methods</a> · <a href="{SLUG}_source_bindings.json">Source hashes</a> · <a href="../../data-preprocessing/three_ward_final_evidence_v1/Three_Ward_Final_Evidence.xlsx">Final evidence workbook</a></p></body></html>'
 (R/(SLUG+'_review.html')).write_text(body)
 print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
