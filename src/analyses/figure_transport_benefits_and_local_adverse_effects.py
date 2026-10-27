#!/usr/bin/env python3
"""Approved four-panel presentation of existing traffic and adverse-change evidence.
AnaSOP5: distinct three-parent adverse and September-only traffic scopes.
AnaSOP6: existing native head screen, paired road closure and fixed OD cohort.
AnaSOP7-8: add/bdd/cdd assembly; no simulation, routing or fitting.
"""
from pathlib import Path
import os,json,html,hashlib,sys
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter,MaxNLocator
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import figure_neighboring_flood_transfer as adverse
import figure_vehicle_access_and_traffic_cost as traffic
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
SLUG='transport_benefits_and_local_adverse_effects'
TITLE='Transport Benefits and Local Adverse Effects'
OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')
BIND={}
REFS={'No action':('Q5 relative to no action','#cc6633'),'S1R-Q2.5':('Q5 relative to Q2.5','#4477aa')}
STAGE={'H':('High stage','-','o'),'L':('Low stage','--','s')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1024**2),b''):h.update(z)
 return h.hexdigest()
def bind(p):
 p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def prepare():
 bind(Path(__file__));bind(Path(adverse.__file__));bind(Path(traffic.__file__))
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();assert '| '+TITLE+' |' in doc
 head,road,cross,primary,samples,checks,support=adverse.prepare()
 print('Verified twelve native head contrasts and twenty-four two-branch road contrasts.',flush=True)
 data,within,source,paired,audits,weight=traffic.prepare()
 print('Verified twenty-four existing traffic fits and the fixed all-state OD cohort.',flush=True)
 BIND.update(adverse.BIND);BIND.update(traffic.BIND)
 # Check the assembly against both accepted masters without rerendering them.
 for key,d in [('neighboring_flood_transfer_head_screens',head),('neighboring_flood_transfer_road_side_effects',road),('vehicle_access_and_traffic_cost_plot_data',data)]:
  old=pd.read_csv(bind(R/(key+'.csv')),float_precision='round_trip')
  pd.testing.assert_frame_equal(d.reset_index(drop=True),old,check_dtype=False,rtol=0,atol=1e-8)
 assert len(primary)==9 and len(audits)==24
 for _,g in data.groupby(['Frame','Receiving Water Stage']):
  assert g['Routable Requests Percent'].nunique()==1
 assert paired.restored_request_proxy.eq(0).all() and paired.newly_unreachable_request_proxy.eq(0).all()
 return head,road,primary,samples,support,data,within,paired,audits,weight,checks

def baseline_closure_context(support,primary):
 folder=adverse.contexts('E20240830','Q5')
 run=next(z for z in read(folder/'run_manifest.json')['runs'] if z['stage']=='H')
 source=bind(folder/'exposure'/(run['case_id']+'__baseline.npz'))
 with np.load(source,allow_pickle=False) as z:
  qi=z['road_indices'];times=z['analysis_times'];actual=z['actual_times'];depth=z['lower'];states=z['road_state_codes'][:,0]
 with np.load(bind(adverse.FIVE/'clipped_road_lengths.npz'),allow_pickle=False) as z:length=z['union_length_m']
 assert len(times)==109 and times[0]==0 and times[-1]==32400 and np.max(np.abs(times-actual))<=.001
 assert np.array_equal(qi,np.flatnonzero(support['qualified_roads']))
 assert np.array_equal(states,np.where(depth>=.30,2,np.where(depth>=.10,1,0)))
 closed=depth>=.30;weights=length[qi]/1000;trajectory=closed@weights
 ledger=pd.read_parquet(bind(ROOT/'data/processed/three_ward_final_evidence_v1/road_effects.parquet'))
 row=ledger[ledger['Rainfall Parent'].eq('E20240830')&ledger['Receiving Water Stage'].eq('H')&ledger['Measure Configuration'].eq('Q5')].iloc[0]
 exposure=float(np.trapezoid(trajectory,times)/3600);assert abs(exposure-row['No Action Closed Road Duration'])<1e-8
 keep=closed.any(axis=0)&(length[qi]>0);indices=qi[keep]
 assert len(indices)==len(np.unique(indices)) and support['qualified_roads'][indices].all()
 selected=np.flatnonzero(keep)
 records=[]
 for j in selected:
  frames=np.flatnonzero(closed[:,j]);records.append({'Road Index':int(qi[j]),'Three-Ward Clipped Length m':float(length[qi[j]]),'Maximum Lower Depth m':float(depth[:,j].max()),'Closed Output Frames':int(len(frames)),'First Closed Frame':int(frames[0]),'Last Closed Frame':int(frames[-1])})
 union_overlap=[]
 crossings=primary[primary['Rainfall Parent'].eq('E20240830')&primary['Receiving Water Stage'].eq('H')&primary.Reference.eq('No action')]
 for _,z in crossings.iterrows():
  j=int(np.flatnonzero(qi==int(z['Road Index']))[0]);frame=int(z['Frame']);assert not closed[frame,j]
  union_overlap.append({'road_index':int(qi[j]),'frame':frame,'also_closed_in_baseline_at_other_frames':bool(closed[:,j].any()),'extra_depth_mm':float(z['Mapped Depth Increase mm'])})
 summary={'rainfall_parent':'E20240830','stage':'H','configuration':'none','depth_branch':'lower','threshold_m':.30,'frames':109,'window_hours':9,'any_time_unique_closed_roads':int(len(indices)),'any_time_unique_closed_length_km':float(weights[keep].sum()),'peak_simultaneous_closed_length_km':float(trajectory.max()),'closed_road_exposure_km_h':exposure,'source_exposure_reproduced':True,'same_support_and_clocks_verified':True,'primary_new_closure_baseline_overlap':union_overlap}
 return indices,pd.DataFrame(records),pd.DataFrame({'Frame':np.arange(109),'Elapsed Hours':times/3600,'Closed Road Length km':trajectory}),summary

def map_panel(ax,primary,samples,support):
 togeo=Transformer.from_crs(6677,4326,always_xy=True)
 wards=[transform(togeo.transform,shape(f['geometry'])) for f in read(adverse.FIVE/'wards_6677.geojson')['features']]
 features={int(f['properties']['road_index']):f for f in read(adverse.FIVE/'clipped_roads_6677.geojson')['features']}
 known=[];unknown=[]
 for i,f in features.items():
  lines=adverse.geometry_lines(transform(togeo.transform,shape(f['geometry'])))
  (known if support['qualified_roads'][i] else unknown).extend(lines)
 ax.add_collection(LineCollection(known,colors='#d8e1e7',linewidths=.26,zorder=1))
 ax.add_collection(LineCollection(unknown,colors='#98a6b0',linewidths=.3,linestyles=(0,(2,2)),alpha=.7,zorder=2))
 baseline_ids,baseline_register,baseline_trajectory,baseline_summary=baseline_closure_context(support,primary)
 baseline_lines=[]
 for i in baseline_ids:
  assert int(i) in features,'A positive-length baseline closed road lacks mapped geometry'
  baseline_lines.extend(adverse.geometry_lines(transform(togeo.transform,shape(features[int(i)]['geometry']))))
 ax.add_collection(LineCollection(baseline_lines,colors='#59636d',linewidths=1.0,zorder=4))
 baseline_register.to_csv(R/(SLUG+'_baseline_closed_road_context.csv'),index=False)
 baseline_trajectory.to_csv(R/(SLUG+'_baseline_closed_road_trajectory.csv'),index=False)

 for g in wards:
  for xy in adverse.geometry_lines(g):ax.plot(xy[:,0],xy[:,1],color='#4e626e',lw=.85,zorder=3)
 mapped=samples[samples.event.eq('E20240830')&samples.stage.eq('H')];map_counts={};road_locators=[]
 for ref,(_,color) in REFS.items():
  z=mapped[mapped.reference.eq(ref)].drop_duplicates('grid_index')
  x,y=togeo.transform(z.X.to_numpy(),z.Y.to_numpy())
  ax.scatter(x,y,s=72 if ref=='No action' else 31,marker='o' if ref=='No action' else 'D',facecolor='none',edgecolor=color,lw=1.2,zorder=6)
  ids=primary[primary['Rainfall Parent'].eq('E20240830')&primary['Receiving Water Stage'].eq('H')&primary.Reference.eq(ref)]['Road Index'].unique()
  segments=[]
  for i in ids:segments.extend(adverse.geometry_lines(transform(togeo.transform,shape(features[int(i)]['geometry']))))
  if segments:ax.add_collection(LineCollection(segments,colors=color,linewidths=2.4,zorder=7))
  for i in ids:
   geom=shape(features[int(i)]['geometry']);point=geom.interpolate(.5,normalized=True)
   assert geom.distance(point)<1e-7,'Road locator must lie on its original source geometry'
   px,py=togeo.transform(point.x,point.y)
   ax.scatter([px],[py],s=196,marker='s',facecolor='none',edgecolor=color,lw=1.6,zorder=9)
   road_locators.append({'reference':ref,'road_index':int(i),'longitude':float(px),'latitude':float(py),'source_midpoint_lies_on_road':True,'marker_size_pt2':196})

  map_counts[ref]={'head_cells':len(z),'newly_closed_road_ids':len(ids)}
 bounds=np.array([g.bounds for g in wards]);xmin,ymin=bounds[:,:2].min(0);xmax,ymax=bounds[:,2:].max(0)
 cx,cy=(xmin+xmax)/2,(ymin+ymax)/2;aspect=1/np.cos(np.deg2rad(cy));p=ax.get_position();fig=ax.figure
 ratio=p.height*fig.get_figheight()/(p.width*fig.get_figwidth())
 dx=(xmax-xmin)*1.02;dy=ratio*dx/aspect
 assert dy>=(ymax-ymin)*1.02,'Tight longitude must retain all ward latitudes'
 ax.set(xlim=(cx-dx/2,cx+dx/2),ylim=(cy-dy/2,cy+dy/2),xlabel='Longitude',ylabel='Latitude')
 longitude_ticks=np.arange(np.ceil((cx-dx/2)/.02)*.02,cx+dx/2+1e-10,.02)
 ax.set_aspect(aspect,adjustable='box');ax.set_xticks(longitude_ticks);ax.yaxis.set_major_locator(MaxNLocator(7))
 all_lines=known+unknown
 for xy in all_lines:
  assert (xy[:,0]>=cx-dx/2).all() and (xy[:,0]<=cx+dx/2).all() and (xy[:,1]>=cy-dy/2).all() and (xy[:,1]<=cy+dy/2).all(),'A contextual road would be cropped'
 mx,my=togeo.transform(mapped.X.to_numpy(),mapped.Y.to_numpy())
 assert np.all((mx>=cx-dx/2)&(mx<=cx+dx/2)&(my>=cy-dy/2)&(my<=cy+dy/2)),'An adverse location would be cropped'
 assert xmin>cx-dx/2 and xmax<cx+dx/2 and ymin>cy-dy/2 and ymax<cy+dy/2
 extent={'baseline_closure_context':baseline_summary,'road_location_markers':road_locators,'road_geometries_not_lengthened':True,'longitude_limits':list(ax.get_xlim()),'latitude_limits':list(ax.get_ylim()),'longitude_ticks':longitude_ticks.tolist(),'longitude_tick_interval_degrees':.02,'ward_bounds':[float(xmin),float(ymin),float(xmax),float(ymax)],'all_ward_boundaries_and_adverse_locations_visible':True,'all_context_road_vertices_visible':True,'total_longitude_margin_fraction':.02}

 ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f} E'));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f} N'))
 ax.tick_params(labelsize=9);ax.grid(color='#97a5ae',alpha=.5,lw=.45,zorder=4)
 handles=[Line2D([0],[0],color='#59636d',lw=1.0,label='No action: closed roads (any time)')];legend_encodings=[{'layer':'baseline_anytime_closed_roads','color':'#59636d','line':'solid','reference':'No action','locations':len(baseline_ids)}]
 for ref,(label,color) in REFS.items():
  marker='o' if ref=='No action' else 'D'
  h=Line2D([0],[0],ls='none',marker=marker,markerfacecolor='none',markeredgecolor=color,markeredgewidth=1.2,markersize=6.5,label='Water-level rise, '+label)
  assert h.get_marker()==marker and h.get_markeredgecolor()==color and h.get_markerfacecolor()=='none'
  handles.append(h);legend_encodings.append({'layer':'water_level_rise','reference':ref,'color':color,'marker':marker,'fill':'none','line':'none'})
 for ref,(label,color) in REFS.items():
  if map_counts[ref]['newly_closed_road_ids']==0:continue
  h=Line2D([0],[0],color=color,lw=2.4,marker='s',markerfacecolor='none',markeredgecolor=color,markeredgewidth=1.6,markersize=8,label=('New closure (threshold-sensitive), ' if ref=='No action' else 'New closure location, ')+label)
  assert h.get_color()==color and h.get_linestyle()=='-' and h.get_marker()=='s' and h.get_markerfacecolor()=='none'
  handles.append(h);legend_encodings.append({'layer':'newly_closed_road','reference':ref,'color':color,'line':'solid','marker':'s','fill':'none','marker_meaning':'enlarged_location_only','locations':map_counts[ref]['newly_closed_road_ids']})
 handles.append(Line2D([0],[0],color='#98a6b0',ls='--',label='Unknown road exposure'))
 extent['legend_encodings']=legend_encodings+[{'layer':'unknown_road','color':'#98a6b0','line':'dashed'}]
 legend=ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),frameon=True,facecolor='white',edgecolor='none',fontsize=9,handlelength=2.3,labelspacing=.30)
 return map_counts,legend,extent

def render(head,road,primary,samples,support,data):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.linewidth':.8,'savefig.facecolor':'white','axes.edgecolor':'#52616c','xtick.labelsize':9,'ytick.labelsize':9})
 fig=plt.figure(figsize=(14,12.8))
 positions={'a':[.15,.70,.2115,.26],'b':[.15,.375,.2115,.19],'c':[.15,.09,.2115,.18],'d':[.4765,.09,.4935,.87]}
 axes={k:fig.add_axes(v) for k,v in positions.items()};legends=[]
 groups=[(e,s) for e in adverse.EVENTS for s in ['H','L']]
 for key in ['a','b']:
  ax=axes[key];ax.set_ylim(-.65,5.65)
  ax.set_yticks(range(6),[adverse.date(e)+'  '+('High' if s=='H' else 'Low') for e,s in groups[::-1]])
  ax.tick_params(axis='y',labelsize=8.7);ax.grid(axis='x',color='#dce3e8',lw=.6);ax.set_axisbelow(True)
  for y in [1.5,3.5]:ax.axhline(y,color='#e3e8eb',lw=.6)
 for i,(event,stage) in enumerate(groups):
  for (ref,(_,color)),offset in zip(REFS.items(),[.14,-.14]):
   z=head[head.event.eq(event)&head.stage.eq(stage)&head.reference.eq(ref)].iloc[0];y=5-i+offset
   peak,union=float(z.peak_simultaneous_certain_head_rise_m2),float(z.any_time_certain_head_rise_union_m2)
   axes['a'].plot([peak,union],[y,y],color=color,lw=1,zorder=2)
   axes['a'].scatter([peak],[y],s=33,marker='o',facecolor='white',edgecolor=color,lw=1.2,zorder=3,clip_on=False)
   axes['a'].scatter([union],[y],s=19,marker='D',color=color,lw=.6,zorder=4,clip_on=False)
   z=road[road['Rainfall Parent'].eq(event)&road['Receiving Water Stage'].eq(stage)&road.Reference.eq(ref)&road['Road Depth Branch'].eq('lower')].iloc[0]
   peakroad=float(z['Peak Newly Closed Road Length m'])
   segment,=axes['b'].plot([0,peakroad],[y,y],color=color,lw=1.8,zorder=3,solid_capstyle='butt')
   np.testing.assert_array_equal(segment.get_xdata(),[0,peakroad])
   axes['b'].scatter([z['Peak Newly Closed Road Length m']],[y],s=33,color=color,edgecolor='white',lw=.5,zorder=4,clip_on=False)
 axes['a'].set(xlim=(0,1900),xlabel='Area with additional rise ≥0.10 m (m²)');axes['a'].set_xticks([0,500,1000,1500])
 axes['b'].set(xlim=(0,60),xlabel='Peak additional closed-road length (m)');axes['b'].set_xticks([0,15,30,45,60])
 refhandles=[Line2D([0],[0],color=c,lw=2,label=label) for label,c in REFS.values()]
 metric=[Line2D([0],[0],ls='none',marker='o',markerfacecolor='white',color='#586a75',label='Maximum simultaneous area'),Line2D([0],[0],ls='none',marker='D',color='#586a75',label='Any-time area union')]
 legends.append(axes['a'].legend(handles=refhandles+metric,loc='upper left',bbox_to_anchor=(-.015,-.18),ncol=1,frameon=False,fontsize=8.1,columnspacing=.85,handlelength=1.6))
 legends.append(axes['b'].legend(handles=refhandles,loc='upper left',bbox_to_anchor=(-.015,-.23),ncol=1,frameon=False,fontsize=8.1,columnspacing=.85,handlelength=1.6))
 traces=[]
 axes['c'].axhline(0,color='#7d8b96',ls='--',lw=1,zorder=1)
 for arm in ['Q2p5','Q5']:
  for stage,(_,style,marker) in STAGE.items():
   z=data[data['Receiving Water Stage'].eq(stage)&data['Measure Configuration'].eq(arm)].sort_values('Frame')
   line,=axes['c'].plot(z['Elapsed Hours'],z['Fixed Cohort Cost Reduction Percent'],color=traffic.COLORS[arm],ls=style,marker=marker,markersize=4.6,lw=1.6)
   np.testing.assert_array_equal(line.get_ydata(),z['Fixed Cohort Cost Reduction Percent'])
   traces.append({'panel':'c','stage':stage,'arm':arm,'y':z['Fixed Cohort Cost Reduction Percent'].tolist()})
 axes['c'].set(ylim=(-.1,.45),ylabel='Cost reduction (%)');axes['c'].set_yticks([-.1,0,.1,.2,.3,.4])
 costhandles=[Line2D([0],[0],color=traffic.COLORS[a],lw=1.7,label='Q2.5' if a=='Q2p5' else 'Q5') for a in ['Q2p5','Q5']]
 costhandles += [Line2D([0],[0],color='#52616c',ls=style,marker=marker,lw=1.5,label=label) for label,style,marker in STAGE.values()]
 legends.append(axes['c'].legend(handles=costhandles,loc='upper left',bbox_to_anchor=(-.015,-.25),ncol=2,frameon=False,fontsize=9,handlelength=2.3,columnspacing=1.6))
 for key in ['c']:
  ax=axes[key];ax.set(xlim=(2,9.5),xlabel='Elapsed model time (h:mm)');ax.set_xticks([29/12,4,6,9],['2:25','4:00','6:00','9:00']);ax.grid(color='#dce3e8',lw=.6);ax.set_axisbelow(True)
 mapcounts,maplegend,mapextent=map_panel(axes['d'],primary,samples,support)
 titles={'a':'a: Additional Water-Level Rise','b':'b: Additional Road Closures','c':'c: September Fixed-Cohort Cost','d':'d: August High Stage, Any-Time Locations'}
 headings=[]
 for key,ax in axes.items():
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
  headings.append(ax.text(0,1+.008/ax.get_position().height,titles[key],transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom'))
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 # Tight boxes include tick labels, headings, axis labels and plot-specific legends.
 allboxes=[]
 for key,ax in axes.items():
  box=ax.get_tightbbox(renderer);assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height,(key,box.bounds,fig.bbox.bounds)
  allboxes.append((key,box))
 for i,(key,a) in enumerate(allboxes):
  for other,b in allboxes[i+1:]:assert not a.overlaps(b),('Overlapping panel contents',key,other,a.bounds,b.bounds)
 for key in 'abc':np.testing.assert_allclose(axes[key].get_position().bounds[::2],[.15,.2115],atol=1e-10)
 leftshare=axes['a'].get_position().width/(axes['a'].get_position().width+axes['d'].get_position().width)
 assert np.isclose(leftshare,.3,rtol=0,atol=1e-12)
 assert abs(axes['d'].get_position().y1-axes['a'].get_position().y1)<1e-10
 assert abs(axes['d'].get_position().y0-axes['c'].get_position().y0)<1e-10
 assert all(ax.get_title()=='' for ax in axes.values())
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135)
 plotboxes={key:list(ax.get_position().bounds) for key,ax in axes.items()};plt.close(fig)
 with Image.open(OUT) as im:dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<x<451 for x in dpi)
 return {'dimensions':dimensions,'dpi':dpi,'panel_positions':plotboxes,'panel_contents_do_not_overlap':True,'left_panels_aligned':True,'plotting_width_ratio_left_to_right':[3,7],'left_plotting_width_share':float(leftshare),'inter_column_gap_excluded_from_ratio':True,'right_map_spans_all_three_rows':True,'map_geographic_aspect_preserved':True,'four_sided_map_frame':True,'geographic_graticules':True,'map_counts':mapcounts,'map_extent':mapextent,'map_is_any_time_union':True,'traces':traces,'legends_use_distinct_comparison_and_stage_encodings':True,'b_zero_to_peak_segments':12,'b_segments_match_peak_values':True,'b_line_glyph_matches_line_legend':True}

def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
 head,road,primary,samples,support,data,within,paired,audits,weight,checks=prepare()
 layout=render(head,road,primary,samples,support,data)
 pd.DataFrame(layout['map_extent']['road_location_markers']).to_csv(R/(SLUG+'_road_location_markers.csv'),index=False)
 for suffix,d in [('head_screens',head),('road_side_effects',road),('traffic_plot_data',data),('traffic_within_time_cohort',within),('traffic_paired_effects',paired),('traffic_fit_checks',audits),('supporting_threshold_crossings',primary),('map_head_samples',samples),('head_reconstruction_checks',pd.DataFrame(checks))]:d.to_csv(R/(SLUG+'_'+suffix+'.csv'),index=False)
 caption='''Transport benefits and local adverse effects under higher shared receiving allowance. Panels a and b cover three historical rainfall parents (2024-07-31, 2024-08-30 and 2025-09-11), each under two prescribed stages. Orange compares Q5 with no action and blue compares Q5 with Q2.5. Q2.5 and Q5 share 319 native intake/receiver locations, weighted allocation, eligibility and unit ceilings; only their five hypothetical receiving-pool caps differ, totaling 2.5 and 5.0 cubic meters per second. Panel a screens a conservative native water-surface elevation increase of at least 0.10 m. Open circles show maximum simultaneous area; diamonds show the union of any-time affected locations. These are distinct area summaries, not measured inundation extents or uncertainty intervals. Panel b shows maximum simultaneous unique road length newly crossing the unchanged 0.30 m closure rule on the primary lower-depth branch, not cumulative closure exposure or net road benefit. All twelve event-stage-reference contrasts, including zeros, are retained. Panel b horizontal segments extend from zero to the respective peak, with endpoint markers, matching the colored line legend; they do not connect different events or imply temporal trajectories. No action, Q2.5 and Q5 have identical represented request reachability at every September stage-time; this null result is retained in this caption and Vehicle Request and Common-Cohort Cost, with original denominator 37,113.1985; the standalone request-reachability curves are archived diagnostics. Panel c shows September-only modeled generalized-cost reduction at elapsed times 2:25, 4:00, 6:00 and 9:00 relative to same-time no action for one identical original-weighted OD cohort reachable in all twenty-four traffic states, with weight 26,825.4559. Blue represents Q2.5 and green Q5; line style and marker distinguish stage. Positive values mean lower modeled cost; negative values mean higher cost. Zero is the no-action reference. Benefits remain below 0.4 percent and do not establish large observed congestion relief. Cost excludes unserved requests and is not the average of the full original request population. Lines connect four snapshots without constructing intervening traffic states, dynamic queues or a nine-hour cost integral. The doubled 17:00 small-vehicle driver-request proxy and original capacity, reference-cost, restricted route-pool and mask assumptions are unchanged. Panel d maps the original August high-stage any-time head-rise locations and newly closed roads for both references. Neutral dark-gray lines show no-action roads closed at any of the same 109 output times under the unchanged lower-depth 0.30 m rule. This baseline location union gives total flood-closure context and is not a simultaneous closure state. Its unique road length, peak simultaneous length and integrated closure burden have distinct units and meanings. A road may be in this baseline union yet incur an additional same-time closure at a different output time. Its legend directly pairs orange open circles and blue open diamonds with the respective water-level-rise references, and the orange solid line with the newly closed road relative to no action; enlarged same-color hollow squares locate each short newly closed segment at its road-length midpoint. Real road geometries and lengths are preserved. Squares are position symbols, not affected footprints. The August high-stage diagnostic contains one threshold-sensitive newly closed road location relative to no action, with an approximately 0.049 millimeter mapped-depth increase at a single output time; the same road is already closed in the no-action scenario at other times. It has no newly closed road location relative to Q2.5; this zero result is retained here and in supporting data, while its empty map-legend entry is omitted. Gray dashed lines denote unknown road exposure. It is a spatial diagnostic from a different parent than c, not a simultaneous state or representative event. Enlarged location markers and road widths do not represent affected areas or corridor widths. Unknown road exposure remains gray dashed, not certified safe. Qualified road length is 836.958 km and unknown exposure 277.920 km. Original threshold-crossing records remain supporting material. No new hydraulic simulation, routing, fitting, population conversion or observed impact validation is introduced. Added receiving-river storage, downstream consequences, actual queues and independent observed road flooding remain unverified.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 methods='''# Transport Benefits and Local Adverse Effects

This approved assembly follows AnaSOP Sections 5-8 and decision MILI-D-20261024-040. It preserves the accepted standalone masters and their source receipts. Twelve native head contrasts are independently reconstructed from every original sparse sample and all 109 frames; twenty-four road contrasts on both depth branches are reconstructed from original arrays, clocks, length weights and closure rules. All twenty-four September saved traffic fits are checked without running an optimizer, and original request balances, paired effects, within-time cohorts and the fixed all-state OD cohort are reproduced. The new plotted values are checked against the accepted master exports within their original numerical comparison tolerances. No mask, raw/processed source, quality criterion, scenario or final concept is changed.

The revised approved layout is add / bdd / cdd, with three statistics stacked left and one full-height geographic map right. Statistical plotting width and map plotting width follow the approved 3:7 ratio, excluding the inter-column gap; geographic aspect is preserved. The first two legends stack in single columns to preserve their font size and avoid the map coordinates. Legends are plot-specific: adverse reference colors in a, b and d, and allowance colors and stage line styles in c. The original request-reachability panel is archived rather than generated as a separate publication figure. Identical measure-specific reachability remains verified from all twenty-four saved traffic states and reported in the main caption and Vehicle Request and Common-Cohort Cost. No gray subtitle line is added. Head peaks and unions retain different symbols. Panel b uses zero-to-peak horizontal segments with endpoint markers so that its plotted glyphs match the line legend; values, offsets and axis limits are unchanged. Request reachability curves collapse across measures only after equality and zero paired restoration/new-loss checks. All four traffic traces retain true snapshot spacing and a fixed cost cohort. The closure-margin records and generated diagnostic image remain archived for audit; Road Closure Threshold Sensitivity is excluded from the current publication figure set and has no pending supplementary rendering.

The map uses the same original August high-stage sparse head locations and newly closed road IDs as the accepted master. Its legend additionally identifies the neutral dark-gray no-action any-time closure context. The baseline IDs and geometry are reproduced from all 109 original exposure states on positive-length qualified support, with the full road-closure integral checked against the final ledger. The other entries match each actual layer-reference glyph directly: two hollow point shapes, the orange solid road line with its hollow square locator, and one gray dashed unknown-road key. Every locator is checked to lie at the normalized half-length point of the original projected road geometry; marker size is symbolic and no road geometry is stretched. A comparison with no newly closed locations has no map-legend entry; its zero result remains in the caption and supporting data. Generic gray point and solid-line proxies are removed. Its longitude span uses the full ward envelope plus two percent total padding, with 0.02-degree tick spacing; every ward boundary, mapped adverse location and background road vertex is checked to remain visible. Latitude limits follow the fixed map-frame ratio and true geographic aspect. Its coordinates are checked upstream against the original ward and common-admitted masks. Longitude/latitude transformation is display-only; geographic aspect, full three-ward coverage, graticules, complete borders and unknown road markings are retained. Layout checks include all axis/tick labels, headings and legends, with no overlapping panel tight bounding boxes. Export is English, 450 dpi. The existing publication artifacts are hash-checked after generation. Numerical reproduction is computational consistency, not field validation. This new assembly awaits individual human review.
'''
 (R/(SLUG+'_methods.md')).write_text(methods)
 assert all(sha(ROOT/n)==h for n,h in BIND.items()),'Read-only source changed during generation'
 originals=json.loads((R/'revisions/before_transport_adverse_merger_20261024/publication_hashes.json').read_text())
 assert all(sha(ROOT/n)==h for n,h in originals.items()),'An existing publication artifact changed'
 val={'passed':True,'decision':'MILI-D-20261024-040','head_contrasts_reconstructed':len(head),'road_contrasts_both_branches':len(road),'traffic_states_reconstructed':len(data),'original_fit_checks_verified':len(audits),'fixed_cohort_request_weight':float(weight),'original_request_denominator':float(data['Original Request Proxy'].iloc[0]),'zero_restored_or_newly_unreachable_requests':True,'primary_threshold_records_retained_as_supporting':len(primary),'request_reachability_retained_in_caption_and_table':True,'existing_publication_artifacts_preserved':len(originals),'accepted_master_values_reproduced':True,'source_files_unchanged':True,'original_masks_and_numerical_gates_unchanged':True,'head_road_parent_count':3,'traffic_parent_count':1,'traffic_frames':[29,48,72,108],'map_parent':'E20240830','map_stage':'H','max_simultaneous_head_rise_m2':int(head.peak_simultaneous_certain_head_rise_m2.max()),'max_any_time_head_rise_union_m2':int(head.any_time_certain_head_rise_union_m2.max()),'max_peak_newly_closed_road_m':float(road[road['Road Depth Branch'].eq('lower')]['Peak Newly Closed Road Length m'].max()),'new_simulations':0,'new_traffic_fits':0,'new_training':False,'sealed_test_access':False,'awaiting_human_review':True,'output_sha256':sha(OUT),**layout}
 dump(R/(SLUG+'_validation.json'),val);dump(R/(SLUG+'_source_bindings.json'),BIND)
 dump(R/(SLUG+'_completion.json'),{'status':'generated_pending_human_review','title':TITLE,'output':str(OUT.relative_to(ROOT)),'validation':str((R/(SLUG+'_validation.json')).relative_to(ROOT)),'output_sha256':sha(OUT)})
 body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transport Benefits and Local Adverse Effects</title><style>body{font:16px/1.7 system-ui;background:#f6f9fb;color:#24404f;max-width:1400px;margin:25px auto;padding:0 24px}img{width:100%;height:auto;background:white}aside{background:#e5eff4;padding:18px}a{color:#176a96}</style><h1>Transport Benefits and Local Adverse Effects</h1><aside>New assembly pending individual visual review. Panels a/b: three rainfall parents; panel c: September only, four snapshots; panel d: August high-stage any-time diagnostic. Existing qualified results, English annotations, 450 dpi.</aside>'''
 body+=f'<img src="../../../results/figures/Figure_{SLUG}.png?v={val["output_sha256"][:12]}" alt="Three stacked local adverse and cost panels, with a full-height August diagnostic map">'
 body+='<h2>Interpretation</h2><p>Request reachability is identical across measures at each sampled stage-time. Fixed-cohort modeled cost benefits remain below 0.4 percent and are negative at the first snapshot. Local adverse water-level and road screens are distinct from regional net benefits. The map is an any-time location union from August, whereas the traffic-cost panel shows September.</p><h2>Caption</h2><p>'+html.escape(caption)+'</p>'
 body+=f'<p><a href="../../../results/figures/Figure_{SLUG}.png">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Validation</a> · <a href="{SLUG}_methods.md">Methods</a> · <a href="{SLUG}_source_bindings.json">Source bindings</a> · <a href="september_vehicle_request_reachability_review.html">Archived request-reachability diagnostic</a> · <a href="neighboring_flood_transfer_review.html">Original adverse-transfer master</a> · <a href="vehicle_access_and_traffic_cost_review.html">Original traffic master</a></p></html>'
 (R/(SLUG+'_review.html')).write_text(body)
 print(json.dumps({k:val[k] for k in ['passed','dimensions','dpi','output_sha256','existing_publication_artifacts_preserved','awaiting_human_review']},indent=2),flush=True)
if __name__ == '__main__':
 main()
 from publication_review_state import refresh_review_page
 refresh_review_page(Path(__file__).resolve().parents[2], 'transport_benefits_and_local_adverse_effects')
