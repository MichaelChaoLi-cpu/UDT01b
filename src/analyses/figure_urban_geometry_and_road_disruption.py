#!/usr/bin/env python3
"""Urban Geometry and Road Disruption.
AnaSOP 5: distinct geometry and five-parent road evidence, not causal attribution.
AnaSOP 6: conservative peak bounds, simultaneous road thresholds, length-time integral.
AnaSOP 7-8: approved six-panel ab/cff/dff/eff assembly. Existing sources only.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
import rasterio
import seaborn as sns
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
B=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
C=ROOT/'data/processed/three_ward_formal_transport_experiments_v1/factor_comparison_v1'
SLUG='urban_geometry_and_road_disruption'
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'
EVENTS=['E20240618','E20240731','E20240830','E20250710','E20250911']
COLORS=['#4477aa','#ee7733','#228833','#aa3377','#cc3311']
DPI=450
BIND={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def dump(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def date(e):return f'{e[1:5]}-{e[5:7]}-{e[7:9]}'
def load(p,names):
 with np.load(bind(p),allow_pickle=False) as z:return {k:z[k].copy() for k in names}
def pieces(g):return list(g.geoms) if hasattr(g,'geoms') else [g]

def prepare():
 doc=bind(ROOT/'docs/AnaSOP.md').read_text()
 assert '| Urban Geometry and Road Disruption |' in doc and 'Approved six-panel assembly workflow' in doc
 for slug in ['geometry_and_water_level_sensitivity','road_interruption_and_recovery']:
  v=read(A/f'{slug}_validation.json');assert v['passed']
  assert sha(bind(ROOT/'data/results/figures'/f'Figure_{slug}.png'))==v['output_sha256']
 assert read(ROOT/'data/exp/estimation/three_ward_formal_transport_experiments_v1/B2_B3_forcing_geometry_comparison/validation.json')['all_passed']
 assert read(ROOT/'data/exp/estimation/three_ward_five_event_temporal_analysis_v1/A1_validation.json')['passed']
 support=load(C/'common_support.npz',['common_admitted','ward_code'])
 region=support['common_admitted']&(support['ward_code']>0);assert region.shape==(1012,770)
 contrasts={};bounds={}
 for stage in ['L','H']:
  for arm in ['T','U']:
   z=load(C/f'R1__{stage}__{arm}_peak_bounds.npz',['ground_lower','ground_upper'])
   assert np.isfinite(z['ground_lower'][region]).all() and np.isfinite(z['ground_upper'][region]).all()
   assert (z['ground_lower'][region]<=z['ground_upper'][region]).all();bounds[stage,arm]=z
  lo=bounds[stage,'U']['ground_lower']-bounds[stage,'T']['ground_upper']
  hi=bounds[stage,'U']['ground_upper']-bounds[stage,'T']['ground_lower']
  contrasts[stage]=np.where(lo>0,lo,np.where(hi<0,hi,0.))
  assert (lo[region]<=hi[region]).all()
 dem=ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif'
 dem_stat={'size':dem.stat().st_size,'mtime_ns':dem.stat().st_mtime_ns}
 with rasterio.open(dem) as ds:bb=ds.bounds;assert ds.shape==(20240,15400)
 to_geo=Transformer.from_crs(6677,4326,always_xy=True)
 to_metric=Transformer.from_crs(4326,6677,always_xy=True)
 xx,yy=np.meshgrid(np.linspace(bb.left,bb.right,region.shape[1]+1),np.linspace(bb.top,bb.bottom,region.shape[0]+1))
 lon,lat=to_geo.transform(xx,yy)
 extent=[float(lon.min()),float(lon.max()),float(lat.min()),float(lat.max())]
 aspect=1/np.cos(np.deg2rad((extent[2]+extent[3])/2))
 ex,ny=to_metric.transform(lon[::253,::192],lat[::253,::192])
 roundtrip=float(max(np.max(abs(ex-xx[::253,::192])),np.max(abs(ny-yy[::253,::192]))));assert roundtrip<1e-5
 wards=[shape(f['geometry']) for f in read(B/'wards_6677.geojson')['features']]
 full=pd.read_csv(bind(B/'road_trajectories.csv'))
 data=full[full.bound.eq('lower')&full.arm.eq('flood')&full.scope.eq('Three wards')].copy()
 summary=pd.read_csv(bind(B/'road_temporal_summary.csv'))
 summary=summary[summary.bound.eq('lower')&summary.arm.eq('flood')&summary.scope.eq('Three wards')]
 assert len(data)==1090 and len(summary)==10
 lengths=load(B/'clipped_road_lengths.npz',['qualified','union_length_m'])
 qualified=lengths['qualified'];weights=lengths['union_length_m'];known=float(weights[qualified].sum()/1000);unknown=float(weights[~qualified].sum()/1000)
 np.testing.assert_allclose(data.qualified_total_km,known,rtol=0,atol=1e-9)
 np.testing.assert_allclose(data.unknown_exposure_km,unknown,rtol=0,atol=1e-9)
 np.testing.assert_allclose(data.qualified_open_km+data.qualified_slow_km+data.qualified_closed_km,known,rtol=0,atol=1e-7)
 curves={};exposure=np.empty((5,2));checks=[]
 for ei,event in enumerate(EVENTS):
  for si,stage in enumerate(['L','H']):
   g=data[data.event_id.eq(event)&data.stage.eq(stage)].sort_values('frame')
   assert len(g)==109 and np.array_equal(g.frame,np.arange(109))
   times=g.analysis_elapsed_s.to_numpy();assert times[0]==0 and times[-1]==32400 and np.all(np.diff(times)>0)
   assert np.max(abs(times-g.actual_elapsed_s.to_numpy()))<=.001
   integral=float(np.trapezoid(g.qualified_closed_km,times)/3600)
   row=summary[summary['case'].eq(f'{event}__{stage}__U')];assert len(row)==1
   np.testing.assert_allclose(integral,row.closed_km_hours.iloc[0],rtol=0,atol=1e-8)
   curves[event,stage]=g;exposure[ei,si]=integral
   checks.append({'event':event,'stage':stage,'frames':109,'closure_exposure_km_h':integral,'peak_closed_km':float(g.qualified_closed_km.max())})
 refs=pd.read_csv(bind(B/'reference_states.csv'))
 anchor=refs[refs.event_id.eq('E20250911')&refs.selection_reasons.str.contains('H_noaction_closed_length_peak')]
 assert len(anchor)==1 and not bool(anchor.selected_using_intervention_benefit.iloc[0]);frame=int(anchor.frame.iloc[0]);assert frame==29
 z=load(B/'exposure/E20250911__H__U.npz',['qualified_road_indices','lower','upper','analysis_times'])
 indices=z['qualified_road_indices'];assert np.array_equal(indices,np.flatnonzero(qualified))
 assert np.isfinite(z['lower']).all() and (z['lower'][frame]<=z['upper'][frame]).all()
 states=np.where(z['lower']>=.30,2,np.where(z['lower']>=.10,1,0)).astype('u1')
 assert abs(float(z['analysis_times'][frame])-8700)<=.001
 for code,name in [(0,'qualified_open_km'),(1,'qualified_slow_km'),(2,'qualified_closed_km')]:
  reconstructed=(states==code)@(weights[indices]/1000)
  np.testing.assert_allclose(reconstructed,curves['E20250911','H'][name],rtol=0,atol=1e-7)
 original=read(A/'road_interruption_and_recovery_validation.json')
 np.testing.assert_allclose(exposure,original['panel_c_original_values'],rtol=0,atol=1e-8)
 road_state=np.full(len(weights),-1,dtype='i1');road_state[indices]=states[frame]
 features=read(B/'clipped_roads_6677.geojson')['features'];ids=np.array([int(f['properties']['road_index']) for f in features])
 assert len(np.unique(ids))==len(ids) and set(ids)==set(np.flatnonzero(weights>0))
 segments={code:[] for code in [-1,0,1,2]}
 for feature,idx in zip(features,ids):
  g=shape(feature['geometry']);assert abs(g.length-weights[idx])<1e-6
  parts=[g] if g.geom_type=='LineString' else list(g.geoms)
  for part in parts:
   xy=np.asarray(part.coords);x,y=to_geo.transform(xy[:,0],xy[:,1]);segments[int(road_state[idx])].append(np.column_stack([x,y]))
 return locals()

def render(z):
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.10)
 plt.rcParams.update({'axes.linewidth':.8,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
 width=14.;left=.085;right=.90;topwidth=.355;leftwidth=.235;mapleft=.43;mapwidth=.47
 ratio=(z['extent'][3]-z['extent'][2])/(z['extent'][1]-z['extent'][0])*z['aspect']
 lower_height=mapwidth*width*ratio;upper_height=topwidth*width*ratio
 footer=1.65;gap=1.05;header=.65;height=footer+lower_height+gap+upper_height+header
 fig=plt.figure(figsize=(width,height))
 bottom=footer/height;lh=lower_height/height;upper_bottom=(footer+lower_height+gap)/height;uh=upper_height/height
 axes={}
 axes['a']=fig.add_axes([left,upper_bottom,topwidth,uh]);axes['b']=fig.add_axes([right-topwidth,upper_bottom,topwidth,uh])
 rowgap=1.05/height;rowheight=(lh-2*rowgap)/3
 for i,k in enumerate(['e','d','c']):axes[k]=fig.add_axes([left,bottom+i*(rowheight+rowgap),leftwidth,rowheight])
 axes['f']=fig.add_axes([mapleft,bottom,mapwidth,lh])
 def mapstyle(ax):
  ax.set_xlim(z['extent'][:2]);ax.set_ylim(z['extent'][2:]);ax.set_aspect(z['aspect'],adjustable='box')
  ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
  ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2f}°E'));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:.2f}°N'))
  ax.tick_params(labelsize=9.5);ax.set_axisbelow('line');ax.grid(color='#75838b',lw=.55,linestyle=(0,(3,3)),alpha=.55)
  for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8);spine.set_color('#303030')
  for g in z['wards']:
   for part in pieces(g):
    x,y=part.exterior.xy;x,y=z['to_geo'].transform(x,y);ax.plot(x,y,color='#53616b',lw=.65,zorder=4)
 for key,stage in [('a','L'),('b','H')]:
  ax=axes[key];mesh=ax.pcolormesh(z['lon'],z['lat'],np.ma.masked_where(~z['region'],z['contrasts'][stage]),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.30,vcenter=0,vmax=.30),shading='flat',antialiased=False,rasterized=True)
  mapstyle(ax)
 cax=fig.add_axes([.918,upper_bottom+.04*uh,.014,.92*uh])
 cb=fig.colorbar(mesh,cax=cax,extend='both',ticks=[-.3,-.2,-.1,0,.1,.2,.3]);cb.set_label('Urban minus terrain peak-depth contrast (m)',labelpad=9);cb.ax.tick_params(labelsize=9)
 ymax=float(np.ceil(z['data'].qualified_closed_km.max()/25)*25)
 for key,stage in [('c','L'),('d','H')]:
  ax=axes[key]
  for event,color in zip(EVENTS,COLORS):
   g=z['curves'][event,stage];ax.plot(g.analysis_elapsed_s/3600,g.qualified_closed_km,color=color,lw=1.65)
  ax.set(xlim=(0,9),ylim=(0,ymax),xlabel='Elapsed model time (h)',ylabel='Closed road length (km)');ax.set_xticks([0,3,6,9]);ax.set_yticks([0,50,100,150,200]);ax.grid(axis='y',color='#dce2e6',lw=.6);ax.set_axisbelow(True)
 ax=axes['e'];ypos=np.arange(5)[::-1];ex=z['exposure']
 ax.hlines(ypos,ex[:,0],ex[:,1],color='#8a98a1',lw=1.6,zorder=2)
 ax.scatter(ex[:,0],ypos,color='#4477aa',s=33,zorder=3);ax.scatter(ex[:,1],ypos,color='#ee7733',s=33,zorder=3)
 for i,y in enumerate(ypos):
  ax.annotate(f'{ex[i,0]:.1f}',(ex[i,0],y),xytext=(0,-11),textcoords='offset points',ha='center',va='top',fontsize=8.8,color='#345f88')
  ax.annotate(f'{ex[i,1]:.1f}',(ex[i,1],y),xytext=(0,8),textcoords='offset points',ha='center',va='bottom',fontsize=8.8,color='#b95a24')
 ax.set_yticks(ypos,[date(e) for e in EVENTS],fontsize=9)
 ax.set(xlabel='Closure exposure (km h)',ylabel='Rainfall event',xlim=(0,np.ceil(ex.max()*1.08/50)*50),ylim=(-.85,4.85));ax.grid(axis='x',color='#dce2e6',lw=.6);ax.set_axisbelow(True)
 rc={0:'#b8c8d4',1:'#ee7733',2:'#bb2233',-1:'#6e7478'};rw={0:.32,1:.65,2:.8,-1:.35}
 ax=axes['f']
 for code in [0,-1,1,2]:ax.add_collection(LineCollection(z['segments'][code],colors=rc[code],linewidths=rw[code],linestyles='dashed' if code==-1 else 'solid',rasterized=True,zorder=3 if code in (1,2) else 2))
 mapstyle(ax)
 labels={'a':'a: Low stage','b':'b: Raised stage','c':'c: Low stage','d':'d: Raised stage','e':'e: Nine-hour closure exposure','f':'f: 2025-09-11, raised stage'}
 for key,ax in axes.items():
  ax.annotate(labels[key],(0,1),xycoords='axes fraction',xytext=(-24,13 if key!='f' else 27),textcoords='offset points',fontsize=12.5,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
 axes['f'].annotate('Elapsed 2:25',(0,1),xycoords='axes fraction',xytext=(-24,9),textcoords='offset points',fontsize=10.5,ha='left',va='bottom',annotation_clip=False)
 # The three keys share the bottom band, each aligned to its relevant block.
 events=[Line2D([0],[0],color=c,lw=1.8,label=date(e)) for e,c in zip(EVENTS,COLORS)]
 eventlegend=fig.legend(handles=events,loc='lower center',bbox_to_anchor=(.51,.12/height),ncol=5,frameon=False,fontsize=10,handlelength=2,columnspacing=1.8)
 stages=[Line2D([0],[0],marker='o',linestyle='none',color=c,label=l) for c,l in [('#4477aa','Low stage'),('#ee7733','Raised stage')]]
 stagelegend=fig.legend(handles=stages,loc='upper center',bbox_to_anchor=(left+leftwidth/2,1.12/height),ncol=2,frameon=False,fontsize=9,handletextpad=.45,columnspacing=1.2)
 roadlabels={0:'Open by rule',1:'Restricted by rule',2:'Closed by rule',-1:'Unknown exposure'}
 roadhandles=[Line2D([0],[0],color=rc[c],lw=1.5,ls='--' if c==-1 else '-',label=roadlabels[c]) for c in [0,1,2,-1]]
 roadlegend=fig.legend(handles=roadhandles,loc='upper center',bbox_to_anchor=(mapleft+mapwidth/2,1.12/height),ncol=2,frameon=False,fontsize=9.3,columnspacing=2)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes={k:ax.get_position() for k,ax in axes.items()}
 align={'a_b_top':abs(boxes['a'].y1-boxes['b'].y1),'a_b_bottom':abs(boxes['a'].y0-boxes['b'].y0),'c_f_top':abs(boxes['c'].y1-boxes['f'].y1),'e_f_bottom':abs(boxes['e'].y0-boxes['f'].y0),'c_d_left':abs(boxes['c'].x0-boxes['d'].x0),'c_e_right':abs(boxes['c'].x1-boxes['e'].x1),'f_double_left_width':abs(boxes['f'].width-2*boxes['c'].width)}
 assert max(align.values())<1e-9,align
 assert fig._suptitle is None and all(not ax.get_title() for ax in axes.values())
 # Distinct data axes may not overlap another panel's labels or legend boxes.
 for k,ax in axes.items():
  for j,other in axes.items():
   if k!=j:assert not ax.get_tightbbox(renderer).overlaps(other.get_tightbbox(renderer)),(k,j,'panel-label collision')
 for ax in list(axes.values())+[cax]:
  tb=ax.get_tightbbox(renderer);assert tb.x0>=0 and tb.y0>=0 and tb.x1<=fig.bbox.width and tb.y1<=fig.bbox.height,('clipped axis text',tb.bounds)
 legends=[eventlegend,stagelegend,roadlegend]
 lbs=[l.get_window_extent(renderer) for l in legends]
 for i,lb in enumerate(lbs):
  for other in lbs[i+1:]:assert not lb.overlaps(other),'legend collision'
  assert lb.x0>=0 and lb.y0>=0 and lb.x1<=fig.bbox.width and lb.y1<=fig.bbox.height
  assert all(not lb.overlaps(ax.get_tightbbox(renderer)) for ax in axes.values()),'legend/panel collision'
 OUT.parent.mkdir(exist_ok=True,parents=True);A.mkdir(exist_ok=True,parents=True)
 fig.savefig(OUT,dpi=DPI,facecolor='white');fig.savefig(A/f'{SLUG}_preview.png',dpi=120,facecolor='white');plt.close(fig)
 with Image.open(OUT) as im:dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<=d<=451 for d in dpi)
 return {'figure_inches':[width,height],'dimensions_px':dimensions,'dpi':dpi,'panel_positions':{k:list(b.bounds) for k,b in boxes.items()},'alignment_error_figure_fraction':align,'map_geographic_aspect_preserved':True,'identical_map_extent':True,'shared_event_legend':True,'road_map_elapsed_annotation':'2:25','no_panel_or_global_titles':True,'legend_collisions':False,'panel_label_collisions':False,'layout':['ab','cff','dff','eff']}

def main():
 existing={str(p.relative_to(ROOT)):sha(p) for folder in ['data/results/figures','data/results/tables'] for p in (ROOT/folder).iterdir() if p.is_file() and p!=OUT}
 z=prepare();layout=render(z)
 assert all(sha(ROOT/p)==h for p,h in BIND.items()),'Source changed'
 assert all(sha(ROOT/p)==h for p,h in existing.items()),'Existing publication output changed'
 dem=z['dem'];assert {'size':dem.stat().st_size,'mtime_ns':dem.stat().st_mtime_ns}==z['dem_stat']
 z['data'].to_csv(A/f'{SLUG}_road_plot_data.csv',index=False)
 pd.DataFrame(z['checks']).to_csv(A/f'{SLUG}_exposure_plot_data.csv',index=False)
 validation={'passed':True,'status':'generated_awaiting_human_review','decision':'MILI-D-20261024-019','panels':6,'layout':layout,'geometry_cases_displayed':4,'geometry_parent':'2024-08-21 reference family','road_rainfall_parents':5,'road_event_stage_cases':10,'road_trajectory_rows':1090,'trajectory_frames_per_case':109,'qualified_road_length_km':z['known'],'unknown_road_length_km':z['unknown'],'map_frame':29,'map_elapsed_seconds':8700,'all109_anchor_case_road_states_reconstructed':True,'original10_exposure_values_identical':True,'coordinate_roundtrip_error_m':z['roundtrip'],'geometry_data_resampled':False,'geometry_and_road_experiments_kept_distinct':True,'original_publication_files_unchanged':True,'source_files_unchanged':True,'new_solver_runs':0,'new_routing':0,'training':False,'sealed_test_access':False,'output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT),'script_sha256':sha(Path(__file__)),'source_sha256':BIND,'case_checks':z['checks']}
 dump(A/f'{SLUG}_validation.json',validation);dump(A/f'{SLUG}_source_bindings.json',BIND)
 caption='''Urban geometry sensitivity and road disruption under prescribed rainfall and receiving-water scenarios. Panels a and b retain low- and raised-stage conservative urban-minus-terrain differences between cellwise peak-depth envelopes from the original 2024-08-21 reference-rain geometry family. Red denotes a strictly positive interval contrast, blue a strictly negative contrast, and neutral color includes overlapping numerical bounds. The scale saturates at ±0.30 m; these are not simultaneous flood maps. Panels c and d show lower-depth qualified closed-road length at 109 simultaneous five-minute states for five historical radar parents, under low and raised stages. The shared date legend applies to c and d. Panel e pairs the same nine-hour cumulative exposures by stage in kilometer-hours; these are not unique road kilometers. Panel f is the frozen September raised-stage no-action peak-closure anchor, frame 29, elapsed 2:25. Open by rule means mapped lower depth below 0.10 m, restricted at least 0.10 m but below 0.30 m, and closed at least 0.30 m; structural or hazard unknowns remain separately mapped. The 836.958 km qualified road denominator and 277.920 km unknown exposure are unchanged. The geometry maps and five-parent road analysis are distinct experimental blocks, not a same-event geometry-to-road factorial or evidence of observed accuracy superiority. All histories start dry and retain assumed receiving-water stages. Closure persists at the nine-hour endpoint, so full recovery is unobserved. No new simulation, routing, fitting, training or source adjustment is performed.'''
 (A/f'{SLUG}_caption.md').write_text(caption+'\n')
 (A/f'{SLUG}_methods.md').write_text('''# Urban Geometry and Road Disruption

This six-panel assembly follows the user-approved ab / cff / dff / eff structure in AnaSOP Sections 5-8. It reuses the original conservative interval difference of peak ground-depth envelopes, original lower-depth road thresholds, union-clipped length weights, approved analysis clocks and trapezoidal length-time integrals. The full-domain 10 m reference and geometry source masks are preserved. All ten integrated road histories reproduce the original accepted exposure values. All 109 September road-state length vectors are independently reconstructed from mapped depth bounds; clipped vector lengths reproduce their saved weights. Geographic cell corners and road vectors are transformed from EPSG:6677 to EPSG:4326; no scientific values are resampled. All three maps share extent and geographic aspect. A dedicated colorbar prevents asymmetric map resizing. The right map has twice the lower-left plotting width; its top aligns with c and bottom with e. The top pair has identical plotting dimensions. Separate bottom legends identify rainfall events, exposure stages and road states. The image uses English and 450 dpi. The original accepted figures and workbooks remain byte-identical. Generation does not confer human acceptance or update the new figure to done.
''')
 html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Urban Geometry and Road Disruption</title><style>body{{font:16px/1.65 system-ui;max-width:1300px;margin:32px auto;padding:0 20px;color:#203b4b;background:#f7f9fb}}img{{width:100%;height:auto;background:white}}aside{{background:#e6eef5;padding:18px}}a{{color:#176a96}}</style><h1>Urban Geometry and Road Disruption</h1><aside>Six-panel assembly awaiting visual acceptance. Layout: ab / cff / dff / eff. English labels; 450 dpi. Original images and numerical rules preserved.</aside><img src="../../../results/figures/Figure_{SLUG}.png?v={validation['output_sha256'][:12]}" alt="Two geometry difference maps above three stacked road plots and one large road-state map"><h2>Caption and scope</h2><p>{caption}</p><p><a href="../../../results/figures/Figure_{SLUG}.png">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Numerical and layout checks</a> · <a href="{SLUG}_methods.md">Methods</a> · <a href="{SLUG}_road_plot_data.csv">Road plot data</a></p></html>'''
 (A/f'{SLUG}_review.html').write_text(html)
 print('GENERATED',OUT,flush=True);print(json.dumps({'passed':True,'layout':layout,'qualified_km':z['known'],'unknown_km':z['unknown'],'existing_outputs_preserved':len(existing)},indent=2),flush=True)
if __name__ == '__main__':
 main()
 from publication_review_state import refresh_review_page
 refresh_review_page(Path(__file__).resolve().parents[2], 'urban_geometry_and_road_disruption')
