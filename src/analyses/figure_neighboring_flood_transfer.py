#!/usr/bin/env python3
"""Neighboring Flood Transfer: qualified adverse-change screens, no new models.
AnaSOP5: same parent/stage/reference/support, frozen S1R allowances.
AnaSOP6: head-rise peak vs union, paired new closure, threshold margins.
AnaSOP7-8: whole-area adverse diagnostics and unknown coverage, four panels.
"""
from pathlib import Path
import os,json,hashlib,html
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
import seaborn as sns
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_final_evidence_v1'
H=ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1'
FIVE=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
SLUG='neighboring_flood_transfer';OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')
EVENTS=['E20240731','E20240830','E20250911']
REFS={'No action':('Q5 vs no action','#cc6633'),'S1R-Q2.5':('Q5 vs Q2.5','#4477aa')}
BIND={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(p):p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def csv(p):return pd.read_csv(bind(p))
def load(p):
 with np.load(bind(p),allow_pickle=False)as z:return dict(z)
def dump(p,z):p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def date(e):return f'{e[1:5]}-{e[5:7]}-{e[7:9]}'
def contexts(e,q):
 if e=='E20240731':return ROOT/'data/processed'/('three_ward_s1_reallocation_pilot_v1'if q=='Q2p5'else'three_ward_s1r_shared_allowance_pilot_v1')/'batches/O1'
 return H/'batches'/(e+'_'+q)
def geometry_lines(g):
 if g.geom_type=='LineString':return [np.asarray(g.coords)]
 if g.geom_type=='MultiLineString':return [np.asarray(z.coords)for z in g.geoms]
 if g.geom_type=='Polygon':return [np.asarray(g.exterior.coords)]
 if g.geom_type=='MultiPolygon':return [np.asarray(z.exterior.coords)for z in g.geoms]
 raise ValueError(g.geom_type)
def prepare():
 assert read(ROOT/'data/exp/estimation'/H.name/'final_summary_validation.json')['passed']
 assert read(ROOT/'data/exp/research-design/three_ward_final_evidence_contract_v1/validation.json')['passed']
 assert read(R/'resident_access_loss_and_restoration_acceptance.json')['status']=='human_accepted'
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();assert '| Neighboring Flood Transfer |'in doc
 clip=load(FIVE/'clipped_road_lengths.npz');length=clip['union_length_m'];head=pd.read_parquet(bind(P/'adverse_head_changes.parquet'));roadsummary=pd.read_parquet(bind(P/'three_parent_road_comparison_preprocessed.parquet'))
 # Article-facing normalized exports are the primary summaries. Native states
 # independently reproduce all road clocks, thresholds and masks below.
 oldroad=csv(H/'three_parent_road_comparison.csv')
 assert len(head)==12 and len(roadsummary)==len(oldroad)==96
 samples=[];headchecks=[];roadrows=[];crossings=[];support=None
 for event in EVENTS:
  cp=H/'comparisons'/event;rr=ROOT/'data/exp/estimation'/H.name/'comparisons'/event
  assert read(rr/'validation.json')['passed']
  hs=csv(cp/'adverse_head_samples.csv');ht=csv(cp/'head_transfer_trajectories.csv');hs['event']=event;samples.append(hs)
  assert not hs.duplicated(['stage','reference','frame','grid_index']).any()
  for stage in ['H','L']:
   for ref in REFS:
    h=hs[hs.stage.eq(stage)&hs.reference.eq(ref)];z=head[head.event.eq(event)&head.stage.eq(stage)&head.reference.eq(ref)];assert len(z)==1;z=z.iloc[0]
    counts=np.bincount(h.frame.to_numpy(dtype=int),minlength=109)*100
    traj=ht[ht.stage.eq(stage)&ht.reference.eq(ref)&ht.scope.eq('Three wards')].sort_values('frame');assert len(traj)==109 and np.array_equal(traj.frame.to_numpy(),np.arange(109))
    assert np.array_equal(counts,traj.certain_head_rise_m2.to_numpy())
    assert counts.max()==z.peak_simultaneous_certain_head_rise_m2 and h.grid_index.nunique()*100==z.any_time_certain_head_rise_union_m2
    assert z.any_time_certain_head_rise_union_m2>=z.peak_simultaneous_certain_head_rise_m2
    assert h.certain_head_rise_m.ge(.10).all()and h.cell_area_m2.eq(100).all()and h.ward_code.gt(0).all()
    headchecks.append({'Rainfall Parent':event,'Receiving Water Stage':stage,'Reference':ref,'Peak Area':int(counts.max()),'Union Area':int(h.grid_index.nunique()*100),'source_sparse_samples_reproduce_all109frames':True})
  hp,qp=contexts(event,'Q5'),contexts(event,'Q2p5');runs=read(hp/'run_manifest.json')['runs'];qruns=read(qp/'run_manifest.json')['runs']
  ss=load(hp/'common_road_support.npz')
  if support is None:support=ss
  else:assert np.array_equal(support['qualified_roads'],ss['qualified_roads'])and np.array_equal(support['common_admitted'],ss['common_admitted'])
  for run in runs:
   stage=run['stage'];case=run['case_id'];qrun=next(x for x in qruns if x['stage']==stage)
   target=load(hp/'exposure'/(case+'__drainage.npz'))
   references={'No action':load(hp/'exposure'/(case+'__baseline.npz')),'S1R-Q2.5':load(qp/'exposure'/(qrun['case_id']+'__drainage.npz'))}
   qi=target['road_indices'];t=target['analysis_times'];assert len(qi)==28064 and np.array_equal(qi,np.flatnonzero(support['qualified_roads']))and len(t)==109 and t[0]==0 and t[-1]==32400
   for ref,z in references.items():
    assert np.array_equal(z['road_indices'],qi)and np.array_equal(z['analysis_times'],t)and np.max(abs(z['actual_times']-target['actual_times']))<=.001
    for bi,bound in enumerate(['lower','upper']):
     a,b=z[bound],target[bound];assert np.array_equal(z['road_state_codes'][:,bi],np.where(a>=.30,2,np.where(a>=.10,1,0)))and np.array_equal(target['road_state_codes'][:,bi],np.where(b>=.30,2,np.where(b>=.10,1,0)))
     new=(a<.30)&(b>=.30);w=length[qi]/1000;newlength=new@w;exposure=float(np.trapezoid(newlength,t)/3600)
     prior=oldroad[oldroad.event.eq(event)&oldroad.stage.eq(stage)&oldroad.reference.eq(ref)&oldroad.branch.eq(bound)&oldroad.scope.eq('Three wards')];assert len(prior)==1;prior=prior.iloc[0]
     assert abs(newlength.max()-prior.peak_newly_closed_km)<1e-8 and abs(exposure-prior.newly_closed_km_h)<1e-8
     roadrows.append({'Rainfall Parent':event,'Receiving Water Stage':stage,'Reference':ref,'Road Depth Branch':bound,'Peak Newly Closed Road Length m':float(newlength.max()*1000),'Newly Closed Road Exposure km h':exposure,'Any-Time Newly Closed Unique Length m':float(length[qi[new.any(axis=0)]].sum()),'Frames with Positive-Length New Closure':int((newlength>0).sum()),'Qualified Road Length km':float(w.sum()),'Unknown Road Length km':float(length[~support['qualified_roads']].sum()/1000)})
     # Keep all road-state records in the register; zero clipped length stays
     # context-only and is excluded from the primary scatter and ward totals.
     for ti,j in np.argwhere(new):
      margin=(.30-float(a[ti,j]))*1000;increase=(float(b[ti,j])-float(a[ti,j]))*1000
      assert increase+1e-10>=margin>0
      crossings.append({'Rainfall Parent':event,'Receiving Water Stage':stage,'Reference':ref,'Road Depth Branch':bound,'Frame':int(ti),'Elapsed Hours':float(t[ti]/3600),'Road Index':int(qi[j]),'Three-Ward Clipped Length m':float(length[qi[j]]),'Reference Road Depth m':float(a[ti,j]),'Q5 Road Depth m':float(b[ti,j]),'Margin Below Closure mm':margin,'Mapped Depth Increase mm':increase,'Primary Positive-Length Record':bool(bound=='lower'and length[qi[j]]>0),'Threshold Sensitive Increment Up To10mm':bool(increase<=10)})
 samples=pd.concat(samples,ignore_index=True);road=pd.DataFrame(roadrows);cross=pd.DataFrame(crossings);main=cross[cross['Primary Positive-Length Record']].copy();assert len(road)==24 and len(headchecks)==12 and len(main)>0
 grid=read(ROOT/'data/processed/three_ward_surrogate_inputs/input_schema.json')['grid'];ward=load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz')['Ward Code'].ravel()
 ix=samples.grid_index.to_numpy(dtype=int);assert np.array_equal(ward[ix],samples.ward_code.to_numpy(dtype=int));assert support['common_admitted'].ravel()[ix].all()
 row,col=np.divmod(ix,grid['shape'][1]);samples['X']=grid['left']+(col+.5)*grid['dx_m'];samples['Y']=grid['top']-(row+.5)*grid['dx_m']
 return head,road,cross,main,samples,headchecks,support

def render(head,road,cross,main,samples,support):
 togeo=Transformer.from_crs(6677,4326,always_xy=True)
 wards=[transform(togeo.transform,shape(f['geometry']))for f in read(FIVE/'wards_6677.geojson')['features']]
 raw=read(FIVE/'clipped_roads_6677.geojson')['features'];features={int(f['properties']['road_index']):f for f in raw};known=[];unknown=[]
 for i,f in features.items():
  arr=geometry_lines(transform(togeo.transform,shape(f['geometry'])));(known if support['qualified_roads'][i]else unknown).extend(arr)
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.05)
 plt.rcParams.update({'axes.linewidth':.8,'savefig.facecolor':'white'})
 fig=plt.figure(figsize=(14,12));xs=[.105,.59];width=.355
 axes=[fig.add_axes([x,.625,width,.285])for x in xs]+[fig.add_axes([x,.16,width,.385])for x in xs]
 axa,axb,axc,axd=axes
 groups=[(e,s)for e in EVENTS for s in ['H','L']];ybase={k:5-i for i,k in enumerate(groups)}
 for ax in [axa,axb]:
  ax.set_ylim(-.65,5.65);ax.set_yticks(range(6),[date(e)+' '+('High'if s=='H'else'Low')for e,s in groups[::-1]]);ax.tick_params(labelsize=8.3);ax.set_axisbelow(True);ax.grid(axis='x',color='#dce3e8',lw=.6)
  for y in [1.5,3.5]:ax.axhline(y,color='#e3e8eb',lw=.6)
 for (event,stage),yc in ybase.items():
  for(ref,(label,color)),offset in zip(REFS.items(),[.14,-.14]):
   z=head[head.event.eq(event)&head.stage.eq(stage)&head.reference.eq(ref)].iloc[0];y=yc+offset
   peak,union=float(z.peak_simultaneous_certain_head_rise_m2),float(z.any_time_certain_head_rise_union_m2)
   axa.plot([peak,union],[y,y],color=color,lw=1,zorder=2)
   axa.scatter([peak],[y],s=30,marker='o',facecolor='white',edgecolor=color,lw=1.2,zorder=3,clip_on=False)
   axa.scatter([union],[y],s=18,marker='D',facecolor=color,edgecolor=color,lw=.6,zorder=4,clip_on=False)
   r=road[road['Rainfall Parent'].eq(event)&road['Receiving Water Stage'].eq(stage)&road.Reference.eq(ref)&road['Road Depth Branch'].eq('lower')].iloc[0]
   axb.scatter([r['Peak Newly Closed Road Length m']],[y],s=30,marker='o',facecolor=color,edgecolor='white',lw=.5,zorder=4,clip_on=False)
 axa.set(xlim=(0,1900),xlabel='Screened water-level-rise area (m\N{SUPERSCRIPT TWO})');axa.set_xticks([0,500,1000,1500])
 axb.set(xlim=(0,60),xlabel='Peak simultaneous newly closed road length (m)');axb.set_xticks([0,15,30,45,60])
 marks={'E20240731':'s','E20240830':'o','E20250911':'^'}
 for ref,(_,color)in REFS.items():
  for e in EVENTS:
   z=main[main.Reference.eq(ref)&main['Rainfall Parent'].eq(e)]
   if len(z):axc.scatter(z['Margin Below Closure mm'],z['Mapped Depth Increase mm'],s=52,marker=marks[e],c=color,alpha=.8,edgecolor='white',lw=.6,zorder=3)
 xlow=10**np.floor(np.log10(main['Margin Below Closure mm'].min()));ylow=10**np.floor(np.log10(main['Mapped Depth Increase mm'].min()))
 line=np.array([min(xlow,ylow),2.]);axc.plot(line,line,ls='--',color='#74848e',lw=1,zorder=1)
 axc.set(xscale='log',yscale='log',xlim=(xlow,1.3),ylim=(ylow,2.4),xlabel='Reference margin below closure (mm)',ylabel='Increase in mapped road depth (mm)');axc.grid(which='major',color='#dce3e8',lw=.6);axc.tick_params(labelsize=9)
 assert main['Margin Below Closure mm'].between(*axc.get_xlim()).all()and main['Mapped Depth Increase mm'].between(*axc.get_ylim()).all(),'Every positive-length crossing must be visible'
 axd.add_collection(LineCollection(known,colors='#d8e1e7',linewidths=.25,zorder=1));axd.add_collection(LineCollection(unknown,colors='#98a6b0',linewidths=.28,linestyles=(0,(2,2)),alpha=.7,zorder=2))
 for g in wards:
  for xy in geometry_lines(g):axd.plot(xy[:,0],xy[:,1],color='#4e626e',lw=.8,zorder=3)
 # Disclosed August high-stage diagnostic map. Any-time locations, not a
 # simultaneous state; symbols are enlarged to remain visible at ward scale.
 mapped=samples[samples.event.eq('E20240830')&samples.stage.eq('H')]
 for ref,(_,color)in REFS.items():
  z=mapped[mapped.reference.eq(ref)].drop_duplicates('grid_index');x,y=togeo.transform(z.X.to_numpy(),z.Y.to_numpy());axd.scatter(x,y,s=64 if ref=='No action'else 27,marker='o'if ref=='No action'else'D',facecolor='none',edgecolor=color,lw=1.1,zorder=6)
  ids=main[main['Rainfall Parent'].eq('E20240830')&main['Receiving Water Stage'].eq('H')&main.Reference.eq(ref)]['Road Index'].unique()
  seg=[]
  for i in ids:assert int(i)in features;seg.extend(geometry_lines(transform(togeo.transform,shape(features[int(i)]['geometry']))))
  if seg:axd.add_collection(LineCollection(seg,colors=color,linewidths=2.1,zorder=7))
 bounds=np.array([g.bounds for g in wards]);xmin,ymin=bounds[:,:2].min(0);xmax,ymax=bounds[:,2:].max(0);cx,cy=(xmin+xmax)/2,(ymin+ymax)/2;aspect=1/np.cos(np.deg2rad(cy));ratio=(.385*12)/(width*14);dx=max((xmax-xmin)*1.08,(ymax-ymin)*1.08*aspect/ratio);dy=ratio*dx/aspect
 axd.set(xlim=(cx-dx/2,cx+dx/2),ylim=(cy-dy/2,cy+dy/2),xlabel='Longitude',ylabel='Latitude');axd.set_aspect(aspect,adjustable='box');axd.xaxis.set_major_locator(MaxNLocator(4));axd.yaxis.set_major_locator(MaxNLocator(5));axd.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f} E'));axd.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.2f} N'));axd.tick_params(labelsize=8.5);axd.grid(color='#97a5ae',alpha=.5,lw=.45,zorder=4)
 labels=['a: Water-level-rise screen','b: New road closures','c: Distance to the closure threshold','d: August high stage, any-time locations']
 for ax,label in zip(axes,labels):
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
  ax.text(0,1.05,label,transform=ax.transAxes,fontsize=11.2,fontweight='bold',va='bottom')
 handles=[Line2D([0],[0],color=c,lw=2,label=label)for label,c in REFS.values()]
 handles += [Line2D([0],[0],ls='none',marker='o',markerfacecolor='white',color='#586a75',label='a: simultaneous peak'),Line2D([0],[0],ls='none',marker='D',color='#586a75',label='a: any-time union')]
 for event in EVENTS:
  if main['Rainfall Parent'].eq(event).any():handles.append(Line2D([0],[0],ls='none',marker=marks[event],color='#586a75',label='c: '+date(event)))
 handles += [Line2D([0],[0],color='#74848e',ls='--',label='c: closure-crossing line'),Line2D([0],[0],ls='none',marker='o',markerfacecolor='none',color='#586a75',label='d: head-rise cell location'),Line2D([0],[0],color='#586a75',lw=2.1,label='d: newly closed road'),Line2D([0],[0],color='#98a6b0',ls='--',label='d: unknown road exposure')]
 legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),ncol=4,frameon=False,fontsize=8.6,columnspacing=1.5,handlelength=1.9)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();assert legend.get_window_extent(renderer).y1<min(ax.get_tightbbox(renderer).y0 for ax in axes[2:])
 for upper,lower in [(axa,axc),(axb,axd)]:assert np.allclose([upper.get_position().x0,upper.get_position().x1],[lower.get_position().x0,lower.get_position().x1],atol=1e-8)
 for ax in axes:
  box=ax.get_tightbbox(renderer);assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
 assert all(ax.get_title()==''for ax in axes)
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135);plt.close(fig)
 with Image.open(OUT)as im:dims=list(im.size);dpi=list(im.info['dpi']);assert all(449<=x<=451 for x in dpi)
 return {'dimensions':dims,'dpi':dpi,'columns_aligned':True,'legend_below_panels':True,'four_sided_map_frame':True,'geographic_graticules':True,'positive_road_state_records':len(main),'all_primary_crossings_inside_scatter_axes':True,'map_head_cells_by_reference':{r:int(mapped[mapped.reference.eq(r)].grid_index.nunique())for r in REFS},'map_is_any_time_union':True,'marker_sizes_are_not_flood_areas':True}

def companions(head,road,cross,main,samples,checks,layout):
 for name,d in [('head_screens',head),('road_side_effects',road),('threshold_crossings',cross),('primary_threshold_crossings',main),('head_cell_samples',samples),('head_reconstruction_checks',pd.DataFrame(checks))]:d.to_csv(R/(SLUG+'_'+name+'.csv'),index=False)
 caption='''Neighboring modeled adverse changes under the higher receiving allowance. Q5 retains the same 319 native intake and outlet locations, allocation weights, unit ceiling and solver as Q2.5; only the five shared receiving-pool caps differ, totaling 5.0 versus 2.5 cubic meters per second. Orange compares Q5 with no action; blue compares Q5 with Q2.5. These are hypothetical allowances, not surveyed facilities or downstream receiving capacity. Panel a reports the area with a conservative native water-surface elevation increase of at least 0.10 m: open circles are maximum simultaneous area and diamonds are the union of any-time affected locations. A 10 m cell contributes 100 square meters. Peaks and unions are different statistics, not measured flood extents. All twelve parent-stage-reference contrasts are retained, including zeros. Panel b shows maximum simultaneous unique three-ward road length newly crossing the unchanged 0.30 m closure rule on the primary lower road-depth branch. It is neither cumulative closure exposure nor the union of roads affected at different times. Individual road and head-rise peaks need not coincide. Panel c plots every positive-length lower-branch road-state crossing, using the reference depth margin below 0.30 m and the treatment-minus-reference mapped depth increase, both in millimeters on logarithmic axes. Repeated high/low-stage records can overlap at identical coordinates; there is no jitter or response-based selection. The dashed diagonal is the crossing boundary. Zero clipped-length context roads and the upper-depth sensitivity are retained in the companion register, not added to three-ward length or the primary scatter. Tiny increments close to the rule indicate threshold sensitivity, not field-confirmed road flooding or negligible physical uncertainty. Panel d retains the predeclared August high-stage diagnostic context at whole-three-ward scale. It maps any-time head-rise cell locations and any-time newly closed roads for both references; it is not a simultaneous flood map. Head-cell markers and road line widths are enlarged for visibility and do not represent affected area or corridor width. Road exposure with unresolved structure or hazard mapping remains gray dashed and is not assumed dry. Native screen support, source clocks, shallow censor bounds and all original numerical gates remain unchanged. Qualified road length is 836.958 km and unknown exposure is 277.920 km. The zero newly wet subset of the head-rise screen does not establish zero newly inundated land. Added receiving-river storage, actual downstream consequences and independent observed road-flood accuracy remain unmodeled or unverified. New resident-loss bounds remain separately available in the final evidence ledger; they cannot be inferred from head-rise area or road length.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 methods='''# Neighboring Flood Transfer\n\nAnaSOP Sections 5-7 distinguish paired native water-surface screening from mapped road-threshold crossings. Section 8 fixes four panels and the two explicit Q5 references. The generator uses existing qualified Parquet summaries and immutable native exposure and sparse-head sample files. It verifies every one of the twelve head contrasts against all 109 source frames and the sparse-cell unique union, every road contrast on both original depth branches against stored maximum and integral summaries, and road classification against the original lower/upper depth arrays. Every positive-length primary road-state crossing is retained; zero-length context support stays in the register.\n\nThe spatial panel uses the existing August high-stage diagnostic context. Grid centers use the north-up 10 m schema and are checked against original ward codes and the common admitted mask before transforming to longitude/latitude. All qualifying sparse samples were produced by the admitted original native head screen; no new simulation, native missing-value imputation or road routing is performed. Cell locations and road lines are symbolized for visibility at the full three-ward extent, with no local inset or favorable time selection.\n\nMaximum simultaneous area, any-time location union, peak new closure length, integrated newly closed exposure and resident consequences have different units and denominators. No single-edge causal attribution, actual road-flood validation, engineering acceptability or downstream safety is inferred. The figure retains English annotations, 450 dpi, aligned plotting columns, a complete geographic map frame and a bottom legend. It awaits individual human review; no next figure is automatically generated.\n'''
 (R/(SLUG+'_methods.md')).write_text(methods)
 assert all(sha(ROOT/n)==h for n,h in BIND.items())
 val={'passed':True,'decision':'MILI-D-20261024-003','head_contrasts':12,'road_contrasts_both_branches':24,'all109head_frames_reproduced':True,'all_road_thresholds_and_integrals_reproduced':True,'sparse_cells_on_common_admitted_support':True,'max_simultaneous_head_rise_m2':int(head.peak_simultaneous_certain_head_rise_m2.max()),'max_any_time_head_rise_union_m2':int(head.any_time_certain_head_rise_union_m2.max()),'max_peak_newly_closed_road_m':float(road[road['Road Depth Branch'].eq('lower')]['Peak Newly Closed Road Length m'].max()),'max_primary_new_closure_depth_increment_mm':float(main['Mapped Depth Increase mm'].max()),'original_numerical_and_quality_rules_unchanged':True,'new_simulations':0,'new_routing':0,'source_files_unchanged':True,'awaiting_human_review':True,'output_sha256':sha(OUT),**layout}
 dump(R/(SLUG+'_validation.json'),val);dump(R/(SLUG+'_source_bindings.json'),BIND)
 def table(d):return '<div class="scroll">'+d.to_html(index=False,float_format=lambda x:f'{x:,.6f}')+'</div>'
 body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neighboring Flood Transfer</title><style>body{font:16px/1.75 system-ui;background:#f6f9fb;color:#24404f;max-width:1350px;margin:28px auto;padding:0 24px}img{width:100%;height:auto;background:white}aside{background:#e5eff4;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:8px;border:1px solid #ccd7df;white-space:nowrap}a{color:#176a96}</style><h1>Neighboring Flood Transfer</h1><aside>Pending visual review. Existing qualified model results only; English annotations and 450 dpi. Native water-level increases and new road closures are separate screens, with explicit references and unknown support.</aside>'''
 body+=f'<img src="../../../results/figures/Figure_{SLUG}.png?v={val["output_sha256"][:12]}" alt="Head rise, new closures, threshold margins and whole-area diagnostic map">'
 body+='<h2>How to read the figure</h2><p>Panel a separates maximum simultaneous area from the union of affected locations over time. Panel b shows peak newly closed length, not the net regional road benefit. Panel c shows why a very small mapped-depth change can cross the road rule; the diagonal marks the required increase. Panel d locates sparse adverse signals at ward scale with enlarged location symbols. None of these screens is an observed flood benchmark or proof of downstream safety.</p><h2>Caption</h2><p>'+html.escape(caption)+'</p><h2>Native head screens, including zero cases</h2>'+table(head)+'<h2>Road side effects, both depth branches</h2>'+table(road)+'<h2>Every primary road-state threshold crossing</h2>'+table(main)
 body+=f'<p><a href="../../../results/figures/Figure_{SLUG}.png">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Validation</a> · <a href="{SLUG}_methods.md">Methods</a> · <a href="{SLUG}_threshold_crossings.csv">Full crossing register</a> · <a href="{SLUG}_source_bindings.json">Source hashes</a></p></html>'
 (R/(SLUG+'_review.html')).write_text(body)
 print(json.dumps(val,indent=2))
def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True);bind(Path(__file__))
 head,road,cross,primary,samples,checks,support=prepare();layout=render(head,road,cross,primary,samples,support);companions(head,road,cross,primary,samples,checks,layout)
if __name__=='__main__':main()
