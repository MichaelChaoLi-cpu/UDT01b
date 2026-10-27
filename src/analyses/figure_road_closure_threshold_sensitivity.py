#!/usr/bin/env python3
"""Road Closure Threshold Sensitivity: one supporting scatter, existing evidence.
AnaSOP5: fixed parent, stage, reference and common qualified support.
AnaSOP6: unchanged 0.30 m rule and same-time reference/treatment depths.
AnaSOP7-8: relocate all positive-length lower-depth crossings from the master.
No new simulation, routing, fitting, threshold tuning or training.
"""
from pathlib import Path
import os,json,hashlib,sys,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import figure_neighboring_flood_transfer as master
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
SLUG='road_closure_threshold_sensitivity';TITLE='Road Closure Threshold Sensitivity'
OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')
BIND={}
MARKERS={'E20240731':'s','E20240830':'o','E20250911':'^'}
REFS={'No action':('Q5 relative to no action','#cc6633'),'S1R-Q2.5':('Q5 relative to Q2.5','#4477aa')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def prepare():
 bind(Path(__file__));bind(Path(master.__file__))
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();row=next(x for x in doc.splitlines() if x.startswith('| '+TITLE+' |'));assert '| 1 |' in row
 _,road,full,primary,_,checks,_=master.prepare();BIND.update(master.BIND)
 prior=pd.read_csv(bind(R/'neighboring_flood_transfer_primary_threshold_crossings.csv'),float_precision='round_trip')
 pd.testing.assert_frame_equal(primary.reset_index(drop=True),prior,check_dtype=False,rtol=0,atol=1e-10)
 assert len(primary)==9 and primary['Road Index'].nunique()==4
 assert primary['Three-Ward Clipped Length m'].gt(0).all() and primary['Road Depth Branch'].eq('lower').all()
 gap=(.30-primary['Reference Road Depth m'])*1000;increase=(primary['Q5 Road Depth m']-primary['Reference Road Depth m'])*1000
 np.testing.assert_allclose(gap,primary['Margin Below Closure mm'],rtol=0,atol=1e-10)
 np.testing.assert_allclose(increase,primary['Mapped Depth Increase mm'],rtol=0,atol=1e-10)
 assert (gap>0).all() and (increase>=gap).all() and primary['Reference Road Depth m'].lt(.30).all() and primary['Q5 Road Depth m'].ge(.30).all()
 return primary,full,road,checks

def render(primary):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.linewidth':.8,'axes.edgecolor':'#52616c','xtick.labelsize':10,'ytick.labelsize':10,'savefig.facecolor':'white'})
 fig=plt.figure(figsize=(8.5,7));ax=fig.add_axes([.14,.27,.81,.68]);handles=[];plotted=[]
 # No jitter or favorable-point selection: each original event-stage-road-frame
 # record is plotted, including coincident high/low-stage records.
 for event in MARKERS:
  for ref,(label,color) in REFS.items():
   z=primary[primary['Rainfall Parent'].eq(event)&primary.Reference.eq(ref)]
   if z.empty:continue
   marker=MARKERS[event]
   artist=ax.scatter(z['Margin Below Closure mm'],z['Mapped Depth Increase mm'],s=72,marker=marker,color=color,edgecolor='white',lw=.65,alpha=.9,zorder=3)
   np.testing.assert_allclose(artist.get_offsets(),z[['Margin Below Closure mm','Mapped Depth Increase mm']].to_numpy(),rtol=0,atol=0)
   handles.append(Line2D([0],[0],ls='none',marker=marker,color=color,markerfacecolor=color,markeredgecolor='white',markersize=7,label=master.date(event)+' · '+label))
   plotted.extend(z.index.tolist())
 assert sorted(plotted)==sorted(primary.index.tolist())
 xlow=10**np.floor(np.log10(primary['Margin Below Closure mm'].min()));ylow=10**np.floor(np.log10(primary['Mapped Depth Increase mm'].min()))
 diagonal=np.array([min(xlow,ylow),2.]);ax.plot(diagonal,diagonal,color='#74848e',ls='--',lw=1.2,zorder=1)
 handles.append(Line2D([0],[0],color='#74848e',ls='--',lw=1.2,label='Closure threshold: increase = gap'))
 ax.set(xscale='log',yscale='log',xlim=(xlow,1.3),ylim=(ylow,2.4),xlabel='Reference gap below 0.30 m (mm)',ylabel='Additional mapped road depth (mm)')
 ax.grid(which='major',color='#dce3e8',lw=.6);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(True)
 assert primary['Margin Below Closure mm'].between(*ax.get_xlim()).all() and primary['Mapped Depth Increase mm'].between(*ax.get_ylim()).all()
 legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.025),ncol=2,frameon=False,fontsize=8.7,columnspacing=1.4,handlelength=2.0,handletextpad=.6,labelspacing=.8)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();box=ax.get_tightbbox(renderer);leg=legend.get_window_extent(renderer)
 assert leg.y1<box.y0 and min(box.x0,box.y0,leg.x0,leg.y0)>=0
 assert max(box.x1,leg.x1)<=fig.bbox.width and max(box.y1,leg.y1)<=fig.bbox.height
 assert ax.get_title()=='' and len(fig.axes)==1 and not ax.texts
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135);plt.close(fig)
 with Image.open(OUT) as im:dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<x<451 for x in dpi)
 return {'dimensions':dimensions,'dpi':dpi,'single_panel_without_title_or_letter':True,'all_source_records_plotted':len(plotted),'all_points_inside_axes':True,'logarithmic_axes':True,'legend_below_plot':True,'legend_symbols_match_each_present_event_reference_group':True,'jitter_added':False,'limits':{'x':[float(xlow),1.3],'y':[float(ylow),2.4]}}

def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
 original_files={str(p.relative_to(ROOT)):sha(p) for folder in ['data/results/figures','data/results/tables'] for p in (ROOT/folder).glob('*') if p.is_file() and p!=OUT}
 primary,full,road,checks=prepare();print('Reproduced all nine primary crossings from qualified source arrays.',flush=True)
 layout=render(primary)
 for suffix,d in [('plot_data',primary),('all_branch_crossings',full),('all_road_contrasts',road),('source_head_checks',pd.DataFrame(checks))]:d.to_csv(R/(SLUG+'_'+suffix+'.csv'),index=False)
 coordinate_count=primary[['Margin Below Closure mm','Mapped Depth Increase mm']].drop_duplicates().shape[0]
 caption='''Sensitivity of additional modeled road closures to the unchanged 0.30 m classification threshold. The horizontal axis is the reference road depth gap below 0.30 m; the vertical axis is the Q5-minus-reference mapped depth increase, both in millimeters on logarithmic axes. Orange compares Q5 with no action and blue compares Q5 with Q2.5. Square, circle and triangle markers identify the July 2024, August 2024 and September 2025 rainfall parents. The dashed increase-equals-gap line is the classification boundary; all shown records lie on or above it. All nine positive-three-ward-length lower-depth event-stage-road-frame crossing records are retained, involving four road IDs and three rainfall parents. The records occupy six distinct coordinate pairs because some high/low-stage results coincide; no jitter, event selection or point filtering is introduced. Points are conditional crossing records, not independent storms, road-length weights, observed incidents or a probability of closure. The largest mapped-depth increase is approximately 1.668 mm; the August highlighted case has an increase of approximately 0.049 mm. These magnitudes show sensitivity of the binary road rule near its threshold. The plot alone does not determine whether an increment is a physically resolved redistribution or a numerical/representation-sensitive difference, and does not establish that adverse effects are absent. The complete upper-depth and zero-clipped-length context records and all zero/nonzero road contrasts remain in supporting data; the same lower-depth primary support and original numerical admission are retained. No observed flooding accuracy, substantial physical risk transfer or downstream receiving-river safety is inferred. Q2.5 and Q5 are hypothetical shared receiving allowances using the same 319 intake/receiver locations and allocation rules, not surveyed equipment capacities. No new hydraulic simulation, route search, fit, training or threshold tuning occurs.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 methods='''# Road Closure Threshold Sensitivity

This supporting rendering follows AnaSOP Sections 5-8 and the approved relocation of the crossing scatter from the accepted Neighboring Flood Transfer master. It calls that master's read-only preparation routine, reconstructing the twelve native head contrasts and twenty-four road comparisons across both original depth branches, every source clock, original lower/upper classifications and clipped-length integral. The nine positive-length primary crossing records are checked against the accepted master export and independently recomputed from reference and treatment depths. All existing numerical admission and quality masks remain unchanged.

For each road-state crossing, the horizontal value is one thousand times (0.30 minus reference depth), and the vertical value is one thousand times (Q5 depth minus reference depth). Positive reference gap and treatment depth at least 0.30 m are verified for every record. Each source record is plotted without jitter. Coincident high/low-stage records remain coincident; nine records occupy six distinct coordinate pairs. No point size encodes road length. Legends show only event/reference combinations actually present, with the same glyph shape and color as the corresponding points, plus the dashed classification boundary. All points lie within the disclosed logarithmic limits. Single-panel output has no panel letter or plot title, uses English annotations and 450 dpi, and awaits individual human review. Existing publication files are hash-checked for preservation. No source data or accepted image is overwritten.
'''
 (R/(SLUG+'_methods.md')).write_text(methods)
 assert coordinate_count==6
 assert all(sha(ROOT/n)==h for n,h in BIND.items()) and all(sha(ROOT/n)==h for n,h in original_files.items())
 val={'passed':True,'planning_decisions':['MILI-D-20261024-030','MILI-D-20261024-031'],'primary_crossing_records':len(primary),'distinct_road_ids':int(primary['Road Index'].nunique()),'rainfall_parent_count':int(primary['Rainfall Parent'].nunique()),'distinct_coordinate_pairs':int(coordinate_count),'source_primary_export_reproduced':True,'all_gap_and_increment_values_reconstructed':True,'max_increment_mm':float(primary['Mapped Depth Increase mm'].max()),'min_increment_mm':float(primary['Mapped Depth Increase mm'].min()),'reference_depth_below_and_Q5_depth_at_or_above_closure_rule':True,'full_branch_register_retained':True,'all_original_numerical_and_quality_rules_preserved':True,'existing_publication_files_preserved':len(original_files),'source_files_unchanged':True,'new_simulations':0,'new_traffic_fits':0,'new_training':False,'sealed_test_access':False,'awaiting_human_review':True,'output_sha256':sha(OUT),**layout}
 dump(R/(SLUG+'_validation.json'),val);dump(R/(SLUG+'_source_bindings.json'),BIND);dump(R/(SLUG+'_publication_preservation.json'),original_files)
 dump(R/(SLUG+'_completion.json'),{'title':TITLE,'status':'generated_pending_human_review','output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT)})
 body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><style>body{{font:16px/1.7 system-ui;background:#f6f9fb;color:#24404f;max-width:1150px;margin:25px auto;padding:0 24px}}img{{width:100%;height:auto;background:white}}aside{{background:#e5eff4;padding:18px}}table{{border-collapse:collapse;background:white;font-size:12px}}td,th{{border:1px solid #ccd7df;padding:7px;white-space:nowrap}}.scroll{{overflow:auto}}</style><h1>{TITLE}</h1><aside>Supporting diagnostic, pending individual human review. Nine crossing records, four road IDs, three rainfall parents; six distinct coordinate pairs. Existing model results only, English labels, 450 dpi.</aside><img src="../../../results/figures/Figure_{SLUG}.png?v={val['output_sha256'][:12]}" alt="Logarithmic scatter of reference closure margin and additional mapped road depth"><h2>How to read</h2><p>Points on or above the dashed line cross the fixed road rule. Very small increments can change the binary state when the reference lies close to the threshold. Coincident stages overlap without jitter. These records do not establish observed new flooding or absence of risk transfer.</p><h2>Caption</h2><p>{html.escape(caption)}</p><h2>Every primary crossing record</h2><div class="scroll">{primary.to_html(index=False,float_format=lambda x:f'{x:.9g}')}</div><p><a href="{SLUG}_validation.json">Validation</a> · <a href="{SLUG}_source_bindings.json">Source bindings</a> · <a href="{SLUG}_plot_data.csv">Primary plot data</a> · <a href="{SLUG}_all_branch_crossings.csv">Complete crossing register</a> · <a href="{SLUG}_methods.md">Methods</a></p></html>'''
 (R/(SLUG+'_review.html')).write_text(body)
 print(json.dumps({k:val[k] for k in ['passed','primary_crossing_records','distinct_coordinate_pairs','dimensions','dpi','existing_publication_files_preserved','awaiting_human_review']},indent=2),flush=True)
if __name__=='__main__':main()
