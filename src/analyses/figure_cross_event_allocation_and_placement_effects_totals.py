#!/usr/bin/env python3
"""Revise the frozen cross-event figure to absolute closure totals.

AnaSOP 5: matched event/stage/road support; 6: integrate closed length;
7: no-action and remaining totals; 8: two stage-specific grouped-bar panels.
Read-only reconstruction from existing qualified states; no new model runs.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import hashlib,json,shutil,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from summarize_three_event_controls_v1 import contexts
SLUG='cross_event_allocation_and_placement_effects'
REPORT=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
REV=REPORT/'totals_revision'
SOURCE=ROOT/'data/processed/three_ward_shared_receiver_controls_august_v1'
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'
EVENTS=['E20240731','E20240830','E20250911']
SCHEMES=['Static','O1','S1']
COLORS={'No action':'#222222','Static':'#999999','O1':'#4477aa','S1':'#228833'}
BIND={}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def bind(path):
 path=Path(path);BIND[str(path.relative_to(ROOT))]=sha(path);return path

def dump(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def date(event):return f'{event[1:5]}-{event[5:7]}-{event[7:9]}'

def reconstruct():
 historical=json.loads((REPORT/f'{SLUG}_source_bindings.json').read_text())
 for name,value in historical.items():
  if name!='docs/AnaSOP.md':assert sha(ROOT/name)==value,('Historical source changed',name)
 bind(ROOT/'docs/AnaSOP.md')
 assert json.loads(bind(REV/'contract_validation.json').read_text())['passed']
 with np.load(bind(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz'))as z:weights=z['union_length_m']/1000
 summaries=pd.read_csv(bind(SOURCE/'three_event_road_effects.csv'))
 oldplot=pd.read_csv(bind(REPORT/f'{SLUG}_plot_data.csv'))
 coverage=pd.read_csv(bind(SOURCE/'three_event_road_coverage.csv'))
 rows=[];checks=[];baseline_states={};common=None
 for event,base,report,static,suffix in contexts(SOURCE):
  assert json.loads(bind(report/'final_summary_validation.json').read_text())['passed']
  for scheme,path in [('Static',static),('O1',base/'batches/O1'),('S1',base/'batches/S1')]:
   with np.load(bind(path/'common_road_support.npz'))as z:q=z['qualified_roads']
   if common is None:common=q.copy()
   assert np.array_equal(common,q)and q.sum()==28064
   known=float(weights[q].sum());assert abs(known-836.9583262602582)<1e-7
   cov=coverage[coverage.event.eq(event)&coverage.scheme.eq(scheme)].iloc[0]
   assert abs(float(cov.qualified_length_km)-known)<1e-7
   trajectory=pd.read_csv(bind(path/'road_trajectories.csv'))
   for stage in ['H','L']:
    case=f'{event}__{stage}__U'+(suffix if scheme=='Static'else f'__{scheme.lower()}')
    states=[];times=[]
    for arm in ['baseline','drainage']:
     with np.load(bind(path/'exposure'/f'{case}__{arm}.npz'))as z:
      assert np.array_equal(z['road_indices'],np.flatnonzero(q))
      states.append(z['road_state_codes'][:,0]==2);times.append(z['analysis_times'])
      assert z['road_state_codes'].shape==(109,2,int(q.sum()))
    a,b=states;t=times[0]
    assert np.array_equal(t,times[1])and np.all(np.diff(t)>0)and abs(t[0])<1e-9 and abs(t[-1]-32400)<.001
    key=(event,stage)
    if key not in baseline_states:baseline_states[key]=(a.copy(),t.copy())
    else:assert np.array_equal(a,baseline_states[key][0])and np.array_equal(t,baseline_states[key][1])
    w=weights[q];base_length=a@w;remaining_length=b@w;restored=(a&~b)@w;new=(~a&b)@w
    integ=lambda x:float(np.trapezoid(x,t)/3600)
    baseline=integ(base_length);remaining=integ(remaining_length);rr=integ(restored);nn=integ(new);net=baseline-remaining
    assert abs(net-(rr-nn))<1e-8
    for arm,values in [('baseline',base_length),('drainage',remaining_length)]:
     tr=trajectory[trajectory.Case.eq(case)&trajectory.Scope.eq('Three wards')&trajectory['Depth Branch'].eq('lower')&trajectory.Arm.eq(arm)].sort_values('Frame')
     assert len(tr)==109 and np.allclose(tr['Analysis Time'],t,rtol=0,atol=.001)
     assert np.allclose(tr['Closed Road Length'],values,rtol=0,atol=1e-7)
     if arm=='drainage':
      assert np.allclose(tr['Same Time Restored Road Length'],restored,rtol=0,atol=1e-7)
      assert np.allclose(tr['Same Time Newly Closed Road Length'],new,rtol=0,atol=1e-7)
    old=summaries[summaries.event.eq(event)&summaries.scheme.eq(scheme)&summaries.Stage.eq(stage)&summaries.Scope.eq('Three wards')&summaries['Depth Branch'].eq('lower')]
    assert len(old)==1;old=old.iloc[0]
    assert abs(float(old['Baseline Closure km h'])-baseline)<1e-7
    assert abs(float(old['Closure Reduction km h'])-net)<1e-7
    prior=oldplot[oldplot.Event.eq(event)&oldplot['Receiving Water Stage'].eq(stage)&oldplot['Measure Configuration'].eq(scheme)].iloc[0]
    for field,value in [('Net Avoided Closure Exposure',net),('Restored Closure Exposure',rr),('Newly Closed Exposure',nn)]:assert abs(float(prior[field])-value)<1e-7
    percent=100*net/baseline if baseline>0 else None
    assert percent is not None and abs(percent-float(old['Closure Reduction Percent']))<1e-7
    rows.append({'Event':event,'Rainfall Parent':date(event),'Receiving Water Stage':stage,'Measure Configuration':scheme,'No Action Closure km h':baseline,'Remaining Closure km h':remaining,'Net Avoided Closure km h':net,'Reduction Percent':percent,'Restored Closure km h':rr,'Newly Closed km h':nn,'Maximum Simultaneous Newly Closed km':float(new.max()),'Qualified Road Length km':known,'Unknown Road Length km':float(weights[~q].sum()),'Frames':109,'Window Hours':9,'Depth Branch':'lower'})
    checks.append({'case':case,'baseline_and_treatment_trajectories_verified':True,'independent_state_integrals_verified':True,'previous_benefits_and_side_effects_reproduced':True})
 data=pd.DataFrame(rows)
 assert len(data)==18 and not data.duplicated(['Event','Receiving Water Stage','Measure Configuration']).any()
 assert int((data['Newly Closed km h']>0).sum())==3
 data['sort_event']=data.Event.map({e:i for i,e in enumerate(EVENTS)})
 data['sort_scheme']=data['Measure Configuration'].map({s:i for i,s in enumerate(SCHEMES)})
 data=data.sort_values(['Receiving Water Stage','sort_event','sort_scheme']).drop(columns=['sort_event','sort_scheme'])
 plot=[]
 for stage in ['H','L']:
  for event in EVENTS:
   sub=data[data.Event.eq(event)&data['Receiving Water Stage'].eq(stage)].set_index('Measure Configuration')
   for scheme in ['No action']+SCHEMES:
    row=sub.iloc[0]if scheme=='No action'else sub.loc[scheme]
    plot.append({'Event':event,'Rainfall Parent':date(event),'Receiving Water Stage':stage,'Configuration':scheme,'Cumulative Closure km h':float(row['No Action Closure km h']if scheme=='No action'else row['Remaining Closure km h']),'Reduction Percent':0. if scheme=='No action'else float(row['Reduction Percent'])})
 return data,pd.DataFrame(plot),checks

def render(plot):
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.08)
 plt.rcParams.update({'savefig.facecolor':'white','axes.linewidth':.75})
 fig=plt.figure(figsize=(13,8.2));axes=[fig.add_axes([left,.19,.365,.70])for left in [.105,.585]]
 offsets={'No action':.30,'Static':.10,'O1':-.10,'S1':-.30}
 labels=[];bar_count=0
 for ax,stage,letter in zip(axes,['H','L'],'ab'):
  for ei,event in enumerate(EVENTS):
   sub=plot[plot.Event.eq(event)&plot['Receiving Water Stage'].eq(stage)].set_index('Configuration');center=2-ei
   for scheme in ['No action']+SCHEMES:
    value=float(sub.loc[scheme,'Cumulative Closure km h']);y=center+offsets[scheme]
    bar=ax.barh(y,value,height=.15,color=COLORS[scheme],zorder=3)[0];assert abs(bar.get_width()-value)<1e-10;bar_count+=1
    labels.append(ax.annotate(f'{value:.2f}',(value,y),xytext=(5,0),textcoords='offset points',va='center',ha='left',fontsize=9,color=COLORS[scheme]))
  ax.set(xlim=(0,1000),ylim=(-.55,2.55),xlabel='Cumulative road closure (km h)')
  ax.set_xticks([0,200,400,600,800,1000]);ax.set_yticks([2,1,0],[date(e)for e in EVENTS]);ax.tick_params(labelsize=9)
  ax.set_axisbelow(True);ax.grid(axis='x',color='#dce2e6',lw=.65)
  for y in [.5,1.5]:ax.axhline(y,color='#e8edf0',lw=.6,zorder=0)
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
  ax.text(0,1.05,f'{letter}: '+('High stage'if stage=='H'else'Low stage'),transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=12)
 names={'No action':'No action','Static':'Static: fixed allocation (62 locations)','O1':'O1: reallocation (62 locations)','S1':'S1: placement (319 locations)'}
 handles=[Patch(facecolor=COLORS[s],label=names[s])for s in ['No action']+SCHEMES]
 legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.095),ncol=2,frameon=False,fontsize=9.5,columnspacing=2.8,handlelength=1.5)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes=[x.get_window_extent(renderer)for x in labels]
 assert not any(a.overlaps(b)for i,a in enumerate(boxes)for b in boxes[i+1:]),'Overlapping value labels'
 for box in boxes+[legend.get_window_extent(renderer)]:assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
 assert legend.get_window_extent(renderer).y1<min(ax.get_tightbbox(renderer).y0 for ax in axes)
 assert axes[0].get_position().y0==axes[1].get_position().y0 and axes[0].get_position().y1==axes[1].get_position().y1
 assert not fig.texts and all(ax.get_title()==''for ax in axes)
 assert bar_count==24
 fig.savefig(OUT,dpi=450);fig.savefig(REPORT/f'{SLUG}_preview.png',dpi=135);plt.close(fig)
 with Image.open(OUT)as im:
  dimensions=list(im.size);dpi=list(im.info['dpi']);assert all(449<=d<=451 for d in dpi)
 return {'dimensions':dimensions,'dpi':dpi,'bars':bar_count,'zero_based_identical_axes':True,'labels_overlap':False,'legend_below_panels':True,'no_footer_notes':True,'panel_edges_aligned':True}

def companions(data,plot,validation):
 data.to_csv(REPORT/f'{SLUG}_totals_comparison.csv',index=False)
 plot.to_csv(REPORT/f'{SLUG}_totals_plot_data.csv',index=False)
 side=data[['Rainfall Parent','Receiving Water Stage','Measure Configuration','Restored Closure km h','Newly Closed km h','Maximum Simultaneous Newly Closed km']]
 side.to_csv(REPORT/f'{SLUG}_side_effects.csv',index=False)
 caption='''Cross-event cumulative road closure before and after hypothetical drainage measures. Panels a and b show high and low prescribed receiving-water stages. Each rainfall group compares no action with remaining closure under Static fixed allocation, O1 reallocation and S1 transport-priority placement. Bar labels are absolute nine-hour length-time totals in kilometer-hours; lower is better. They are neither unique kilometers of closed road nor instantaneous closure lengths. All 24 bars use identical zero-based axes, the common 836.958 km qualified three-ward support, the lower road-depth branch, the unchanged 0.30 m closure threshold and 109 synchronized output states. Unknown exposure of 277.920 km remains separate. Static and O1 use 62 native intake locations; S1 uses 319. Location counts are not pump-machine counts. Each of five prescribed-stage receiving pools has an assumed allowance of 0.5 cubic meters per second, totaling 2.5; equal allowance does not imply equal equipment, cost or actual pumping. Across the six event-stage contexts, remaining closure includes any new closures. Compared with same-event no action, positive new-closure exposure occurs only for August high-stage O1 (0.0023605 km h), August high-stage S1 (0.0042806 km h) and August low-stage S1 (0.0045650 km h); exact values and all zero cases are retained in the companion register. These small sampled threshold crossings do not independently establish substantial flood transfer. The registry's exact values take precedence over caption rounding. July informed deployment selection; August and September are already-inspected transfer parents. Historical radar rainfall with assumed stages, dry initial conditions and hypothetical interventions is not observed road-closure reconstruction. Added receiving-river storage and downstream capacity remain unmodeled. Road-closure reductions do not establish observed congestion relief or population-access superiority.'''
 # Populate caption with independently reconstructed values rather than typed approximations.
 positives=data[data['Newly Closed km h']>0].sort_values(['Receiving Water Stage','Measure Configuration'])
 details='; '.join(f"{r['Rainfall Parent']} {'high'if r['Receiving Water Stage']=='H'else'low'}-stage {r['Measure Configuration']}: {r['Newly Closed km h']:.8f} km h"for _,r in positives.iterrows())
 a=caption.index('Compared with same-event no action, positive new-closure exposure occurs only')
 b=caption.index('These small sampled',a)
 caption=caption[:a]+'Compared with same-event no action, positive new-closure exposure occurs only in three contexts ('+details+'). The other 15 contexts have zero newly closed exposure on this branch. Exact values and zero cases are retained in the companion register. '+caption[b:]
 (REPORT/f'{SLUG}_caption.md').write_text(caption+'\n')
 methods='''# Cross-Event Allocation and Placement Effects: absolute totals revision

The user-approved presentation revision uses the existing AnaSOP Sections 5-7 road consequence framework and revised Section 8 two-panel plan. Baseline and treatment closure states are independently reduced to unique, clipped three-ward lengths and integrated over the same 109 actual-time states. The baseline is identical across the three portfolios within each event-stage. Six no-action totals and eighteen remaining totals produce 24 plotted bars. Every reconstructed trajectory is checked against the original baseline or treatment trajectory, every integral against its admitted source summary, and every avoided/restored/newly closed integral against the preceding figure data. No new simulation, training, routing, preprocessing or demand assignment is performed.

Remaining closure equals no-action closure minus restored closure plus newly closed closure. New closure is already included in the remaining bar; it must not be added again. Net benefit equals no-action closure minus remaining closure. Percentage reduction divides this difference by the same nonzero no-action total. Positive restoration and new-closure values are separate road-state comparisons, not inferences from regional differences.

The common zero-based axes preserve the scale of the total burden. Date groups are not a temporal trajectory and stage assumptions are not observed hydrographs. All lower-depth masks, 0.30 m threshold rules, receiving-pool assumptions, frozen placements and rainfall lineage are unchanged. Side effects are separately disclosed in a full eighteen-row register, including zeros, rather than displayed as tiny red labels inside the main figure. Previously accepted output versions and historical numerical validations are preserved. The current revision awaits visual acceptance; later planned outputs remain paused.
'''
 (REPORT/f'{SLUG}_methods.md').write_text(methods)
 visible=plot.rename(columns={'Rainfall Parent':'Rainfall event','Receiving Water Stage':'Water stage','Cumulative Closure km h':'Cumulative closure (km h)','Reduction Percent':'Reduction (%)'}).drop(columns=['Event'])
 body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Road Closure Before and After Measures</title><style>body{{font:16px/1.7 system-ui;color:#203b4b;background:#f7f9fb;max-width:1300px;margin:30px auto;padding:0 22px}}aside{{background:#e4eef4;padding:18px}}img{{width:100%;height:auto;background:white}}.scroll{{overflow:auto}}table{{border-collapse:collapse;max-width:100%;font-size:13px;background:white}}th,td{{border:1px solid #cfdae3;padding:8px;text-align:left;white-space:nowrap}}a{{color:#176a96}}</style></head><body><h1>Road Closure Before and After Measures</h1><aside><b>Absolute totals; lower is better.</b> Black bars show no action. Gray, blue and green bars show the closure remaining after each measure. All bars cover the same nine-hour window. English labels, 450 dpi and a bottom legend are retained. This revised figure awaits visual acceptance; later outputs remain paused.</aside><img src="../../../results/figures/Figure_{SLUG}.png?v={validation['output_sha256'][:12]}" alt="Two stage-specific panels comparing cumulative closure without measures and remaining closure under Static, O1 and S1"><h2>How to read the figure</h2><p>The unit is kilometer-hours: 1 km closed for 1 hour equals 1 km h, as does 0.5 km closed for 2 hours. The bars are total modeled closure exposure, not the amount avoided and not unique road length. The difference between a black bar and a treatment bar is the net avoided exposure. Across these six contexts, reductions range from {data['Reduction Percent'].min():.2f}% to {data['Reduction Percent'].max():.2f}%.</p><h2>Separate side-effect disclosure</h2><p>Newly closed roads are roads open under no action but closed under a measure at the same saved time. Their exposure is already part of the remaining total. Only three of the eighteen portfolio-stage-event comparisons have positive newly closed exposure on this branch. The register includes all zero cases. Small threshold crossings are not a full assessment of flood transfer.</p><div class="scroll">{side.to_html(index=False,float_format=lambda x:f'{x:,.8f}')}</div><h2>All plotted totals</h2><div class="scroll">{visible.to_html(index=False,float_format=lambda x:f'{x:,.6f}')}</div><h2>Deployment and evidence limits</h2><p>Static keeps 62 native intake locations and fixed reservations. O1 keeps the same locations and redistributes eligible allowance within each pool. S1 uses 319 frozen transport-priority locations with static allocation. Locations are model cells, not machines. The same five hypothetical receiving pools each allow 0.5 m³/s, totaling 2.5 m³/s. Equal allowance is not equal equipment, cost or actual extraction.</p><p>All comparisons preserve 836.958 km of common qualified road support, with 277.920 km of unknown exposure kept separate. Historical radar rain, assumed water stages and a dry start do not reconstruct observed road closures. Receiving-river storage and downstream capacity are unmodeled. These road outcomes do not establish measured traffic congestion relief or population-access superiority.</p><p><a href="../../../results/figures/Figure_{SLUG}.png">450 dpi PNG</a> · <a href="{SLUG}_totals_plot_data.csv">Plot data</a> · <a href="{SLUG}_side_effects.csv">Side-effect register</a> · <a href="{SLUG}_totals_validation.json">Validation</a> · <a href="{SLUG}_caption.md">Caption</a> · <a href="{SLUG}_methods.md">Methods</a></p></body></html>'''
 (REPORT/f'{SLUG}_review.html').write_text(body)

def main():
 REV.mkdir(exist_ok=True)
 archive=REV/'previous_output';archive.mkdir(exist_ok=True)
 for p in [OUT]+[REPORT/f'{SLUG}_{s}'for s in ['preview.png','review.html','caption.md','methods.md']]:
  target=archive/p.name
  if p.exists()and not target.exists():shutil.copy2(p,target)
 data,plot,checks=reconstruct();layout=render(plot)
 assert all(sha(ROOT/key)==value for key,value in BIND.items()),'Source changed during revision'
 validation={'passed':True,'status':'revised_awaiting_human_review','output_sha256':sha(OUT),'layout':layout,'source_data_unchanged':True,'main_contrasts':18,'identical_baselines_verified':True,'baseline_and_treatment_frames':109,'newly_closed_included_once_in_remaining_totals':True,'side_effects_separate_all18_cases':True,'minimum_reduction_percent':float(data['Reduction Percent'].min()),'maximum_reduction_percent':float(data['Reduction Percent'].max()),'no_new_simulations':True,'next_figure_started':False,'checks':checks,'script_sha256':sha(Path(__file__))}
 companions(data,plot,validation)
 dump(REPORT/f'{SLUG}_totals_validation.json',validation)
 dump(REPORT/f'{SLUG}_totals_source_bindings.json',BIND)
 print('VALIDATED_24_TOTAL_BARS_18_CONTRASTS_450DPI',OUT,flush=True)
 print(plot.to_string(index=False),flush=True)

if __name__ == '__main__':
 main()
 from publication_review_state import refresh_review_page
 refresh_review_page(Path(__file__).resolve().parents[2], 'cross_event_allocation_and_placement_effects')
