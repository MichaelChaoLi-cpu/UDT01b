#!/usr/bin/env python3
"""Cross-Event Allocation and Placement Effects.

AnaSOP5: three frozen portfolios,three parents,two stages,within-event no-action reference.
AnaSOP6: restored/newly closed unique lengths and their matched-time trapezoidal integrals.
AnaSOP7-8: whole-three-ward road benefits;all six contexts,lower-depth main branch.
Sources are read-only qualified native/summary contracts;no new hydraulic or traffic fit.
"""
from __future__ import annotations
from pathlib import Path
import os,json,hashlib,sys,ast
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-article-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from summarize_three_event_controls_v1 import contexts
TITLE='Cross-Event Allocation and Placement Effects';SLUG='cross_event_allocation_and_placement_effects'
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1';OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')
SOURCE=ROOT/'data/processed/three_ward_shared_receiver_controls_august_v1';EVENTS=['E20240731','E20240830','E20250911'];STAGES=['H','L'];SCHEMES=['Static','O1','S1']
COLORS={'Static':'#666666','O1':'#4477aa','S1':'#228833'};MARKERS={'Static':'s','O1':'o','S1':'^'};RED='#bb2233';BIND={}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def bind(p):
 p=Path(p);key=str(p.relative_to(ROOT))
 if key not in BIND:BIND[key]=sha(p)
 return p

def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def date(x):return x[1:5]+'-'+x[5:7]+'-'+x[7:9]

def render(data):
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.05)
 plt.rcParams.update({'savefig.facecolor':'white','axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.75})
 fig=plt.figure(figsize=(13,11.5));axes=np.empty((2,2),dtype=object)
 for col,left in enumerate([.105,.585]):
  axes[0,col]=fig.add_axes([left,.655,.365,.255]);axes[1,col]=fig.add_axes([left,.14,.365,.375])
 offsets={'Static':.22,'O1':0.,'S1':-.22};labels=[];bars=[];nonzero=[];zero_values=[]
 for col,stage in enumerate(STAGES):
  a=axes[0,col];g=axes[1,col]
  for ei,event in enumerate(EVENTS):
   y=2-ei;sub=data[data.Event.eq(event)&data['Receiving Water Stage'].eq(stage)].set_index('Measure Configuration')
   xx=[float(sub.loc[s,'Net Avoided Closure Exposure'])for s in SCHEMES];yy=[y+offsets[s]for s in SCHEMES];a.plot(xx,yy,color='#aab5bc',lw=1,zorder=1)
   for scheme in SCHEMES:
    x=float(sub.loc[scheme,'Net Avoided Closure Exposure']);yp=y+offsets[scheme];a.scatter([x],[yp],color=COLORS[scheme],marker=MARKERS[scheme],s=47,zorder=4)
    labels.append(a.annotate(f'{x:.2f}',(x,yp),xytext=(6,0),textcoords='offset points',ha='left',va='center',fontsize=9,color=COLORS[scheme],bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.25}))
    restored=float(sub.loc[scheme,'Restored Closure Exposure']);new=float(sub.loc[scheme,'Newly Closed Exposure'])
    patch=g.barh(yp,restored,height=.17,color=COLORS[scheme],alpha=.88,zorder=3)[0];assert abs(patch.get_width()-restored)<1e-12;bars.append({'event':event,'stage':stage,'scheme':scheme,'restored':restored,'newly_closed':new})
    labels.append(g.annotate(f'{restored:.2f}',(restored,yp),xytext=(5,0),textcoords='offset points',ha='left',va='center',fontsize=8.5,color=COLORS[scheme]))
    if new>0:
     p=g.barh(yp,-new,height=.17,color=RED,hatch='////',edgecolor=RED,linewidth=.6,zorder=4)[0];assert abs(p.get_width()+new)<1e-12
     g.plot([-new],[yp],marker='|',markersize=7,color=RED,zorder=5)
     labels.append(g.annotate(f'-{new:.5f}',(-new,yp),xytext=(-6,0),textcoords='offset points',ha='right',va='center',fontsize=8.2,color=RED,bbox={'facecolor':'white','edgecolor':'none','pad':.2}));nonzero.append({'event':event,'stage':stage,'scheme':scheme,'value':new})
    else:zero_values.append((event,stage,scheme))
  for ax in [a,g]:
   ax.set(xlim=(-6.5,32.5),ylim=(-.6,2.6));ax.set_yticks([2,1,0],[date(e)for e in EVENTS]);ax.set_xticks([0,10,20,30]);ax.set_axisbelow(True);ax.grid(axis='x',color='#dce2e6',lw=.65);ax.axvline(0,color='#66747d',lw=.8,zorder=2)
   for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c');spine.set_linewidth(.75)
   for yline in [.5,1.5]:ax.axhline(yline,color='#e8edf0',lw=.6,zorder=0)
  a.set_xlabel('Net avoided closure exposure (km h)');g.set_xlabel('Restored (+) / newly closed (-) exposure (km h)')
  for ax,label in [(a,'ab'[col]),(g,'cd'[col])]:ax.text(0,1.055,label+': '+('High stage'if stage=='H'else'Low stage'),transform=ax.transAxes,fontweight='bold',fontsize=11.5,va='bottom',ha='left')
 handles=[Line2D([0],[0],marker=MARKERS[s],color=COLORS[s],lw=0,markersize=7,label=n)for s,n in [('Static','Static: fixed allocation (62 locations)'),('O1','O1: reallocation (62 locations)'),('S1','S1: placement (319 locations)')]]
 handles.append(Patch(facecolor=RED,hatch='////',label='Newly closed (-)'))
 legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.515,.067),ncol=2,frameon=False,fontsize=9,columnspacing=2.2,handlelength=1.5)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bb=[a.get_position()for a in axes.ravel()]
 assert abs(bb[0].x0-bb[2].x0)<1e-12 and abs(bb[0].x1-bb[2].x1)<1e-12 and abs(bb[1].x0-bb[3].x0)<1e-12 and abs(bb[1].x1-bb[3].x1)<1e-12
 boxes=[l.get_window_extent(renderer)for l in labels];collisions=[]
 for i,x in enumerate(boxes):
  for j,y in enumerate(boxes[i+1:],i+1):
   if x.overlaps(y):collisions.append((i,j))
 assert not collisions,('Value labels overlap',collisions)
 for ax in axes.ravel():assert ax.get_title()==''
 for x in boxes+[legend.get_window_extent(renderer)]:assert x.x0>=0 and x.x1<=fig.bbox.width and x.y0>=0 and x.y1<=fig.bbox.height,'Clipped label'
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135);plt.close(fig)
 with Image.open(OUT)as im:dimensions=im.size;dpi=im.info['dpi'];assert all(449<=d<=451 for d in dpi)
 assert len(nonzero)==3 and all(x['event']=='E20240830'for x in nonzero)
 assert all(sha(ROOT/k)==v for k,v in BIND.items()),'Source altered'
 return dimensions,dpi,nonzero,zero_values

def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True)
 assert not OUT.exists(),'Preserve an existing figure;make an explicit reviewed revision instead'
 sop=bind(ROOT/'docs/AnaSOP.md');sop_text=sop.read_text();assert ('| '+TITLE+' |')in sop_text and '| pending |'in next(x for x in sop_text.splitlines()if x.startswith('| '+TITLE+' |'))
 receipt=ROOT/'data/exp/research-design/three_ward_transport_contract_refresh_r3/validation.json';contract=json.loads(bind(receipt).read_text());assert contract['passed']and contract['after_sha256']==sha(sop)
 # Confirm all required final concepts using the actual plan row.
 s4=sop_text.split('## 4.')[1].split('## 5.')[0];final={x.split('|')[1].strip()for x in s4.splitlines()if x.endswith('| yes |')}
 row=next(x for x in sop_text.splitlines()if x.startswith('| '+TITLE+' |'));keys=row.split('|')[5].strip().split(', ');assert all(x in final for x in keys)
 coverage=pd.read_csv(bind(SOURCE/'three_event_road_coverage.csv'))
 source=pd.read_csv(bind(SOURCE/'three_event_road_effects.csv'));source=source[source['Depth Branch'].eq('lower')&source.Scope.eq('Three wards')].copy();assert len(source)==18
 assert not source.duplicated(['event','scheme','Stage']).any()
 clip_path=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz'
 with np.load(bind(clip_path))as z:weights=z['union_length_m']/1000
 rows=[];checks=[];supports=[]
 for event,base,br,static,suffix in contexts(SOURCE):
  assert json.loads(bind(br/'final_summary_validation.json').read_text())['passed']
  for scheme,path in [('Static',static),('O1',base/'batches/O1'),('S1',base/'batches/S1')]:
   with np.load(bind(path/'common_road_support.npz'))as z:q=z['qualified_roads']
   assert int(q.sum())==28064;supports.append(q.copy());known=float(weights[q].sum());unknown=float(weights[~q].sum())
   assert abs(known-836.9583262602582)<1e-7
   cov=coverage[coverage.event.eq(event)&coverage.scheme.eq(scheme)].iloc[0];assert int(cov.qualified_roads)==int(q.sum())and abs(float(cov.qualified_length_km)-known)<1e-7
   for stage in STAGES:
    case=f'{event}__{stage}__U'+(suffix if scheme=='Static'else '__'+scheme.lower());states=[];times=[]
    for arm in ['baseline','drainage']:
     with np.load(bind(path/'exposure'/(case+'__'+arm+'.npz')))as z:
      qi=z['road_indices'];t=z['analysis_times'];assert np.array_equal(qi,np.flatnonzero(q));assert z['road_state_codes'].shape==(109,2,len(qi))
      states.append(z['road_state_codes'][:,0]==2);times.append(t)
    assert np.array_equal(times[0],times[1])and abs(t[0])<1e-9 and abs(t[-1]-32400)<.001 and np.all(np.diff(t)>0)
    a,b=states;w=weights[qi];restored=(a&~b)@w;new=(~a&b)@w;base_closed=a@w;trt_closed=b@w
    rr=float(np.trapezoid(restored,t)/3600);nn=float(np.trapezoid(new,t)/3600);net=float(np.trapezoid(base_closed-trt_closed,t)/3600)
    assert abs(net-(rr-nn))<1e-8
    trajectories=pd.read_csv(bind(path/'road_trajectories.csv'))
    traj=trajectories[trajectories.Case.eq(case)&trajectories.Scope.eq('Three wards')&trajectories['Depth Branch'].eq('lower')&trajectories.Arm.eq('drainage')].sort_values('Frame')
    assert len(traj)==109 and np.allclose(traj['Analysis Time'],t,rtol=0,atol=.001)
    for field,values in [('Same Time Restored Road Length',restored),('Same Time Newly Closed Road Length',new),('Closed Road Length',trt_closed)]:assert np.allclose(traj[field],values,rtol=0,atol=1e-7),(case,field)
    assert np.allclose(traj['Common Qualified Road Length'],known,rtol=0,atol=1e-7)and np.allclose(traj['Unknown Road Length'],unknown,rtol=0,atol=1e-7)
    old=source[source.event.eq(event)&source.scheme.eq(scheme)&source.Stage.eq(stage)].iloc[0]
    for name,value in [('Closure Reduction km h',net),('Maximum Same Time Newly Closed km',float(new.max()))]:
     assert abs(float(old[name])-value)<1e-7,(case,name,value,float(old[name]))
    rows.append({'Rainfall Parent':date(event),'Event':event,'Receiving Water Stage':stage,'Measure Configuration':scheme,'Net Avoided Closure Exposure':net,'Restored Closure Exposure':rr,'Newly Closed Exposure':nn,'Reduction Percent':float(old['Closure Reduction Percent']),'Qualified Road Length':known,'Unknown Road Length':unknown,'Output Frames':109,'Road Depth Branch':'lower','Reference':'No action','First Analysis Time':float(t[0]),'Last Analysis Time':float(t[-1])})
    checks.append({'case':case,'scheme':scheme,'stage':stage,'same_time_integral_reconstructed':True,'closed_state_arrays_reconstructed':True})
 assert all(np.array_equal(supports[0],x)for x in supports[1:])
 data=pd.DataFrame(rows);data['event_order']=data.Event.map({e:i for i,e in enumerate(EVENTS)});data['scheme_order']=data['Measure Configuration'].map({s:i for i,s in enumerate(SCHEMES)});data=data.sort_values(['Receiving Water Stage','event_order','scheme_order']).drop(columns=['event_order','scheme_order'])
 data.to_csv(R/(SLUG+'_plot_data.csv'),index=False)
 dimensions,dpi,nonzero,zero_values=render(data)
 validation={'passed':True,'title':TITLE,'status':'generated_awaiting_human_review','output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT),'dpi':list(dpi),'dimensions':list(dimensions),'all18_main_contrasts':True,'all109state_integrals_reconstructed':True,'reference':'same-event no action','same_qualified_support':True,'qualified_roads':28064,'qualified_length_km':836.9583262602582,'quality_masks_and_lower_branch_unchanged':True,'restoration_and_new_closure_separate':True,'new_closure_negative_bars_not_magnified':True,'nonzero_new_closure_labels':nonzero,'zero_new_closure_contexts':len(zero_values),'value_label_overlap_count':0,'panel_column_edges_aligned':True,'no_titles':True,'all_source_hashes_unchanged':True,'coverage_from_frozen_masks_and_coverage_register':True,'legacy_missing_repeated_coverage_not_imputed':True,'new_simulations':0,'training':False,'AnaSOP_status_unchanged_pending':True,'numerical_checks':checks,'script_sha256':sha(Path(__file__))}
 dump(R/(SLUG+'_validation.json'),validation);dump(R/(SLUG+'_source_bindings.json'),BIND)
 caption='''Cross-event road effects of frozen shared-receiver portfolios. Panels a and b compare net avoided closure exposure under high and low prescribed receiving-water stages. Connected points within each rainfall parent represent the original static portfolio at 62 native intake locations, O1 reallocation at the same locations, and S1 transport-priority placement at 319 locations. Panels c and d separate restored closure exposure (positive bars) from newly closed exposure (negative red bars) at the corresponding stages. Every comparison uses same-event no action, the lower road-depth branch, the unchanged 0.30 m closure rule, all 109 synchronized states, and identical qualified three-ward road support of 836.958 km. Units are kilometer-hours, not unique road length or people. Positive values are rounded to two decimals. Red values retain five decimals because their true-width bars may be smaller than a pixel on the common linear scale; zero new-closure exposure has no red label. The accompanying data retain original precision. The three positive new-closure portfolio-stage contexts occur in August; the existing diagnosis classifies their primary road crossings as single-sample, small-depth threshold shifts, not observed flooding. All portfolios have the same assumed five-pool allowance totaling 2.5 cubic meters per second. Equal allowance does not imply equal actual pumping, equipment or financial cost. July supplied deployment-selection evidence; August and September were already-inspected transfer parents. These nine-hour dry-start scenarios omit complete antecedent conditions, and prescribed receiving stages omit added receiving-river storage. Unknown road exposure of 277.920 km remains separate. Road benefit does not establish population-access superiority or actual congestion relief.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 methods='''# Cross-Event Allocation and Placement Effects: comparison contract

The figure follows the general-reference road framework in AnaSOP Sections 5-7 and the confirmed four-panel plan in Section 8. It reads qualified native road-state arrays and source summaries without editing raw or processed data. Every contrast is independently reconstructed from Boolean closure states, unique three-ward length weights and approved actual clocks. Same-time restored and newly closed lengths are integrated separately using the original trapezoidal rule; their difference reproduces net avoided exposure. All 18 main-branch contrasts and all 109 output states are retained. No favorable event, stage or treatment peak is selected.

Some older source summaries do not repeat coverage or separate restored/newly closed exposure columns. These missing summary entries are not imputed. Coverage is independently checked against the frozen qualification masks, length weights and coverage register. Restored and newly closed histories are reconstructed from road-state arrays and checked against every original trajectory value before integration.

Static uses fixed reservations at 62 native intake cells. O1 preserves these locations and reservation weights, redistributing eligible wet-intake allowance within each named pool under the frozen unit, storage, trigger and lift bounds. S1 uses 319 July-derived transport-priority locations with the original static allocation operator. Each of five receiving pools has an assumed allowance of 0.5 cubic meters per second, totaling 2.5. The reallocation operator was separately qualified; per-arm images and treatment files remain frozen. Native locations are not physical machine counts, and equal flow allowance is not equal equipment or economic cost. Added receiving-river storage, complete sewer hydraulics, observed road flooding and actual congestion response are outside this figure.

English labels, 450 dpi and aligned four-panel plotting areas retain the approved style. New-closure bars use their true signed linear width; red labels expose tiny values without changing scale. The graph shows road outcomes only. Human output review is required before updating its AnaSOP status or generating the next figure. No new hydraulic simulation, traffic assignment or model training is run.
'''
 (R/(SLUG+'_methods.md')).write_text(methods)
 q=validation['output_sha256'][:12]
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三场暴雨的调配与布局对照</title><style>body{{font:16px/1.8 system-ui;color:#203b4b;background:#f7f9fb;max-width:1300px;margin:30px auto;padding:0 22px}}aside{{background:#e4eef4;padding:18px}}img{{width:100%;height:auto;background:white}}table{{border-collapse:collapse;display:block;max-width:100%;overflow:auto;background:white;font-size:13px}}th,td{{border:1px solid #cfdae3;padding:8px;text-align:left}}a{{color:#176a96}}</style><h1>三场暴雨：改调配与改布局如何影响封路</h1><aside><b>已生成并通过数值与排版核验，等待图面审阅。</b> a／b比较净减少的累计封路；c／d将恢复与新增封路分别列出。所有面板都相对同场、同水位的无措施情景。英文标注、450dpi，a与c、b与d边框对齐。</aside><img src="../../../results/figures/Figure_{SLUG}.png?v={q}" alt="Four panels comparing frozen static allocation,reallocation and placement across three historical rainfall parents"><p>上排是净改善，下排是恢复（正）与新增封路（负）。红色数字单独标出极小的新增封路积分；条形保留真实线性宽度，没有人为放大。零值不显示红字。km h是道路封闭长度与时间的积分，不是单独公里数。</p><h2>措施与比较范围</h2><p>Static为原62个原生取水位置的固定分配；O1保留位置并在池内调配额度；S1使用319个冻结的交通优先位置、保留静态分配算子。位置数不是设备台数。五个接收池各假设0.5m³/s，共2.5m³/s。同额度不等于同成本，也不代表实际抽排量相同。7月是布点选择背景；8月和9月是已查看过的迁移场景。</p><p>共同准入道路836.958km，未知暴露277.920km单列；0.30m阈值、水深下界、原质量掩膜、109帧及原雨型保持不变。三区封路改善不代表人口通达性全面改善或实际拥堵缓解。人口与副作用由后续图分别表达。</p><h2>18组数据与核验</h2>{data.drop(columns=['Event']).to_html(index=False,float_format=lambda x:f'{x:,.6f}')}<p><a href="../../../results/figures/Figure_{SLUG}.png">450dpi原图</a> · <a href="{SLUG}_plot_data.csv">原精度绘图数据</a> · <a href="{SLUG}_validation.json">核验</a> · <a href="{SLUG}_caption.md">English caption</a> · <a href="{SLUG}_methods.md">English methods</a></p><p>本轮只生成这一张图，尚未改为done；确认后再进入人口通达性图。</p></html>'''
 (R/(SLUG+'_review.html')).write_text(body)
 print('Saved:',OUT,flush=True);print('VALIDATED_18_CONTRASTS_450DPI_AWAITING_REVIEW',flush=True)
def revise_layout():
 import shutil
 previous=json.loads((R/(SLUG+'_validation.json')).read_text())
 bindings=json.loads((R/(SLUG+'_source_bindings.json')).read_text())
 accepted=json.loads((R/(SLUG+'_acceptance.json')).read_text())
 assert previous['passed'] and accepted['status']=='human_accepted'
 assert sha(ROOT/'docs/AnaSOP.md')==accepted['after_sha256']
 for key,value in bindings.items():
  if key!='docs/AnaSOP.md':assert sha(ROOT/key)==value,('Source changed',key)
 data_path=R/(SLUG+'_plot_data.csv');data_hash=sha(data_path)
 archive=R/'layout_before_bottom_legend';archive.mkdir(exist_ok=True)
 for path in [OUT,R/(SLUG+'_preview.png'),R/(SLUG+'_review.html')]:
  target=archive/path.name
  if not target.exists():shutil.copy2(path,target)
 data=pd.read_csv(data_path)
 dimensions,dpi,nonzero,zero_values=render(data)
 assert sha(data_path)==data_hash
 for key,value in bindings.items():
  if key!='docs/AnaSOP.md':assert sha(ROOT/key)==value
 assert sha(ROOT/'docs/AnaSOP.md')==accepted['after_sha256']
 review=R/(SLUG+'_review.html');body=review.read_text()
 import re
 body=re.sub(r'(Figure_'+SLUG+r'\.png\?v=)[a-f0-9]+',lambda m:m.group(1)+sha(OUT)[:12],body)
 body=body.replace('已生成并通过数值与排版核验，等待图面审阅。','已按要求移除图内底部小字，图例移至图下部；数据与450dpi保持不变。')
 body=body.replace('本轮只生成这一张图，尚未改为done；确认后再进入人口通达性图。','本轮仅调整当前图排版；后续图表保持暂停。数值解释保留于配套图注和本页。')
 review.write_text(body)
 dump(R/(SLUG+'_bottom_legend_validation.json'),{'passed':True,'revision':'remove figure footer notes and move legend below panels','dpi':list(dpi),'dimensions':list(dimensions),'output_sha256':sha(OUT),'previous_output_sha256':previous['output_sha256'],'plot_data_sha256':data_hash,'plot_data_unchanged':True,'source_data_unchanged':True,'AnaSOP_unchanged':True,'legend_at_bottom':True,'figure_footer_notes_removed':True,'new_simulations':0,'next_item_started':False,'script_sha256':sha(Path(__file__))})
 print('LAYOUT_REVISION_VALIDATED',OUT)

if __name__=='__main__':
 if '--revise-layout' in sys.argv:revise_layout()
 else:main()
