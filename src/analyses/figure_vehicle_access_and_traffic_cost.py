#!/usr/bin/env python3
"""Vehicle Access and Traffic Cost: existing September static comparisons only.
AnaSOP5: fixed requests, capacities, masks, stage and allowance comparisons.
AnaSOP6: request ledger, within-time cohort and fixed all-state cohort costs.
AnaSOP7-8: distinguish route availability from modeled cost in four panels.
No fitting, hydraulic simulation, routing, training or source-data mutation.
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
import seaborn as sns
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_final_evidence_v1'
OLD=ROOT/'data/processed/three_ward_s1r_september_traffic_v1'
T=ROOT/'data/processed/three_ward_s1r_september_traffic_phases_v1'
R=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
SLUG='vehicle_access_and_traffic_cost';OUT=ROOT/'data/results/figures'/('Figure_'+SLUG+'.png')
FRAMES=[29,48,72,108];ARMS=['none','Q2p5','Q5'];STAGES=['H','L'];BIND={}
COLORS={'Q2p5':'#4477aa','Q5':'#228833'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def bind(p):p=Path(p);BIND[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def load(p,keys=None):
 with np.load(bind(p),allow_pickle=False)as z:return {k:z[k]for k in (z.files if keys is None else keys)}
def dump(p,z):p.write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def close(a,b):assert np.isclose(a,b,rtol=0,atol=1e-8),(a,b)
def prepare():
 assert read(R/'neighboring_flood_transfer_acceptance.json')['status']=='human_accepted'
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();assert '| Vehicle Access and Traffic Cost |'in doc
 assert read(ROOT/'data/exp/research-design/three_ward_final_evidence_contract_v1/validation.json')['passed']
 val=read(ROOT/'data/exp/estimation'/T.name/'validation.json');assert val['passed']and val['new_traffic_states']==18 and val['reused_states']==6 and val['max_gap']<=.001
 source=pd.read_parquet(bind(P/'traffic_states.parquet'));paired=pd.read_parquet(bind(P/'paired_traffic_effects.parquet'));fixed=pd.read_parquet(bind(P/'fixed_temporal_cohort.parquet'))
 assert len(source)==len(paired)==len(fixed)==24
 for d in [source,fixed]:assert not d.duplicated(['frame','stage','portfolio']).any()and set(d.frame)==set(FRAMES)
 dem=load(ROOT/'data/processed/three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz',['base','period_weights']);core=load(ROOT/'data/processed/three_ward_demand_protection_feedback_v1/protection_masks.npz',['core_endpoint_od'])['core_endpoint_od']&dem['base'];q=dem['period_weights'][:,3]*dem['base']*2.
 original=float(q[core].sum());fits={};audits=[];globalcohort=core&(q>0)
 for frame in FRAMES:
  bp=OLD if frame==29 else T/'batches'/f'f{frame:03d}';br=ROOT/'data/exp/estimation'/bp.relative_to(ROOT/'data/processed')
  assert read(br/'validation.json')['passed']
  meta=read(bp/'state_manifest.json')['states'];assert len(meta)==6 and all(v['frame']==frame for v in meta)
  for stage in STAGES:
   for arm in ARMS:
    case=f'E20250911__{stage}__{arm}';name=case+'__flood__defer00__UE'
    v=read(br/(name+'_fit_validation.json'));assert v['passed']and all(v['checks'].values())and v['gap']<=.001
    z=load(bp/'fits'/(name+'.npz'),['original_requested','retained_requested','reachable','mean_od_cost']);assert np.array_equal(z['original_requested'],q)and np.array_equal(z['retained_requested'],q)
    fits[(frame,stage,arm)]=z;globalcohort&=z['reachable'];audits.append({'frame':frame,'stage':stage,'portfolio':arm,'Original Fit Passed':True,'Original Gap':v['gap'],'Original Balance Residual':v['balance_residual']})
 weight=float(q[globalcohort].sum());assert weight>0
 ids=pd.read_csv(bind(T/'fixed_all_times_cohort.csv'));assert np.array_equal(ids.expanded_od_index.to_numpy(),np.flatnonzero(globalcohort));assert np.allclose(ids.original_request_weight,q[globalcohort],rtol=0,atol=1e-12)
 plot=[];within=[]
 for frame in FRAMES:
  for stage in STAGES:
   local=core&(q>0)
   for arm in ARMS:local&=fits[(frame,stage,arm)]['reachable']
   lw=float(q[local].sum());assert lw>0
   baseline=fits[(frame,stage,'none')];basefixed=float(q[globalcohort]@baseline['mean_od_cost'][globalcohort]/weight/60);baselocal=float(q[local]@baseline['mean_od_cost'][local]/lw/60)
   for arm in ARMS:
    z=fits[(frame,stage,arm)];s=source[source.frame.eq(frame)&source.stage.eq(stage)&source.portfolio.eq(arm)];f=fixed[fixed.frame.eq(frame)&fixed.stage.eq(stage)&fixed.portfolio.eq(arm)];assert len(s)==len(f)==1;s=s.iloc[0];f=f.iloc[0]
    served=float(q[core&z['reachable']].sum());unreach=float(q[core&~z['reachable']].sum());fraction=served/original;close(served+unreach,original)
    for a,b in [(s.original_core_request_proxy,original),(s.served_core_request_proxy,served),(s.unreachable_core_request_proxy,unreach),(s.completion_fraction_original,fraction),(s.common_cohort_original_request_proxy,lw)]:close(a,b)
    ml=float(q[local]@z['mean_od_cost'][local]/lw/60);close(ml,s.common_cohort_mean_cost_min);close(100*ml/baselocal,s.common_cohort_cost_index)
    mf=float(q[globalcohort]@z['mean_od_cost'][globalcohort]/weight/60);index=100*mf/basefixed;excluded=float(q[core&baseline['reachable']&~globalcohort].sum())
    for a,b in [(f.all24states_common_request_proxy,weight),(f.common_cohort_mean_model_cost_min,mf),(f.same_time_noaction_cost_index,index),(f.baseline_reachable_excluded_proxy,excluded)]:close(a,b)
    plot.append({'Frame':frame,'Elapsed Hours':frame/12,'Receiving Water Stage':stage,'Measure Configuration':arm,'Original Request Proxy':original,'Routable Request Proxy':served,'Unreachable Request Proxy':unreach,'Routable Requests Percent':100*fraction,'Fixed Temporal Cohort Weight':weight,'Fixed Cohort Mean Model Cost min':mf,'Fixed Cohort Cost Index':index,'Fixed Cohort Cost Reduction Percent':100-index,'Baseline-Reachable Excluded From Temporal Cost':excluded})
    within.append({'Frame':frame,'Receiving Water Stage':stage,'Measure Configuration':arm,'Within-Time Cohort Weight':lw,'Within-Time Mean Model Cost min':ml,'Within-Time Cost Index':100*ml/baselocal,'Within-Time Cost Reduction Percent':100-100*ml/baselocal})
   for reference,treatment in [('none','Q2p5'),('none','Q5'),('Q2p5','Q5')]:
    a,b=fits[(frame,stage,reference)],fits[(frame,stage,treatment)];rest=float(q[core&~a['reachable']&b['reachable']].sum());new=float(q[core&a['reachable']&~b['reachable']].sum());ca=float(q[local]@a['mean_od_cost'][local]/lw/60);change=float(q[local]@(b['mean_od_cost'][local]-a['mean_od_cost'][local])/lw/60)
    z=paired[paired.frame.eq(frame)&paired.stage.eq(stage)&paired.reference.eq(reference)&paired.treatment.eq(treatment)];assert len(z)==1;z=z.iloc[0]
    for x,y in [(z.restored_request_proxy,rest),(z.newly_unreachable_request_proxy,new),(z.net_avoided_unreachable_request_proxy,rest-new),(z.common_cohort_cost_change_min,change),(z.common_cohort_cost_change_percent,100*change/ca)]:close(x,y)
 data=pd.DataFrame(plot);within=pd.DataFrame(within)
 for _,g in data.groupby(['Frame','Receiving Water Stage']):
  assert g['Routable Request Proxy'].max()-g['Routable Request Proxy'].min()<1e-8
 assert paired.restored_request_proxy.eq(0).all()and paired.newly_unreachable_request_proxy.eq(0).all()
 return data,within,source,paired,pd.DataFrame(audits),weight

def render(data):
 sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',font_scale=1.08)
 plt.rcParams.update({'savefig.facecolor':'white','axes.linewidth':.8})
 fig=plt.figure(figsize=(14,11));xs=[.10,.585];width=.355
 axes=[fig.add_axes([x,.625,width,.285])for x in xs]+[fig.add_axes([x,.18,width,.335])for x in xs]
 texts=[];traces=[]
 for ax,stage,letter in zip(axes[:2],STAGES,'ab'):
  g=data[data['Receiving Water Stage'].eq(stage)&data['Measure Configuration'].eq('none')].sort_values('Frame');x=g['Elapsed Hours'].to_numpy();y=g['Routable Requests Percent'].to_numpy()
  line,=ax.plot(x,y,color='#263746',marker='o',markersize=5,lw=1.7,zorder=3);assert np.array_equal(line.get_xdata(),x)and np.array_equal(line.get_ydata(),y);traces.append(1)
  for xx,yy in zip(x,y):texts.append(ax.annotate(f'{yy:.1f}%',(xx,yy),xytext=(0,10),textcoords='offset points',ha='center',va='bottom',fontsize=9.3,color='#263746'))
  ax.set(ylim=(0,100),ylabel='Requests with a modeled route (%)');ax.set_yticks([0,25,50,75,100])
  ax.text(0,1.05,f'{letter}: '+('High stage'if stage=='H'else'Low stage')+', request reachability',transform=ax.transAxes,fontweight='bold',fontsize=11.2,va='bottom')
 for ax,stage,letter in zip(axes[2:],STAGES,'cd'):
  ax.axhline(0,color='#7d8b96',ls='--',lw=1,zorder=1)
  for arm in ['Q2p5','Q5']:
   g=data[data['Receiving Water Stage'].eq(stage)&data['Measure Configuration'].eq(arm)].sort_values('Frame');x=g['Elapsed Hours'].to_numpy();y=g['Fixed Cohort Cost Reduction Percent'].to_numpy()
   line,=ax.plot(x,y,color=COLORS[arm],marker='s'if arm=='Q2p5'else'^',markersize=5,lw=1.7,zorder=3);assert np.array_equal(line.get_xdata(),x)and np.array_equal(line.get_ydata(),y);assert ((y>=-.10)&(y<=.45)).all();traces.append(1)
   texts.append(ax.annotate(f'{y[-1]:.2f}%',(x[-1],y[-1]),xytext=(7,0),textcoords='offset points',ha='left',va='center',fontsize=9.3,color=COLORS[arm]))
  ax.set(ylim=(-.10,.45),ylabel='Modeled cost reduction vs no action (%)');ax.set_yticks([-.1,0,.1,.2,.3,.4])
  ax.text(0,1.05,f'{letter}: '+('High stage'if stage=='H'else'Low stage')+', fixed-cohort cost',transform=ax.transAxes,fontweight='bold',fontsize=11.2,va='bottom')
 for ax in axes:
  ax.set(xlim=(2.0,9.8),xlabel='Elapsed model time (h:mm)');ax.set_xticks([29/12,4,6,9],['2:25','4:00','6:00','9:00']);ax.grid(color='#dce3e8',lw=.6);ax.set_axisbelow(True);ax.tick_params(labelsize=9)
  for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#52616c')
 handles=[Line2D([0],[0],lw=1.7,marker='o',color='#263746',label='Request reachability: all three portfolios coincide'),Line2D([0],[0],lw=1.7,marker='s',color=COLORS['Q2p5'],label='Cost: Q2.5, lower receiving allowance'),Line2D([0],[0],lw=1.7,marker='^',color=COLORS['Q5'],label='Cost: Q5, higher receiving allowance'),Line2D([0],[0],lw=1,ls='--',color='#7d8b96',label='Cost: no-action reference (zero reduction)')]
 legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.035),ncol=2,frameon=False,fontsize=9,columnspacing=2.8,handlelength=2.2)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();assert legend.get_window_extent(renderer).y1<min(ax.get_tightbbox(renderer).y0 for ax in axes[2:])
 for upper,lower in [(axes[0],axes[2]),(axes[1],axes[3])]:assert np.allclose([upper.get_position().x0,upper.get_position().x1],[lower.get_position().x0,lower.get_position().x1],atol=1e-10)
 boxes=[t.get_window_extent(renderer)for t in texts]
 assert not any(a.overlaps(b)for i,a in enumerate(boxes)for b in boxes[i+1:])
 for ax in axes:
  box=ax.get_tightbbox(renderer);assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
 assert all(ax.get_title()==''for ax in axes)
 fig.savefig(OUT,dpi=450);fig.savefig(R/(SLUG+'_preview.png'),dpi=135);plt.close(fig)
 with Image.open(OUT)as im:dims=list(im.size);dpi=list(im.info['dpi']);assert all(449<=d<=451 for d in dpi)
 return {'dimensions':dims,'dpi':dpi,'columns_aligned':True,'legend_below_panels':True,'annotations_do_not_overlap':True,'zero_to100_request_axes':True,'common_cost_axis_limits':[-.10,.45],'discrete_snapshot_values_unchanged':True,'request_traces_collapsed_only_after_equality_check':True}

def companions(data,within,source,paired,audits,weight,layout):
 for name,d in [('plot_data',data),('within_time_cohort',within),('original_request_states',source),('paired_effects',paired),('fit_checks',audits)]:d.to_csv(R/(SLUG+'_'+name+'.csv'),index=False)
 q5=data[data['Measure Configuration'].eq('Q5')]['Fixed Cohort Cost Reduction Percent']
 val={'passed':True,'decision':'MILI-D-20261024-004','traffic_states_reconstructed':24,'original_fit_checks_verified':24,'within_time_paired_effects_reproduced':24,'request_denominator':float(data['Original Request Proxy'].iloc[0]),'fixed_temporal_cohort_request_weight':weight,'global_cohort_reconstructed_from_all24reachable_masks':True,'all_requests_accounted':True,'within_time_cohort_means_and_exclusions_preserved':True,'restored_requests_all_sampled_pairs':0.,'newly_unreachable_requests_all_sampled_pairs':0.,'Q5_fixed_cohort_cost_reduction_percent_range':[float(q5.min()),float(q5.max())],'max_original_gap':float(audits['Original Gap'].max()),'full_original_admission_preserved':True,'new_simulations':0,'new_traffic_fits':0,'new_training':False,'sealed_test_access':False,'source_files_unchanged':True,'awaiting_human_review':True,'output_sha256':sha(OUT),**layout}
 caption='''Vehicle request reachability and modeled cost under the completed September receiving-allowance scenarios. Panels a and b show the share of original represented core vehicle-request proxy weight with a modeled route at four fixed snapshots, under high and low prescribed receiving-water stages. The denominator is the unchanged original core request proxy of 37,113.1985. No action, Q2.5 and Q5 have identical reachability at every stage-time, so their curves are drawn once after direct equality and zero-restoration/new-loss checks. Changes along these shared curves are time/stage changes, not treatment-restored requests. They are not observed completed trips, all vehicles, unique people or a dynamic nine-hour trajectory. Panels c and d show modeled generalized-cost reduction relative to same-time no action on one identical original-weighted OD cohort reachable in all twenty-four traffic states; the common cohort has weight 26,825.4559. Positive values mean lower modeled cost; negative values mean a higher cost. Zero is the no-action reference. Both panels use the same expanded scale from -0.10 to 0.45 percent, so the largest plotted benefit remains below 0.4 percent rather than a large congestion reduction. Absolute modeled costs, excluded baseline-reachable weight and each time's original three-portfolio common-cohort contrasts are retained in companion tables. The fixed temporal cohort prevents changes in survivor composition from creating artificial cross-time cost changes; a time-specific common cohort is a different comparison population and need not produce exactly the same percentage. Neither cohort's average is the cost of unserved requests. Lines connect only the declared snapshots for readability, with true spacing; they do not supply intervening traffic states, a dynamic queue or a nine-hour cost integral. All states hold the survey-derived 17:00 small-vehicle driver-request proxy at twice its reference value, together with original capacity, route-pool, reference-cost, mask and road-rule assumptions. Lower mapped depth is the fitted branch: roads at least 0.30 m deep close, and roads from 0.10 to below 0.30 m have doubled reference cost and halved capacity. Unknown road exposure remains conditionally available, not certified safe. Q2.5 and Q5 use the same 319 native intake locations, weighted allocator, unit ceiling and eligibility rules; five hypothetical receiving-pool caps total 2.5 and 5.0 cubic meters per second. They are not installed equipment or surveyed downstream capacity. July/August hydraulic and resident evidence and the older traffic pilot remain separate blocks, not additional fitted traffic parents here. All original numerical reference checks pass, but historical rainfall forcing and assumed stages do not establish measured traffic, realized delay, calibrated demand, full-network equilibrium or independent congestion validation.'''
 (R/(SLUG+'_caption.md')).write_text(caption+'\n')
 methods='''# Vehicle Access and Traffic Cost\n\nAnaSOP Sections 5-7 bind the fixed demand, restricted two-path assignment, original request accounting and cohort rules. Section 8 calls for four stage-paired panels separating route availability from cost. This generator reads saved fitted arrays without importing or running the routing/optimization engine. It independently reconstructs original core demand from the frozen 17:00 request weights, all twenty-four request balances and original fit receipts, each time-stage's three-portfolio cohort means, all twenty-four direct request/cost contrasts, and the all-state OD intersection and its original weights. All results reproduce their saved CSV/Parquet summaries within 0.00000001 in their stated units. No optimizer-success flag, tolerance, mask or source value is changed. Original admission is retained as defined by its original source checks and restricted gap, rather than a newly imposed success criterion.\n\nThe main cost panels deliberately use the fixed all-state cohort because they connect four times. Each original within-time three-portfolio cohort remains in a separate companion table rather than being mixed into the temporal means. Main cost reduction is 100 minus the no-action-indexed fixed-cohort cost. Request shares retain the original full represented core request denominator, not the smaller cost cohort. Equal treatment reachability is checked before collapsing the three identical curves into one, and every plotted snapshot, annotation and common axis is numerically verified. The cost scale is disclosed and positive/negative values are retained.\n\nThe figure uses English labels, 450 dpi, aligned panels, complete plot borders and a bottom legend without footer notes. It is pending human review. No new hydraulic simulation, routing, traffic fit, training or sealed-test access occurs.\n'''
 (R/(SLUG+'_methods.md')).write_text(methods)
 assert all(sha(ROOT/n)==h for n,h in BIND.items())
 dump(R/(SLUG+'_validation.json'),val);dump(R/(SLUG+'_source_bindings.json'),BIND)
 def table(d):return '<div class="scroll">'+d.to_html(index=False,float_format=lambda x:f'{x:,.6f}')+'</div>'
 body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vehicle Access and Traffic Cost</title><style>body{font:16px/1.75 system-ui;background:#f6f9fb;color:#24404f;max-width:1350px;margin:28px auto;padding:0 24px}img{width:100%;height:auto;background:white}aside{background:#e5eff4;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:8px;border:1px solid #ccd7df;white-space:nowrap}a{color:#176a96}</style><h1>Vehicle Access and Traffic Cost</h1><aside>Pending visual review. September only: four snapshots, two assumed stages and three portfolios. Request reachability and modeled cost are different outcomes. English annotations, 450 dpi; existing fitted results only.</aside>'''
 body+=f'<img src="../../../results/figures/Figure_{SLUG}.png?v={val["output_sha256"][:12]}" alt="Shared request-reachability curves and fixed-cohort modeled cost reductions by stage">'
 body+='<h2>How to read the figure</h2><p>Upper panels: all three portfolios have the same represented request reachability at each snapshot. A rising curve therefore does not show a treatment benefit. Lower panels: positive means a small modeled cost reduction relative to no action; negative means a small increase. The vertical range is less than one percentage point and must not be interpreted as a large reduction in real congestion. One identical OD cohort is used across all times.</p><h2>Caption</h2><p>'+html.escape(caption)+'</p><h2>Main figure values: fixed temporal cohort</h2>'+table(data)+'<h2>Original within-time common-cohort values</h2>'+table(within)+'<h2>All direct within-time request and cost contrasts</h2>'+table(paired)+'<h2>Original numerical fit receipts</h2>'+table(audits)
 body+=f'<p><a href="../../../results/figures/Figure_{SLUG}.png">450 dpi PNG</a> · <a href="{SLUG}_validation.json">Validation</a> · <a href="{SLUG}_methods.md">Methods</a> · <a href="{SLUG}_plot_data.csv">Plot values</a> · <a href="{SLUG}_source_bindings.json">Source hashes</a></p></html>'
 (R/(SLUG+'_review.html')).write_text(body)
 print(json.dumps(val,indent=2))
def main():
 R.mkdir(parents=True,exist_ok=True);OUT.parent.mkdir(parents=True,exist_ok=True);bind(Path(__file__))
 data,within,source,paired,audits,weight=prepare();layout=render(data);companions(data,within,source,paired,audits,weight,layout)
if __name__=='__main__':main()
