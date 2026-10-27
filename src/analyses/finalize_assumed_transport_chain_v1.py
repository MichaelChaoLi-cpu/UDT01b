"""Interpret completed numerical results without rerouting or changing protocol.

Distinguish prior-driven scenarios from real losses and baseline-coverage stress.
Retains the initial generated review and figure before adding interpretation.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_assumed_transport_chain_v1'
R=ROOT/'data/exp/estimation/three_ward_assumed_transport_chain_v1'

def main():
 assert json.loads((R/'validation.json').read_text())['all_passed']
 df=pd.read_csv(P/'paired_access_metrics.csv');ef=pd.read_csv(P/'measure_effects.csv');st=pd.read_csv(P/'road_state_clipped_lengths.csv');pressure=pd.read_csv(P/'screenline_stress_summary.csv')
 sub=df[df['shape'].eq('base')&df.load.eq('central')&df.period.eq('hour17')&df.scope.eq('core_endpoint_proxy')]
 def row(b,arm):return sub[(sub.branch==b)&sub.arm.eq(arm)].iloc[0]
 eff=ef[ef['shape'].eq('base')&ef.load.eq('central')&ef.period.eq('hour17')&ef.scope.eq('core_endpoint_proxy')&ef.branch.eq(0)].iloc[0]
 n=row(0,'normal');f=row(0,'flood');p=row(0,'protected');nn=row(1,'normal')
 covered=float(nn.normal_reachable/nn.requested)
 closed=float(st[(st.branch==0)&st.arm.eq('flood')&st.state.eq('known_closed')].ward_clipped_length_km.iloc[0]);slow=float(st[(st.branch==0)&st.arm.eq('flood')&st.state.eq('known_slow')].ward_clipped_length_km.iloc[0])
 notes={
 'primary_case':'E20250911__H__U elapsed14400s; historical radar with hypothetical river-stage rise',
 'primary_demand':'central original PT-derived prior,base external shape,core-endpoint proxy subset',
 'unknown_available_branch_is_an_assumption':True,
 'clipped_core_closed_length_km':closed,'clipped_core_slow_length_km':slow,
 'normal_reachable_hour17_trip_proxy':float(n.normal_reachable),
 'flood_newly_unreachable_hour17_trip_proxy':float(f.newly_unreachable),
 'flood_newly_unreachable_fraction':float(f.newly_unreachable_share),
 'ideal_corridor_restored_hour17_trip_proxy':float(eff.restored_vehicle_trip_proxy),
 'restored_share_of_flood_lost':float(eff.restoration_fraction_of_flood_lost),
 'protected_newly_unreachable_fraction':float(p.newly_unreachable_share),
 'unknown_withheld_normal_core_demand_coverage':covered,
 'withheld_branch_interpretation':'Coverage stress diagnostic only; deleting unknown roads across10km context destroys normal accessibility. Its storm loss is conditional on a different tiny baseline and is not an empirical upper bound.',
 'measure_interpretation':'This frozen4.77km plan restores only about0.9%of newly lost assumed trips in the primary snapshot. Do not claim zero engineering benefit or optimality; no drainage construction simulated.',
 'demand_sensitivity_interpretation':'Uniform demand factors change weighted volumes/pressure but not fixed-cost route fractions; invariant fractions do not constitute empirical robustness.',
 'next_proposed_analysis':'Demand-weighted corridor comparison across existing events/times,followed by explicitly assumed capacity feedback for congestion; population impact requires a separate person-trip linkage,not direct use of vehicle proxies.',
 'no_additional_simulation':True}
 stress_rows=pd.read_csv(P/'screenline_stress12h.csv');matched=[]
 for (b,sh,ld),g in stress_rows.groupby(['branch','shape','load']):
  common=set.intersection(*[set(g[g.arm.eq(a)&g.capacity_comparable].screen_index)for a in ['normal','flood','protected']])
  for a in ['normal','flood','protected']:
   h=g[g.arm.eq(a)&g.screen_index.isin(common)]
   matched.append(dict(branch=b,shape=sh,load=ld,arm=a,common_screens=len(common),small_component_vc_gt1=int((h.small_component_vc>1).sum()),with_fixed_large_background_gt1=int((h.with_fixed_observed_large_background_vc>1).sum())))
 matched=pd.DataFrame(matched);matched.to_csv(P/'matched_screenline_stress.csv',index=False,encoding='utf-8-sig')
 notes['stress_comparison_requires_same_screen_intersection']=True
 notes['unreachable_requested_trips_remain_in_ledger_not_in_traffic_queues']=True
 (R/'interpretation.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
 for name in ['review.html','paired_outcomes.png']:
  dst=R/(Path(name).stem+'_initial'+Path(name).suffix)
  if not dst.exists():shutil.copyfile(R/name,dst)
 text=(R/'review_initial.html').read_text()
 intro=f'''<div class="note"><b>首轮最直接的结果：</b>三区内约{closed:.1f}km道路触发关闭、{slow:.1f}km触发缓行；中需求开放分支约{f.newly_unreachable_share*100:.1f}%的重点出行代理新增不可达。既有通道方案恢复约{eff.restoration_fraction_of_flood_lost*100:.1f}%的新增不可达需求。本轮证明流程可以跑通，也显示原保障清单对这套需求的覆盖有限。</div>'''
 text=text.replace('<h2>这一轮的假设</h2>',intro+'<h2>这一轮的假设</h2>')
 warning=f'''<h2>未知分支的覆盖限制与措施解释</h2><p class="note">扣除全部未知道路后，重点出行代理在正常状态下仅{nn.normal_reachable:,.1f}/{nn.requested:,.1f}可达（{covered*100:.2f}%）。大部分10km上下文没有地表积水证据，全部扣除会人为缩减正常交通网络，因此该分支只作为证据覆盖压力诊断，不能作为暴雨真实损失上界；也不能把它与开放分支拼成损失区间。开放分支则假设未知道路保持正常可用，这项假设仍需在结论中保留。</p><p>在正常可用假设下，既有方案恢复{eff.restored_vehicle_trip_proxy:,.1f}次17时出行代理，占新增不可达的{eff.restoration_fraction_of_flood_lost*100:.2f}%；其余{eff.remaining_new_lost:,.1f}次仍不可达。这个结果提示应将下一轮措施比较与实际研究目标——需求加权的道路可达性——衔接。它不证明排水工程无效，也不证明原方案已最优；本轮没有模拟工程本身。</p>'''
 text=text.replace('<h2>流量转移与拥堵风险能说到哪里</h2>',warning+'<h2>流量转移与拥堵风险能说到哪里</h2>')
 stressnote='<h2>容量压力的配对范围</h2><p>以上各状态覆盖断面数不同，配对比较须使用三种状态均可计算容量的同一交集，数据已另存。下表为开放分支、不额外穿越的共同74断面：</p><div class="scroll">'+matched[(matched.branch==0)&matched['shape'].eq('base')].to_html(index=False,border=0)+'</div><p class="note">断连需求仍在请求账本，但静态分配没有把它留在道路上排队。因此受淹后过载断面减少，可能来自出行无法完成、流量移出这些断面或漏掉未知容量道路，不能解读为暴雨缓解拥堵。通道保障也可能因恢复车辆出行而增加个别断面压力。真实拥堵时长与排队仍需容量反馈情景，当前压力指标仅用于筛选。</p><p><a href="../../../processed/three_ward_assumed_transport_chain_v1/matched_screenline_stress.csv">同一断面交集的容量压力表</a></p>'
 text=text.replace('<h2>核验与接下来的完整实验</h2>',stressnote+'<h2>核验与接下来的完整实验</h2>')
 text=text.replace('<li><a href="validation.json">','<li><a href="interpretation.json">结果解释与证据边界</a></li><li><a href="validation.json">')
 (R/'review.html').write_text(text)
 fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 axs[0].bar(['Flood','Protected'],[100*f.newly_unreachable_share,100*p.newly_unreachable_share],color=['#ba5149','#468a7d']);axs[0].set(ylabel='Newly unreachable proxy (%)',title='Unknown roads kept available')
 axs[1].bar(['Lost in flood','Restored'],[eff.flood_new_lost,eff.restored_vehicle_trip_proxy],color=['#ba5149','#468a7d']);axs[1].set(ylabel='Assumed vehicle trips (17h proxy)',title='Fixed4.77km ideal protection')
 axs[2].bar(['Available','Withheld'],[100*n.normal_reachable/n.requested,100*covered],color=['#5b8094','#b8a072']);axs[2].set(ylabel='Normal demand reachable (%)',title='Baseline evidence coverage',ylim=[0,105])
 fig.suptitle('First snapshot scenario | assumed central demand | no congestion feedback')
 fig.savefig(R/'paired_outcomes.png',dpi=150);plt.close(fig)
 # Cross-field checks supplement the original solver validation,without changing it.
 assert abs(eff.flood_new_lost-eff.restored_vehicle_trip_proxy-eff.remaining_new_lost)<1e-8
 assert covered<.02 and not notes['withheld_branch_interpretation'].startswith('Upper')
 assert abs(f.newly_unreachable_share-row(0,'flood').newly_unreachable_share)<1e-12
 checks={'measure_loss_ledger_conserved':True,'strict_branch_low_baseline_coverage_disclosed':True,'scenario_weights_not_people':True,'no_empirical_robustness_from_uniform_scaling_claimed':True,'initial_review_and_figure_retained':True}
 (R/'interpretation_validation.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
 print(json.dumps(notes,ensure_ascii=False,indent=2))
 print(pressure[(pressure.branch==0)&pressure['shape'].eq('base')].to_string(index=False))
if __name__=='__main__':main()
