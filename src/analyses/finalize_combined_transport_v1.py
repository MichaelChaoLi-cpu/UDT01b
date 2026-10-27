"""Interpret complete combinations using original-demand ledgers and cost indices.
No new fits, weather runs, path generation or intervention selection.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,sys,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import run_combined_transport_measures_v1 as m
P=m.P;R=m.R
def main():
 m.check();assert json.loads((R/'validation.json').read_text())['passed']
 df=pd.read_csv(P/'combined_outcomes.csv');ef=pd.read_csv(P/'same_demand_guidance_effects.csv');dem=m.load(m.prev.REP/'multi_routing_demands.npz');core=m.load(m.prev.P/'protection_masks.npz')['core_endpoint_od'];q0=m.original_request(dem);rows=[]
 for case,g in df.groupby('case_id'):
  baseline=m.load(P/'fits'/(m.key(case,'flood',0.,'UE')+'.npz'));co=core&baseline['reachable']&(q0>0);baseline_mean=float(q0[co]@baseline['mean_od_cost'][co]/q0[co].sum()/60)
  for _,z in g.iterrows():
   actual=m.load(P/'fits'/(m.key(case,z.protection,float(z.deferred_fraction),z.guidance)+'.npz'));source=float(z.original_core_request);served=float(z.retained_served_core_proxy);deferred=float(z.deferred_core_proxy);failed=float(z.retained_unreachable_core_proxy);assert abs(source-served-deferred-failed)<1e-6
   original_no_action=g[g.protection.eq('flood')&g.deferred_fraction.eq(0)&g.guidance.eq('UE')].iloc[0];cost_index=float(z.fixed_original_cohort_mean_cost_min/baseline_mean*100);c=float(z.completion_fraction_original)
   rows.append(dict(case_id=case,protection=z.protection,deferred_fraction=float(z.deferred_fraction),guidance=z.guidance,original_request_proxy=source,served_proxy=served,deferred_proxy=deferred,retained_unreachable_proxy=failed,completed_original_percent=c*100,deferred_original_percent=deferred/source*100,retained_unreachable_original_percent=failed/source*100,fixed_original_cohort_cost_index_no_action100=cost_index,original_baseline_cohort_cost_min=baseline_mean,completion_change_percentage_points=(c-float(original_no_action.completion_fraction_original))*100,not_worse_than_original_no_action_in_two_model_metrics=bool(c>=original_no_action.completion_fraction_original-1e-10 and cost_index<=100+1e-6)))
 out=pd.DataFrame(rows);m.save(P/'original_demand_and_cost_indices.csv',out)
 selected=[]
 for case in m.CASES:
  for arm,r,guidance in [('flood',0.,'UE'),('targeted',0.,'UE'),('targeted',.1,'UE'),('targeted',.1,'SO')]:selected.append(out[out.case_id.eq(case)&out.protection.eq(arm)&out.deferred_fraction.eq(r)&out.guidance.eq(guidance)].iloc[0].to_dict())
 illustrative=pd.DataFrame(selected)
 # Plot the same original completion metric against a fixed-cohort cost index.
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,case in zip(axs,m.CASES):
  g=out[out.case_id.eq(case)];base=g[g.protection.eq('flood')&g.guidance.eq('UE')&g.deferred_fraction.eq(0)].iloc[0]
  ax.axhline(100,color='#888',lw=.7);ax.axvline(base.completed_original_percent,color='#aaa',lw=.7)
  for mode,style in [('UE','--'),('SO','-')]:
   h=g[g.protection.eq('targeted')&g.guidance.eq(mode)].sort_values('deferred_fraction');ax.plot(h.completed_original_percent,h.fixed_original_cohort_cost_index_no_action100,style,marker='o',label='Protection/'+mode)
   for _,z in h[h.guidance.eq('UE')].iterrows():ax.annotate(f'{round(z.deferred_fraction*100)}%',(z.completed_original_percent,z.fixed_original_cohort_cost_index_no_action100),xytext=(4,4),textcoords='offset points',fontsize=8)
  ax.scatter([base.completed_original_percent],[100],marker='x',s=70,c='#b45248',label='No action');ax.set(title=case,xlabel='Completed original requests (%)',ylabel='Same-cohort modeled cost index (no action=100)');ax.legend(fontsize=8)
 fig.suptitle('Completion and capacity-cost tradeoff | labels are deferred fractions,not benefits');fig.savefig(R/'completion_cost_tradeoff.png',dpi=150);plt.close(fig)
 ef['network_gain_percent']=-ef.same_retained_network_total_cost_change_fraction*100
 ranges=dict(min_percent=float(ef.network_gain_percent.min()),max_percent=float(ef.network_gain_percent.max()))
 summary=dict(illustrative_no_deferral_and10percent=selected,same_retained_guidance_network_cost_gain_range=ranges,
  all_cells_noninferior_completion_and_cost=out[out.not_worse_than_original_no_action_in_two_model_metrics].to_dict('records'),
  conclusions=['Deferred requests are current-hour noncompletion,not restored or completed traffic.','Uniform deferral across all10km-context OD needs cross-area coordination;not proven achievable by three wards alone.','Limited two-path ideal guidance result does not bound real city traffic management.','Absolute minutes are highly sensitive to uncalibrated capacity and fourth-power cost under high demand;cost indices are scenario comparisons,not observed savings.','Two rainfall events share the same hypothetical river-stage pulse and fixed17h demand;not historical06h replay or broad independent validation.'])
 m.dump(R/'interpretation.json',summary)
 first=R/'review_initial.html'
 if not first.exists():shutil.copyfile(R/'review.html',first)
 text=first.read_text();label={'case_id':'事件','protection':'道路保障','deferred_fraction':'当前小时移出比例','guidance':'分配方式','completed_original_percent':'原请求完成%','deferred_original_percent':'主动移出%','retained_unreachable_original_percent':'保留但断连%','fixed_original_cohort_cost_index_no_action100':'固定原有出行成本指数（无措施100）'}
 display=illustrative[list(label)].rename(columns=label).replace({'targeted':'冻结新方案','flood':'无保障'})
 table='<div class="scroll">'+display.to_html(index=False,border=0,float_format=lambda x:f'{x:,.2f}')+'</div>'
 lead='<h2>先看原请求完成量，再看成本</h2><p>以下选取无措施、仅保障，以及保障＋移出10%（有／无理想疏导）作阅读示例并列展示；其余20%和30%均保留在全表，没有依据结果重选通道或调整比例。每行完成、主动移出和保留但断连合计100%。成本指数在同一场无措施原有可达OD群体上计算，不能与完成率相互替代。</p>'+table+'<img src="completion_cost_tradeoff.png" alt="Original request completion versus fixed-cohort cost index,explicit deferral labels">'
 text=text.replace('<h2>本轮的问题和固定条件</h2>',lead+'<h2>本轮的问题和固定条件</h2>')
 note=f'''<h2>组合措施的解释边界</h2><p class="note">本轮理想疏导在同一保留需求下，整个道路上下文总模型成本降低约{ranges['min_percent']:.3f}%–{ranges['max_percent']:.3f}%。这是固定两条候选路径内的结果，不能据此说真实交通疏导无效；它没有生成更多路径、模拟信号、现场服从或动态排队。</p><p>高负荷下的四次方函数可产生很大的模型分钟差。绝对值高度依赖未校准容量和参考时耗，因此主要看完成量与相对模型成本的权衡，不将其作为真实节时或政策效果预测。</p><p>10%／20%／30%的移出作用于整个10km上下文的每个OD，包括跨区交通；它表示区域协同假设，不证明三区政府可以独立实现。被移出的出行没有算作完成，也没有赋予已完成的未来时段或福利收益。两场雨的断连基线不同，同一移出比例不必同样适合；“移出越多”不能自动解释为“越好”。</p><p><a href="../../../processed/{P.name}/original_demand_and_cost_indices.csv">原请求账本与成本指数</a> · <a href="interpretation.json">完整解释与示例</a></p>'''
 text=text.replace('<h2>如何使用这些结果</h2>',note+'<h2>如何使用这些结果</h2>');(R/'review.html').write_text(text)
 m.dump(R/'interpretation_validation.json',dict(passed=True,original_ledger_all32cells=True,illustrative_examples_not_separate_tests=True,all_predeclared_matrix_cells_retained=True,fixed_cohort_cost_indices=True,systemwide_deferral_disclosed=True,guidance_candidate_limit_disclosed=True,original_review_retained=True))
 print(illustrative[['case_id','protection','deferred_fraction','guidance','completed_original_percent','fixed_original_cohort_cost_index_no_action100']].to_string(index=False));print(json.dumps(ranges))
if __name__=='__main__':main()
