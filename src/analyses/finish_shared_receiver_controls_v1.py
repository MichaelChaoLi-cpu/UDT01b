#!/usr/bin/env python3
"""Complete summaries from four already-qualified runs; never start a solver.

Reuse the installed NetCDF binary only for legacy helper imports. Main pandas,
numpy and Arrow stay in the project environment. No package installation.
"""
from pathlib import Path
import sys,json,hashlib,shutil,datetime,html
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
EXTRA=ROOT/'.frame-control/runtimes/physical-measures-py312/lib/python3.12/site-packages'
sys.path.append(str(EXTRA));sys.path.insert(0,str(ROOT/'src/analyses'))
import netCDF4,numpy as np,pandas as pd,pyarrow
import shared_receiver_controls_v1 as c
from shared_receiver_pilot_common_v1 import sha,dump
from temporal_transport_common_v1 import population_inputs
from run_temporal_road_population_v1 import signed_envelope,envelope
P=c.P;R=c.R

def direct_pair():
 blocks,stock,normal,infos=population_inputs();ward=blocks.ward.to_numpy(dtype=str)
 sels=[np.ones(len(stock),bool)]+[ward==w for w in ['13101','13102','13103']];scopes=['Three wards','Chiyoda','Chuo','Minato']
 temp=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1';rows=[]
 def node(access,key):
  f=access/'node_cache'/(str(key)+'.npz')
  if not f.exists():f=temp/'reachability_cache'/(str(key)+'.npz')
  with np.load(f) as z:
   assert hashlib.sha256(z['closed_packed'].tobytes()).hexdigest()==str(key)
   return np.unpackbits(z['node_ok_packed'],count=len(normal)).astype(bool)
 for stage in ['H','L']:
  aa=P/'batches/O1/access';bb=P/'batches/S1/access'
  with np.load(aa/'population'/f'E20240731__{stage}__U__o1.npz') as z:a=z['node_cache_keys'];ts=z['analysis_times'];ids=z['block_ids']
  with np.load(bb/'population'/f'E20240731__{stage}__U__s1.npz') as z:b=z['node_cache_keys'];assert np.array_equal(ts,z['analysis_times']) and np.array_equal(ids,z['block_ids'])
  for ti in range(109):
   for bi,bound in enumerate(['lower','upper']):
    x=node(aa,a[ti,bi,1]);y=node(bb,b[ti,bi,1]);delta=x.astype('i1')-y.astype('i1');delta[~normal]=0
    for buf,info in infos.items():
     lo,hi=signed_envelope(delta,info,stock);ra,rb=envelope(normal&~x&y,info,stock);na,nb=envelope(normal&x&~y,info,stock)
     for sel,scope in zip(sels,scopes):rows.append({'stage':stage,'bound':bound,'buffer_m':buf,'scope':scope,'frame':ti,'elapsed_s':float(ts[ti]),'net_lower':int(-hi[sel].sum()),'net_upper':int(-lo[sel].sum()),'restored_lower':int(stock[sel&ra].sum()),'restored_upper':int(stock[sel&rb].sum()),'new_loss_lower':int(stock[sel&na].sum()),'new_loss_upper':int(stock[sel&nb].sum())})
 df=pd.DataFrame(rows);assert (df.net_lower<=df.net_upper).all();df.to_csv(P/'s1_vs_o1_population_trajectories.csv',index=False)
 summary=[]
 for keys,g in df.groupby(['stage','bound','buffer_m','scope']):
  g=g.sort_values('frame');row=dict(zip(['stage','bound','buffer_m','scope'],keys))
  for k in ['net_lower','net_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper']:row[k+'_resident_hours']=float(np.trapezoid(g[k],g.elapsed_s)/3600)
  summary.append(row)
 out=pd.DataFrame(summary);out.to_csv(P/'s1_vs_o1_population_summary.csv',index=False);return out

def main():
 state=json.loads((R/'campaign_state.json').read_text());manifest=json.loads((P/'run_manifest.json').read_text());assert state['new_runs_started']==4 and len(state['completed_cases'])==4
 # Confirm the exact executed code is still present before invoking its finalizer.
 bound=json.loads((P/'execution_code_binding.json').read_text());assert all(sha(ROOT/k)==v for k,v in bound.items())
 archive=R/'finalization_failure_01'
 if state['status']=='held' and not archive.exists():
  archive.mkdir()
  for name in ['campaign_state.json','campaign_error.txt','review.html','delivery_validation.json']:
   if (R/name).exists():shutil.copy2(R/name,archive/name)
 audits=[]
 for run in manifest['runs']:
  ar=R/'batches'/run['contrast']/'execution'/run['case_id'];a=json.loads((ar/'audit.json').read_text());b=json.loads((ar/'pump_budget_summary.json').read_text())
  assert a['all_checks_passed'] and a['admission']['technical_reference'] and len(a['checks'])==28 and all(x['pass'] for x in a['checks'])
  assert b['passed'] and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values()) and a['summary']['negative_cells']==0
  audits.append({'case':run['case_id'],'original_gates':28,'pump_checks':8,'maximum_budget_residual_m3':a['summary']['max_budget_residual_m3'],'negative_cells':0})
 for arm in ['O1','S1']:
  ar=R/'batches'/arm
  for name in ['postprocessing_validation.json','expanded_transfer_validation.json','access/validation.json']:assert json.loads((ar/name).read_text())['passed']
 c.finalize();direct=direct_pair()
 comp=json.loads((R/'comparison_summary.json').read_text());assert comp['passed'] and comp['road_common_support_exact']
 # Verify disjoint group partitions reproduce the regional paired road integral.
 groups=pd.read_csv(P/'group_continuity_vs_static.csv');regional=pd.read_csv(P/'comparison_summary.csv')
 for row in regional.to_dict('records'):
  g=groups[groups.contrast.eq(row['contrast'])&groups.stage.eq(row['stage'])&groups.bound.eq('lower')]
  assert abs(g.net_restored_edge_km_h.sum()-row['incremental_closure_reduction_vs_static_km_h'])<1e-8
 pair=pd.read_csv(P/'paired_population_vs_static_summary.csv');assert (pair.net_lower_resident_hours<=pair.net_upper_resident_hours+1e-8).all()
 # Assemble a compact overview with the explicit no-action comparator.
 tables=[]
 old=pd.read_csv(c.OLD/'paired_road_effects.csv');old=old[old['Depth Branch'].eq('lower')&old.Scope.eq('Three wards')]
 for name,bp in [('Original static',c.OLD),('O1 allocation',P/'batches/O1'),('S1 placement',P/'batches/S1')]:
  d=pd.read_csv(bp/'paired_road_effects.csv');d=d[d['Depth Branch'].eq('lower')&d.Scope.eq('Three wards')]
  for x in d.to_dict('records'):tables.append({'Scheme':name,'Stage':x['Stage'],'Avoided closure exposure km h':x['Closure Reduction km h'],'Reduction percent':x['Closure Reduction Percent']})
 road=pd.DataFrame(tables);road.to_csv(P/'final_road_overview.csv',index=False)
 pop=pair[pair.bound.eq('lower')&pair.buffer_m.eq(10)&pair.scope.eq('Three wards')].copy();pop['Net paired benefit vs static (resident h)']=pop.apply(lambda x:f'[{x.net_lower_resident_hours:,.1f}, {x.net_upper_resident_hours:,.1f}]',axis=1)
 dd=direct[direct.bound.eq('lower')&direct.buffer_m.eq(10)&direct.scope.eq('Three wards')].copy()
 dd['Net S1 benefit vs O1 (resident h)']=dd.apply(lambda x:f'[{x.net_lower_resident_hours:,.1f}, {x.net_upper_resident_hours:,.1f}]',axis=1)
 shift=[]
 for arm in ['O1','S1']:
  df=pd.read_csv(P/'batches'/arm/'expanded_head_transfer_summary.csv');df=df[df.Scope.eq('Three wards')].copy();df.insert(0,'Contrast',arm);shift.append(df)
 transfer=pd.concat(shift,ignore_index=True)
 transfer['Maximum head-rise area m2']=transfer['All Supported Certain Head Rise Area']*1e6
 transfer[['Contrast','Stage','Maximum head-rise area m2']].to_csv(P/'final_transfer_overview.csv',index=False)
 pop_crosses_zero=bool(((dd.net_lower_resident_hours<=0)&(dd.net_upper_resident_hours>=0)).all())
 comp['s1_vs_o1_population_primary']=dd.to_dict('records');comp['interpretation']={'roads':'S1 has greater regional avoided closure exposure under both stages.','population':'Direct S1-vs-O1 candidate intervals must govern;not ordered by marginal benefit bounds.','transfer':'No >=0.10m certain head-rise cells in O1;S1 low stage has100m2 in Chiyoda. No new qualified road closures versus no-action. Downstream receiver storage remains excluded.','scope':'one July rainfall parent,two prescribed stage scenarios;same total flow allowance,not same financial cost.'};dump(R/'comparison_summary.json',comp)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();state.pop('error',None);state.update(status='qualified_complete',current_case=None,final_summary_status='complete',final_summary_completed_utc=now,finalization_recovery='Existing installed runtime reused;no simulation or gate changed.')
 dump(R/'campaign_state.json',state)
 c.report()
 conclusion='<section id="final-comparison"><h2>最终比较：道路收益与人口权衡</h2><p><b>4／4组通过原28项数值检查及8项抽排核验，负体积均为零。</b>最后统一汇总曾因主环境缺少NetCDF4而中断，现已复用既有依赖补齐，没有重跑模拟或改动准入规则。</p><p><b>改布局S1的封路累计暴露改善更大；改调配O1的人口通达性改善更稳健。</b>下表的减少比例均相对同一场暴雨、同水位的无措施情景，指标为封路长度对时间的积分。</p>'+road.to_html(index=False,float_format=lambda x:f'{x:,.3f}')
 conclusion+='<h3>人口通达性：与原静态部署直接配对</h3>'+pop[['contrast','stage','Net paired benefit vs static (resident h)']].to_html(index=False)+'<p>单位为居民·小时，不是独立受困人数，也不是置信区间。O1在主规则下未增加相对原静态部署的人口通达性损失；S1撤去部分原位置的保护，因此部分接入会比原静态方案变差，净改善区间跨零。不能因S1相对无措施的改善区间端点较大，就宣称其人口效果全面优于O1。</p>'
 conclusion+='<h3>改布局与改调配直接比较</h3>'+dd[['stage','Net S1 benefit vs O1 (resident h)']].to_html(index=False)+'<p>所有比较都在同一时刻、同一候选接入位置上配对后汇总，未使用两个边际区间相减。</p><h3>积水转移</h3>'+transfer[['Contrast','Stage','Maximum head-rise area m2']].to_html(index=False)+'<p>面积表示相对无措施情景、水头确定抬升≥0.10m的最大同时面积。O1高低水位及S1高水位为零；S1低水位在千代田区出现100m²（一个10m网格单元），未引发共同已知道路上的新增封路。不能排除更小变化；规定水位接收端没有计算抽排水进入河道后的额外储水，不能据此排除真实下游风险。</p><p>这些结论来自一场暴雨和两种规定水位。同额度是相同总流量额度，不是相同设备、管线或经济成本。尚不能推广成普遍优劣。</p><p><a href="final_summary_validation.json">最终汇总核验</a> · <a href="comparison_summary.json">完整结果JSON</a></p></section>'
 f=R/'review.html';s=f.read_text();assert '<h2>O1</h2>' in s;s=s.replace('<h2>O1</h2>',conclusion+'<h2>O1</h2>',1);f.write_text(s)
 validation={'passed':True,'four_original_numerical_audits_passed':True,'four_pump_audits_passed':True,'original_rules_unchanged':True,'group_road_partition_reconciles':True,'direct_population_pairs_ordered':True,'direct_s1_vs_o1_intervals_cross_zero_primary':pop_crosses_zero,'no_new_hydraulic_runs':True,'runtime_bridge':{'path':str(EXTRA),'netCDF4':netCDF4.__version__,'numpy':np.__version__,'pyarrow':pyarrow.__version__},'audit_receipts':audits,'script_sha256':sha(Path(__file__)),'completed_jst':datetime.datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()}
 dump(R/'final_summary_validation.json',validation)
 delivery=json.loads((R/'delivery_validation.json').read_text());delivery.update(final_summary_complete=True,final_summary_validation_sha256=sha(R/'final_summary_validation.json'));dump(R/'delivery_validation.json',delivery)
 if (R/'campaign_error.txt').exists():
  target=archive/'campaign_error.txt'
  assert target.exists() and target.read_bytes()==(R/'campaign_error.txt').read_bytes();(R/'campaign_error.txt').unlink()
 with (R/'README.md').open('a') as out:out.write('\n\nFinal summary recovery: `.venv/bin/python src/analyses/finish_shared_receiver_controls_v1.py`. This reads already-qualified outputs only and reuses the installed NetCDF4 site path for legacy helper imports; no installation or hydraulic retry. Direct S1-vs-O1 population pairing is included, rather than marginal interval subtraction. Old finalization failure is preserved in finalization_failure_01/.\n')
 files=list(P.glob('*.csv'))+[R/'review.html',R/'comparison_summary.json',R/'final_summary_validation.json',R/'delivery_validation.json',R/'campaign_state.json',Path(__file__)]
 dump(R/'final_delivery_manifest.json',{'passed':True,'files':[{'path':str(x.relative_to(ROOT)),'sha256':sha(x)} for x in files]})
 print('FINAL_SUMMARY_COMPLETE',pop_crosses_zero,flush=True);print(dd[['stage','net_lower_resident_hours','net_upper_resident_hours']].to_string(index=False))
if __name__=='__main__':main()
