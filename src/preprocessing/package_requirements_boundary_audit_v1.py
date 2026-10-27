#!/usr/bin/env python3
"""Package reference-only boundary-flow coverage and target-review evidence."""
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_target_boundary_budget_audit_v1';OLD='three_ward_protection_requirements_register_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG;OP=ROOT/'data/processed'/OLD

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 doc=sha(ROOT/'docs/AnaSOP.md');nv=json.loads((R/'native_validation.json').read_text());assert nv['passed'] and len(nv['cases'])==10
 conversions=[('boundary_flow_evidence.csv','zone_boundary_flow_evidence_preprocessed.parquet'),('receiving_exchange_trajectories.csv','zone_receiving_exchange_preprocessed.parquet'),('zone_flow_coverage.csv','zone_budget_readiness_preprocessed.parquet')]
 frames={}
 for source,name in conversions:
  d=pd.read_csv(R/source,float_precision='round_trip');d.to_parquet(P/name,index=False);pd.testing.assert_frame_equal(d,pd.read_parquet(P/name));frames[source]=d
 b=frames['boundary_flow_evidence.csv'];tr=frames['receiving_exchange_trajectories.csv'];z=frames['zone_flow_coverage.csv']
 assert not b.duplicated(['Case','Zone','Native Face']).any() and not tr.duplicated(['Case','Zone','Frame']).any()
 assert len(z)==207 and len(tr)==22563 and tr.groupby(['Case','Zone']).size().eq(109).all()
 assert z['Required Pump Capacity'].isna().all() and not z['Budget Complete'].any()
 assert tr['Unknown Ground Exchange'].isna().all() and tr['Unknown Applied Rainfall'].isna().all()
 storage=pd.read_parquet(OP/'zone_storage_trajectories_preprocessed.parquet')
 m=storage.merge(tr,on=['Case','Zone','Frame'],validate='one_to_one',suffixes=('',' Flow'))
 clock_difference=float(np.max(abs(m['Actual Time']-m['Actual Time Flow'])))
 assert clock_difference<1e-8
 m=m.rename(columns={'Actual Time Flow':'Flow Actual Time'}).sort_values(['Case','Zone','Frame'])
 first=m.groupby(['Case','Zone'])['Stored Volume'].transform('first');m['Storage Change Since First Map']=m['Stored Volume']-first
 m['Unresolved Combined Net Supply']=m['Storage Change Since First Map']+m['Receiving Outflow Since First Map']-m['Receiving Inflow Since First Map']
 m['Unresolved Supply Meaning']='combined applied rainfall plus unobserved ground exchange and numerical/storage output differences;not pump demand or numerical residual'
 m.to_parquet(P/'zone_partial_budget_trajectories_preprocessed.parquet',index=False)
 assert np.allclose(m['Storage Change Since First Map'],m['Unresolved Combined Net Supply']-m['Receiving Outflow Since First Map']+m['Receiving Inflow Since First Map'],atol=1e-7,rtol=0)
 # Retain all existing groups. Separate two candidate service families without selecting top-N.
 roads=pd.read_parquet(OP/'road_target_groups_preprocessed.parquet').copy()
 roads['Protection Review Family']=np.where(roads['Nonincident Anchor Loss Under Group Probe']>0,'area_access_dependency',np.where(roads['Backbone Class Candidate'],'backbone_with_portal_bypass','local_exposure_context'))
 roads['Final Target Frozen']=False;roads['Protection Review Limit']='structural closure probe,not observed simultaneous isolation;unknown exposure remains assumed available'
 roads.to_parquet(P/'road_protection_review_preprocessed.parquet',index=False)
 families=roads.groupby('Protection Review Family',as_index=False).agg(**{'Road Groups':('Road Group','size'),'Exposure Candidate Length':('Exposure Candidate Length','sum')})
 families.to_parquet(P/'road_protection_family_summary_preprocessed.parquet',index=False)
 cs=[]
 for case,zz in z.groupby('Case'):
  bb=b[b.Case==case];tt=m[m.Case==case]
  cs.append({'Case':case,'Zones':len(zz),'Zones With Saved Receiving Flow':int((zz['Receiving Face Memberships']>0).sum()),'Unobserved Open Ground Face Memberships':int(zz['Unobserved Open Ground Faces'].sum()),'Additional Configured Closed Memberships':int(bb['Flow Evidence'].eq('configured_closed_ground_no_saved_flow').sum()),'Receiving Outflow Since First Map':float(zz['Saved Receiving Outflow'].sum()),'Receiving Inflow Since First Map':float(zz['Saved Receiving Inflow'].sum()),'Complete Zone Budgets':int(zz['Budget Complete'].sum())})
 case=pd.DataFrame(cs);case.to_parquet(P/'case_boundary_budget_summary_preprocessed.parquet',index=False)
 recv=b['Boundary Type'].eq('prescribed_water_exchange');assert b.loc[recv,'Flow Evidence'].eq('saved_receiver_flow_verified').all()
 assert b.loc[~recv,'Section Name'].isna().all()
 assert ((z['Receiving Face Memberships']==0)|(z['Receiving Face Memberships']>0)).all()
 # Ground-ground faces shared by two zones are two oriented memberships,not two case-wide faces.
 duplicate_memberships=len(b)-len(b[['Case','Native Face']].drop_duplicates())
 stats={'road_families':families.to_dict(orient='records'),'cases':len(case),'zones':len(z),'high_stage_cases_with_receiving_inflow':int(((case['Case'].str.contains('__H__'))&(case['Receiving Inflow Since First Map']>0)).sum()),'zones_with_saved_receiving_faces':int((z['Receiving Face Memberships']>0).sum()),'zones_without_saved_receiving_faces':int((z['Receiving Face Memberships']==0).sum()),'boundary_memberships':len(b),'receiving_memberships_verified':int(recv.sum()),'unobserved_open_ground_memberships':int(z['Unobserved Open Ground Faces'].sum()),'additional_configured_closed_ground_memberships':int(b['Flow Evidence'].eq('configured_closed_ground_no_saved_flow').sum()),'shared_oriented_boundary_memberships':duplicate_memberships,'complete_local_budgets':0,'max_previous_parquet_clock_representation_difference_s':clock_difference,'largest_global_prefix_check_difference_m3':max(c['max_global_prefix_budget_difference_m3'] for c in nv['cases'])}
 decisions={'scope':'Reference-only existing-output audit continuing requirements-led measure design','is_final_variable':'no','AnaSOP_changed':False,'source_variables_renamed':False,'rules':['Preserve all road candidates and original evidence;no targets frozen','All receiver crosssections match unique native faces with verified land-to-water orientation','Use actual step dt and solver float32 applied-volume convention;not nominal output timestamps or trapezoids','Integrate each face signed flow first;report gross directions separately','109-frame cumulative exchange starts at actual first map,not an assumed zero state','Full-topology perimeter includes configured closed ground faces omitted from open graph;no saved ground flux claimed','Unobserved local rainfall and ground flux remain missing;their combined storage identity is not an independently closed budget','Fixed receiver water level is not receiving allowance;no required flow,pump count or new simulations']}
 dump(R/'decisions.json',decisions)
 units={'Flow Actual Time':'s','Service Closure Threshold':'m','Stricter Slow Threshold':'m','Qualified Group Length':'km','Chiyoda Length':'km','Chuo Length':'km','Minato Length':'km','Actual Time':'s','Analysis Time':'s','Exposure Candidate Length':'km','Maximum Net Receiving Rate':'m3/s,positive out','Minimum Net Receiving Rate':'m3/s,negative in','Required Pump Capacity':'m3/s,unresolved'}
 dictionary=[];exports=[]
 previous_dictionary=pd.read_csv(ROOT/'data/exp/data-preprocessing'/OLD/'data_dictionary.csv')
 for _,entry in previous_dictionary.iterrows():
  units.setdefault(entry['Readable Field'],entry['Unit'])
 for f in sorted(P.glob('*_preprocessed.parquet')):
  d=pd.read_parquet(f);assert all(c.isascii() and c==c.strip() for c in d)
  for c in d.columns:
   unit=units.get(c,'m3' if any(k in c for k in ['Storage','Volume','Supply','Outflow','Inflow','Exchange','Rainfall']) and c not in ['Unresolved Supply Meaning'] else 'identifier/category/boolean/count;see source register')
   dictionary.append({'Dataset':f.name,'Readable Field':c,'Unit':unit,'Role':'reference-only construction diagnostic','Missing Rule':'retain missing;no zero filling of unobserved terms'})
  exports.append({'path':str(f.relative_to(ROOT)),'rows':len(d),'sha256':sha(f)})
 pd.DataFrame(dictionary).to_csv(R/'data_dictionary.csv',index=False)
 dump(P/'export_manifest.json',{'outputs':exports,'native_validation':str((R/'native_validation.json').relative_to(ROOT)),'scripts':{n:sha(ROOT/'src/preprocessing'/n) for n in ['audit_requirements_boundary_flows_v1.py','package_requirements_boundary_audit_v1.py']},'source_register':str(OP.relative_to(ROOT)),'source_register_sha256':{n:sha(OP/n) for n in ['road_target_groups_preprocessed.parquet','zone_storage_trajectories_preprocessed.parquet','zone_demand_feasibility_readiness_preprocessed.parquet']}})
 style='body{font:16px/1.75 system-ui;max-width:1250px;margin:28px auto;padding:0 24px;color:#203b49}h1{font-size:27px}h2{font-size:22px}aside{padding:18px;background:#edf4f7;border-left:4px solid #467e91}.warn{padding:16px;background:#fff3df}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border:1px solid #d4e0e6;text-align:left}.scroll{overflow:auto}a{color:#14667e}'
 links=''.join(f'<li><a href="../../../processed/{TAG}/{f.name}">{html.escape(f.name)}</a></li>' for f in sorted(P.glob('*_preprocessed.parquet')))
 preview=roads.sort_values('Nonincident Anchor Loss Under Group Probe',ascending=False).head(15)[['Road Group','Road Name','Protection Review Family','Exposure Candidate Length','Nonincident Anchor Loss Under Group Probe']]
 text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>保护道路复核与控制区水量证据</title><style>{style}</style><h1>保护道路复核与控制区水量证据</h1><aside>完成5场历史降雨×2水位的已有输出核查。全部10组断面映射、流向和逐步积分通过核验。<b>这推进了接收端交换证据，但尚未闭合局部水量收支，不能据此定泵数。</b>未启动新模拟、训练或人口计算。</aside><h2>1. 要保护的道路分开讨论</h2><p>2152组候选道路全部保留。577组在结构压力检查下影响组外端点对既定外部联系点的双向可达性；另93组为仍存在该类绕行的骨干路；其余1482组作为局部暴露背景。不是从三区任取前19组，也未将节点数当成人数。这里只整理保护理由，没有冻结最终目标。</p><div class="scroll">{families.to_html(index=False,float_format=lambda v:f'{v:,.3f}')}</div><p>结构压力检查合并了不同事件出现的候选封闭路段，<b>不能等同于某场暴雨的实际同时断连</b>。有外部联系绕行不代表旅行时间或所有OD不受损；未知桥隧暴露仍保留原假设。下表仅作浏览，完整清单保留。</p><div class="scroll">{preview.to_html(index=False,float_format=lambda v:f'{v:,.3f}')}</div><h2>2. 河道交换：已有断面能恢复什么</h2><p>每组6763条保存断面全部唯一匹配原生陆水边界面，几何方向明确为陆地→指定水位边界。按每一步实际dt和原求解器float32流量×dt口径积分，分别保存流出和倒灌，随后精确对应109帧的实际步号。没有用名义输出时刻作梯形积分。全域累计出入水与原同一步预算的最大前缀差为{stats['largest_global_prefix_check_difference_m3']:.3g} m³，所有已关闭陆水面的保存流量为零。</p><p>207个情景内控制区记录中，{stats['zones_with_saved_receiving_faces']}个接触有保存断面的接收边界，{stats['zones_without_saved_receiving_faces']}个没有这类断面。后者不表示该区不排水，仍可经地表交换。累计量从第一帧的真实时刻起算，第一帧不是严格0秒；原时刻未修改。与上一轮Parquet时刻表示的最大差为{clock_difference:.3g}秒，属于文本解析浮点表示差；两列均保留并按原帧号及实际步号连接。</p><div class="scroll">{case.to_html(index=False,float_format=lambda v:f'{v:,.1f}')}</div><p>表中体积单位m³；同一组情景内按唯一陆水格面归属汇总，控制区可含必要缓冲区，因此不把它称作“三区内排水总量”。两种水位情景和不同事件分别报告，不能累加成一次排水需求。</p><p><b>高水位5组都有边界向这些控制区的来水，单情景累计约43.5–56.0万m³；低水位5组这一项为零。</b>这是原指定水位情景下的模型交换，不是东京实测倒灌量，也不表示可接收新增抽排水。抽排目标和接收端不能分开定。</p><h2>3. 控制区边界还缺什么</h2><p>本次用完整原生拓扑复核边界，补列{stats['additional_configured_closed_ground_memberships']:,}条原开放连通图没有列入的已配置封闭地面格面记录。它们标为“配置封闭但无保存地面流量”，不将配置状态冒充逐步实测诊断。另有{stats['unobserved_open_ground_memberships']:,}条开放地面边界成员记录缺少逐步格面流量；同一地面格面可分别属于相邻两控制区，必须按方向处理，不能重复当全域流量相加。</p><div class="warn">Δ存水＋已知流出−已知流入，只能得到尚未拆分的净供水组合项。它混合了持续降雨、地面来去水及数值／储量输出差异，<b>既不是独立验证的水量残差，也不是需要抽走的水量</b>。局部降雨和地面交换缺失值仍为空；完整局部预算为0/207，所需抽排能力全部未定。</div><h2>4. 接下来怎样形成可模拟的部署</h2><ol><li>沿用两类保护理由明确目标道路与服务时段，并保留所有其他道路作风险转移对照。</li><li>核对固定控制区内的原屋面分配降雨；对缺失地面格面，先检查求解器是否已有可导出的同一步通量接口。需要新输出时另列最小诊断重放方案，本轮不执行。</li><li>在需求和共享来水可解释后，再联结可布点位置、扬程能力、接收水位及允许接收量，推导有效排水量与点数。指定水位边界不等于河道有无限接收容量。</li></ol><p>原0.30m封路边界仅用于筛查，严格可通行余量和完整控制区目标水位仍待明确。未做最终保护承诺、工程优化或新的措施效果结论。AnaSOP、原准入规则和三张已认可图未修改。</p><h2>完整记录</h2><ul>{links}</ul><p><a href="validation.json">整体验证</a> · <a href="native_validation.json">逐情景原生核验</a> · <a href="data_dictionary.csv">字段与单位</a> · <a href="README.md">复现方法</a> · <a href="../{OLD}/review.html">上一轮道路与储量清单</a></p></html>'''
 (R/'review.html').write_text(text)
 (R/'README.md').write_text('''# Existing-output target and boundary budget audit

Reference-only extension of the requirements-led road and storage register. No final variables or targets are promoted. AnaSOP, source runs and accepted figures remain unchanged. No solver, training, new population computation, or sealed-test access.

Run from the project root:

1. `.frame-control/runtimes/physical-measures-py312/bin/python src/preprocessing/audit_requirements_boundary_flows_v1.py`
2. `.venv/bin/python src/preprocessing/package_requirements_boundary_audit_v1.py`

Each receiver section is matched one-to-one to native topology and oriented land-to-water. History nominal labels are checked, but actual step times and dt drive integration. Float32(q*dt),matching original applied-volume budgets,is accumulated in float64. Global prefix directional volumes must agree with the original ledger within1e-5m3. Closed receiver faces must have exactly zero saved flow. Map cumulative exchanges subtract the actual first-map baseline;the first map occurs after a tiny initialization step,not exactly at zero.

Full native zone perimeters add configured closed ground faces omitted from the open graph. No saved ground-face flow is claimed. Shared ground faces have separate zone-oriented memberships. Receiver faces belong to their single land-side zone;do not sum repeated many-to-many road-zone joins. Fixed case zones are observed-wet envelopes,not catchments. Unknown rainfall and ground exchanges remain missing. Storage change plus known net receiving outflow is a combined unidentified supply term,not a numerical-residual test or pump-demand estimate. A boundary with no saved receiver section is not proof of no drainage.

All road groups are retained. Area-access dependency and backbone-with-bypass families are proposed service-review categories based on existing structural probes,not observed flood isolation,traffic volume,people or a frozen allocation. No point count,unit capacity or receiver allowance is prescribed.
''')
 assert sha(ROOT/'docs/AnaSOP.md')==doc
 dump(R/'validation.json',{'passed':True,**stats,'109_actual_frames_per_zone':True,'source_values_and_masks_preserved':True,'missing_terms_preserved':True,'no_capacity_estimate':True,'AnaSOP_unchanged':True,'no_final_targets_frozen':True,'new_simulations':0,'new_people_counts':False,'all_native_case_checks_passed':True})
 print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
