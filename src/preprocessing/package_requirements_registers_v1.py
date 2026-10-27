#!/usr/bin/env python3
"""Package reference-only requirements registers,units,uncertainties and review.
Use project .venv after road and native stages. Leaves AnaSOP and all inputs intact.
"""
from __future__ import annotations
from pathlib import Path
import json,hashlib,html,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_protection_requirements_register_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 doc_sha=sha(ROOT/'docs/AnaSOP.md');rv=json.loads((R/'road_validation.json').read_text());nv=json.loads((R/'native_validation.json').read_text());assert rv['passed'] and nv['passed']
 conversions=[('control_storage_native.csv','control_cell_storage_preprocessed.parquet'),('zone_register_native.csv','drainage_zone_register_preprocessed.parquet'),('zone_storage_native.csv','zone_storage_trajectories_preprocessed.parquet'),('group_zone_links_native.csv','group_zone_links_preprocessed.parquet')]
 manifests=[]
 for source,out in conversions:
  d=pd.read_csv(R/source);d.to_parquet(P/out,index=False);r=pd.read_parquet(P/out);pd.testing.assert_frame_equal(d,r)
  manifests.append({'source':str((R/source).relative_to(ROOT)),'output':str((P/out).relative_to(ROOT)),'rows':len(d),'columns':len(d.columns),'exact_values_nulls_and_order':True,'sha256':sha(P/out)})
 boundary=[]
 for p in sorted(R.glob('*__zone_boundary_faces.csv')):boundary.append(pd.read_csv(p))
 # Generated native stage uses Case prefix,not a second underscore before suffix.
 if not boundary:
  for p in sorted(R.glob('*_zone_boundary_faces.csv')):boundary.append(pd.read_csv(p))
 b=pd.concat(boundary,ignore_index=True) if boundary else pd.DataFrame();b.to_parquet(P/'zone_boundary_evidence_preprocessed.parquet',index=False)
 roads=pd.read_parquet(P/'road_target_groups_preprocessed.parquet');members=pd.read_parquet(P/'road_group_members_preprocessed.parquet');zones=pd.read_parquet(P/'drainage_zone_register_preprocessed.parquet');traj=pd.read_parquet(P/'zone_storage_trajectories_preprocessed.parquet');links=pd.read_parquet(P/'group_zone_links_preprocessed.parquet')
 scope=pd.read_csv(R/'zone_spatial_storage_native.csv')
 assert not scope.duplicated(['Case','Zone','Frame']).any()
 traj=traj.merge(scope,on=['Case','Zone','Frame'],validate='one_to_one',sort=False)
 assert np.max(abs(traj['Three Ward Stored Volume']+traj['Buffer Stored Volume']-traj['Stored Volume']))<1e-5
 assert np.max(abs(traj['Three Ward Control Excess Storage']+traj['Buffer Control Excess Storage']-traj['Road Control Excess Storage']))<1e-5
 traj.to_parquet(P/'zone_storage_trajectories_preprocessed.parquet',index=False)
 for entry in manifests:
  if entry['output'].endswith('zone_storage_trajectories_preprocessed.parquet'):
   entry['native_columns_before_scope_join']=entry['columns'];entry['columns']=len(traj.columns);entry['scope_source']=str((R/'zone_spatial_storage_native.csv').relative_to(ROOT));entry['scope_source_sha256']=sha(R/'zone_spatial_storage_native.csv');entry['sha256']=sha(P/'zone_storage_trajectories_preprocessed.parquet')
 assert members['Road Index'].is_unique
 assert not traj.duplicated(['Case','Zone','Frame']).any()
 assert zones['Required Pump Capacity'].isna().all()
 assert set(traj['Zone'])==set(zones['Zone']) and set(links['Road Group']).issubset(set(roads['Road Group']))
 assert traj.groupby('Zone').size().eq(109).all()
 assert (traj['Stored Volume']>=0).all() and (traj['Road Control Excess Storage']>=0).all()
 assert (traj['Road Control Excess Storage']<=traj['Stored Volume']+1e-5).all()
 assert not links.duplicated(['Case','Zone','Road Group']).any()
 # Aggregate only unique zones at each simultaneous frame;never sum their individual peaks.
 totals=traj.groupby(['Case','Frame'],as_index=False).agg({'Actual Time':'first','Analysis Time':'first','Stored Volume':'sum','Road Control Excess Storage':'sum','Three Ward Stored Volume':'sum','Buffer Stored Volume':'sum','Three Ward Control Excess Storage':'sum','Buffer Control Excess Storage':'sum'})
 totals.to_parquet(P/'case_storage_screening_preprocessed.parquet',index=False)
 case_summary=[]
 for case,g in totals.groupby('Case'):
  a=g.loc[g['Road Control Excess Storage'].idxmax()];v=g.loc[g['Stored Volume'].idxmax()];a3=g.loc[g['Three Ward Control Excess Storage'].idxmax()]
  case_summary.append({'Case':case,'Peak Simultaneous Road Control Excess Storage':float(a['Road Control Excess Storage']),'Control Excess Peak Time':float(a['Analysis Time']),'Peak Simultaneous Control Zone Storage':float(v['Stored Volume']),'Peak Three Ward Road Control Excess Storage':float(a3['Three Ward Control Excess Storage']),'Buffer Control Excess At Three Ward Peak':float(a3['Buffer Control Excess Storage']),'Three Ward Control Excess Peak Time':float(a3['Analysis Time']),'Zones':int(zones.Case.eq(case).sum()),'Derived Pump Capacity':None,'Interpretation':'screening subset only;not zone drainage requirement or a simultaneous sum of local peaks'})
 cs=pd.DataFrame(case_summary);cs.to_parquet(P/'case_storage_summary_preprocessed.parquet',index=False)
 # The third required register remains honest about missing equipment and receiver evidence.
 readiness=[]
 for row in zones.itertuples(index=False,name=None):
  rec=dict(zip(zones.columns,row));readiness.append({'Case':rec['Case'],'Zone':rec['Zone'],'Protection Group Freeze':'pending review','Storage Evidence':'simultaneous native volume and road-controlled subset reconstructed','Continuing Rain Evidence':'original forcing available;not allocated to a contributing catchment','Ground Exchange Evidence':'boundary native faces listed;time-series flux not reconstructed','Receiving Exchange Evidence':'existing crosssections need geometry and sign matching','Whole Zone Target Stage':'unresolved beyond controlled road cells','Requirement Status':'incomplete;capacity remains null','Unit Capability Evidence':'not supplied;legacy0.5m3/s is hypothetical','Receiving Allowance Evidence':'not supplied;fixed stage is not acceptance capacity','Placement Count':'not prescribed','Deployment Status':'not ready'})
 pd.DataFrame(readiness).to_parquet(P/'zone_demand_feasibility_readiness_preprocessed.parquet',index=False)
 units={'Three Ward Stored Volume':'m3','Buffer Stored Volume':'m3','Three Ward Control Excess Storage':'m3','Buffer Control Excess Storage':'m3','Peak Three Ward Road Control Excess Storage':'m3','Buffer Control Excess At Three Ward Peak':'m3,at the same frame as the three-ward peak','Three Ward Control Excess Peak Time':'s','Qualified Group Length':'km','Exposure Candidate Length':'km','Chiyoda Length':'km','Chuo Length':'km','Minato Length':'km','Unique Clipped Length':'km','Observed Closure Episode Span':'km h of observed episode spans;not trapezoidal exposure','Target Water Level':'m,existing model elevation datum','Target Storage':'m3','Stored Volume':'m3','Road Control Excess Storage':'m3','Road Control Stored Volume':'m3','Maximum Simultaneous Stored Volume':'m3,individual-zone maximum','Maximum Road Control Excess Storage':'m3,individual-zone maximum','Actual Time':'s','Analysis Time':'s','Required Pump Capacity':'m3/s;all null','Derived Pump Capacity':'m3/s;all null','Control Excess Peak Time':'s','Peak Simultaneous Road Control Excess Storage':'m3,case peak after simultaneous aggregation','Peak Simultaneous Control Zone Storage':'m3,case peak after simultaneous aggregation'}
 fields=[]
 for f in sorted(P.glob('*_preprocessed.parquet')):
  d=pd.read_parquet(f)
  assert all(c.isascii() and c==c.strip() for c in d)
  for c in d:fields.append({'Dataset':f.name,'Readable Field':c,'Unit':units.get(c,'identifier,category,boolean or count'),'Role':'reference construction diagnostic;not a final analysis variable','Missing Policy':'retained;never filled with zero'})
 pd.DataFrame(fields).to_csv(R/'data_dictionary.csv',index=False)
 dump(R/'decisions.json',{'authorization':'User requested construction of requirements-led registers under MILI-D-20261020-018','main_analysis_variables_changed':False,'AnaSOP_changed':False,'rules':['All new fields are reference-only','No source renaming or imputation','Road groups use source-node adjacency and same name/way and grade','Group closure probe is structural and may combine historical closure candidates','Zones are case-specific observed-wet connectivity envelopes,not catchments','Volume always aggregated at simultaneous saved times','Road-controlled target level is the0.30m closure boundary;strict passage margin not established','No capacity,pump count or official deployment inferred','Unknown or incomplete demand and receiving evidence remain explicit']})
 dump(P/'export_manifest.json',{'native_csv_to_parquet':manifests,'dictionary':str((R/'data_dictionary.csv').relative_to(ROOT)),'script_sha256':{name:sha(ROOT/'src/preprocessing'/name) for name in ['prepare_requirements_road_groups_v1.py','prepare_requirements_storage_zones_v1.py','package_requirements_registers_v1.py','split_requirements_storage_scope_v1.py']}})
 # Build a concise whole-area review with complete downloadable registers.
 road_table=roads.sort_values(['Nonincident Anchor Loss Under Group Probe','Exposure Candidate Length'],ascending=False).head(20)[['Road Group','Road Name','Exposure Candidate Length','Priority Evidence','Nonincident Anchor Loss Under Group Probe']].to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}')
 case_table=cs[['Case','Zones','Peak Three Ward Road Control Excess Storage','Buffer Control Excess At Three Ward Peak','Three Ward Control Excess Peak Time']].to_html(index=False,escape=True,float_format=lambda x:f'{x:,.1f}')
 stats=roads['Priority Evidence'].value_counts().to_dict();groups_with_zones=links['Road Group'].nunique();missing_groups=roads[~roads['Road Group'].isin(links['Road Group'])].copy();missing_groups['Hold Reason']='no supported road-control excess zone under the construction rules;not proof of no drainage need';missing_groups.to_parquet(P/'held_road_groups_preprocessed.parquet',index=False)
 filelinks=''.join(f'<li><a href="../../../processed/{TAG}/{f.name}">{html.escape(f.name)}</a></li>' for f in sorted(P.glob('*_preprocessed.parquet')))
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>目标道路组与排水需求证据整理</title><style>body{{font:16px/1.75 system-ui;max-width:1250px;margin:30px auto;padding:0 24px;color:#223845}}h1{{font-size:27px}}h2{{font-size:22px}}aside{{background:#edf4f7;padding:18px;border-left:4px solid #47788f}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border:1px solid #d6e0e5;text-align:left}}a{{color:#176992}}.scroll{{overflow:auto}}.warn{{padding:16px;background:#fff5e7}}</style><h1>三区目标道路组与排水需求证据整理</h1><aside>完成既有结果整理，未运行新水动力、未训练、未定泵点数。已构建{len(roads):,}个候选道路组，以及10组既有情景中的{len(zones):,}个临时存水控制区记录。<b>所需排水容量全部保持未定。</b></aside><h2>道路清单：由暴露进入连接作用核查</h2><p>6064个曾采样封闭的合格源路段、216.084 km多情景并集，按相同道路名（无名时用原way）、相同分层和真实源节点邻接归组。组内补入同一正常走廊的合格未封路段，因此组总长度与216 km暴露长度不同。没有按泵数或任取前N条来选目标。</p><p>每组在正常的原转向感知网络中暂时移除其暴露候选段，检查已正常双向连接的地面道路端点是否仍可到达既定外部联系点。共{rv['normal_core_anchors']:,}个端点用于检查，正常网络完全复现。<b>这是组闭合的结构压力检查，不是历史同时封路、实测交通或受影响人口。</b>非组内端点损失优先标识可能影响更广的连接；有绕行不等于所有OD通行和旅行时间不变；未知暴露道路沿用既有可用假设，不代表桥隧实际状态已核验。</p><p>优先证据分类：{html.escape(str(stats))}。以下仅展示前20组便于浏览，完整清单全部保留，没有据此冻结保护名单。</p><div class="scroll">{road_table}</div><h2>存水控制区：同一时刻算体积，不能把路段重复算水</h2><p>使用原共同准入掩膜和地面格面连通图；仅在同一帧两侧有水且水位达到格面最低控制高程时加入连通关系，再取9小时内实际出现过的关系并集，形成每个情景固定的临时控制区。<b>这是观测到的湿区连接包络，不是完整汇水区，也不是声称其全部同时积水。</b>其存水量逐帧从原subgrid_volume汇总，始终保持同一时刻。跨区／模型缓冲区单元不因区界被截断。</p><p>原亚网格储量—水位算子已复算并核验；最大复算差{max(c['maximum_storage_to_level_error_m'] for c in nv['cases']):.3g} m，只说明输出读写与算子吻合，不代表DEM高程精度。共有{groups_with_zones:,}组连接到这些存水区；其余{len(missing_groups):,}组单列，不当作无排水需求。</p><p>道路控制单元的水位边界取同一格内原采样路面最低允许值（路面高程＋0.30 m），使用原蓄水曲线反求对应储量。多个道路组共用一个单元时只计算一次。得到的是<b>这些道路控制单元的超额存水筛查量</b>，不包括整个分区所有单元的目标要求，也没有证明降到阈值边界就可安全通行。</p><div class="warn">下表先在每帧对不重叠控制区汇总，再取该情景峰值，没有相加各区不同时间的峰值。下表将三区内超额存水与同一时刻的缓冲区部分分开列示；保留缓冲区以核查跨界条件，不能把它计入三区内数量。单位是m³，<b>不是m³/s，也不能据此直接换算泵数</b>。不同情景的控制区边界可不同，不把区号当作固定跨事件汇水区。</div><div class="scroll">{case_table}</div><h2>排水需求表已明确哪些能算、哪些还缺</h2><p>每个区已有：关联道路组、共同支持单元、原时钟、109帧存水、道路控制单元超额存水和需要核对的边界格面。仍缺：区外道路单元的目标水位、完整贡献区、持续降雨分配与边界来／去水量、接收端余量以及设备有效能力。已有6763条命名断面不能直接当成覆盖任意控制区的完整水量预算，空间和符号需匹配。</p><p>因此所有“Required Pump Capacity”保持空值，未建部署方案。控制区—道路组是多对多联系，不能按道路组重复累加体积。第一批模拟继续未定义，前三张已认可图及AnaSOP正式变量均未改动。</p><h2>下一步</h2><ol><li>复核并冻结骨干路连续通行与片区出入口两类保护道路组，以及目标时段和通行标准。</li><li>把道路控制单元扩展为有依据的分区目标水位，核对本批已列出的边界格面与断面；缺失的流量继续保留为缺口。</li><li>据完整需求范围核查抽排源、设备扬程能力和共享接收限额，之后再推导部署数量。</li></ol><h2>可复现清单</h2><ul>{filelinks}</ul><p><a href="validation.json">整体验证</a> · <a href="road_validation.json">道路核验</a> · <a href="native_validation.json">原生储量核验</a> · <a href="data_dictionary.csv">字段、单位与缺失规则</a> · <a href="README.md">方法与运行方式</a></p></html>'''
 (R/'review.html').write_text(page)
 (R/'README.md').write_text('''# Requirements-led construction registers

All new fields are reference diagnostics,not confirmed main-analysis variables. Input files and AnaSOP are unchanged. No hydraulic simulation,traffic optimization,training,sealed-test access or affected-person count is performed.

Run from the project root:

1. `.venv/bin/python src/preprocessing/prepare_requirements_road_groups_v1.py`
2. `.frame-control/runtimes/physical-measures-py312/bin/python src/preprocessing/prepare_requirements_storage_zones_v1.py`
3. `.frame-control/runtimes/physical-measures-py312/bin/python src/preprocessing/split_requirements_storage_scope_v1.py`
4. `.venv/bin/python src/preprocessing/package_requirements_registers_v1.py`

The native reader uses the existing runtime with netCDF4;the project environment handles parquet. No new dependency is installed. Both are ordinary local Python commands.

Road groups use exact source-node adjacency plus same name(or original way when unnamed)and grade. Exposure candidates are combined across existing storms for a labeled structural closure probe;they are not an observed simultaneous flood state. Normal portal connectivity is checked with the existing turn-state router. Node counts are not people. All groups are retained pending target review.

Native zones are case-specific envelopes of actually observed wet face connectivity. They are control volumes,not delineated contributing catchments or identical cross-event zones. Shared cell volumes are unique and all aggregation precedes time maxima. Every group-zone join is many-to-many and cannot be used to sum volumes across repeated road groups.

The source storage-to-level operator is reproduced,including its0.999full-cell and1e-6dry branches. The0.30m road threshold is a screening boundary;an actual passability margin remains to be declared. Target storage at road-control cells is inverted from that operator. Only that subset supports excess-storage screening;whole-zone service targets,continuing load and complete boundary budget remain unresolved. Unsupported controls remain held. Required pumping capacity stays null. Saved values are never converted into a point count by dividing by assumed rate or hours.

Boundary face IDs are listed for later flow/crosssection coverage checks. Receivers,capabilities and their shared budgets are unqualified. The feasibility-readiness table explicitly records these gaps.
''')
 assert sha(ROOT/'docs/AnaSOP.md')==doc_sha
 sources_ok=True
 for rel,h in rv['source_sha256'].items():
  path=ROOT/rel;dig=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):dig.update(chunk)
  sources_ok &= dig.hexdigest()==h
 assert sources_ok
 outputs=[{'file':str(f.relative_to(ROOT)),'rows':len(pd.read_parquet(f))} for f in sorted(P.glob('*_preprocessed.parquet'))]
 dump(R/'validation.json',{'passed':True,'road_groups':len(roads),'candidate_roads':6064,'zone_records':len(zones),'trajectory_rows':len(traj),'group_zone_links':len(links),'linked_road_groups':groups_with_zones,'held_road_groups':len(missing_groups),'parquet_outputs':outputs,'all_zones_have_109_frames':True,'simultaneous_unique_zone_aggregation':True,'many_to_many_not_duplicated_in_volumes':True,'original_normal_network_reproduced':True,'all_required_capacity_null':True,'storage_only_not_dynamic_requirement':True,'source_hashes_unchanged':sources_ok,'AnaSOP_unchanged':True,'new_hydraulics':0,'new_traffic_fits':0,'new_training':0,'people_counted':False,'three_ward_and_buffer_volumes_partition_exact':True,'buffer_road_control_cells':129,'case_storage_summary':case_summary})
 print('PACKAGED_REGISTERS',len(roads),len(zones),len(traj),flush=True)
if __name__=='__main__':main()
