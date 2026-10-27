#!/usr/bin/env python3
"""Package rain-reconstructed partial budgets and inactive ground-flux requests."""
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_zone_rain_flux_readiness_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
PRIOR=ROOT/'data/processed/three_ward_target_boundary_budget_audit_v1'

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ana=sha(ROOT/'docs/AnaSOP.md');rain=json.loads((R/'rain_validation.json').read_text());interface=json.loads((R/'interface_validation.json').read_text());plan=json.loads((R/'ground_flux_interface_plan.json').read_text())
 assert rain['passed'] and interface['passed'] and len(rain['cases'])==10 and not plan['execution_enabled']
 conversion=[('zone_rain_trajectories.csv','zone_rain_trajectories_preprocessed.parquet'),('zone_rain_summary.csv','zone_rain_summary_preprocessed.parquet'),('zone_encoded_rain_intervals.csv','zone_encoded_rain_intervals_preprocessed.parquet'),('case_rain_partition.csv','case_rain_partition_preprocessed.parquet')]
 loaded={}
 for csvname,output in conversion:
  d=pd.read_csv(R/csvname,float_precision='round_trip');d.to_parquet(P/output,index=False);pd.testing.assert_frame_equal(d,pd.read_parquet(P/output));loaded[csvname]=d
 t=loaded['zone_rain_trajectories.csv'];s=loaded['zone_rain_summary.csv'];c=loaded['case_rain_partition.csv']
 assert len(t)==22563 and len(s)==207 and t.groupby(['Case','Zone']).size().eq(109).all()
 assert not t.duplicated(['Case','Zone','Frame']).any() and not s.duplicated(['Case','Zone']).any()
 assert not t['Independent Local Applied Rain Saved'].any()
 assert np.max(abs(c['Selected Zone Rain Since First Map']+c['Unselected Ground Rain Since First Map']-c['Whole Model Rain Since First Map']))<1e-5
 assert np.max(abs(t['Chiyoda Rain Since First Map']+t['Chuo Rain Since First Map']+t['Minato Rain Since First Map']+t['Buffer Rain Since First Map']-t['Reconstructed Rain Since First Map']))<1e-5
 prior=pd.read_parquet(PRIOR/'zone_partial_budget_trajectories_preprocessed.parquet');m=prior.merge(t,on=['Case','Zone','Frame'],validate='one_to_one',suffixes=('',' Rain'))
 assert np.array_equal(m['Actual Step'],m['Actual Step Rain'])
 assert np.array_equal(m['Flow Actual Time'],m['Actual Time Rain'])
 m=m.rename(columns={'Actual Time Rain':'Rain Actual Time'}).drop(columns=['Actual Step Rain','Unresolved Combined Net Supply','Unresolved Supply Meaning','Unknown Applied Rainfall','Unknown Ground Exchange'])
 m['Unresolved Ground Exchange And Rounding']=m['Storage Change Since First Map']+m['Receiving Outflow Since First Map']-m['Receiving Inflow Since First Map']-m['Reconstructed Rain Since First Map']
 m['Ground Flux Independently Saved']=False;m['Budget Complete']=False;m['Required Pump Capacity']=np.nan
 m['Remainder Meaning']='unobserved net ground input plus source/continuity rounding and stored-volume quantization;not independently measured flow or numerical residual'
 m.to_parquet(P/'zone_rain_partial_budget_preprocessed.parquet',index=False)
 assert np.allclose(m['Storage Change Since First Map'],m['Reconstructed Rain Since First Map']+m['Receiving Inflow Since First Map']-m['Receiving Outflow Since First Map']+m['Unresolved Ground Exchange And Rounding'],atol=1e-7,rtol=0)
 assert m['Required Pump Capacity'].isna().all() and not m['Budget Complete'].any()
 cov=pd.read_parquet(PRIOR/'zone_budget_readiness_preprocessed.parquet');final=t[t.Frame==108].groupby('Case').sum(numeric_only=True);rows=[]
 for d in rain['cases']:
  case=d['case'];f=final.loc[case];zc=cov[cov.Case==case]
  rows.append({'Case':case,'Zones':int(d['zones']),'Reconstructed Zone Rain':float(f['Reconstructed Rain Since First Map']),'Three Ward Zone Rain':float(f['Chiyoda Rain Since First Map']+f['Chuo Rain Since First Map']+f['Minato Rain Since First Map']),'Buffer Zone Rain':float(f['Buffer Rain Since First Map']),'Zone Receiving Inflow':float(zc['Saved Receiving Inflow'].sum()),'Zone Receiving Outflow':float(zc['Saved Receiving Outflow'].sum()),'Maximum Global Applied Rain Check Difference':d['max_applied_global_prefix_difference_m3'],'Complete Local Budgets':0})
 cs=pd.DataFrame(rows);cs.to_parquet(P/'case_rain_budget_readiness_preprocessed.parquet',index=False)
 # The high/low pair has identical forcing,while selected fixed cells may differ.
 pairchecks=[]
 for event,g in cs.groupby(cs.Case.str.split('__').str[0]):
  cases=g.Case.to_list();part=c[c.Case.isin(cases)].pivot(index='Frame',columns='Case',values='Whole Model Rain Since First Map')
  delta=float(np.max(abs(part.iloc[:,0]-part.iloc[:,1])))
  hashes=[v['sha256'] for k,v in rain['source_bindings'].items() if any(k.endswith('/runs/'+x+'/historical_rain.nc') for x in cases)]
  assert len(hashes)==2 and len(set(hashes))==1
  assert delta<=max(1.,float(np.max(abs(part.to_numpy()))))*1e-6
  pairchecks.append({'parent':event,'encoded_roof_rain_fields_identical_between_stages':True,'maximum_partition_reduction_difference_m3':delta})
 budgets=pd.read_parquet(P/'ground_request_budget_preprocessed.parquet');inc=pd.read_parquet(P/'ground_zone_incidence_preprocessed.parquet');req=pd.read_parquet(P/'ground_face_requests_preprocessed.parquet');assert not budgets['Execution Enabled'].any() and not req['Runtime Face Mapping Verified'].any()
 readiness=cov[['Case','Zone','Unobserved Open Ground Faces','Configured Closed Ground Faces','Receiving Face Memberships']].merge(s,on=['Case','Zone'],validate='one_to_one')
 readiness['Rain Evidence']='original roof-adjusted source verified and actual-step applied rain reconstructed;local log unavailable'
 readiness['Ground Flow Evidence']='native request geometry and signed incidence prepared;runtime data not collected'
 readiness['Whole Zone Target Stage']='unresolved beyond road-controlled cells'
 readiness['Receiving Allowance']='unresolved;prescribed stage is not available discharge allowance'
 readiness['Required Pump Capacity']=np.nan;readiness['Deployment Ready']=False
 readiness.to_parquet(P/'zone_requirements_readiness_preprocessed.parquet',index=False)
 decisions={'scope':'Reference-only continuation of MILI-D-20261020-018;local rain reconstruction and ground-flow interface audit','is_final_variable':'no','AnaSOP_changed':False,'raw_or_original_runs_changed':False,'rules':['Rainfall,parent lineage,split,roof allocation,quality masks and actual solver dt remain unchanged','New applied-rain reconstruction uses original encoded roof-recipient field,source float32 arithmetic and actual block-step overlaps','Global and ward/buffer partitions retain all unselected active ground as a separate category','Fixed observed-wet control volumes are not contributing catchments;high/low selected volumes may differ even with identical rainfall','Partial-budget ground remainder is not independently observed ground flow or a numerical residual;capacity remains null','Generated ground-face requests have static geometry checks only,not runtime mapping qualification','Existing crosssection interface can observe ground faces without new physics;new logging hooks are not implemented','No new simulation,training,people calculation or automatic ten-case replay']}
 dump(R/'decisions.json',decisions)
 # Dictionary retains source meanings and assigns explicit units to new fields.
 previous={}
 for tag in ['three_ward_protection_requirements_register_v1','three_ward_target_boundary_budget_audit_v1']:
  d=pd.read_csv(ROOT/'data/exp/data-preprocessing'/tag/'data_dictionary.csv')
  for _,r in d.iterrows():previous[r['Readable Field']]=r['Unit']
 units={'Rain Interval':'interval index','Actual Time':'s','Actual Time Rain':'s','Rain Actual Time':'s','Flow Actual Time':'s','Interval Start':'s','Interval End':'s','Face Width':'m','Required Pump Capacity':'m3/s,unresolved','Original Runtime Seconds':'s','Proposed Wallclock Stop Seconds':'s','Additional Raw Discharge Bytes':'bytes'}
 dictionary=[];exports=[]
 for path in sorted(P.glob('*_preprocessed.parquet')):
  d=pd.read_parquet(path);assert all(v.isascii() and v==v.strip() for v in d)
  for col in d:
   quantity=(pd.api.types.is_numeric_dtype(d[col]) and not pd.api.types.is_bool_dtype(d[col]) and any(x in col for x in ['Rain','Inflow','Outflow','Storage','Volume','Rounding','Difference']))
   unit=units.get(col,previous.get(col,'m3' if quantity else 'identifier/category/count/boolean'))
   dictionary.append({'Dataset':path.name,'Readable Field':col,'Unit':unit,'Role':'reference construction diagnostic;not a final manuscript variable','Missing Policy':'retain missing and unobserved status;no zero filling'})
  exports.append({'path':str(path.relative_to(ROOT)),'rows':len(d),'sha256':sha(path)})
 pd.DataFrame(dictionary).to_csv(R/'data_dictionary.csv',index=False)
 bind={str((PRIOR/n).relative_to(ROOT)):sha(PRIOR/n) for n in ['zone_partial_budget_trajectories_preprocessed.parquet','zone_budget_readiness_preprocessed.parquet']}
 scripts={n:sha(ROOT/'src/preprocessing'/n) for n in ['reconstruct_requirements_zone_rain_v1.py','prepare_requirements_ground_flux_requests_v1.py','package_requirements_rain_flux_readiness_v1.py']}
 dump(P/'export_manifest.json',{'outputs':exports,'source_parquet_sha256':bind,'script_sha256':scripts,'rain_source_bindings':str((R/'rain_validation.json').relative_to(ROOT)),'interface_evidence':str((R/'ground_flux_interface_plan.json').relative_to(ROOT))})
 style='body{font:16px/1.75 system-ui;max-width:1280px;margin:28px auto;padding:0 24px;color:#203b49}h1{font-size:27px}h2{font-size:22px}aside{padding:18px;background:#edf4f7;border-left:4px solid #467e91}.warn{padding:16px;background:#fff3df}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border:1px solid #d4e0e6;text-align:left}.scroll{overflow:auto}a{color:#14667e}'
 maxerr=max(d['max_applied_global_prefix_difference_m3'] for d in rain['cases']);sources=''.join(f'<li><a href="../../../../{x["file"]}:{x["lines"][0]}">{html.escape(Path(x["file"]).name)}:{x["lines"][0]}</a>：{html.escape(x["expression"])}</li>' for x in plan['source_evidence'])
 # File anchors use ordinary relative files,with line references as visible text only.
 sources=''.join(f'<li><a href="../../../../{x["file"]}">{html.escape(Path(x["file"]).name)}</a>（第{x["lines"][0]}行）：{html.escape(x["expression"])}</li>' for x in plan['source_evidence'])
 links=''.join(f'<li><a href="../../../processed/{TAG}/{p.name}">{html.escape(p.name)}</a></li>' for p in sorted(P.glob('*_preprocessed.parquet')))
 first=plan['proposed_minimum_replay'];hours=first['combined_wallclock_stop_seconds']/3600;gb=first['additional_raw_q_bytes']/1e9
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>控制区降雨核算与地表通量导出方案</title><style>{style}</style><h1>控制区降雨核算与地表通量导出方案</h1><aside>已核算5场历史降雨×2水位、207个情景内控制区、各109帧的屋面再分配降雨输入。10组均复现原输入并通过全域雨量核验。<b>地表通量请求已生成但未运行；本轮新增水动力模拟为0。</b></aside><h2>1. 控制区降雨已能核算</h2><p>直接读取每组原historical_rain.nc，并逐帧与冻结的urban_effective_rainfall及原雨量倍数比较，屋面再分配字段逐值一致；原雨型、父事件、划分、质量掩膜和实际dt全部保留。在本批网格上，原生单元中心与降雨双线性采样位置一致。按原START时间戳雨强保持规则和每一步实际起止时刻求重叠，复现原float32平均雨强×100m²×dt的施加口径。</p><p>重建全域累计施加雨量与原逐步雨量预算的最大前缀差为{maxerr:.3g} m³，沿用原1e-6相对雨量核验门槛。未选入控制区的活跃地面单元单列，三区内与缓冲区分开核算，未把区外贡献默默删除。原分区累计与全域之和通过检查。</p><div class="scroll">{cs.to_html(index=False,float_format=lambda v:f'{v:,.3f}',formatters={'Maximum Global Applied Rain Check Difference':lambda v:f'{v:.2e}'})}</div><p>体积单位m³，累计区间为第一张真实地图至最后一张真实地图。高低水位两组编码雨场文件的哈希一致；分区后全域雨量求和仅允许原门槛内的浮点归约差。固定湿区包络边界不同，因此控制区内雨量可不同。控制区并非完整汇水区；表中雨量不是直接泵排需求。<b>局部降雨是按原算子重建，并非原运行独立保存的局部施加日志。</b></p><h2>2. 地表交换可以怎样导出</h2><p>原求解器的命名断面没有限定在河道：sfincs_crosssections可读取普通地面格面，并从原生q和宽度输出带符号流量。现有6763条B断面只覆盖陆水边界，因此旧结果缺少任意控制区地面边界的逐步通量。5分钟单元中心u/v及负体积触发记录都不能替代这些正性限制之后、连续方程所用的同一步q。</p><p>已为10组全部缺失地面边界生成一面一段的G断面请求，共{len(req):,}条“情景×唯一格面”记录，以及{len(inc):,}条带方向的控制区成员关系。共享格面两侧外向符号相反，双计问题已静态核验；断面长度9.8m、围绕10m格面中心，仅为唯一截取该原生格面的观测几何，不是新挡水设施或路宽参数。开放和配置封闭格面均保留，后者须在未来实测输出中核验为零。<b>实际运行时的断面匹配尚未核验，原运行文件未改。</b>新请求表中的原生单元和格面索引从0开始，G名称编码格面索引＋1，名称不代表新增实体设施。</p><div class="scroll">{budgets[['Case','New Ground Sections','Open Ground Sections','Configured Closed Ground Sections','Additional Raw Discharge Bytes','Execution Enabled']].to_html(index=False)}</div><p>请求文件在<a href="../../../processed/{TAG}/requests/">requests目录</a>；每个情景使用自己的固定控制区。部署措施改变流路后必须复核诊断边界的适用性，不能将本轮湿区边界当作永久汇水分界。</p><h2>3. 已准备最小诊断重放方案，本轮不执行</h2><p>先选择{html.escape(first['execute_first_only'])}，因为它的开放地面边界最多，适合检验接口覆盖；通过后再处理同一父事件的另一水位组。这与保护道路优先级无关。仅在独立新目录复制原物理输入并追加G观测断面，保留原B断面顺序、原镜像及逐步历史输出，首选现有接口，无需先改水动力代码或重建镜像。</p><p>两组新增q数组的未压缩净载荷约{gb:.2f}GB；还需原输出、元数据和额外磁盘余量。操作性停机上限按每组“原记录耗时×2＋600秒”预设，两组合计{hours:.2f}小时，<b>这不是耗时预测</b>。启动前复核空闲磁盘及可用计算资源。</p><ol><li>检查实际步号／dt和原109帧水深／储量／质量掩膜逐位不变；原B断面输出保持不变。</li><li>验证所有新G断面一一对应原生格面、方向正确，所有配置封闭格面逐步q为零。</li><li>沿用原负体积、雨量和水量预算准入规则；按实际dt积分两向流量，共享内部格面按符号抵消。</li><li>任何物理输入、时钟、原图面、原断面或准入规则不一致即停止；未通过不扩展到其余8组。</li></ol><h2>4. 仍不能把剩余项当作所需抽排量</h2><div class="warn">现在可写“Δ存水＝重建降雨＋已知河道来水−已知河道去水＋未观测地面交换及舍入组合项”。新表显式保留这一未拆分项。它不是独立测量的地面流量，也不是数值误差或必须抽排的体积。完整局部预算仍为0/207，所需抽排能力全部为空。</div><p>源码已有rain_term、horizontal_term、rain_increment逐单元内存数组，但没有保存完整局部过程。原连续方程把多个格面项先组合为float32 dvol，再加雨量；“各格面分别乘宽度和dt后再相加”与这一分组算术可有舍入差。若后续要独立归因局部残差，需另行核验被动分区汇总日志，不能因新增断面就声称全部舍入项已闭合。完整目标水位、通行余量、接收限额和设备有效能力也仍未确定。</p><h2>源码证据与可复现记录</h2><ul>{sources}</ul><ul>{links}</ul><p><a href="validation.json">整体验证</a> · <a href="rain_validation.json">10组降雨核验</a> · <a href="ground_flux_interface_plan.json">未启用重放方案</a> · <a href="interface_validation.json">断面静态核验</a> · <a href="README.md">复现命令</a> · <a href="data_dictionary.csv">字段与单位</a></p><p>AnaSOP、正式变量和三张已认可图均未改动；未计算新增人口影响。</p></html>'''
 (R/'review.html').write_text(page)
 (R/'README.md').write_text('''# Zone rainfall and ground-flux readiness

Reference-only construction diagnostics. No solver execution,physical input change,training,population calculation or sealed-test access. AnaSOP and original source runs remain unchanged.

Run from the project root:

1. `.frame-control/runtimes/physical-measures-py312/bin/python src/preprocessing/reconstruct_requirements_zone_rain_v1.py`
2. `.venv/bin/python src/preprocessing/prepare_requirements_ground_flux_requests_v1.py`
3. `.venv/bin/python src/preprocessing/package_requirements_rain_flux_readiness_v1.py`

Rain reconstruction reads unchanged encoded roof-adjusted fields,verifies the frozen source export and uses actual native step times. Native centers coincide with forcing sample positions;otherwise the script refuses direct indexing. Original START-stamped block overlap and float32 rain mean,area and dt products are reproduced. Source integration and applied-rain cumulative checks use the original1e-6relative gate,reported with actual absolute differences. All unselected active ground remains a separate partition. Map totals subtract the actual first-map baseline. Ward/buffer totals are separate. Different-stage zones are fixed case-specific wet-connectivity envelopes,not common catchments. Local applied rain is reconstructed,not independently saved.

New inactive G crosssection requests use the existing solver q-based crosssection interface. Each unique native ground face has one9.8m segment,centered on the face and oriented with positive left normal from native endpoint A to B. Zone signs are explicit;shared face signs cancel. Static geometry checks are not runtime mapping qualification. Original B sections and run folders are untouched. Native cell/face identifiers in new request registers are zero-based;G names encode native face+1. No new logger or image is implemented.

The proposed minimal replay is disabled. It selects the largest open-ground-face workload and then its paired stage only if all gates pass. Original109map fields and clocks must remain bitwise unchanged. Negative volumes,global budgets and original B flow stay under existing rules. Configured closed G faces must have zero q. Wallclock ceilings and raw-q byte estimates are operational budgets,not predicted performance. Full float32 grouped-continuity/source attribution remains a separate diagnostic requirement if local residual certification is needed.

The ground remainder computed from storage,rain and receiving exchange is an unidentified combined term including roundoff and map write quantization. It is not a measured ground flux,numerical error or pump requirement. Whole-zone target stage,receiving allowance and effective equipment capability remain unresolved. All required capacities stay missing.
''')
 assert sha(ROOT/'docs/AnaSOP.md')==ana
 dump(R/'validation.json',{'passed':True,'cases':10,'zone_records':207,'actual_map_rows':len(t),'rain_interval_rows':len(loaded['zone_encoded_rain_intervals.csv']),'max_global_applied_rain_prefix_difference_m3':maxerr,'paired_stage_rain_checks':pairchecks,'ward_buffer_and_unselected_partitions_checked':True,'ground_face_requests':len(req),'runtime_ground_face_mapping_verified':False,'local_budget_complete':False,'required_capacity_all_missing':True,'partial_budget_is_identity_not_independent_closure':True,'original_rule_relative_rain_tolerance':1e-6,'AnaSOP_sha256':ana,'AnaSOP_unchanged':True,'new_hydraulic_runs':0,'new_training':0,'new_people_counts':False,'replay_execution_enabled':False})
 print('RAIN_FLUX_PACKAGE_COMPLETE',len(t),'rows',len(req),'inactive ground-face requests',flush=True)
if __name__=='__main__':main()
