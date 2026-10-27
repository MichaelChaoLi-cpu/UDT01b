"""Offline source-readiness report, without assigning traffic or counting affected people."""
from pathlib import Path
import os,json,collections,csv,hashlib,html,datetime as dt
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1';R=ROOT/'data/exp/feasibility-check/three_ward_traffic_baseline_inputs_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 s=json.loads((R/'summary.json').read_text());v=json.loads((R/'validation.json').read_text());assert v['passed']
 zones=gpd.read_file(P/'pt_2018_core_planning_zones_6677.geojson');external=gpd.read_file(P/'pt_2018_candidate_external_zones_6677.geojson')
 colors={'001':'#3a789c','002':'#d48848','003':'#648e65'};names={'001':'Chiyoda','002':'Chuo','003':'Minato'}
 fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[1,1.2]});fig.patch.set_facecolor('#f8f9f8')
 for prefix,col in colors.items():zones[zones.planning_zone.str.startswith(prefix)].plot(ax=ax,color=col,edgecolor='white',linewidth=.7)
 ax.set_aspect('equal');ax.set_axis_off();ax.set_title('14 planning zones / 35 small zones',fontsize=11);ax.legend(handles=[Patch(facecolor=c,label=names[p])for p,c in colors.items()],loc='lower left',frameon=False,fontsize=8)
 external.plot(ax=bx,color='#e0e4e6',edgecolor='white',linewidth=.4)
 for prefix,col in colors.items():zones[zones.planning_zone.str.startswith(prefix)].plot(ax=bx,color=col,edgecolor='white',linewidth=.4)
 bx.set_aspect('equal');bx.set_axis_off();bx.set_title('133 zones in acquisition pool (core included)',fontsize=11)
 fig.text(.5,.015,'Source: Tokyo Metropolitan Region Transportation Planning Council, H30 PT; processed for this study.\n10 km buffer is a data-acquisition pool, not a final OD boundary. EPSG:6677.',ha='center',fontsize=8)
 fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(R/'pt_zone_scope.png',dpi=160);plt.close(fig)
 prof=list(csv.DictReader((P/'hourly_profile_quality.csv').open(encoding='utf-8-sig')));types=collections.Counter((r['observed_estimated_code'],r['duration_code'],r['available_hour_count'])for r in prof)
 table=''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{n}</td></tr>'for (a,b,c),n in sorted(types.items()))
 import os as posix
 def link(name,label):return f'<a href="{html.escape(posix.path.relpath(P/name,R))}">{html.escape(label)}</a>'
 doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区常态交通输入：资料准备与缺口</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#203039;background:#f5f7f7;margin:0}}main{{max-width:1100px;overflow-wrap:anywhere;margin:auto;padding:40px 28px}}h1{{font-size:30px;line-height:1.25}}h2{{font-size:22px;margin-top:36px}}.lead{{font-size:19px}}.box{{background:white;border:1px solid #dce4e6;border-radius:12px;padding:20px;margin:18px 0}}.pending{{border-left:5px solid #bd8338}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{background:white;padding:16px;border-radius:8px}}strong.big{{font-size:28px;display:block}}small,.muted{{color:#55636b}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce4e6;vertical-align:top}}a{{color:#136b91}}img{{max-width:100%;height:auto;background:white}}code{{font-size:13px}}li{{margin:8px 0}}@media(max-width:650px){{.metrics{{grid-template-columns:repeat(2,1fr)}}main{{padding:24px 14px}}table{{font-size:13px}}}}</style><main>
<p class="muted">港区 · 千代田区 · 中央区 ｜ 常态交通资料准备 ｜ 2026-10-17</p><h1>有数据可以约束交通需求与道路运行，<br>目前尚未形成校准后的拥堵基线。</h1>
<p class="lead">已准备道路流量、车速与道路属性候选表，并下载官方 PT 分区边界和代码。OD 数值表需要准确所属信息后才能下载；支路容量和暴雨出行变化仍需显式情景。当前没有交通分配、拥堵结果或受影响人数。</p>
<div class="metrics"><div class="metric"><strong class="big">391</strong>普查基本路段<br><small>按表内三区代码筛选</small></div><div class="metric"><strong class="big">174 / 157</strong>调查单元 / 有小时记录<br><small>同一流量可对应多个路段</small></div><div class="metric"><strong class="big">14 / 35</strong>PT 规划基本分区 / 小分区<br><small>2018 年分区</small></div><div class="metric"><strong class="big">12 / 12</strong>准备检查通过<br><small>未进行路网校准</small></div></div>
<div class="box pending"><b>OD 下载的具体缺项</b><p>官方 <a href="https://www.tokyo-pt.jp/data/01download">下载入口</a>会转到申请者类别、所属与使用目的表单。已核验页面和使用条件，但尚未提交，也未填写虚构身份。公开目录可查并不代表 OD 文件已取得。无需为预制表注册自定义集计账号。</p></div>
<h2>已得到什么，能够做什么</h2><table><tr><th>资料</th><th>准备结果</th><th>适用与限制</th></tr><tr><td>2021 道路普查基本表</td><td>三区391路段：千代田117、中央102、港172。保留12/24小时流量、方向车速、车道数、宽度及原始标记。</td><td>可约束常态运行；普查覆盖不等于所有区道。61条标为机动车专用道路，330条非专用，不能据此判定全部为地面道路。</td></tr><tr><td>2021 小时流量</td><td>618条“方向×车型”记录，展开为14,832个小时槽；10,344个有值，4,488个缺失。17个调查单元没有匹配小时行。</td><td>实测与估算分开；白天空缺或夜间空缺保留未知。不把618条记录当作618个独立观测点。</td></tr><tr><td>2018 PT 分区</td><td>千代田4/14、中央5/10、港5/11（规划基本/小分区）；跨区候选池共133个规划分区，包含核心14个。</td><td>支持 OD 空间组织。10km缓冲只用于候选资料获取，最终须保留跨界出入和穿越交通，不能截断圈外 OD。</td></tr><tr><td>2021 东京都区部小时表</td><td>原文件保留，作为补充核对来源，未重复并入主约束。</td><td>有大型/小型/合计三行结构；不能把合计和分车型重复相加。</td></tr></table>
<img src="pt_zone_scope.png" alt="2018 PT 三区规划基本分区及跨区获取候选范围"><p class="muted">源几何保持不变；核心14个规划分区、35个小分区及133个候选分区均有效。官方说明部分分区有其他市区町村飞地，分区不是独立重建的现行区界。</p>
<h2>小时记录的质量分层</h2><table><tr><th>观测标记</th><th>观测时长标记</th><th>有值小时数</th><th>记录数</th></tr>{table}</table>
<p>观测标记1=观测、2=估算；时长1=12小时、2=24小时，空白保持未知。244条完整24小时记录中，204条为观测来源、40条为估算来源。所有可核对的12/24小时记录，其小时和与原表合计一致。</p><p>调查日以7时至次日7时排列，午夜后保留次日偏移；如 <code>20211000</code> 只提供月份，不编造具体观测日。同一个调查单元的流量被多个基本路段引用时，校准中只能作为同一来源约束，不能累加成需求。</p>
<h2>还要下载的 PT 表：先看字段，再确定变量</h2><table><tr><th>表</th><th>候选用途</th><th>须核验</th></tr><tr><td>d-1</td><td>目的×主要交通方式的分区间 OD</td><td>汽车类是人次，含乘员，不能直接当车辆；小样本、未知和合计码须检查。</td></tr><tr><td>b-3</td><td>驾驶/非驾驶与车型分组，约束车辆换算</td><td>目录标题为 trips，使用指南说明有每人原单位表述；以下载文件字段、单位和合计核验后决定，暂不进行数值换算。</td></tr><tr><td>c-2 / c-3</td><td>出发/到达时段分布</td><td>是出行端点边际分布，不是完整“小时×OD”；不得把起终点相加成行程数，也不能默认所有 OD 共用同一时刻比例。</td></tr><tr><td>5-3 / e-2</td><td>汽车 OD 补充与正常出行时间参照</td><td>先查汽车 OD 驾驶员/乘员口径；平均时间按分区/方式使用，不能充当每条道路的自由流车速。</td></tr><tr><td>b-4</td><td>白天时点滞留人口辅助约束</td><td>调查推算覆盖与访客不足需保留；不代替完整昼间人口或受影响的独立人数。</td></tr></table>
<h2>常态基线的最短落地路线</h2><ol><li><b>需求：</b>取得并核验 OD 数值，保留三区内、跨区和穿越行程；按已有交通网络构造分区连接。时段化和缺失车辆类型分别留假设标记。</li><li><b>道路：</b>批量对应普查调查单元与已有车行网络，使用流量/速度检查常态运行；未覆盖支路采用按道路类别的容量与速度假设，保留桥隧未知分支。</li><li><b>暴雨对照：</b>常态检查后，接入已有积水关闭/降速情景，比较不能完成的行程、延误、可达性与措施改善。需求变化、容量降幅分别设情景，不把结果称为历史实测拥堵。</li></ol>
<div class="box"><b>三个单位不能混用</b><p>道路普查是车辆通过次数；PT 常见表是人的出行人次；受影响人数必须对人去重或使用明确人口模型。道路混杂度使用12小时交通量/容量比，不能把反推的12小时容量除以12后称为实测小时容量。当前尚未指定低/中/高需求和容量的数值。</p></div>
<p>年份组合：2018 PT、2020 常住人口、2021 道路普查、2024/2025 雷达降雨、2026路网。后续是由公开统计约束的情景实验，尚不具备“还原某场暴雨真实车流”的依据。</p>
<h2>本地产物与复现</h2><p>{link('core_road_attributes.csv','道路属性候选表')} · {link('core_directional_observations.csv','方向流量与车速')} · {link('core_count_hourly_long.csv','小时流量长表')} · {link('hourly_profile_quality.csv','小时质量记录')} · {link('section_to_count_unit.csv','路段—调查单元关系')} · {link('pt_2018_core_zone_codes.csv','三区PT分区代码')} · {link('preparation_contract.json','字段映射、单位与准备规则')}</p>
<p><a href="validation.json">检查结果</a> · <a href="summary.json">汇总</a>。输入校验包括源文件哈希不变、逐值回读一致、缺失保留、来源键唯一、区代码前导零与几何对应、坐标参考一致。它们验证资料整理，不验证拥堵模型。</p>
<p>复现：先运行 <code>src/preprocessing/acquire_tokyo_pt_baseline_sources_v1.cjs</code>，再运行 <code>src/preprocessing/prepare_three_ward_traffic_baseline_inputs_v1.py</code> 与 <code>src/analyses/report_three_ward_traffic_baseline_inputs_v1.py</code>。地理Python依赖使用本项目已有运行环境；文件来源与SHA-256保存在输入目录。</p>
<h2>官方依据</h2><p><a href="https://www.tokyo-pt.jp/data/01_01">PT类别与分区</a> · <a href="https://www.tokyo-pt.jp/data/H30">2018 PT表目录</a> · <a href="https://www.tokyo-pt.jp/static/hp/file/data/tebiki.pdf">PT使用指南</a> · <a href="https://www.mlit.go.jp/road/census/r3/index.html">2021道路普查</a> · <a href="https://www.mlit.go.jp/road/census/r3/data/pdf/kasyorep.pdf">道路普查说明与代码</a> · <a href="https://www.tokyo-pt.jp/terms">PT使用条件与出典规则</a></p>
<p class="muted">上述官方资料经本研究整理加工。原始下载字节保留、未提交申请者表单。既有8组水动力实验未被本次资料整理改变；本报告不包含新模拟、训练、拥堵或受影响人数。Framelog：MILI-D-20261017-014。</p></main></html>'''
 (R/'review.html').write_text(doc)
 files=[p for p in P.rglob('*')if p.is_file()]+[R/'review.html',R/'summary.json',R/'validation.json',R/'pt_zone_scope.png',ROOT/'src/preprocessing/acquire_tokyo_pt_baseline_sources_v1.cjs',ROOT/'src/preprocessing/prepare_three_ward_traffic_baseline_inputs_v1.py',Path(__file__).resolve()]
 dump(R/'delivery_manifest.json',{'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files]})
 print('Report prepared; OD pending truthful affiliation.12 preparation checks passed.')
if __name__=='__main__':main()
