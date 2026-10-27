import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,csv,hashlib
import numpy as np,rasterio
from rasterio.transform import Affine
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from shapely.geometry import shape,mapping,box
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='exp' else Path('/Volumes/Extension/Research/UDT01b');OUT=ROOT/'data/exp/data-briefing/three_ward_road_connectivity';PRO=ROOT/'data/processed/three_ward_road_connectivity';OLD=ROOT/'data/processed/three_ward_surface_trial'
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def cs(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def main():
 audit=json.loads((OUT/'connectivity_audit.json').read_text());g=dict(np.load(PRO/'road_graph.npz'));a=dict(np.load(OLD/'surface_inputs.npz'));v=json.loads((OUT/'numerical_validation.json').read_text());r0=dict(np.load(PRO/'R0_result.npz'));r1=dict(np.load(PRO/'R1_result.npz'));fine=dict(np.load(PRO/'R1_timestep_check.npz'));nr,nc=a['land'].shape;tf=Affine(*audit['transform']);size=nr*nc;flag=np.zeros(size,dtype='uint8');np.maximum.at(flag,g['parent'],(fine['node_abs_difference']>.1).astype('uint8'));flag=flag.reshape(nr,nc)
 with rasterio.open(PRO/'R1_area_weighted_peak.tif') as d:
  m=d.meta.copy();m.update(dtype='uint8',nodata=0)
  with rasterio.open(PRO/'timestep_sensitive_cells.tif','w',**m) as p:p.write(flag,1)
 # Strict comparison uses exactly the same node bed, area and forcing in R0/R1.
 rows=list(csv.DictReader((OUT/'ward_model_comparison.csv').open(encoding='utf-8-sig')));groups=['千代田区','中央区','港区'];changes=[]
 for name in groups:
  x=next(r for r in rows if r['ward']==name and r['scenario']=='R0' and r['surface_group']=='all_open_surface');y=next(r for r in rows if r['ward']==name and r['scenario']=='R1' and r['surface_group']=='all_open_surface');changes.append({'ward':name,'suspect_gt2m_R0_m2':float(x['peak_gt2m_open_area_m2']),'suspect_gt2m_R1_m2':float(y['peak_gt2m_open_area_m2']),'screen_ge0_3m_R0_m2':float(x['peak_ge0_3m_open_area_m2']),'screen_ge0_3m_R1_m2':float(y['peak_ge0_3m_open_area_m2']),'interpretation':'Synthetic diagnostic, not observed flood extent. More connectivity redistributes water and may increase accumulation downstream.'})
 cs(OUT/'controlled_comparison_summary.csv',changes)
 total0=sum(r['suspect_gt2m_R0_m2'] for r in changes);total1=sum(r['suspect_gt2m_R1_m2'] for r in changes)
 conclusion={'reopened_cells':audit['reopened_coarse_cells_with_road'],'strict_control':'Only cross-cell road portal edges switched; node terrain, storage, source weights,15mm/h loss andclosed water/outer boundaries identical.','graph_isolated_nodes_R0':13125,'graph_isolated_nodes_R1':311,'suspect_gt2m_open_area_R0_m2':total0,'suspect_gt2m_open_area_R1_m2':total1,'uniform_reduction_of_deep_artifacts':False,'local_depths_validated':False,'upstream_complete':False,'new_water_inlets_created':False};js(OUT/'conclusion.json',conclusion)
 plots(audit,a,g,r0,r1,flag,changes,tf)
 needs=[
 {'priority':'P1','system':'跨区地表直接来水','spatial_scope':'先从三区外缘的连续地表汇水区向外追踪：港区西侧渋谷/南侧品川，千代田西北侧新宿/文京，中央北侧千代田/台东；具体范围由地形和路网划分，不能用固定1km或整区替代。跨河对岸不默认连入。','needed':'裸地DEM覆盖、地面道路/建筑轮廓、主要桥涵/地下通道层级；分区实际降雨；若截断域则提供地表边界入流Q(t)。','time_reference':'2024地貌/设施版本；DEM测量时间、T.P.高程成果版本、JGD2011坐标；不自动拼接2025修订高程。','availability':'现有三区TIN较完整，跨区完整汇水域未圈定；0.50m地形只下载日比谷；东京23区0.50mDEM为补齐候选。','efficient_route':'先圈定直接汇水区并按数据瓦片批量下载；上游河川全流域可用外部Q(t)/H(t)替代全域高分辨率2D网格。','source':'https://www.geospatial.jp/ckan/dataset/tokyopc-23ku-2024'},
 {'priority':'P1','system':'渋谷川—古川','spatial_scope':'官方流域22.8km²，包含港区之外的渋谷川上游及暗渠来水；不是港区1km缓冲。','needed':'渋谷橋、四ノ橋、古川取水口上/下游的1–10分钟H(t)；上游Q(t)或率定关系；古川地下调节池取水/蓄水/运行时序；河床/岸顶断面。','time_reference':'2024-08-21 00:00至2024-08-22 06:00 JST；水尺零点、岸顶/基准点T.P./A.P.标高、变更生效日期。','availability':'已有2024事件水位图、约4.3m相对上涨和42200m³蓄水量；缺原始时序和站点零点。现行站目录有上述站名，2024站址仍需核对。','efficient_route':'先用区内实测水位作地表边界；如需预测新暴雨，再用流域雨量—径流或上游Q(t)，不必把全上游做成0.50m2D模型。','source':'https://www.kensetsu.metro.tokyo.lg.jp/river/seibi/shibuya'},
 {'priority':'P1','system':'神田川—日本橋川—亀島川','spatial_scope':'官方神田川流域约105km²，包括善福寺川、妙正寺川、江古田川；向日本橋川分派，接隅田川。','needed':'饭田桥等区界上游H(t)/Q(t)、日本桥分派量与分水路运行；日本桥川/亀島川下游H(t)与断面/岸顶；如需推算Q，则需率定曲线和回水条件。','time_reference':'同一2024事件时窗，所有站的水尺零点及A.P./T.P.基准点成果；设计高水位不能替代实测事件水位。','availability':'W05线与三区水域已匹配，但只是几何；完整上游、分派流量和事件河段水位尚缺。','efficient_route':'优先获取上下游水位和分派边界，用一维河道/实测H(t)支撑三区2D地表；按需要追加流域DEM和分布雨量。','source':'https://www.kensetsu.metro.tokyo.lg.jp/jimusho/sanken/kasen/kasen_kanri'},
 {'priority':'P1','system':'隅田川—沿岸运河—东京湾','spatial_scope':'隅田川及中央/港沿岸水域；日本桥川/亀島川和古川河口需考虑潮汐及防潮设施。','needed':'佃橋、小台等候选水位站的事件H(t)、入海段水位梯度；晴海未滤波或1–10分钟实测潮位；水门/泵站当日运行，仅用作边界条件而非管网优化。','time_reference':'同一事件JST；海侧DL基准、站点水尺零点、A.P./T.P.成果版本；2024晴海DL=-188.4cm，不能用天文潮位表基准代替。','availability':'已新增晴海8月21–22日48条小时平滑实测潮位；不是分钟河道过程，不能给所有河段统一套晴海H(t)。现行东京站目录有佃桥/小台。','efficient_route':'先确认海侧背景和区内河段H(t)，再补主要水门/泵站边界运行；不必先重建整个地下管网。','source':'https://www.kensetsu.metro.tokyo.lg.jp/river/seibi/kasenbu0078'},
 {'priority':'P1','system':'真实暴雨降雨场','spatial_scope':'三区与上述直接来水区/上游河川模型范围；港区2024事件核心不是JMA东京站单站代表。','needed':'港区9站及東京都中央/高浜/渋谷橋等1–10分钟雨量，或同期XRAIN数值格网与地面站校验。','time_reference':'同一事件时窗和JST，测器精度/缺测/站点迁移记录；峰值统计与固定时段雨量分开。','availability':'已有JMA4站10分钟值和149站汇总；缺核心多站原始曲线和数值雷达场。','efficient_route':'获取事件档案/公开数值网格；当前公开历史下载仅近一年，2024时序需官方档案途径。','source':'https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/tsim0110g.html'},
 {'priority':'P2','system':'模型几何和数值表达','spatial_scope':'本次高坡差、道路/屋顶重叠、5mTIN缺值和时间步敏感块；优先按图块批量复核。','needed':'桥/地下道路层级；可靠道路裸地剖面；地表子网的面积—水位曲线与更严格数值检查。','time_reference':'保持测绘时期版本，后续另更新改造区。','availability':'已排除3306个道路/屋顶重叠5m像元，未连接桥梁或地下层；缺1898个合格道路5m地形样本。数值敏感格已保存。','efficient_route':'先改进子格储水与标高表达，不能把所有深水格直接列成现场补测点。','source':'local:model diagnostics'},
 ]
 cs(OUT/'upstream_and_stage_data_needs.csv',needs);js(OUT/'upstream_and_stage_data_needs.json',needs)
 text=f'''三区道路连通批量修正与跨区边界需求（2026-10-14）
结论：保住道路通道是有效的结构改进，但本轮不能证明异常深水整体减少。
原先关闭的20m格中恢复道路表示{audit['reopened_coarse_cells_with_road']}格；{audit['road_faces_previously_blocked_by_coarse_mask']}条道路子网连接原被粗格掩膜挡住，{audit['road_portals_more_than0_5m_below_old_center_threshold']}条子网连接的原中心阈值比道路通道高超过0.5m。这些是格/子网连接数，不是已证实堵水的街道或现场入口数。
相同模型节点下，关闭/打开跨格道路连接，孤立节点从13125降到311；部分地方积水减少、另一些地方增加。>2m模型疑义深水的开放面面积从{total0:,.0f}增至{total1:,.0f}m²。因此不能把“连通改善”说成“所有假洼地已消除”，也不能把增加解读为实地风险升高。

实现：沿已提取5m道路掩膜从原始PLATEAU TIN重新取样高程；同一20m格内不同连通道路片段各自保留，不强行合并。通道内部的地形凸起用最低最大高程路径保留为格面阈值，原20m地形不改写、不填洼、不沿道路统一烧低。非道路开放地表仍按20m聚合，可能通过聚合节点间接连接道路片段；本轮没有验证完整的5m开放面拓扑或道路层级。道路片段采用均值高程和恒定面积储水，仍是近似；未建立完整面积—水位曲线。
排除道路与屋顶代理重叠像元和水域覆盖像元，缺值不补；不自动打通桥梁、地下层或河岸入口。原5m几何不足以保留路缘细节。未使用WTR高程作事件水位。

严格控制：R0与R1采用相同节点、地形、储水面积、雨量分配、15mm/h损失和关闭的河渠/外缘边界，只切换跨格道路连接。合成降雨域平均60mm持续60min、退水30min，与既有试算的分区权重一致。关闭河渠是控制条件，不代表现场封闭，试验不能评价真实河道排水。
地图是各子节点峰值的面积加权均值（mean of temporal maxima），不是各时刻面积均值的最大值，亦非实测水深；统计开放面面积，不是整格面积。与旧版20m的全格面积指标不能直接混用。
检验：水量守恒、非负、复杂节点最大水面原理和平水面静止已通过。2.5/1.25s比较的>=0.30m开放面面积差{v['timestep_check']['over0_3m_area_relative_difference']:.2%}，但局部峰值最大差{v['timestep_check']['max_node_abs_difference_m']:.2f}m；局部深度仍不收敛，不能用于街道水深预测。图中与资料导出中保留数值敏感格。由相同道路TIN取样不等于独立高程精度验证。

需要补齐：详见upstream_and_stage_data_needs.csv，按P1/P2排序。
推荐先区分两种来水：跨区地表径流要扩到实际连续汇水区或接Q(t)；上游河道来水可由实测水位/流量或一维河道模型提供，不必把神田川105km²全部纳入0.50m2D计算。
港侧优先渋谷川—古川全流域/区内H(t)；千代田/中央优先神田川分派日本桥川、亀島川以及隅田川回水；海侧优先晴海潮位和关键河口/水门运行。河段高水位是边界条件，水面几何接触不等于地表入口。
新增实测资料：2024-08-21/22晴海48条JMA小时潮位，核对2024DL=-188.4cm；观测基准与标高显示一致。它是去除约3h以下周期分量的平滑值，只可提供海侧背景；需要更细实测潮位和各河段H(t)。未把这48条值用于本次合成道路对照，也未完成与PLATEAU高程成果版本的衔接。
所有事件资料建议统一收集2024-08-21 00:00至2024-08-22 06:00 JST；水位优先1–10分钟，并附水尺零点、岸顶/河床断面、基准点T.P./A.P.标高及变更日期。不凭通用常数换算，不用规划高水位代替事件水位。
下一步先拿实际河道边界与跨区来水，完善道路子格储水/数值表达；在完成这些前不对地图深水节点做真实风险排序。申请/邮件尚未发送。

复现：依次执行src/exp/prepare_road_connectivity.py、src/exp/run_road_connectivity.py、src/exp/acquire_boundary_evidence.py、src/exp/finalize_road_connectivity.py。基线数据和原ZIP保留。
'''
 (OUT/'assessment.txt').write_text(text)
 table=''.join(f'<tr><td>{r["ward"]}</td><td>{r["suspect_gt2m_R0_m2"]:,.0f}</td><td>{r["suspect_gt2m_R1_m2"]:,.0f}</td></tr>' for r in changes)
 needtable=''.join(f'<tr><td>{r["priority"]}</td><td>{r["system"]}</td><td>{r["needed"]}</td></tr>' for r in needs)
 (OUT/'review.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区道路连通修正</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:28px auto;padding:0 18px;color:#24364a}}h1{{font-size:26px}}img{{max-width:100%;height:auto}}.note{{background:#fff2d5;padding:15px}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #dde3ea;padding:8px;text-align:left}}a{{color:#1765a4}}.scroll{{overflow-x:auto}}</style><h1>三区道路连通：保住通道，继续检查模型储水</h1><p>恢复202个粗格内的道路表示，孤立模型节点从13,125降到311。R0/R1采用相同储水与合成降雨，仅切换跨格道路连接，水域边界均关闭。</p><p class="note">连通性改善，但异常深水没有整体下降，部分区域收到重新分配的来水。局部深度尚未数值收敛，不能将地图当实况洪深或风险排名。原始DEM未改写，没有新增河岸出口。</p><img src="road_connectivity_comparison.png" alt="道路连通开关比较"><img src="road_connectivity_metrics.png" alt="合成模型诊断指标"><div class="scroll"><table><tr><th>区</th><th>R0 >2m疑义面积 m²</th><th>R1 >2m疑义面积 m²</th></tr>{table}</table></div><p>面积为模型节点开放面，>2m只是诊断阈值，不是已证实假洼地或实际洪水。</p><h2>需要补齐的跨区与水位资料</h2><div class="scroll"><table><tr><th>优先级</th><th>系统</th><th>所需资料</th></tr>{needtable}</table></div><p>优先实测区内河段H(t)与跨区地表来水，避免先做全上游高分辨率二维模型。晴海48条事件小时平滑实测潮位已补到，但不能代替分钟级河道水位。</p><p><a href="assessment.txt">完整判断与限制</a> · <a href="connectivity_audit.json">连通修正来源</a> · <a href="controlled_comparison_summary.csv">控制对照</a> · <a href="numerical_validation.json">数值检查</a> · <a href="upstream_and_stage_data_needs.csv">资料需求清单CSV</a> · <a href="../../../processed/three_ward_road_connectivity/tokyo_harumi_tide_20240821_22.csv">晴海潮位CSV</a> · <a href="../../../processed/three_ward_road_connectivity/R1_area_weighted_peak.tif">修正后模型栅格</a> · <a href="../../../processed/three_ward_road_connectivity/timestep_sensitive_cells.tif">时间步敏感格</a></p><p>官方参考：<a href="https://www.kensetsu.metro.tokyo.lg.jp/river/seibi/shibuya">渋谷川—古川</a> · <a href="https://www.kensetsu.metro.tokyo.lg.jp/river/seibi/kanda">神田川水系</a> · <a href="https://www.kensetsu.metro.tokyo.lg.jp/river/seibi/kasenbu0078">隅田川</a> · <a href="https://www.data.jma.go.jp/kaiyou/db/tide/genbo/explanation.html">JMA实测潮位定义</a></p></html>''')

 defhash=[ROOT/'src/exp/prepare_road_connectivity.py',ROOT/'src/exp/run_road_connectivity.py',PRO/'road_graph.npz',PRO/'road_ground_5m.tif',OLD/'surface_inputs.npz'];js(OUT/'input_manifest.json',[{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in defhash]);print('FINAL',conclusion,flush=True)

def plots(audit,a,g,r0,r1,flag,changes,tf):
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False;nr,nc=a['land'].shape;extent=[tf.c/1000,(tf.c+nc*20)/1000,(tf.f-nr*20)/1000,tf.f/1000];toxy=Transformer.from_crs(6668,6677,always_xy=True);ws=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];polys=[transform(toxy.transform,shape(f['geometry'])) for f in ws]
 def boundaries(ax):
  for g0 in polys:
   for p in g0.geoms if g0.geom_type=='MultiPolygon' else [g0]:xy=np.asarray(p.exterior.coords)/1000;ax.plot(xy[:,0],xy[:,1],c='#555',lw=.55)
  ax.set_xlabel('東西 km');ax.set_ylabel('南北 km')
 fig,axes=plt.subplots(1,3,figsize=(16,7));data=[r0['area_weighted_peak'],r1['area_weighted_peak'],r1['area_weighted_peak']-r0['area_weighted_peak']];titles=['R0 跨格道路连接关闭','R1 跨格道路连接开启','R1 - R0（重分配，非实地风险变化）']
 for j,(ax,d,title) in enumerate(zip(axes,data,titles)):
  im=ax.imshow(d,extent=extent,cmap='Blues' if j<2 else 'RdBu_r',vmin=0 if j<2 else -.3,vmax=.5 if j<2 else .3);fig.colorbar(im,ax=ax,shrink=.65,label='面积加权节点峰值 m' if j<2 else '节点峰值加权均值差 m',extend='max' if j<2 else 'both');boundaries(ax);ax.set_title(title,fontsize=11)
 fig.suptitle('三区道路连通控制试算／同一合成降雨与储水\n原DEM未烧低；水域边界关闭；局部水深尚未验证',fontsize=14);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'road_connectivity_comparison.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.8));x=np.arange(3);axes[0].bar(x-.18,[r['suspect_gt2m_R0_m2']/10000 for r in changes],.36,label='R0');axes[0].bar(x+.18,[r['suspect_gt2m_R1_m2']/10000 for r in changes],.36,label='R1');axes[0].set_xticks(x,[r['ward'] for r in changes]);axes[0].set_ylabel('>2m模型疑义开放面 ha');axes[0].set_title('没有整体下降／不能解读为实测洪深');axes[0].legend();axes[1].imshow(flag,extent=extent,cmap=ListedColormap(['#f0f4f8','#ea9861']));boundaries(axes[1]);axes[1].set_title('橙：含>0.10m时间步峰值差的格');fig.suptitle('连通改善不等于水深验证：仍需储水与边界资料',fontsize=14);fig.tight_layout();fig.savefig(OUT/'road_connectivity_metrics.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
