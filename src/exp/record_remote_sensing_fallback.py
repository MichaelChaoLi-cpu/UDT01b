"""Record the user's no-inquiry, numerical/remote-sensing fallback data route."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import csv,json,hashlib,shutil,html
import requests
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
PRO=ROOT/'data/processed/three_ward_terrain_fallback'
OUT=ROOT/'data/exp/data-briefing/remote_sensing_fallback'
RAW=ROOT/'data/raw/remote_sensing_reference/2026-10-14'
SOURCES=[
 ('imerg_overview','https://gpm.nasa.gov/data/imerg'),
 ('imerg_v07','https://gpm.nasa.gov/resources/documents/imerg-v07-technical-documentation'),
 ('gsmap_guide','https://sharaku.eorc.jaxa.jp/GSMaP/guide.html'),
 ('jma_radar_product','https://www.data.jma.go.jp/suishin/cgi-bin/catalogue/make_product_page.cgi?id=KaisekiU'),
 ('jma_radar_access','https://www.data.jma.go.jp/developer/weatherdataguide/appendix/1-1-d.html'),
 ('swot_reach_D','https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_RiverSP_reach_D'),
 ('swot_sampling','https://swot.jpl.nasa.gov/resources/136/swot-river-products/'),
 ('sentinel_flood_method','https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-community-examples/python/50_thematic-notebooks/flood-mapping/flood-sar.html'),
 ('sentinel_catalog_api','https://documentation.dataspace.copernicus.eu/APIs/OData.html'),
 ('aw3d30','https://www.eorc.jaxa.jp/ALOS/jp/dataset/aw3d30/aw3d30_j.htm'),
 ('worldcover','https://esa-worldcover.org/en/data-access')]
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def main():
 for p in [RAW,OUT]:p.mkdir(parents=True,exist_ok=True)
 def snapshot(entry):
  name,url=entry;p=RAW/(name+'.html')
  if not p.exists():
   r=requests.get(url,timeout=(15,40));p.write_bytes(r.content);status=r.status_code
  else:status=200
  return {'name':name,'url':url,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':status}
 with ThreadPoolExecutor(max_workers=4) as pool:sources=list(pool.map(snapshot,SOURCES))
 for name in ['tokyo_sentinel1_catalog.json','tokyo_sentinel1_august.json','imerg_event_granules.json','gsmap_event_graph.html','gsmap_portal.js']:
  p=Path('/private/tmp')/name
  if p.exists():shutil.copyfile(p,RAW/name)
 products=[
  dict(id='IMERG_Final_V07B',role='降雨主卫星备选',space='0.1°，东京约9×11km',time='30分钟平均',access='NASA Earthdata/PPS正式文件需认证；已核实2024-08-21 09:00/09:30/10:00 UTC颗粒存在，数据文件未下载',use='事件雨量时序和区域分布情景；与可得JMA原始数值站对比',limit='Final含月尺度雨量计校准，非纯卫星独立地面验证；局地峰值/街区雨差无法恢复',url=dict(SOURCES)['imerg_overview']),
  dict(id='GSMaP',role='降雨第二卫星备选',space='0.1°',time='小时',access='JAXA公开图表可查看2024；CSV/FTP正式数值下载要求注册，未注册未下载CSV',use='与IMERG对照雨量总量和时段，保留具体STD/Gauge产品版本',limit='比IMERG时间更粗；小时平均不是短时峰值，Gauge产品含雨量计调整，不能混称纯卫星',url=dict(SOURCES)['gsmap_guide']),
  dict(id='JMA_Radar_Analysis',role='有现成数值文件时优先的地面遥感雨量',space='1km',time='30分钟更新前1小时累积；速報10分钟更新同为前1小时累积',access='公开图片易得；2024数值档案通常由数据服务渠道提供，本轮不询问机构/购买',use='若已可直接取得，优先用于局地暴雨；用原始数值及区间定义',limit='滚动1小时量不能直接当30或10分钟增量；不从公报最大值或彩图构造时序',url=dict(SOURCES)['jma_radar_product']),
  dict(id='SWOT_RiverSP_D',role='宽河道水位空间旁证',space='单次过境河段/节点产品',time='21日重复轨道，东京过境和具体河段尚未核实',access='NASA PO.DAAC VersionD，数据需常规获取/质量标记核验；本轮没有下载河段数据',use='隅田川等较宽河段的水面高程/坡度候选约束',limit='无法代替古川等窄河分钟H(t)；垂直基准及质量必须核对。流量为检索/估计产品，不是直接量测流量',url=dict(SOURCES)['swot_reach_D']),
  dict(id='Sentinel1_IW_GRDH',role='事件积水影像验证候选',space='SAR影像，具体分辨率依产品',time='东京8月目录最近事件前后为8月17日和29日JST；8月18–24日UTC窗无匹配IW_GRDH',access='已读取Copernicus公开目录，未下载影像；重复处理版本已按过境时刻去重',use='仅选确实捕获积水的事件/位置；可补水域几何和较持久洪水范围',limit='这场短时街面积水没有同期匹配过境；不能以无影像/未见水体证明未积水，也不输出水深',url=dict(SOURCES)['sentinel_flood_method']),
  dict(id='AW3D30',role='最后层级区域地形背景',space='约30m，DSM',time='多期光学立体观测及补源，需读tile元数据',access='JAXA正式产品，通常要求注册；本轮未下载',use='区域背景/范围核对；PLATEAU缺口仍残余时另评估可靠地形源',limit='城市DSM含建筑/树冠，不能直接当道路裸地DTM；本项目优先PLATEAU ground TIN而不是AW3D30',url=dict(SOURCES)['aw3d30']),
  dict(id='WorldCover2021_v200',role='土地覆盖/粗糙度辅助',space='约10m，11类土地覆盖',time='2021年分类',access='ESA公开S3可匿名获取，未在本轮下载',use='树木、草地、建成区等参数分区参考',limit='不能直接给出入渗能力、排水能力或2024地貌；需要与PLATEAU建筑/道路优先层叠合',url=dict(SOURCES)['worldcover'])]
 with (OUT/'remote_sensing_options.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(products[0]));w.writeheader();w.writerows(products)
 dump(OUT/'remote_sensing_options.json',products);dump(OUT/'source_manifest.json',sources)
 check=json.loads((OUT/'terrain_fallback_validation.json').read_text());registry=json.loads((PRO/'elevation_sources.json').read_text())
 policy={'updated_at':datetime.now(timezone.utc).isoformat(),'user_preferences':{'external_inquiries':False,'wait_for_institution_archive_reply':False,'primary_inputs':'原始数值观测/地形，不能取得时采用适配的遥感估计','bulletins':'only_optional_context_not_main_forcing_or_depth_validation'},'terrain':{'primary':'native0.50m DEM where present','gap_fallback':'original PLATEAU udx/dem ground TIN','fallback_file':'data/processed/three_ward_terrain_fallback/plateau_ground_tin_5m.tif','source_registry':'data/processed/three_ward_terrain_fallback/elevation_sources.json','roof_geometry_used_as_ground':False,'coarser_source_not_promoted_to_fine_accuracy':True,'future_residual_fallback':'evaluate actual bare-earth regional source;30mDSM not automatically used for street flow'},'rain':{'preferred_if_available':'numeric weather radar product with valid interval metadata','satellite_primary_candidate':'IMERG Final V07B half-hourly','satellite_comparison_candidate':'GSMaP hourly product version to verify','satellite_files_downloaded':False,'no_fabricated_spatial_detail_from_interpolation':True,'no_bulletin_peak_as_hyetograph':True,'use_existing_JMA_raw_station_data':True},'water_boundary':{'observed_sea_background':'existing48hourly JMA Harumi observed values with filter metadata','satellite':'SWOT only occasional wide-river constraints, not continuous event replacement','unobserved_river_stages':'documented boundary scenarios / later routed river model, not claimed observed2024hydrographs','no_wtr_Z_as_stage':True},'validation':{'remote_event_images_if_timed':True,'no_Sentinel1_IW_GRDH_match_in_checked_event_window':True,'satellite_does_not_certify_street_depth':True,'mode':'remote-sensing-constrained scenario analysis until actual inputs and validation support event replay'},'five_stage_order_preserved':True,'interventions_after_first_four_steps':True}
 dump(PRO/'data_policy.json',policy)
 # Reconcile framework inputs, without marking calibration or numerical stages complete.
 frame=ROOT/'data/processed/three_ward_sfincs_framework';spec=json.loads((frame/'model_spec.json').read_text());spec['data_policy']=policy;spec['terrain_source_priority']=registry['source_priority'];dump(frame/'model_spec.json',spec)
 with (frame/'input_inventory.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 for r in rows:
  if r['id']=='terrain':r['source_path']='data/processed/three_ward_terrain_fallback/elevation_sources.json';r['gap']='已保留0.50m DEM并准备PLATEAU原始地形TIN补缺层，5m预览三区覆盖约99.8%–100%；成果时期/独立精度仍未认证。';r['next_action']='按来源优先生成正式子网格，保留源间距/质量标记；不再等待机构询问。'
  if r['id']=='rain':r['source_path']='data/exp/data-briefing/remote_sensing_fallback/remote_sensing_options.json';r['next_action']='采用可得原始数值雷达或IMERG/GSMaP正式卫星时序；保留分辨率，公报不作为强迫。';r['gap']='已确认IMERG2024事件档案存在，但卫星正式文件未下载且需认证；不能重建港区分钟雨峰。'
  if r['id']=='river_stage':r['next_action']='不等待机构档案；使用可得数值边界，缺时转为明确水位情景，SWOT仅宽河段旁证。'
  if r['id']=='validation':r['next_action']='优先时间匹配的遥感影像、已有原始数值观测及物理/数值检验；公报只作旁证。'
 with (frame/'input_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(frame/'data_policy.json',policy)
 # Visual source map: equal display sampling, distinct original source fidelity.
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
 with rasterio.open(PRO/'terrain_preview_source_5m.tif') as d:a=d.read(1);bounds=d.bounds
 fig,ax=plt.subplots(figsize=(8,9),constrained_layout=True);img=np.where(a>0,a,np.nan);ax.imshow(img,extent=[bounds.left,bounds.right,bounds.bottom,bounds.top],cmap=ListedColormap(['#23828c','#dc9b41']),vmin=1,vmax=2,interpolation='nearest');ax.set_aspect('equal');ax.set_title('地形来源：保留原生DEM，PLATEAU地形补缺');ax.set_xlabel('EPSG:6677 Easting(m)');ax.set_ylabel('Northing(m)');ax.legend(handles=[Patch(color='#23828c',label='0.50m DEM在5m预览中心的采样'),Patch(color='#dc9b41',label='PLATEAU原始地形TIN备用')],loc='lower left');fig.savefig(OUT/'terrain_source_map.png',dpi=150);plt.close(fig)
 s1=json.loads((RAW/'tokyo_sentinel1_august.json').read_text())['response']['value'];unique=sorted(set(x['ContentDate']['Start'] for x in s1));imerg=json.loads((RAW/'imerg_event_granules.json').read_text())['response']['feed']['entry']
 evidence={'sentinel1_matching_utc_passes_august':unique,'sentinel1_event_window_matches':len(json.loads((RAW/'tokyo_sentinel1_catalog.json').read_text())['value']),'imerg_core_event_granules_found':[x['title'] for x in imerg],'imerg_official_entry_status_checked':401,'gsmap_public_graph_http_status':200,'gsmap_registration_required_for_csv':True,'gsmap_protected_csv_download_attempted':False,'institutional_inquiries_sent':False,'accounts_registered':False,'remote_numeric_files_downloaded':False}
 dump(OUT/'access_validation.json',evidence)
 text=f'''数据路线已按用户决定更新：不再询问机构、不等待历史档案回复，优先原始数值资料与适配的遥感估计，公报不作为主要强迫或水深验证。
地形：保留590个原生0.50m DEM；用已有三区CityGML的udx/dem原始地形TIN补缺。新增备用样本格距5m，候选预览补缺{check['filled_preview_area_km2']:.5f}km²，残余预览缺值{check['residual_preview_missing_area_km2']:.5f}km²。千代田99.8523%、中央100%、港99.8829%是5m预览非水域覆盖，不是原生0.50m完整性或绝对精度。
全部原生DEM有效值在预览采样中保持不变；不用屋顶标高，不进行全域高程偏移/填洼/外推。重叠样本差中位数均0，绝对差p95分别0.07/0.05/0.09m；两源可能共享测绘来源，不能称独立精度验证。
降雨：IMERG Final V07B（0.1°/30min）主卫星候选，GSMaP（0.1°/1h）第二候选；东京0.1°约9×11km，不能保留港区局地分钟雨峰或伪生成街区分布。已有JMA原始数值站可继续作参考。JMA1km雷达解析雨量有现成数值时优先，但前1h滚动量不能当30/10min增量。
获取核查：NASA公开目录已确认2024-08-21 18:00/18:30/19:00JST的IMERG颗粒存在；正式文件入口需认证，未下载。GSMaP公开图表可查询事件，CSV规定须注册，不以提取图表脚本绕过正式下载条件。未创建账户、未发送申请。
水位：SWOT VersionD能约束宽河段的过境水面高程/坡度，不代替窄河分钟H(t)。没有可靠事件河位时做明确边界情景，海侧可用已有晴海小时平滑观测作背景，不能将其等同所有河段水位。
影像：已核查东京Sentinel-1 IW_GRDH2024年8月目录，实际过境UTC为{', '.join(unique)}，事件附近JST8月17日和29日，没有8月21日短时积水同期匹配；不得把未见水当未积水。
模型表述：在强迫与验证不充分时称遥感约束情景分析，不能宣称已真实复现并验证2024街道积水深度。五步顺序保留，主动措施仍在前四步之后。
'''
 (OUT/'assessment.txt').write_text(text)
 escaped=html.escape;trs=''.join(f'<tr><td>{escaped(p["role"])}</td><td>{escaped(p["id"])}</td><td>{escaped(p["space"])} / {escaped(p["time"])}</td><td>{escaped(p["access"])}</td><td>{escaped(p["limit"])}</td></tr>' for p in products)
 cov=''.join(f'<li>{name}：5m预览覆盖{r["preview_coverage_fraction"]*100:.2f}%，PLATEAU补缺{r["plateau_gap_fallback_cells"]*25/1e6:.3f}km²。</li>' for name,r in zip(['千代田区','中央区','港区'],registry['coverage']))
 (OUT/'review.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PLATEAU补缺与遥感备选</title><style>body{{font:16px/1.65 -apple-system,sans-serif;background:#f4f7f8;color:#203b48;margin:0}}main{{max-width:1150px;margin:auto;padding:28px}}section{{background:white;padding:18px;margin:12px 0;border-radius:8px}}.note{{background:#fff0d5;padding:18px;border-left:5px solid #c98b2a}}img{{max-width:100%;height:auto}}.table{{overflow:auto}}table{{border-collapse:collapse;min-width:980px;background:white}}td,th{{padding:10px;border:1px solid #dbe4e7;text-align:left;vertical-align:top}}a{{color:#176b82}}</style><main><h1>PLATEAU地形补缺与遥感数据备选</h1><div class="note">不再等待机构询问。采用原始数值资料和适配的遥感估计，公报只保留可选背景。正式卫星雨量文件仍未下载；输入缺口与场景假设均保留标记。</div><h2>已完成：地形备用层</h2><section><p>保留原生0.50m DEM，已从三区CityGML的原始udx/dem地形TIN生成5m备用层，补上约{check['filled_preview_area_km2']:.2f}km²预览缺口。</p><ul>{cov}</ul><p>上述百分比是5m预览覆盖；约5m的TIN不能因插值变成0.50m精度。没有使用屋顶标高，没有偏移高程或填洼。源基准成果时期尚未独立核实。</p></section><img src="terrain_source_map.png" alt="DEM与PLATEAU地形备用来源图"><h2>遥感产品如何使用</h2><div class="table"><table><tr><th>用途</th><th>产品</th><th>空间/时间</th><th>实际获取状态</th><th>适用边界</th></tr>{trs}</table></div><h2>新的模型数据路线</h2><section><p>地形：原生DEM → PLATEAU ground TIN补缺；保留来源与分辨率。</p><p>雨量：可得数值雷达优先；IMERG为主卫星候选，GSMaP作对照；用已有数值站检查总量与时段。</p><p>水位：可得数值边界与海侧背景；SWOT作宽河空间约束。缺少真实河段时序则用明确的边界情景。</p><p>验证：时间匹配的影像和数值观测优先；缺证据处报告不确定性。此事件Sentinel-1目录没有同期匹配，不能以影像缺失作为未积水证据。</p><p>前四步完成后再比较主动措施；在观测不足时明确标记情景结果，不宣称实测复现。</p></section><h2>官方来源</h2><ul>{''.join(f'<li><a href="{escaped(u)}">{escaped(n)}</a></li>' for n,u in SOURCES)}</ul></main></html>''')
 # Point older snapshots to the current policy; preserve historical evidence and drafts.
 for old in [ROOT/'data/exp/data-briefing/three_ward_dem_domain/review.html',ROOT/'data/exp/data-briefing/three_ward_sfincs_framework/review.html']:
  s=old.read_text()
  if 'remote-policy-update' not in s:s=s.replace('<h2>', '<section id="remote-policy-update"><b>最新数据路线</b><p>已按用户要求停止机构询问，以PLATEAU地形补缺和遥感数据作备选。旧询问稿仅是历史未发送材料；后续路线见<a href="../remote_sensing_fallback/review.html">本轮数据报告</a>。</p></section><h2>',1);old.write_text(s)
 print({'terrain_preview_gap_filled_km2':check['filled_preview_area_km2'],'products_considered':len(products),'institutional_inquiries':False,'remote_files_downloaded':False,'imerg_event_granules_confirmed':len(imerg),'sentinel_event_window_matches':evidence['sentinel1_event_window_matches']})

if __name__=='__main__':main()
