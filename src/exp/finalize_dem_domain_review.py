"""Summarize native DEM availability, provisional scope and forcing access."""
import os
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,csv,html,hashlib,importlib.util
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling
from rasterio.features import rasterize
from shapely.geometry import shape,mapping,box
from shapely.ops import transform,unary_union
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
PROC=ROOT/'data/processed/three_ward_dem_domain'
OUT=ROOT/'data/exp/data-briefing/three_ward_dem_domain'
XY=Transformer.from_crs(6668,6677,always_xy=True)
LL=Transformer.from_crs(6677,6668,always_xy=True)
def load(p):return json.loads(p.read_text())
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def read_fc(p):return [(transform(XY.transform,shape(f['geometry'])),f['properties']) for f in load(p)['features']]
def outlines(ax,g,**kw):
 parts=list(g.geoms) if hasattr(g,'geoms') else [g]
 for p in parts:
  if p.geom_type=='Polygon':x,y=p.exterior.xy;ax.plot(x,y,**kw)
def main():
 manifest=load(OUT/'acquisition_manifest.json');scope=load(OUT/'domain_screening.json');ws=read_fc(ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson');domain=read_fc(PROC/'acquisition_domain.geojson')[0][0]
 waters=unary_union([g for g,p in read_fc(ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson')])
 with rasterio.open(PROC/'regional_screening_20m.tif') as d:regional=d.read(1);tf=d.transform;crs=d.crs
 native=np.full(regional.shape,np.nan,dtype='float32')
 with rasterio.open(PROC/'native_dem_0_50m.vrt') as d:
  assert d.res==(.5,.5)
  reproject(source=rasterio.band(d,1),destination=native,src_transform=d.transform,src_crs='EPSG:6677',src_nodata=-32767,dst_transform=tf,dst_crs='EPSG:6677',dst_nodata=np.nan,resampling=Resampling.nearest,warp_mem_limit=128)
 with rasterio.open(PROC/'native_dem_20m_center_audit.tif','w',driver='GTiff',height=native.shape[0],width=native.shape[1],count=1,dtype='float32',transform=tf,crs='EPSG:6677',nodata=np.nan,compress='deflate',tiled=True) as d:d.write(native,1)
 regions=[(p['ward'],g.difference(waters)) for g,p in ws]+[('acquisition_domain',domain),('external_candidate',read_fc(PROC/'surface_contributing_candidates.geojson')[0][0])]
 valid_area={name:0. for name,g in regions};represented_area={name:0. for name,g in regions}
 footprints=[]
 for count,row in enumerate(manifest['resources'],1):
  p=ROOT/row['tif_path'];gbox=box(*row['bounds']);footprints.append(gbox)
  with rasterio.open(p) as d:
   a=d.read(1,masked=True);valid=~np.ma.getmaskarray(a)&np.isfinite(a.data)
   for name,g in regions:
    if not g.intersects(gbox):continue
    intersect=g.intersection(gbox);mask=rasterize([(mapping(intersect),1)],out_shape=d.shape,transform=d.transform).astype(bool)
    valid_area[name]+=float((valid&mask).sum())*.25
    represented_area[name]+=float(mask.sum())*.25
  if count%100==0:print('native pixel coverage audited',count,'/',len(manifest['resources']),flush=True)
 footprint_union=unary_union(footprints);coverage=[]
 for name,g in regions:
  coverage.append({'region':name,'geometric_target_km2':g.area/1e6,'raster_footprint_intersection_km2':g.intersection(footprint_union).area/1e6,'valid_native_pixel_area_km2':valid_area[name]/1e6,'valid_native_pixel_fraction':valid_area[name]/g.area if g.area else None,'area_with_no_valid_native_elevation_km2':max(0,g.area-valid_area[name])/1e6,'note':'Ward target excludes PLATEAU wtr geometry. Native0.50m pixel-center area; no gap filling, overlapping sources counted once by distinct non-overlapping original sheet grid.'})
 with (OUT/'native_dem_coverage.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
 missing_ids={x['sheet'] for x in manifest['failures']};missing_geo=[(g,{**p,'source_status':'official_grd50_archive_HTTP403','not_inferred_zero_height':True}) for g,p in read_fc(PROC/'selected_sheet_index.geojson') if p['sheet'] in missing_ids]
 dump(PROC/'unavailable_native_sheets.geojson',{'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(transform(LL.transform,g)),'properties':p} for g,p in missing_geo]})
 acquire_mask=rasterize([(mapping(domain),1)],out_shape=native.shape,transform=tf).astype(bool)
 gap=acquire_mask&~np.isfinite(native)
 with rasterio.open(PROC/'no_native_elevation_20m_audit.tif','w',driver='GTiff',height=gap.shape[0],width=gap.shape[1],count=1,dtype='uint8',transform=tf,crs='EPSG:6677',nodata=0,compress='deflate',tiled=True) as d:d.write(gap.astype('uint8'),1)
 stats={'official_selected_sheets':manifest['expected_sheets'],'verified_native_sheets':manifest['verified_sheets'],'new_source_sheets':manifest['downloaded_new_sheets'],'reused_prior_study_sheets':manifest['reused_sheets'],'missing_source_ids':sorted(missing_ids),'archives_gb':manifest['total_archive_bytes']/1e9,'native_source_resolution_m':.5,'actual_cropped_source_sheets':sum(r['shape']!=[600,800] for r in manifest['resources']),'coverage':coverage,'all_official_urls_obtained':manifest['complete'],'height_datum_and_survey_epoch_verified':False,'regional_GSI_heights_used_to_fill_native_gaps':False,'final_hydraulic_domain_certified':False,'source_grid_footprints_nonoverlap_check':abs(sum(g.area for g in footprints)-footprint_union.area)<1e-4}
 assert stats['source_grid_footprints_nonoverlap_check']
 dump(OUT/'dem_delivery_summary.json',stats)
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
 fig,axs=plt.subplots(1,2,figsize=(17,10),constrained_layout=True);ext=(tf.c,tf.c+tf.a*regional.shape[1],tf.f+tf.e*regional.shape[0],tf.f);l,b,r,t=domain.bounds
 im=axs[0].imshow(regional,extent=ext,cmap='terrain',vmin=0,vmax=50,interpolation='nearest');axs[0].set_title('初划数据获取范围：三区 + 跨区候选来水 + 核查余量')
 with rasterio.open(PROC/'potential_contribution_mask.tif') as d:potential=d.read(1)>0
 with rasterio.open(PROC/'ward_id.tif') as d:ward=d.read(1)
 overlay=np.where(potential&(ward==0),1,np.nan);axs[0].imshow(overlay,extent=ext,cmap=matplotlib.colors.ListedColormap(['#ee9b2b']),alpha=.8,vmin=0,vmax=1,interpolation='nearest')
 axs[1].imshow(np.where(np.isfinite(native),1,np.nan),extent=ext,cmap=matplotlib.colors.ListedColormap(['#318987']),interpolation='nearest',vmin=0,vmax=1);axs[1].imshow(np.where(gap,1,np.nan),extent=ext,cmap=matplotlib.colors.ListedColormap(['#e09c9c']),alpha=.8,interpolation='nearest',vmin=0,vmax=1);axs[1].set_title('原生0.50 m DEM有效覆盖：右图按20 m中心抽查显示')
 names={'13101-chiyoda-ku':'千代田区','13102-chuo-ku':'中央区','13103-minato-ku':'港区'}
 for ax in axs:
  outlines(ax,domain,color='#486fa1',linewidth=.8)
  for g,p in ws:
   outlines(ax,g,color='#273a4b',linewidth=1.2);pt=g.representative_point();ax.text(pt.x,pt.y,names[p['ward']],fontsize=12,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
  for g,p in missing_geo:outlines(ax,g,color='#8e1b57',linewidth=1.6)
  ax.set_xlim(l-500,r+500);ax.set_ylim(b-500,t+500);ax.set_aspect('equal');ax.set_xlabel('EPSG:6677 Easting (m)');ax.set_ylabel('Northing (m)');ax.grid(alpha=.12)
 axs[0].legend(handles=[Patch(color='#ee9b2b',label='潜在跨区地表来水（待复核）'),Patch(edgecolor='#486fa1',facecolor='none',label='数据获取范围')],loc='lower left',fontsize=10)
 axs[1].legend(handles=[Patch(color='#318987',label='20m中心有原生DEM高程'),Patch(color='#e09c9c',label='获取域内缺高程（含水域/缺测）'),Patch(edgecolor='#8e1b57',facecolor='none',label='4个不可获取官方图幅')],loc='lower left',fontsize=10)
 fig.suptitle('三区原生 DEM 批量获取与范围初划｜不是已验证的水动力边界',fontsize=18);fig.savefig(OUT/'domain_and_dem_coverage.png',dpi=170);plt.close(fig)
 routes=load(OUT/'forcing_acquisition_routes.json');es=html.escape
 covrows=''.join('<tr>'+''.join(f'<td>{es(str(row[k])) if k=="region" else f"{row[k]:.3f}"}</td>' for k in ['region','geometric_target_km2','valid_native_pixel_area_km2','area_with_no_valid_native_elevation_km2'])+'</tr>' for row in coverage)
 rt=''.join(f'<section><b>{es(row["product"])}</b><p>{es(row["available"])}</p><p>{es(row["route"])}</p><small>{es(row["need"])}</small><p><a href="{es(row["link"])}">官方入口</a> · {es(row["status"])}</p></section>' for row in routes)
 text=f'''三区及跨区地表汇水范围/0.50m DEM批量获取
三区边界{scope['ward_polygon_area_km2']:.3f}km²；区域静态地形潜在外部来水{scope['filled_potential_external_km2']:.3f}km²；含100m核查余量的数据获取域{scope['acquisition_area_km2']:.3f}km²。
这是初划数据获取域。GSI区域地形、D8优先填洼和W05中心线截断仅用于筛查；填洼影响的外部候选{scope['external_uncertain_km2']:.3f}km²。河岸、暗渠与多方向漫流未证实，不能把整个候选当真实必要汇水量。搜索边缘没有候选来水，仍需用原生DEM/正式二维计算检验边界。
官方594图幅已核验获取590；新源{stats['new_source_sheets']}，复用{stats['reused_prior_study_sheets']}；ZIP共{stats['archives_gb']:.3f}GB。{stats['actual_cropped_source_sheets']}个官方图幅采用裁切尺寸，已按真实范围保留；4个grd50源HTTP403：{', '.join(stats['missing_source_ids'])}。NoData、没有图幅及水域都不填0，不用GSI替补假装0.50m。
各区非水域和候选区有效高程面积见native_dem_coverage.csv。数据空间分辨率0.50m不代表高程精度、连续地形或已核实测量时期，垂直基准成果版本仍待官方元数据。
暴雨/水位：已有参考资料保留；2024核心历史时序仍未取得。东京都官网最多过去1年；港区公开3小时的前1小时滚动雨量不能转成原始3分钟降雨；DIAS XRAIN元数据当前注明下载不可用，需询问现行档案和研究许可途径；FRIC有偿历史包仅备选。10个都雨量候选和7个水位候选已从当前官方坐标目录列出，但2024站址/零点未核实。日语询问稿已备，未发送。
下一步：优先复核原生DEM未覆盖的非水域和与范围外连通的热点，确认完整模型域；同步申请2024-08-21至22日核心站雨量/河段水位及零点。河川大上游通过Q(t)/H(t)或1D模型提供，不默认全部纳入0.50m二维域。
'''
 (OUT/'assessment.txt').write_text(text)
 (OUT/'review.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区汇水范围与DEM数据获取</title><style>body{{font:16px/1.65 -apple-system,sans-serif;background:#f3f6f8;color:#243c48;margin:0}}main{{max-width:1150px;margin:auto;padding:28px}}section{{background:white;padding:18px;border-radius:8px;margin:12px 0}}.note{{padding:18px;background:#fff0d9;border-left:5px solid #d69a24}}img{{max-width:100%;height:auto}}.table{{overflow:auto}}table{{min-width:650px;border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #dce5e8;padding:10px;text-align:left}}small{{color:#586976}}a{{color:#116f88}}</style><main><h1>三区地表汇水范围与0.50 m DEM</h1><div class="note">已完成初划及可获取图幅的批量核验；4个官方源仍不可获取。范围用于数据获取，正式水动力边界尚待原生地形与河岸核查。</div><section><b>本轮获取</b><p>三区{scope['ward_polygon_area_km2']:.2f} km² · 潜在跨区来水{scope['filled_potential_external_km2']:.2f} km² · 数据获取域{scope['acquisition_area_km2']:.2f} km²</p><p>594个索引图幅中590个核验成功；{stats['new_source_sheets']}个新增，{stats['reused_prior_study_sheets']}个复用，原生ZIP约{stats['archives_gb']:.3f} GB。4个不可获取图幅：{es(', '.join(stats['missing_source_ids']))}。</p><small>0.50m是源网格间距。原生裁切与NoData均保留，未改变高程或用区域GSI填补。测绘日期和高程成果版本仍未全部核实。</small></section><img src="domain_and_dem_coverage.png" alt="候选来水范围、原生DEM覆盖及4个不可获取图幅"><h2>实际有效高程覆盖</h2><p>区内目标面积扣除PLATEAU wtr几何；采样面积为原生0.50m有效像元的统计。水域、建成区缺测和缺失图幅的成因另行区分，不能把缺值都当作可补裸地。</p><div class="table"><table><tr><th>区域</th><th>目标km²</th><th>有有效DEM km²</th><th>无有效DEM km²</th></tr>{covrows}</table></div><section><b>范围的不确定性</b><p>外部候选中{scope['external_uncertain_km2']:.2f}km²对填洼和河渠截断敏感。区域地形筛查不能证明岸顶、地下流路或暴雨时多方向交换；100m余量也不是汇水流域。</p><small>河川上游来水单列H(t)/Q(t)边界，不能因为有河川连线就把105km²神田川流域全部当直接地表来水。</small></section><h2>核心暴雨和河段水位：获取途径</h2>{rt}<h2>交付文件</h2><section><p>GIS：acquisition_domain.geojson、surface_contributing_candidates.geojson、native_dem_sheets.geojson、unavailable_native_sheets.geojson</p><p>原生虚拟拼接：native_dem_0_50m.vrt；查看时按原生0.50m读取，未生成伪填满的整域栅格。</p><p>获取清单、CRC/哈希与覆盖统计：acquisition_manifest.json、native_dem_coverage.csv。候选站与日语未发送询问稿已保存。</p></section></main></html>''')
 (PROC/'README.txt').write_text('''Provisional DEM acquisition domain, not certified hydraulic catchment.
acquisition_domain.geojson:3wards + screened external contribution +100m margin.
surface_contributing_candidates.geojson:static20m D8/priority-flood screening.
regional_screening_20m.tif:GSI display elevations, screening only, never model fill.
native_dem_0_50m.vrt:virtual mosaic of verified original Tokyo2024 grd50 sheets.
native_dem_20m_center_audit.tif and no_native_elevation_20m_audit.tif:coverage display,
not a replacement terrain dataset or proof of accuracy.
native_dem_sheets.geojson:actual source TIFF footprint, including cropped sheets.
unavailable_native_sheets.geojson:official index footprints of inaccessible sources.
forcing_station_candidates.csv/geojson:current official coordinate candidates;
2024site validity and stage zero/datum have not yet been verified.
No source heights corrected, no depressions filled in the native DEM.
Review acquisition_manifest.json and native_dem_coverage.csv for source gaps.
All original files are kept under data/raw/tokyo_lidar_dem/2024.
''')
 framework=ROOT/'data/processed/three_ward_sfincs_framework'
 if framework.exists():
  spec=load(framework/'model_spec.json');spec['domain_data_update']={'acquisition_domain':'data/processed/three_ward_dem_domain/acquisition_domain.geojson','native_dem_vrt':'data/processed/three_ward_dem_domain/native_dem_0_50m.vrt','coverage':'data/exp/data-briefing/three_ward_dem_domain/native_dem_coverage.csv','scope_status':'provisional','available_native_sheets':590,'inaccessible_native_sources':sorted(missing_ids),'height_epoch_alignment_complete':False};dump(framework/'model_spec.json',spec)
  with (framework/'input_inventory.csv').open(encoding='utf-8-sig',newline='') as f:inventory=list(csv.DictReader(f))
  for row in inventory:
   if row['id']=='domain':row['source_path']='data/processed/three_ward_dem_domain/acquisition_domain.geojson';row['gap']='已初划56.69km²获取域，跨区潜在来水10.02km²；对填洼/河渠截断敏感，正式边界仍需原生地形和水动力复核。'
   if row['id']=='terrain':row['source_path']='data/processed/three_ward_dem_domain/native_dem_0_50m.vrt';row['gap']='594图幅取得590，4官方源HTTP403；实际有效覆盖已统计，NoData未填补；测绘时间/高程成果版本待核实。'
   if row['id'] in ['rain','river_stage']:row['next_action']='按forcing_acquisition_routes.csv及未发送日语询问稿获取2024事件核心档案和站点元数据。'
  with (framework/'input_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
 print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
