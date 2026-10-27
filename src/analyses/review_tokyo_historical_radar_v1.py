"""Reproducible whole-domain numerical-radar readiness review; no solver calls."""
from pathlib import Path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt_historical_radar_mpl')
import csv,datetime as dt,hashlib,html,json,sys
import numpy as np
import rasterio
from rasterio.features import shapes
from affine import Affine
from scipy.sparse import load_npz
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'preprocessing'))
from acquire_tokyo_historical_radar_v1 import decode, self_test
ROOT=Path(__file__).resolve().parents[2]
TAG='tokyo_historical_radar_20240821_v1';P=ROOT/'data/processed'/TAG;RAW=ROOT/'data/raw'/TAG;R=ROOT/'data/exp/data-preprocessing'/TAG
JST=dt.timezone(dt.timedelta(hours=9))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

def main():
    R.mkdir(parents=True,exist_ok=True)
    c=json.loads((P/'rainfall_contract.json').read_text());mapping=json.loads((P/'source_recipient_map_contract.json').read_text());q=json.loads((P/'spatial_forcing_qa.json').read_text())
    a=np.load(P/'radar_intervals.npz');rate=a['representative_intensity_mm_h'];ends=a['interval_end_unix_s'];start=a['interval_start_unix_s']
    hours=15+(ends-ends[0]+300)/3600
    checks=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=detail))
        if not ok:raise AssertionError(name+': '+detail)
    self_test();check('decoder_independent_expansion_and_malformed_fixtures',True)
    manifest=json.loads((RAW/'acquisition_manifest.json').read_text())
    check('108_unique_raw_frames',len(manifest['files'])==108==len({x['url'] for x in manifest['files']}))
    check('continuous_exact_300s_windows',np.all(ends-start==300) and np.array_equal(ends[:-1],start[1:]))
    for i,entry in enumerate(manifest['files']):
        p=ROOT/entry['local_path'];check('raw_sha256_'+str(i),sha(p)==entry['sha256'])
        level,r,meta=decode(p.read_bytes())
        check('raw_decode_roundtrip_'+str(i),np.array_equal(level,a['level'][i]) and np.array_equal(r,rate[i],equal_nan=True))
    for entry in manifest['source_documents']:check('document_sha_'+Path(entry['path']).name,sha(ROOT/entry['path'])==entry['sha256'])
    check('original_script_sha_bound',manifest['script_sha256']==sha(ROOT/'src/preprocessing/acquire_tokyo_historical_radar_v1.py'))
    check('netcdf_sha_bound',sha(P/'forcing_intervals_10m.nc')==q['output_sha256'])
    matrices={k:load_npz(P/k) for k in ['terrain_source_area_matrix.npz','urban_source_area_matrix.npz','roof_source_area_matrix.npz']}
    for name,expected in mapping['matrix_sha256'].items():check('map_sha_'+name,sha(P/name)==expected)
    for name,expected in mapping['source_sha256'].items():check('unchanged_source_'+Path(name).name,sha(ROOT/name)==expected)
    t=matrices['terrain_source_area_matrix.npz'];u=matrices['urban_source_area_matrix.npz'];roof=matrices['roof_source_area_matrix.npz']
    source_area=np.asarray(t.sum(0)).ravel();check('source_area_marginals_identical',np.array_equal(source_area,np.asarray(u.sum(0)).ravel()))
    with rasterio.open(ROOT/'data/processed/three_ward_fixed_runoff/contributing_area_10m.tif') as ds:
        frozen=ds.read(1);tf=ds.transform
    check('unchanged_urban_contributing_area',np.array_equal(np.asarray(u.sum(1)).reshape(frozen.shape),frozen))
    intervals=list(csv.DictReader((P/'interval_water_volume.csv').open()))
    with Dataset(P/'forcing_intervals_10m.nc') as nc:
        check('nc_exact_time_bounds',np.array_equal(nc['time_bounds'][:],np.column_stack([start,ends])))
        check('nc_x_y_centres_and_orientation',np.allclose(nc['x'][:],tf.c+(np.arange(frozen.shape[1])+.5)*10) and np.allclose(nc['y'][:],tf.f-(np.arange(frozen.shape[0])+.5)*10))
        max_error=0
        for i in range(108):
            expected=float(intervals[i]['source_rain_volume_m3'])
            for var in ['urban_effective_rainfall','terrain_effective_rainfall']:
                v=nc[var][i].data.astype('f8');check('nc_valid_'+var+'_'+str(i),np.isfinite(v).all() and (v>=0).all())
                err=abs(float(v.sum()*100/12000)-expected);max_error=max(max_error,err)
                check('nc_source_volume_'+var+'_'+str(i),err<=max(1e-7,expected*1e-7))
    check('no_missing_or_saturated_catchment_inputs',q['missing_values_over_catchment']==q['saturated_values_over_catchment']==0)
    check('no_solver_execution',q['solver_runs_started']==0 and q['solver_ready'] is False)
    support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');ward=support['Ward Code']
    all_rate=rate.reshape(108,-1).astype('f8');all_total=all_rate.sum(0)/12;ward_stats=[];hyeto={}
    for label,code in [('Whole catchment',None),('Three wards',-1),('Chiyoda',1),('Chuo',2),('Minato',3),('Cross-ward buffer',0)]:
        sel=np.ones(ward.shape,bool) if code is None else ward>0 if code==-1 else ward==code
        areas=np.asarray(t[sel.ravel()].sum(0)).ravel();used=areas>0
        weights=areas/areas.sum();mean=all_rate@weights;hyeto[label]=mean
        cells=all_total[used]
        ward_stats.append(dict(scope=label,original_catchment_area_km2=float(areas.sum()/1e6),
            source_cells=int(used.sum()),weighted_reconstructed_total_mm=float(mean.sum()/12),
            radar_cell_total_min_mm=float(cells.min()),radar_cell_total_max_mm=float(cells.max()),
            catchment_average_peak_representative_rate_mm_h=float(mean.max()),
            average_peak_end_jst=dt.datetime.fromtimestamp(int(ends[mean.argmax()]),JST).isoformat(),
            area_weighted_max_rolling60_mm=float(np.convolve(mean/12,np.ones(12),'valid').max())))
    with (P/'ward_rainfall_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ward_stats[0]);w.writeheader();w.writerows(ward_stats)
    # Native 1 km rainfall grid remains visibly native; ward lines are model's 10 m classification.
    tr=Transformer.from_crs(6677,6668,always_xy=True)
    polygons=[]
    for geom,value in shapes(ward.astype('u1'),mask=ward>0,transform=tf):polygons.append((transform(tr.transform,shape(geom)),int(value)))
    def outlines(ax):
        for poly,code in polygons:
            seq=list(poly.geoms) if poly.geom_type=='MultiPolygon' else [poly]
            for part in seq:
                x,y=part.exterior.xy;ax.plot(x,y,color='#243142',lw=.65)
        for name,x,y in [('Chiyoda',139.751,35.695),('Chuo',139.78,35.668),('Minato',139.735,35.643)]:
            ax.text(x,y,name,fontsize=8,ha='center',bbox=dict(fc='white',alpha=.75,ec='none',pad=1))
        ax.set(xlim=(139.58,139.90),ylim=(35.59,35.78),xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)')
        ax.set_aspect(1/np.cos(np.deg2rad(35.68)))
    xedge=np.r_[a['longitude']-.00625,a['longitude'][-1]+.00625]
    yedge=np.r_[a['latitude']+1/240,a['latitude'][-1]-1/240]
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for ax,time_label in zip(axes.ravel()[:3],['18:00','18:30','19:00']):
        target=dt.datetime.fromisoformat('2024-08-21T'+time_label+':00+09:00').timestamp();idx=int(np.where(ends==target)[0][0])
        im=ax.pcolormesh(xedge,yedge,rate[idx],cmap='YlGnBu',vmin=0,vmax=150,shading='flat');outlines(ax)
        ax.set_title('Radar representative intensity at '+time_label+' JST');fig.colorbar(im,ax=ax,label='mm/h')
    im=axes.ravel()[3].pcolormesh(xedge,yedge,all_total.reshape(48,40),cmap='YlGnBu',vmin=0,vmax=150,shading='flat');outlines(axes.ravel()[3])
    axes.ravel()[3].set_title('Reconstructed rain, 15:00-24:00 JST');fig.colorbar(im,ax=axes.ravel()[3],label='mm')
    fig.savefig(R/'native_radar_structure.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    for name in ['Chiyoda','Chuo','Minato','Whole catchment']:
        axes[0].step(hours,hyeto[name],where='pre',label=name)
        axes[1].plot(np.r_[15,hours],np.r_[0,np.cumsum(hyeto[name]/12)],label=name)
    axes[0].set_ylabel('Area-weighted representative intensity (mm/h)');axes[1].set_ylabel('Reconstructed accumulated rain (mm)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=9)
    axes[1].set(xlim=(15,24),xticks=np.arange(15,25),xlabel='2024-08-21 local hour (JST); 24 = next midnight')
    fig.savefig(R/'catchment_hyetographs.png',dpi=170);plt.close(fig)
    pairs=list(csv.DictReader((P/'gauge_aligned_pairs.csv').open(encoding='utf-8-sig')))
    comparisons=list(csv.DictReader((P/'gauge_comparison.csv').open(encoding='utf-8-sig')))
    station_labels={'47662':'Tokyo / Kitanomaru','0368':'Setagaya','0370':'Edogawa Rinkai','1002':'Nerima'}
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
    for ax,g in zip(axes.ravel(),comparisons):
        rows=[x for x in pairs if x['station_id']==g['station_id']];h=[15+(dt.datetime.fromisoformat(x['interval_end_jst']).timestamp()-start[0])/3600 for x in rows]
        ax.step(h,[float(x['gauge_10min_mm']) for x in rows],where='pre',label='JMA point gauge')
        ax.step(h,[float(x['radar_reconstructed_10min_mm']) for x in rows],where='pre',label='Radar containing cell')
        ax.set(title=station_labels[g['station_id']],ylabel='10-minute rain (mm)',xlim=(15,24));ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[1,0].set_xlabel('Hour (JST)');axes[1,1].set_xlabel('Hour (JST)');fig.savefig(R/'radar_gauge_comparison.png',dpi=170);plt.close(fig)
    check('scope_totals_match_nc_source',abs(ward_stats[0]['weighted_reconstructed_total_mm']-q['catchment_area_weighted_reconstructed_rain_mm'])<1e-9)
    summary=dict(rainfall=c,spatial_forcing=q,ward_statistics=ward_stats,delivery_check_count=len(checks),all_checks_passed=all(x['passed'] for x in checks),
        original_data_and_split_untouched=True,historical_candidate_not_added_to_formal_training_library=True,
        max_independent_nc_volume_error_m3=max_error,
        next_required='Qualify a preceding-five-minute block-intensity adapter in the actual CPU solver, preserving exact second clocks, same-step budgets, corners and units. Then launch one historical-rainfall case with explicit river-stage scenarios; assess inundation before any transport closure assumptions.')
    (R/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (R/'delivery_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    body='<h1>历史雷达雨场：2024年8月21日东京</h1><p class="lead">已获取数值雨场，并完成到全域10 m地表／屋面汇流网格的守恒映射。当前交付为降雨输入准备与核验；尚未运行历史暴雨积水模拟。</p>'
    body+='<div class="cards">'+''.join('<div><b>'+x+'</b><br>'+y+'</div>' for x,y in [('108幅','15:00—24:00 JST连续时段'),('5分钟／约1 km','气象厅分析产品，大学公开存档'),('0缺测／0饱和','本次裁切雨场与实际汇水范围'),(f"{q['catchment_area_weighted_reconstructed_rain_mm']:.1f} mm",'全计算汇水范围面积加权重建雨量')])+'</div>'
    body+='<h2>雨落在哪里，比全域同一条雨型更接近这次事件</h2><p>图中雨场保留原始约1 km格网。城市边界线来自既有10 m分区；10 m接收网格不代表降雨观测达到10 m精度。各时段数值为GRIB2内置强度等级代表值，使用前300秒恒定强度重建雨量。该积分是明确的输入重建规则，不能称为直接观测的5分钟累计雨量。</p><img src="native_radar_structure.png" alt="Native rainfall fields and reconstructed accumulation">'
    body+=table(['范围','原汇水范围 km²','加权重建雨量 mm','区域平均强度峰值 mm/h','源格网累计范围 mm'],[[x['scope'],f"{x['original_catchment_area_km2']:.3f}",f"{x['weighted_reconstructed_total_mm']:.2f}",f"{x['catchment_average_peak_representative_rate_mm_h']:.1f}",f"{x['radar_cell_total_min_mm']:.1f}–{x['radar_cell_total_max_mm']:.1f}"] for x in ward_stats])
    body+='<p>范围统计使用既有原始汇水掩膜内的面积及10 m区划代码，不代表整个行政区所有陆地／水面降雨。跨区缓冲范围也保留在输入内。</p><img src="catchment_hyetographs.png" alt="Whole-domain and ward hyetographs">'
    body+='<h2>地面站核验：可用作事件驱动，不能当作无误差真值</h2><p>所有比较均为同一15:00—24:00窗口，先对齐至10分钟累计。点雨量与格网代表值支持不同，站坐标也有公开精度限制；雷达雨量换算可能使用雨量站信息，不能宣称这是完全独立验证。没有根据这4站对雨场作整体比例校正。</p>'
    body+=table(['站点','站雨量 mm','雷达重建 mm','差值 mm','10分钟RMSE mm','站／雷达最大60分钟 mm'],[[g['station_name'],f"{float(g['gauge_total_mm']):.2f}",f"{float(g['radar_reconstructed_total_mm']):.2f}",f"{float(g['bias_total_mm']):+.2f}",f"{float(g['rmse_10min_mm']):.2f}",f"{float(g['gauge_rolling60_peak_mm']):.2f} / {float(g['radar_rolling60_peak_mm']):.2f}"] for g in comparisons])
    body+='<img src="radar_gauge_comparison.png" alt="Gauge and radar rainfall comparison">'
    body+='<h2>空间输入与屋面水量守恒</h2><p>以每个0.50 m源像元中心确定降雨源格网；开放地表雨水留在原10 m格网，屋面雨水按既有规则转移至最近的、与道路相连的开放地表。先在屋面源位置取雨，再转移水量，避免直接用接收位置雨量乘旧屋面权重。</p>'
    body+=table(['检查','结果'],[['既有地表接收面积与屋面接收面积','逐格完全一致'],['仅地形与城市约束的降雨源面积','逐源格网完全一致'],['源汇水范围',f"{q['catchment_area_m2']/1e6:.4f} km²；{q['original_radar_source_cells_with_catchment']}个雷达源格网"],['重建总雨量体积',f"{q['total_reconstructed_source_rain_volume_m3']:,.3f} m³"],['float32导出单时段最大相对体积误差',f"{q['max_interval_export_relative_volume_error']:.3e}"],['原人口、质量掩膜及12＋4正式划分','未修改；历史候选事件未并入训练库'],['交付核查',f"{len(checks)}项全部通过"]])
    body+='<p>屋面规则仍为全部即时流向地表、无损失无延迟的假设，不是实测落水管／地下管网。相同屋面规则被保留，以免把降雨变化与排水设定变化混在一起。</p>'
    body+='<h2>模拟前还缺的接口与边界</h2><ol><li><b>5分钟时段输入接口：</b>现有CPU求解器的已核验步积分只接受分段线性AMPR，且明确拒绝ampr_block。当前导出的NetCDF使用时间窗与格网中心，不能直接替换求解器的角点坐标文件。需要保留秒级时钟、同一步水量预算的适配与接口测试；不应把数据强行连成斜线或制造1秒过渡。</li><li><b>河道边界：</b>尚无统一高程基准的本次实测河道水位时序。首次历史降雨驱动模拟应把低／抬升水位标为边界假设，而非声称复现了当时完整河况。</li><li><b>后续证据：</b>模拟完成并通过原数值准入后，再叠合道路和站口。积水暴露与站口关闭、整站关闭、区段停运假设继续分别记录。雷达雨场本身不能验证积水深度或实际停运。</li></ol>'
    body+='<h2>数据、脚本与来源</h2><ul>'
    for label,p in [('处理数据与合同',P),('原始108幅GRIB2与SHA256清单',RAW),('获取与解码脚本',ROOT/'src/preprocessing/acquire_tokyo_historical_radar_v1.py'),('守恒空间映射脚本',ROOT/'src/preprocessing/prepare_tokyo_spatial_radar_v1.py'),('本报告复现脚本',Path(__file__))]:body+='<li><a href="'+html.escape(str(p))+'">'+label+'</a></li>'
    body+='</ul><p>产品与字节格式：<a href="https://www.data.jma.go.jp/suishin/jyouhou/pdf/568.pdf">气象厅配信资料技术信息568</a>。文件来自<a href="http://database.rish.kyoto-u.ac.jp/arch/jmadata/">京都大学RISH气象厅数据档案</a>，按研究用途访问；产品生产者为气象厅，大学为分发存档机构。公开HTTP传输已在清单记录，SHA256用于本地文件身份和复现，不等于对远端内容进行认证。产品注明GRS80；本次给定JGD2011经纬度用于对接，是明确的坐标近似，不宣称观测历元已经核准。</p>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>东京历史雷达雨场核验</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#233043;background:#f7f9fc;max-width:1200px;margin:auto;padding:32px}h1,h2{line-height:1.35}h2{margin-top:36px}.lead{font-size:20px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #d5deea;padding:9px;text-align:left}th{background:#e9eff6}img{max-width:100%;background:white;border-radius:8px;margin:18px 0}.cards{display:flex;flex-wrap:wrap;gap:14px}.cards div{background:#e7f0fa;flex:1;min-width:210px;padding:18px}.cards b{font-size:24px}a{color:#176b9c}</style>'+body+'</html>'
    (R/'review.html').write_text(page)
    print(json.dumps({'checks':len(checks),'all_passed':True,'ward_statistics':ward_stats,'report':str(R/'review.html')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
