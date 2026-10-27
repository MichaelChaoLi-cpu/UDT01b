"""Publish native-source scope refinement from already checked diagnostic masks."""
from pathlib import Path
import importlib.util,json,re,zipfile,csv
import numpy as np
import rasterio
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('acq',ROOT/'src/preprocessing/acquire_three_ward_dem.py');acq=importlib.util.module_from_spec(spec);spec.loader.exec_module(acq)
def main():
 with rasterio.open(acq.PROC/'native_filled_scope_mask.tif') as d:potential=d.read(1)>0;tf=d.transform
 with rasterio.open(acq.PROC/'ward_id.tif') as d:ward=d.read(1)
 with rasterio.open(acq.PROC/'native_dem_20m_center_audit.tif') as d:native=d.read(1)
 with rasterio.open(acq.PROC/'potential_contribution_mask.tif') as d:initial=d.read(1)>0
 ws=acq.wards();core=unary_union([g for g,p in ws]);g=acq.geometry(potential&(ward==0),tf).difference(core)
 with zipfile.ZipFile(ROOT/'data/raw/ksj/N03-20240101_13_GML.zip') as z:admin=json.loads(z.read('N03-20240101_13.geojson'))
 koto=unary_union([acq.transform(acq.TOXY.transform,acq.shape(f['geometry'])) for f in admin['features'] if f['properties']['N03_007']=='13108'])
 across_water=g.intersection(koto);g=g.difference(koto);candidate=unary_union([core,g])
 acq.fc(acq.PROC/'native_unverified_across_water_candidates.geojson',[(across_water,{'role':'across-water candidate excluded from baseline until physical surface connection confirmed','ward':'江東区'})])
 rows=[]
 grouped={}
 for f in admin['features']:
  if f['properties']['N03_007'] in ['13101','13102','13103','13108']:continue
  gg=acq.transform(acq.TOXY.transform,acq.shape(f['geometry']))
  if gg.intersects(g):
   overlap=gg.intersection(g)
   if overlap.area:grouped.setdefault((f['properties']['N03_007'],f['properties']['N03_004']),[]).append(overlap)
 for (code,name),parts in grouped.items():rows.append({'ward_code':code,'ward_name':name,'native_candidate_external_km2':unary_union(parts).area/1e6,'hydraulic_connection_verified':False})
 rows.sort(key=lambda r:-r['native_candidate_external_km2'])
 with (acq.OUT/'native_external_candidates_by_ward.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 acq.fc(acq.PROC/'native_surface_contributing_candidates.geojson',[(g,{'role':'available-native-source static surface contribution candidate','sample_m':20,'full_resolution_flow_delineation':False,'hydraulic_connection_verified':False})])
 acq.fc(acq.PROC/'native_model_domain_candidate.geojson',[(candidate,{'role':'provisional model-domain core, not final hydraulic boundary','needs_boundary_exchange_check':True,'region_area_km2':candidate.area/1e6})])
 check=json.loads((acq.OUT/'native_scope_check.json').read_text());check['native_model_domain_core_km2']=candidate.area/1e6;check['candidate_including_external_polygon_km2']=g.area/1e6;check['excluded_unverified_across_water_km2']=across_water.area/1e6;check['already_acquired_native_elevation_covers_extension_candidates']=True;acq.dump(acq.OUT/'native_scope_check.json',check)
 font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
 fig,ax=plt.subplots(figsize=(9,11),constrained_layout=True);ext=[tf.c,tf.c+tf.a*native.shape[1],tf.f+tf.e*native.shape[0],tf.f]
 ax.imshow(np.where(initial&(ward==0),1,np.nan),extent=ext,cmap=matplotlib.colors.ListedColormap(['#ddd7c9']),vmin=0,vmax=1,interpolation='nearest')
 filtered=acq.rasterize([(acq.mapping(g),1)],out_shape=potential.shape,transform=tf).astype(bool)
 ax.imshow(np.where(filtered,1,np.nan),extent=ext,cmap=matplotlib.colors.ListedColormap(['#dd8933']),vmin=0,vmax=1,interpolation='nearest')
 def line(geom,**kw):
  for p in list(geom.geoms) if hasattr(geom,'geoms') else [geom]:
   x,y=p.exterior.xy;ax.plot(x,y,**kw)
 labels={'13101-chiyoda-ku':'千代田区','13102-chuo-ku':'中央区','13103-minato-ku':'港区'}
 for gg,p in ws:line(gg,color='#285364',linewidth=1.4);pt=gg.representative_point();ax.text(pt.x,pt.y,labels[p['ward']],fontsize=12)
 l,b,r,t=core.bounds;ax.set_xlim(l-1600,r+600);ax.set_ylim(b-600,t+5800);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('EPSG:6677 Easting (m)');ax.set_ylabel('Northing (m)');ax.set_title('新DEM缩小候选范围：保守获取范围保留，模型域可收紧',fontsize=14)
 ax.legend(handles=[Patch(color='#ddd7c9',label='区域地形初筛（10.02km²）'),Patch(color='#dd8933',label=f'原生DEM候选，扣除待核对岸（{g.area/1e6:.2f}km²）'),Patch(edgecolor='#285364',facecolor='none',label='三区边界')],loc='lower left')
 fig.savefig(acq.OUT/'native_domain_refinement.png',dpi=160);plt.close(fig)
 paragraph=f"原生DEM范围复核：同类20m中心筛查的外部候选为{check['filled_external_candidate_km2']:.2f}km²，未填洼直接D8候选为{check['unfilled_external_candidate_km2']:.2f}km²。除6个20m像元（0.0024km²、其原生DEM也已取得）外均在初划获取域内。扣除尚未证实地表连接的江東对岸候选{across_water.area/1e6:.2f}km²后，候选模型核心约{candidate.area/1e6:.2f}km²，可从已取得的保守56.69km²数据域裁出；最终边界仍需河岸/NoData和二维交换检验。"
 page=acq.OUT/'review.html';s=page.read_text();block='<section id="native-refinement"><b>用新DEM复核范围</b><p>'+paragraph+'</p><small>1.99km²也是静态地形候选，存在洼地、缺值截断和河渠近似；不作为实测来水量。20m采样诊断不等于原生0.50m水动力求解。</small></section><img src="native_domain_refinement.png" alt="原生DEM复核后的较小候选地表来水范围">'
 if 'id="native-refinement"' in s:s=re.sub(r'<section id="native-refinement">.*?</section><img src="native_domain_refinement.png"[^>]*>',lambda m:block,s)
 else:s=s.replace('<h2>实际有效高程覆盖</h2>',block+'<h2>实际有效高程覆盖</h2>')
 page.write_text(s)
 text=acq.OUT/'assessment.txt';s=text.read_text()
 if '原生DEM范围复核：' in s:s=re.sub('原生DEM范围复核：[^\n]*',lambda m:paragraph,s)
 else:s+='\n'+paragraph+'\n'
 text.write_text(s)
 framework=ROOT/'data/processed/three_ward_sfincs_framework/model_spec.json';s=json.loads(framework.read_text());s['domain_data_update'].update({'provisional_native_model_core':'data/processed/three_ward_dem_domain/native_model_domain_candidate.geojson','native_scope_check':'data/exp/data-briefing/three_ward_dem_domain/native_scope_check.json','native_model_domain_core_km2':candidate.area/1e6,'domain_is_certified':False});framework.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
 print(paragraph)

if __name__=='__main__':main()
