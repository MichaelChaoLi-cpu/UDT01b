"""Regional source-road list and fixed19:00 map; no new model or investigation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,gzip,collections,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_motor_flood_access_v1 import P,R,N,read,write,dump,load

def main():
 p=load(P/'road_projection.npz');data=np.load(P/'exposures/R1__H__U.npz');lower=data['lower'][5];upper=data['upper'][5];wet=p['qualified']&(lower>=.3);rows=[];meta={};count=collections.Counter();lengths=collections.Counter()
 with(P/'road_inventory.csv').open(encoding='utf-8-sig')as f:
  for row in csv.DictReader(f):
   if row['inside_core']!='True':continue
   j=int(row['road_index']);meta[row['source_edge_id']]=(j,row)
   if wet[j]:
    rows.append(dict(**row,case_id='R1__H__U',time_s=14400,depth_threshold_m=.3,water_depth_lower_m=float(lower[j]),water_depth_upper_m=float(upper[j]),role='Analyst road-interruption scenario, not observed closure.'));name=row['name']or'(unnamed source road)';count[name]+=1;lengths[name]+=float(row['length_m'])/1000
 write(R/'closed_source_roads_19jst.csv',rows);summary=[dict(source_name=n,exceedance_source_segments=c,whole_intersecting_source_length_km=lengths[n],not_capacity_or_flow=True)for n,c in count.most_common()];write(R/'closed_road_name_summary_19jst.csv',summary)
 normal=[];closed=[];unknown=[];features=[]
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as f:
  for line in f:
   e=json.loads(line)
   if e['edge_id']not in meta:continue
   j,row=meta[e['edge_id']];xy=np.asarray(e['geometry']['coordinates'])/1000
   if wet[j]:closed.append(xy);features.append(dict(type='Feature',geometry=e['geometry'],properties=dict(source_edge_id=e['edge_id'],name=row['name'],highway=row['highway'],depth_lower_m=float(lower[j]),depth_upper_m=float(upper[j]),case_id='R1__H__U',time_s=14400,threshold_m=.3,not_observed_closure=True)))
   elif p['qualified'][j]:normal.append(xy)
   else:unknown.append(xy)
 fig,ax=plt.subplots(figsize=(8,9),layout='constrained')
 for lines,color,width,alpha in [(normal,'#bcc8ce',.35,.75),(unknown,'#618ca5',.45,.65),(closed,'#b63f3f',1.,.95)]:ax.add_collection(LineCollection(lines,colors=color,linewidths=width,alpha=alpha))
 ax.set_xlim(-11.36,-3.66);ax.set_ylim(-42.44,-32.32);ax.set_aspect('equal');ax.set_xlabel('JGD2011 / IX easting (km)');ax.set_ylabel('Northing (km)');ax.set_title('Motorcar source roads | R1 / raised stage / urban geometry\n19:00 JST | interruption hypothesis >=0.3 m',fontsize=12)
 ax.legend(handles=[Line2D([0],[0],color=c,label=l,linewidth=w)for c,l,w in [('#b63f3f','Modeled threshold exceedance',1.8),('#bcc8ce','Supported ground, below threshold',1),('#618ca5','Unknown road height/exposure',1)]],loc='lower left',fontsize=9);fig.savefig(R/'vehicle_road_impacts_19jst.png',dpi=160);plt.close(fig)
 dump(P/'closed_vehicle_roads_19jst_6677.geojson',dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=features))
 page=R/'review.html';text=page.read_text();marker='<!-- VEHICLE_ROAD_IMPACTS -->'
 block=marker+'<h2>哪些道路源路段触发中断假设</h2><p>固定原雨量、水位抬升、城市几何U组19:00和0.3 m情景，三区相交源路段中有'+str(len(rows))+'段地面源线段达到深度假设。它们是模型投影下的源线段，不是整条道路实际关闭。蓝色为高度或积水对应未知，不能画成干路。</p><img src="vehicle_road_impacts_19jst.png" alt="Regional motor-road threshold exceedance map"><p><a href="closed_source_roads_19jst.csv">受阻源路段清单</a> · <a href="closed_road_name_summary_19jst.csv">按来源道路名称汇总</a> · <a href="'+os.path.relpath(P/'closed_vehicle_roads_19jst_6677.geojson',R)+'">路段GIS</a>。名称汇总使用完整相交源线段长度，不是区界裁剪道路长度、车道公里或交通容量。</p>'
 if marker not in text:text=text.replace('<h2>方向损失和绕行</h2>',block+'<h2>方向损失和绕行</h2>');page.write_text(text)
 dump(R/'road_impact_summary.json',dict(case_id='R1__H__U',time_s=14400,threshold_m=.3,core_intersecting_source_segments_triggered=len(rows),gis_features=len(features),unknown_source_segments=len(unknown),not_actual_closure=True));print('VEHICLE ROAD MAP',len(rows),'threshold-triggered source segments')
if __name__=='__main__':main()
