"""Inspect official spatial maximum inundation depths and historical observations.
Reproduce locally: .venv/bin/python src/exp/analyze_inundation_reference.py
No flood simulation, synthetic observations, geocoded town centroids, or datum correction.
"""
import os
for k in ('PROJ_LIB','PROJ_DATA','GDAL_DATA'):os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,html,hashlib
import numpy as np
import shapely as S
from shapely.geometry import shape,mapping,Polygon,MultiPolygon
from shapely.ops import unary_union
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch,Patch
from matplotlib.path import Path as MPath
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'data/raw/tokyo_inundation_reference/2026-10-13';OUT=ROOT/'data/exp/data-briefing/tokyo_inundation_reference';PROC=ROOT/'data/processed/tokyo_inundation_reference'
XY=Transformer.from_crs(6668,6677,always_xy=True);LL=Transformer.from_crs(6677,6668,always_xy=True)
WARD={'13101-chiyoda-ku':'千代田区','13102-chuo-ku':'中央区','13103-minato-ku':'港区'}
SOURCES=[('kanda','shinsui_kandagawa.csv','神田川流域','fld_8_2','2018-03-30'),('sumida','shinsui_sumidagawa.csv','隅田川・新河岸川流域','fld_10_2','网页说明2020-03；PLATEAU目录2021-03-30，版本待协调'),('jonan','shinsui_jyounantiku.csv','城南地区河川流域','fld_14_2','2018-12-20／2019-06-27修正'),('koto','shinsui_koutounaibu.csv','江東内部河川流域','fld_16_2','2020-03-26')]
FONT=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf');plt.rcParams['font.family']=FONT.get_name()
COLORS=['#fae1b0','#eea354','#d66751','#a93e59','#69386e','#332644']
BINLABELS=['0 < h < 0.1 m','0.1–0.5 m','0.5–1 m','1–2 m','2–3 m','≥3 m']

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def project(g,t=XY):return S.transform(g,t.transform,interleaved=False)
def draw(ax,g,**kw):
 if isinstance(g,Polygon):
  g=S.orient_polygons(g);v=[];c=[]
  for ring in [g.exterior,*g.interiors]:
   q=np.array(ring.coords);v.extend(q);c.extend([MPath.MOVETO]+[MPath.LINETO]*(len(q)-2)+[MPath.CLOSEPOLY])
  ax.add_patch(PathPatch(MPath(v,c),**kw))
 elif hasattr(g,'geoms'):
  for q in g.geoms:draw(ax,q,**kw)
def point_read(path):
 head=path.open('rb').read(1024)
 try:head.decode('utf-8-sig');enc='utf-8-sig'
 except UnicodeDecodeError:enc='cp932'
 with path.open(encoding=enc,newline='') as f:
  rd=csv.reader(f);headers=next(rd);rows=[];skipped=0
  for q in rd:
   if not q or (len(q)==1 and q[0].strip()=='\x1a'):skipped+=1;continue
   if len(q)!=5:raise ValueError('Unexpected data row: '+repr(q))
   rows.append([float(v) if v.strip() else np.nan for v in q])
  a=np.array(rows,dtype=np.float64)
 if len(headers)!=5 or not all(t in headers for t in ('浸水深','地盤高','緯度','経度')):raise ValueError(headers)
 return a,headers,enc,skipped
class HTMLTables(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  if t=='tr':self.row=[]
  if t in ('td','th'):self.cell=[]
 def handle_data(self,s):
  if self.cell is not None:self.cell.append(s)
 def handle_endtag(self,t):
  if t in ('td','th') and self.cell is not None:
   if self.row is not None:self.row.append(' '.join(''.join(self.cell).split()))
   self.cell=None
  if t=='tr' and self.row:
   self.rows.append(self.row);self.row=None

def main():
 OUT.mkdir(parents=True,exist_ok=True);PROC.mkdir(parents=True,exist_ok=True)
 wards={WARD[f['properties']['ward']]:project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features']};domain=unary_union(list(wards.values()));bounds=project(domain,LL).bounds
 groups={};stats=[];fileaudit=[];validation={};csv_fields=['source','source_row','sheet_no','raw_inundation_value','depth_m','source_ground_m','latitude','longitude','ward','source_date_context']
 for key,fn,label,sid,date in SOURCES:
  path=RAW/fn;a,headers,enc,skipped=point_read(path);valid=np.all(np.isfinite(a[:,3:5]),axis=1)
  bbox=(a[:,4]>=bounds[0])&(a[:,4]<=bounds[2])&(a[:,3]>=bounds[1])&(a[:,3]<=bounds[3])&valid
  ix=np.flatnonzero(bbox);x,y=XY.transform(a[ix,4],a[ix,3]);inside=S.intersects_xy(domain,x,y);ix=ix[inside];x=x[inside];y=y[inside];b=a[ix]
  # Preserve conflicting co-located depths. Only duplicate coordinates + depth + terrain rows are removed.
  sig=np.column_stack([np.round(b[:,4],8),np.round(b[:,3],8),b[:,1],b[:,2]]);_,keep=np.unique(sig,axis=0,return_index=True);keep=np.sort(keep)
  removed=len(b)-len(keep);b=b[keep];x=x[keep];y=y[keep];ix=ix[keep]
  _,coordcounts=np.unique(np.round(b[:,3:5],8),axis=0,return_counts=True);conflict=int(np.sum(coordcounts>1))
  wname=np.full(len(b),'',dtype=object)
  for name,wg in wards.items():wname[S.intersects_xy(wg,x,y)]=name
  records=[dict(zip(csv_fields,[key,int(i+2),int(v[0]) if np.isfinite(v[0]) else None,float(v[1]) if np.isfinite(v[1]) else None,float(v[1]) if key!='kanda' and np.isfinite(v[1]) else None,float(v[2]) if np.isfinite(v[2]) else None,float(v[3]),float(v[4]),str(w),date])) for i,v,w in zip(ix,b,wname)]
  table(PROC/(key+'_model_points_three_wards.csv'),records)
  np.savez_compressed(PROC/(key+'_model_points_three_wards.npz'),source_row=ix+2,sheet=b[:,0],depth=b[:,1] if key!='kanda' else np.full(len(b),np.nan),raw_inundation_value=b[:,1],ground=b[:,2],lat=b[:,3],lon=b[:,4],x=x,y=y,ward=wname.astype(str))
  groups[key]=(b,x,y,wname,label)
  fileaudit.append({'source':key,'file':fn,'source_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'encoding':enc,'source_nondata_rows_skipped':skipped,'original_headers':headers,'source_rows':len(a),'rows_with_finite_coordinates':int(valid.sum()),'rows_missing_coordinates':int((~valid).sum()),'missing_counts_by_original_field':{h:int(np.sum(~np.isfinite(a[:,j]))) for j,h in enumerate(headers)},'retained_three_ward_points':len(b),'identical_point_value_duplicates_removed':removed,'co_located_conflicting_value_groups':conflict,'source_date_context':date,'coordinate_use':'Raw lon/lat preserved; EPSG6668 assumed for spatial clip following package metadata; no resampling','geometry_role':'source grid point positions, not independently surveyed water depths'})
  for name in wards:
   q=b[wname==name];d=q[:,1];wet=d[d>=.1]
   stats.append({'source':key,'source_name':label,'ward':name,'points':len(q),'zero_depth_points':int(np.sum(d==0)),'depth_below_0_1m_points':int(np.sum((d>0)&(d<.1))),'depth_ge_0_1m_points':len(wet),'depth_max_m':float(np.nanmax(d)) if np.isfinite(d).any() else None,'wet_depth_median_m':float(np.median(wet)) if len(wet) else None,'wet_depth_p95_m':float(np.quantile(wet,.95)) if len(wet) else None,'ground_min_m':float(np.nanmin(q[:,2])) if np.isfinite(q[:,2]).any() else None,'ground_max_m':float(np.nanmax(q[:,2])) if np.isfinite(q[:,2]).any() else None,'missing_ground_points':int(np.sum(~np.isfinite(q[:,2]))),'missing_depth_points':int(np.sum(~np.isfinite(d))),'negative_depth_points':int(np.sum(d<0)),'meaning':'modeled maximum scenario depths at provided grid points; count is not inundation area'})
  print(key,'source rows',len(a),'三区',len(b),'duplicates',removed,'conflicts',conflict,flush=True)
 for st in stats:
  if st['source']=='kanda':
   st['raw_inundation_field_max']=st['depth_max_m'];st['raw_nonzero_points']=st['depth_ge_0_1m_points'];st['value_use_status']='Nonzero source values are only integers 1–7 in three wards; rounding/code semantics unresolved, not used as precise metre depth.'
   for fld in ('depth_ge_0_1m_points','depth_max_m','wet_depth_median_m','wet_depth_p95_m'):st[fld]=None
  else:st['value_use_status']='Numeric model depth values; not observed water depths.'
 table(OUT/'scenario_depth_summary.csv',stats);dump(OUT/'scenario_depth_summary.json',stats);dump(OUT/'numeric_source_audit.json',fileaudit)
 # Compare only reference bins for the corresponding PLATEAU source; do not compare geometry Z to new DEM.
 hazard=json.loads((ROOT/'data/processed/plateau_hazards/scenario_depth_class_footprints.geojson').read_text())['features'];matches=[]
 for key,fn,label,sid,date in SOURCES:
  b,x,y,wname,_=groups[key];geoms=[];ranks=[]
  if key=='kanda':
   matches.append({'source':key,'plateau_scenario_id':sid,'status':'Not assessed as metre depth; integer field semantics require verification','raw_nonzero_values':sorted(set(b[b[:,1]>0,1].tolist()))});continue
  for f in hazard:
   if f['properties']['scenario_id']==sid:geoms.append(S.make_valid(project(shape(f['geometry']))));ranks.append(int(f['properties']['rank_code']))
  tr=S.STRtree(geoms);points=S.points(x,y);pairs=tr.query(points,predicate='intersects');assigned=np.zeros(len(b),dtype=int)
  for pi,gi in pairs.T:assigned[pi]=max(assigned[pi],ranks[gi])
  d=b[:,1];expected=np.select([d<.5,d<3,d<5,d<10,d<20],[1,2,3,4,5],default=6);wet=d>=.1;covered=assigned>0
  matches.append({'source':key,'plateau_scenario_id':sid,'points_depth_ge_0_1m':int(wet.sum()),'wet_points_inside_plateau_footprint':int((wet&covered).sum()),'wet_points_outside_plateau_footprint':int((wet&~covered).sum()),'wet_covered_points_same_depth_class':int((wet&covered&(assigned==expected)).sum()),'wet_covered_points_different_depth_class':int((wet&covered&(assigned!=expected)).sum()),'scope':'point versus 2D source footprint class; boundary/grid representation and version differences remain; not independent flood validation'})
 dump(OUT/'plateau_reference_class_comparison.json',matches)
 # Historical town observations: no inferred coordinates or absence labels.
 alltown=json.loads((RAW/'town_history.json').read_text());town=[r for r in alltown if r.get('city') in wards]
 table(PROC/'historical_town_records_three_wards.csv',town);dump(PROC/'historical_town_records_three_wards.json',town)
 events=defaultdict(list)
 for r in town:events[(r['city'],r['st'],r['ed'])].append(r)
 erows=[{'ward':k[0],'start_date':k[1],'end_date':k[2],'town_records':len(v),'towns':sorted(set(r['cho'] for r in v)),'reported_causes':sorted(set(r['sge'] for r in v if r.get('sge'))),'reported_building_count_sum':sum(float(r['ht']) for r in v if r.get('ht') is not None),'use':'reported incident counts, not unique buildings or measured depth'} for k,v in sorted(events.items(),key=lambda v:(tuple(int(x) for x in v[0][1].split('/')),v[0][0]))]
 table(OUT/'historical_event_inventory.csv',erows);dump(OUT/'historical_event_inventory.json',erows)
 # Historical spatial archive: documented map period is 1989 onward. Earlier public API features kept separately.
 hj=json.loads((RAW/'historical_inundation_map.geojson').read_text());selected=[];earlier=[];outside=[]
 for f in hj['features']:
  p=f['properties']
  if p.get('munic') not in wards:continue
  g=S.make_valid(project(shape(f['geometry'])));clipped=g.intersection(domain)
  if clipped.is_empty:outside.append(p['id']);continue
  q={**p,'source_geometry_role':'approximate historical incident location/footprint; not surveyed extent','clipped_to_three_wards':True,'observed_depth_m':None}
  rec={'type':'Feature','geometry':mapping(project(clipped,LL)),'properties':q}
  if str(p.get('dis_date_s') or '')>='1989-01-01':selected.append(rec)
  else:earlier.append(rec)
 dump(PROC/'historical_inundation_areas_1989_onward.geojson',{'type':'FeatureCollection','features':selected});dump(PROC/'historical_inundation_areas_before_1989_api_only.geojson',{'type':'FeatureCollection','features':earlier})
 ht=HTMLTables();ht.feed((RAW/'chiyoda_history_page.html').read_text());roadrows=[r for r in ht.rows if len(r)==6 and any(t in r[1] for t in ['昭和','平成'])]
 table(PROC/'chiyoda_road_flood_event_list.csv',[{'source_year':r[0],'source_date':r[1],'weather':r[2],'total_rain_mm_raw':r[3],'hour_rain_mm_raw':r[4],'map_number':r[5],'geometry_role':'range inferred from road ponding locations; not actual surveyed flood footprint'} for r in roadrows])
 historystats={'town_archive_all_tokyo_rows':len(alltown),'town_three_ward_rows':len(town),'town_rows_by_ward':dict(Counter(r['city'] for r in town)),'town_date_range':[min(r['st'] for r in town),max(r['st'] for r in town)],'town_year_range':[min(r['year'] for r in town),max(r['year'] for r in town)],'town_distinct_ward_date_ranges':len(events),'town_coordinates_available':False,'map_api_all_tokyo_features':len(hj['features']),'map_documented_period_three_ward_features':len(selected),'map_features_by_ward':dict(Counter(f['properties']['munic'] for f in selected)),'map_dates':sorted(set(f['properties']['dis_date_s'] for f in selected)),'earlier_public_api_features_saved_separately':len(earlier),'out_of_domain_named_features':outside,'chiyoda_road_event_rows':len(roadrows),'historical_measured_depth_records':0,'important_limits':['Different collection thresholds and update periods; do not compare counts as ward risk rankings.','Absent reports are not negative flood observations.','Town and GIS records overlap; never sum them as independent observations.','Design-maximum maps and actual events have different rainfall and infrastructure states.']}
 dump(OUT/'historical_data_inventory.json',historystats)
 maps(groups,wards,domain,selected,stats)
 validation={'all_four_numeric_sources_acquired':len(fileaudit)==4,'negative_depth_records':sum(s['negative_depth_points'] for s in stats),'all_retained_points_have_ward':all(np.all(g[3]!='') for g in groups.values()),'all_retained_raw_inundation_values_finite':all(np.isfinite(g[0][:,1]).all() for g in groups.values()),'missing_ground_points_preserved':sum(s['missing_ground_points'] for s in stats),'kanda_inundation_value_semantics_unresolved':True,'no_temporal_depth_series_available':True,'no_datum_correction_applied':True,'no_source_data_overwritten':True,'no_new_simulation':True,'historical_depth_not_fabricated':True,'source_hashes':{r['file']:r['source_sha256'] for r in fileaudit}}
 dump(OUT/'validation.json',validation);report(stats,fileaudit,historystats,matches)
 print(json.dumps({'total_three_ward_source_points':sum(len(g[0]) for g in groups.values()),'history':historystats,'validation':validation},ensure_ascii=False),flush=True)

def maps(groups,wards,domain,history,stats):
 water=[project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']]
 def base(ax):
  for name,wg in wards.items():draw(ax,wg,fc='#f5f3ee',ec='#586e77',lw=.65,zorder=0)
  for g in water:draw(ax,g,fc='#c3dce0',ec='none',zorder=3)
  for name,wg in wards.items():
   p=wg.representative_point();ax.text(p.x,p.y,name,ha='center',fontsize=9,fontproperties=FONT,bbox={'fc':'white','ec':'none','alpha':.8},zorder=6)
  xmin,ymin,xmax,ymax=domain.bounds;ax.set_xlim(xmin-250,xmax+250);ax.set_ylim(ymin-250,ymax+250);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.text(xmax-150,ymax-100,'N ↑',ha='right',va='top',fontsize=9)
  ax.plot([xmin+300,xmin+2300],[ymin+300]*2,color='#173446',lw=2);ax.text(xmin+1300,ymin+400,'2 km',ha='center',fontsize=8)
 fig,axes=plt.subplots(2,2,figsize=(12,14))
 for ax,(key,(b,x,y,w,label)) in zip(axes.flat,groups.items()):
  base(ax);d=b[:,1]
  if key=='kanda':
   ax.scatter(x[d==0],y[d==0],s=.6,c='#d1d4d0',marker='s',linewidths=0,zorder=1);ax.scatter(x[d>0],y[d>0],s=.8,c='#374f5c',marker='s',linewidths=0,zorder=2);ax.set_title(label+'｜非零原字段位置（整数1–7）',fontproperties=FONT,fontsize=12);ax.text(.5,.02,'含义待核对；此面板不按米制深度着色',transform=ax.transAxes,ha='center',fontsize=9,fontproperties=FONT);continue
  ax.scatter(x[d==0],y[d==0],s=.6,c='#d1d4d0',marker='s',linewidths=0,zorder=1)
  idx=np.select([d<.1,d<.5,d<1,d<2,d<3],[0,1,2,3,4],default=5)
  for j,col in enumerate(COLORS):
   m=(d>0)&(idx==j);ax.scatter(x[m],y[m],s=.75,c=col,marker='s',linewidths=0,zorder=2)
  ax.set_title(label+'｜模型最大浸水深',fontproperties=FONT,fontsize=12)
 handles=[Patch(fc='#d1d4d0',label='提供数据点：h=0')]+[Patch(fc=c,label=l) for c,l in zip(COLORS,BINLABELS)]+[Patch(fc='#c3dce0',label='水域')]
 fig.suptitle('都心三区｜官方水深字段与数据覆盖',fontproperties=FONT,fontsize=18,y=.985);fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.045),prop=FONT,frameon=False);fig.text(.5,.018,'数值随空间变化；不是水深时间序列。空白为本来源未提供数据点，四来源不合成为同一事件。',ha='center',fontproperties=FONT,fontsize=10);fig.subplots_adjust(top=.95,bottom=.12,hspace=.06,wspace=.08);fig.savefig(OUT/'continuous_depth_four_sources.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,9));base(ax)
 for f in history:draw(ax,project(shape(f['geometry'])),fc='#8675ab',ec='#59487f',alpha=.3,lw=.7,zorder=4)
 ax.set_title('公开历史浸水空间记录｜1989年以后筛选',fontproperties=FONT,fontsize=14);fig.text(.5,.05,'图中是历次事件的大致位置／范围，不是同一场洪水；中央区无该图层记录，不代表未发生积水。',ha='center',fontproperties=FONT,fontsize=9);fig.subplots_adjust(bottom=.1,top=.94);fig.savefig(OUT/'historical_spatial_reference.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4.5));names=list(wards);x=np.arange(3);by={k:np.array([next(s['depth_ge_0_1m_points'] for s in stats if s['source']==k and s['ward']==n) for n in names]) for k in groups if k!='kanda'}
 for j,(k,g) in enumerate([(k,g) for k,g in groups.items() if k!='kanda']):ax.bar(x+(j-1.5)*.19,by[k],.18,label=g[4])
 ax.set_xticks(x,names,fontproperties=FONT);ax.set_ylabel('h ≥ 0.1 m 的模型数据点数',fontproperties=FONT);ax.set_title('分区数据分布｜三个小数水深来源；神田川待核对',fontproperties=FONT);ax.legend(prop=FONT,ncol=2,frameon=False);fig.text(.5,.015,'用于了解数据规模；不能相加为独立网格数，也不是实测受灾面积。',ha='center',fontproperties=FONT,fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(OUT/'depth_point_counts.png',dpi=150);plt.close(fig)

def report(stats,audit,history,matches):
 notes=['本步取得三个来源的小数水深和神田川原字段数值；都是空间数据，不是水深随时间变化的序列。','神田川三区内非零原值只出现整数1–7。是否取整或等级编码尚未确认；暂不作为精确米制水深，导出保留raw_inundation_value，depth_m为空，地图仅显示非零位置。','四来源分别保存，零水深点及地盘高缺失的水深记录保留；空白、零值、未报灾三者不能混为一谈。','原地盘高与0.50m DEM用途不同。隅田川官网明确地盘高使用TP，并为5m地盘高按10m网格平均；其余文件未自带高程基准字段，本步不跨版本校正或重算深度。','隅田川网页说明写2020-03，而PLATEAU来源目录写2021-03-30；版本关系保留待核对。','历史记录可以核查事件、热点和大致范围，尚无实测水深或完整道路关闭时间序列。','町丁记录缺经纬度，保留原地址，不放置虚构点；空间图层有大致范围，精度不能视为实测边界。','町丁、空间图、区级PDF存在重复事件与不同报告门槛；不将它们相加，不按记录数量排名三区风险。','原始CSV尾数不表示测量精度。仅删除同来源坐标和水深/地盘高相同的重复点；同位置冲突保留并计数。']
 lines=['三区官方浸水数值与历史记录数据检查','']+notes+['','数值数据：']
 for a in audit:lines.append(f"{a['source']}: 原始{a['source_rows']}行；三区{a['retained_three_ward_points']}点；同值重复删除{a['identical_point_value_duplicates_removed']}；冲突位置{a['co_located_conflicting_value_groups']}。")
 lines+=['','历史记录：',json.dumps(history,ensure_ascii=False,indent=2),'','PLATEAU等级与原数值对照：',json.dumps(matches,ensure_ascii=False,indent=2),'','建议：先取得匹配实际事件的降雨时序；优先考虑2024年港区事件（资料较新、有区级实绩图），与官方最大规模基线分开使用。历史事件的地形与设施条件需按时期解释。','官方数据目录：https://catalog.data.metro.tokyo.lg.jp/dataset/t000014d0000000029','历史资料说明：https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/kako']
 (OUT/'assessment.txt').write_text('\n'.join(lines)+'\n')
 esc=html.escape
 rows=''.join('<tr>'+''.join(f'<td>{esc('待核对' if s['source']=='kanda' and k in ('depth_ge_0_1m_points','depth_max_m') else '—' if s[k] is None else str(s[k]))}</td>' for k in ['source_name','ward','points','zero_depth_points','depth_ge_0_1m_points','depth_max_m'])+'</tr>' for s in stats)
 hs=''.join(f'<li>{esc(k)}：{v}条町丁记录；{history["map_features_by_ward"].get(k,0)}条空间记录（1989年以后）。</li>' for k,v in history['town_rows_by_ward'].items())
 sources=[('町丁与空间实绩说明','https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/kako'),('千代田区道路冠水履历','https://www.city.chiyoda.lg.jp/koho/kurashi/bosai/bosai-taisaku/fusui-dosha-saigai/rirekizu.html'),('港区水害实绩','https://www.city.minato.tokyo.jp/dobokukeikaku/bosai-anzen/bosai/shinsui/suigaizissekizu.html'),('中央区道路／地下室实绩','https://d3ri4jk2y3zd2r.cloudfront.net/a0011/bousaianzen/bousai/bousaitaisaku/suigaisonae/kouzuihazardmap/kozui02.html')]
 sourcehtml=''.join(f'<li><a href="{u}">{esc(t)}</a></li>' for t,u in sources)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区连续水深与历史记录</title><style>body{background:#f4f2ed;color:#263c43;font:16px/1.7 system-ui;margin:0}main{max-width:1100px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:23px}section{background:#fff;border-radius:12px;padding:22px;margin:20px 0}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#f5f4ef}.scroll{overflow-x:auto}.note{border-left:4px solid #bd7b45;padding:14px;background:#fff6e9}a{color:#22637a}@media(max-width:600px){main{padding:14px}section{padding:14px}h1{font-size:25px}table{min-width:700px}}</style><main><h1>三区连续数值水深与历史积水资料</h1><p class="note">三个来源提供小数水深；神田川三区内只出现整数1–7，含义待核对，暂不按米制水深解释。此处均为空间数据，历史记录另列。</p><section><h2>官方模型数值：四个来源（神田川字段待核对）</h2><img src="continuous_depth_four_sources.png" alt="四来源连续水深地图"><div class="scroll"><table><thead><tr><th>来源</th><th>行政区</th><th>数据点</th><th>零水深点</th><th>≥0.1m 点</th><th>最大水深 m</th></tr></thead><tbody>ROWS</tbody></table></div><p><a href="scenario_depth_summary.csv">数值清单 CSV</a> · <a href="numeric_source_audit.json">原始字段与下载校验</a></p></section><section><h2>分区数据规模</h2><img src="depth_point_counts.png" alt="分区数据规模"></section><section><h2>历史记录：可用于哪些验证</h2><ul>HISTORY</ul><p>町丁记录所含事件为1978—2024年；本次筛选的空间图层记录为1989—2008年。时间范围和报告门槛不同，不能据此判断某区风险更小。三个区的PDF图件另提供道路冠水或地下室浸水参考。</p><img src="historical_spatial_reference.png" alt="历史空间记录"><p><a href="historical_event_inventory.csv">按日期整理的事件清单</a> · <a href="historical_data_inventory.json">历史资料覆盖说明</a></p></section><section><h2>使用说明</h2><ul>NOTES</ul><p><a href="assessment.txt">完整数据检查说明</a> · <a href="plateau_reference_class_comparison.json">与PLATEAU深度等级对照</a></p><h2>官方资料入口</h2><ul>SOURCES</ul><p>来源：東京都建设局、下水道局及三区政府。此页为数据整理，未运行新的洪水模拟。</p></section></main></html>'''
 page=page.replace('ROWS',rows).replace('HISTORY',hs).replace('NOTES',''.join('<li>'+esc(n)+'</li>' for n in notes)).replace('SOURCES',sourcehtml)
 (OUT/'review.html').write_text(page)
if __name__=='__main__':main()
