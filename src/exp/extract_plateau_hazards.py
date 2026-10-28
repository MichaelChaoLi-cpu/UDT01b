"""Read-only inspection of PLATEAU flood/high-tide scenarios in three Tokyo wards.
Run .venv/bin/python src/exp/extract_plateau_hazards.py; --render-only uses saved GeoJSON.
Source 3D water-surface Z is audited, never interpreted as depth or subtracted from new DEM.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for k in ('PROJ_LIB','PROJ_DATA','GDAL_DATA'): os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from collections import Counter,defaultdict
import xml.etree.ElementTree as ET
import io,csv,json,html,sys,time
import numpy as np
import shapely as S
from shapely.geometry import Polygon,MultiPolygon,shape,mapping
from shapely.ops import unary_union
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon,Patch
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/data-briefing/plateau_hazard_overview'
PROC=ROOT/'data/processed/plateau_hazards'
RAW=moga_raw(ROOT, 'plateau/2025')
G='{http://www.opengis.net/gml}';W='{http://www.opengis.net/citygml/waterbody/2.0}';C='{http://www.opengis.net/citygml/2.0}'
XY=Transformer.from_crs(6668,6677,always_xy=True);LL=Transformer.from_crs(6677,6668,always_xy=True)
WARD={'13101-chiyoda-ku':'千代田区','13102-chuo-ku':'中央区','13103-minato-ku':'港区'}
COLORS=['#fae1b0','#eea354','#d66751','#a93e59','#69386e','#332644']
LABELS=['<0.5 m','0.5–3 m','3–5 m','5–10 m','10–20 m','≥20 m']
FONT=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams['font.family']=FONT.get_name()
GENERAL='https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/menu'
QA='https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/shinsuiyosoukuikizuQ_A_R803'
SOURCES={
 '8':{'title':'神田川流域浸水予想区域图','date':'2018-03-30','type':'内水＋外水','url':'https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/menu/menu02-01','rain_l2':'1小时最大153 mm；总雨量690 mm','facilities_epoch':'河川2016-08；下水道2016-09'},
 '10':{'title':'隅田川及び新河岸川流域浸水予想区域图','date':'2021-03-30','type':'内水＋外水','url':GENERAL,'rain_l2':'1小时最大153 mm；总雨量690 mm（官方Q&A）','facilities_epoch':'河川、下水道FY2018'},
 '14':{'title':'城南地区河川流域浸水予想区域图','date':'2018-12-20；2019-06-27局部修正','type':'内水＋外水','url':'https://www.gesui.metro.tokyo.lg.jp/living/amesh/inundation/jonankaisei','rain_l2':'1小时最大153 mm；总雨量690 mm','facilities_epoch':'河川、下水道FY2017'},
 '16':{'title':'江東内部河川流域浸水予想区域图','date':'2020-03-26','type':'内水（该情景无河川溢水）','url':'https://www.gesui.metro.tokyo.lg.jp/living/amesh/inundation/kotonaibukaisei','rain_l2':'1小时最大153 mm；总雨量690 mm','facilities_epoch':'河川、下水道FY2018'},
 '31':{'title':'神田川流域洪水浸水想定区域图','date':'2024-02-15','type':'外水洪水','url':'https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/menu/menu02-01','rain_l2':'1小时最大153 mm；24小时总雨量690 mm','facilities_epoch':'本次未逐项核对'},
 '34':{'title':'城南地区河川洪水浸水想定区域图','date':'2024-02-15','type':'外水洪水','url':GENERAL,'rain_l2':'本次未单独核对原图降雨条件','facilities_epoch':'本次未逐项核对'},
 'htd':{'title':'東京都高潮浸水想定区域图','date':'2024-12-19','type':'高潮','url':'https://www.kensetsu.metro.tokyo.lg.jp/river/suibo/takashiosoutei','rain_l2':'最大规模高潮；需另查台风、潮位、设施失效条件','facilities_epoch':'本次未逐项核对'}
}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def polyonly(g):
 if isinstance(g,(Polygon,MultiPolygon)):return g
 if hasattr(g,'geoms'):return unary_union([polyonly(q) for q in g.geoms])
 return Polygon()
def project(g,t=XY):return S.transform(g,t.transform,interleaved=False)
def fc(p,items):dump(p,{'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(project(g,LL)),'properties':a} for g,a in items]})
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def boundaries():
 return {f['properties']['ward']:project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features']}
def codes(z,n):
 r=ET.fromstring(z.read(n))
 return {a.find(G+'name').text:(a.find(G+'description').text or '') for a in r.iter(G+'Definition')}
class HashReader:
 def __init__(self,f):self.f=f;self.h=sha256()
 def read(self,n=-1):b=self.f.read(n);self.h.update(b);return b

def extract():
 wards=boundaries();domain=unary_union(list(wards.values()));llbounds=project(domain,LL).bounds
 records=[];audit=[];metadata=[];resources=[];codebook={};seenfiles={};seenobjects={};dups=0;sourceobjects=0;zeroarea=0
 for package in sorted(RAW.glob('*/*citygml*.zip')):
  with ZipFile(package) as z:
   for n in z.namelist():
    if n.startswith('metadata/') and (n.endswith('_fld_op.xml') or n.endswith('_htd_op.xml')):
     raw=z.read(n);r=ET.fromstring(raw)
     metadata.append({'package':package.name,'member':n,'sha256':sha256(raw).hexdigest(),'fields':[{'tag':e.tag.split('}')[-1],'text':e.text.strip()} for e in r.iter() if e.tag.split('}')[-1] in ('abstract','statement','value','dateStamp','title','code') and e.text and e.text.strip()]})
    if n.endswith('resource_op.csv'):
     for q in csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))):
      if q.get('feature')=='wtr:WaterBody' and any(t in (q.get('sourceName') or '') for t in ('浸水','洪水','高潮')):
       row={k:(q.get(k) or '').strip() for k in ('sourceName','authority','date','srs','URL')};row['package']=package.name
       if row not in resources:resources.append(row)
   cb={Path(n).stem:codes(z,n) for n in z.namelist() if n.startswith('codelists/') and any(q in n for q in ('RiverFloodingRiskAttribute','HighTideRiskAttribute','DataQualityAttribute_geometrySrcDesc')) and n.endswith('.xml')}
   codebook[package.name]=cb
   names=[n for n in z.namelist() if n.startswith(('udx/fld/','udx/htd/')) and n.endswith('.gml')]
   for ix,n in enumerate(names):
    info=z.getinfo(n);finger=(info.CRC,info.file_size);a={'package':package.name,'member':n,'layer':n.split('/')[1],'size_bytes':info.file_size,'crc32':f'{info.CRC:08x}'}
    if finger in seenfiles:
     with z.open(n) as f:
      h=sha256()
      for b in iter(lambda:f.read(1<<20),b''):h.update(b)
     a['sha256']=h.hexdigest()
     if a['sha256']==seenfiles[finger]['sha256']:
      a['duplicate_of']=seenfiles[finger]['package']+'::'+seenfiles[finger]['member'];audit.append(a);continue
    layer=a['layer'];objectpolys=[];zmin=np.inf;zmax=-np.inf;zcount=0;tri=0;invalid=0;membercount=0;crs=set();loddepth=0;inbody=False
    with z.open(n) as f:
     hf=HashReader(f)
     for event,e in ET.iterparse(hf,events=('start','end')):
      tag=e.tag.split('}')[-1]
      if event=='start':
       if e.get('srsName'):crs.add(e.get('srsName'))
       if e.tag==W+'WaterBody':inbody=True;objectpolys=[];zmin=np.inf;zmax=-np.inf;zcount=0;tri=0;invalid=0
       if tag=='lod1MultiSurface':loddepth+=1
       continue
      if tag=='Polygon' and inbody and loddepth:
       tri+=1;rings=[]
       for ring in e.iter(G+'LinearRing'):
        pos=ring.find(G+'posList')
        if pos is None:raise ValueError('Expected explicit posList')
        v=np.fromstring(pos.text or '',sep=' ')
        if len(v)%3:raise ValueError('Expected 3D coordinates')
        v=v.reshape(-1,3);zmin=min(zmin,float(v[:,2].min()));zmax=max(zmax,float(v[:,2].max()));zcount+=len(v)
        rings.append(v)
       if rings:
        q=rings[0];xmin,ymin,xmax,ymax=llbounds
        if not (q[:,1].max()<xmin or q[:,1].min()>xmax or q[:,0].max()<ymin or q[:,0].min()>ymax):
         xy=[np.column_stack(XY.transform(v[:,1],v[:,0])) for v in rings]
         p=Polygon(xy[0],xy[1:])
         if not p.is_valid:invalid+=1;p=polyonly(S.make_valid(p))
         if not p.is_empty:objectpolys.append(p)
       e.clear()
      elif tag=='lod1MultiSurface':loddepth-=1
      elif e.tag==W+'WaterBody':
       sourceobjects+=1;membercount+=1;sid=e.get(G+'id');name=e.findtext(G+'name');created=e.findtext(C+'creationDate');attrs=[]
       for f0 in e.iter():
        t0=f0.tag.split('}')[-1]
        if len(f0)==0 and f0.text and f0.text.strip() and t0 not in ('posList','name','creationDate'):
         val=f0.text.strip();book=Path(f0.get('codeSpace','')).stem;attrs.append({'field':t0,'value':val,'decoded':cb.get(book,{}).get(val),'codeSpace':f0.get('codeSpace'),'uom':f0.get('uom')})
       values={q['field']:q['value'] for q in attrs};decoded={q['field']:q['decoded'] for q in attrs}
       scale=values.get('scale','NA');desc=values.get('description','NA');scenario=f'{layer}_{desc}_{scale}'
       p0={'source_id':sid,'scenario_id':scenario,'layer':layer,'name':name,'creation_date':created,'description_code':desc,'description_label':decoded.get('description'),'scale_code':scale,'scale_label':decoded.get('scale'),'rank_code':values.get('rank'),'rank_label':decoded.get('rank'),'depth_m':float(values['depth']) if 'depth' in values else None,'duration':values.get('duration'),'source_z_min_m':None if not zcount else zmin,'source_z_max_m':None if not zcount else zmax,'source_vertex_count':zcount,'source_polygon_count':tri,'repaired_source_polygons':invalid,'source_package':package.name,'source_member':n,'raw_attributes':attrs}
       if objectpolys:
        chunks=[unary_union(objectpolys[j:j+4000]) for j in range(0,len(objectpolys),4000)]
        geom=polyonly(unary_union(chunks).intersection(domain))
       else:geom=Polygon()
       if geom.area>0.0001:
        signature=sha256(S.to_wkb(S.normalize(geom))+json.dumps([name,attrs,zmin,zmax],sort_keys=True,ensure_ascii=False).encode()).hexdigest();key=(scenario,sid)
        if key in seenobjects and seenobjects[key]==signature:dups+=1
        else:
         if key in seenobjects:raise ValueError('Same scenario/ID has conflicting geometry or attributes')
         seenobjects[key]=signature;p0['area_m2']=geom.area;p0['wards']=[WARD[k] for k,g in wards.items() if g.intersection(geom).area>.0001];records.append((geom,p0))
       else:zeroarea+=1
       e.clear();inbody=False;objectpolys=[]
      elif tag=='cityObjectMember':e.clear()
     a['sha256']=hf.h.hexdigest()
    if crs!={'http://www.opengis.net/def/crs/EPSG/0/6697'}:raise ValueError(f'Unexpected CRS {crs}')
    a['crs']=sorted(crs);a['source_objects']=membercount;a['retained_so_far']=len(records);seenfiles[finger]=a;audit.append(a)
    if ix%10==0 or layer=='htd':print(package.parent.name,ix+1,len(names),n,'retained',len(records),flush=True)
 dump(OUT/'source_member_audit.json',audit);dump(OUT/'source_metadata.json',metadata);dump(OUT/'source_resource_dates.json',resources);dump(OUT/'codebook.json',codebook)
 fc(PROC/'hazard_source_features.geojson',records)
 dump(OUT/'extraction_audit.json',{'gml_file_occurrences':len(audit),'unique_gml_contents':sum('duplicate_of' not in q for q in audit),'source_objects_in_unique_files':sourceobjects,'duplicate_object_occurrences_removed':dups,'objects_outside_three_wards_or_zero_area':zeroarea,'retained_source_objects':len(records),'clip_crs':'EPSG:6677','geojson_coordinate_order':'longitude, latitude; 2D','source_z_use':'LOD1 inundation surface elevation; not depth and not a new DEM based calculation'})
 return records

def readrecords():return [(polyonly(S.make_valid(project(shape(f['geometry'])))),f['properties']) for f in json.loads((PROC/'hazard_source_features.geojson').read_text())['features']]
def draw(ax,g,**kw):
 if isinstance(g,Polygon):
  # Holes are paths, not filled as risk polygons.
  from matplotlib.path import Path as MPath
  from matplotlib.patches import PathPatch
  q=S.orient_polygons(g);verts=[];codes0=[]
  for ring in [q.exterior,*q.interiors]:
   xy=np.array(ring.coords);verts.extend(xy);codes0.extend([MPath.MOVETO]+[MPath.LINETO]*(len(xy)-2)+[MPath.CLOSEPOLY])
  ax.add_patch(PathPatch(MPath(verts,codes0),**kw))
 elif hasattr(g,'geoms'):
  for q in g.geoms:draw(ax,q,**kw)
def main():
 for d in (OUT,PROC):d.mkdir(parents=True,exist_ok=True)
 records=readrecords() if '--render-only' in sys.argv else extract();wards=boundaries();domain=unary_union(list(wards.values()))
 grouped=defaultdict(list)
 for g,p in records:grouped[p['scenario_id']].append((g,p))
 scenario_rows=[];rank_records=[];ward_rows=[];field_rows=[]
 for sid,rs in sorted(grouped.items()):
  p=rs[0][1];src=SOURCES.get('htd' if p['layer']=='htd' else p['description_code'],{'title':p['name'],'type':'未分类','date':'未核对','url':GENERAL,'rain_l2':'未核对','facilities_epoch':'未核对'})
  ranks={r:S.union_all([S.set_precision(polyonly(S.make_valid(g)),.001) for g,q in rs if q['rank_code']==r],grid_size=.001) for r in sorted({q['rank_code'] for g,q in rs})}
  footprint=unary_union(list(ranks.values()));rank_sum=sum(g.area for g in ranks.values());overlap=rank_sum-footprint.area
  row={'scenario_id':sid,'name':p['name'],'source_title':src['title'],'hazard_type':src['type'],'source_date':src['date'],'scale':p['scale_label'] or ('最大规模高潮' if p['layer']=='htd' else '来源未填scale'),'rain_or_tide_condition':src['rain_l2'] if p['scale_code']!='1' else 'L1计划规模；本次未核对具体雨量','facilities_epoch':src['facilities_epoch'],'source_url':src['url'],'retained_objects':len(rs),'inundation_footprint_km2':round(footprint.area/1e6,6),'rank_overlap_m2':round(overlap,3),'creation_dates':sorted({q['creation_date'] for g,q in rs}),'ranks_present':sorted(ranks),'precise_depth_present':sum(q['depth_m'] is not None for g,q in rs),'duration_present':sum(q['duration'] is not None for g,q in rs),'geometry_z_min_m':min(q['source_z_min_m'] for g,q in rs),'geometry_z_max_m':max(q['source_z_max_m'] for g,q in rs),'role':'existing scenario footprint; not observations or a runnable model'}
  scenario_rows.append(row)
  for r,g in ranks.items():
   rp={'scenario_id':sid,'rank_code':r,'rank_label':next(q['rank_label'] for g0,q in rs if q['rank_code']==r),'area_km2':g.area/1e6};rank_records.append((g,rp))
  for wk,wg in wards.items():
   area=footprint.intersection(wg).area
   ward_rows.append({'scenario_id':sid,'ward':WARD[wk],'ward_boundary_area_km2':round(wg.area/1e6,6),'scenario_footprint_area_km2':round(area/1e6,6),'fraction_of_ward_boundary_percent':round(100*area/wg.area,3),'meaning':'area showing inundation in this scenario, not total model coverage; water areas remain included','rank_area_km2':{r:round(g.intersection(wg).area/1e6,6) for r,g in ranks.items()}})
  for field in ('name','creation_date','rank_code','scale_label','depth_m','duration','source_z_min_m','source_z_max_m'):
   n=sum(q.get(field) is not None for g,q in rs);field_rows.append({'scenario_id':sid,'field':field,'present':n,'missing':len(rs)-n,'objects':len(rs)})
 fc(PROC/'scenario_depth_class_footprints.geojson',rank_records);dump(OUT/'scenario_inventory.json',scenario_rows);csvout(OUT/'scenario_inventory.csv',scenario_rows);dump(OUT/'ward_scenario_areas.json',ward_rows);csvout(OUT/'ward_scenario_areas.csv',ward_rows);csvout(OUT/'attribute_availability.csv',field_rows)
 # Overview context: known water footprints and W05 lines, no inferred drainage connection.
 water=[project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']]
 lines=[project(shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/ksj_tokyo/W05_central_three_wards.geojson').read_text())['features']]
 def panel(ax,sids,title):
  for wk,g in wards.items():draw(ax,g,facecolor='#f6f3ee',edgecolor='#52636a',lw=.6,zorder=0)
  # Single source scenario or clearly labelled cross-source spatial envelope; higher rank is only display priority.
  for r in map(str,range(1,7)):
   gg=[g for g,p in rank_records if p['scenario_id'] in sids and p['rank_code']==r]
   if gg:draw(ax,unary_union(gg).simplify(.5,preserve_topology=True),facecolor=COLORS[int(r)-1],edgecolor='none',zorder=1+int(r)/10)
  for g in water:draw(ax,g,facecolor='#b8d6dc',edgecolor='none',zorder=3)
  for g in lines:
   for q in (g.geoms if hasattr(g,'geoms') else [g]):
    if hasattr(q,'xy'):ax.plot(*q.xy,color='#487c91',lw=.5,zorder=4)
  for wk,g in wards.items():
   p=g.representative_point();ax.text(p.x,p.y,WARD[wk],ha='center',va='center',fontsize=9,fontproperties=FONT,bbox={'fc':'white','ec':'none','alpha':.75,'pad':2},zorder=5)
  # Hibiya is an orientation reference only.
  hp=XY.transform(139.755,35.674);ax.plot(*hp,'o',ms=3,color='#173446',zorder=6);ax.annotate('日比谷',hp,xytext=(5,-12),textcoords='offset points',fontsize=8,fontproperties=FONT,zorder=6)
  xmin,ymin,xmax,ymax=domain.bounds;ax.set_xlim(xmin-250,xmax+250);ax.set_ylim(ymin-250,ymax+250);ax.set_aspect('equal');ax.set_title(title,fontproperties=FONT,fontsize=12,pad=10);ax.set_xticks([]);ax.set_yticks([])
  ax.plot([xmin+300,xmin+2300],[ymin+350]*2,color='#173446',lw=2);ax.text(xmin+1300,ymin+450,'2 km',ha='center',fontsize=8)
  ax.text(xmax-200,ymax-200,'N ↑',ha='right',va='top',fontsize=10)
 panels=[([r['scenario_id'] for r in scenario_rows if r['hazard_type'].startswith('内水')],'暴雨浸水予想：四来源范围叠加'),([r['scenario_id'] for r in scenario_rows if r['hazard_type']=='外水洪水' and 'L2' in r['scale']],'河川洪水：L2来源范围叠加'),([r['scenario_id'] for r in scenario_rows if r['hazard_type']=='高潮'],'高潮：2024-12-19 来源')]
 fig,axes=plt.subplots(1,3,figsize=(15,7.5))
 for ax,(ids,title) in zip(axes,panels):panel(ax,ids,title)
 fig.suptitle('都心三区｜PLATEAU 已有浸水风险数据',fontproperties=FONT,fontsize=18,y=.98)
 handles=[Patch(fc=c,label=l) for c,l in zip(COLORS,LABELS)]+[Patch(fc='#b8d6dc',label='水域 / 河川线')]
 fig.legend(handles=handles,ncol=7,loc='lower center',bbox_to_anchor=(.5,.07),frameon=False,prop=FONT)
 fig.text(.5,.035,'空白 = 本批图层未显示浸水；不等于无风险。多来源叠加只作空间概览，不是联合事件模拟。',ha='center',fontproperties=FONT,fontsize=10)
 fig.subplots_adjust(left=.025,right=.985,bottom=.16,top=.88,wspace=.08);fig.savefig(OUT/'three_ward_hazard_overview.png',dpi=170);plt.close(fig)
 for row in scenario_rows:
  fig,ax=plt.subplots(figsize=(8,8));panel(ax,[row['scenario_id']],row['name']);fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,prop=FONT);fig.subplots_adjust(bottom=.14,top=.92,left=.04,right=.97);fig.savefig(OUT/(row['scenario_id']+'.png'),dpi=140);plt.close(fig)
 validate={'all_geometries_valid':all(g.is_valid for g,p in records),'all_outputs_within_domain':all(g.difference(domain.buffer(.001)).area<.01 for g,p in records),'unique_scenario_source_ids':len({(p['scenario_id'],p['source_id']) for g,p in records})==len(records),'rank_labels_decoded':all(p['rank_label'] is not None for g,p in records),'scenario_count':len(scenario_rows),'retained_objects':len(records),'ifld_files_found':0,'source_data_modified':False,'new_hydraulic_simulation_run':False,'rank_union_numerical_precision_m':.001,'rank_overlap_m2_by_scenario':{r['scenario_id']:r['rank_overlap_m2'] for r in scenario_rows}}
 dump(OUT/'validation.json',validate)
 report(scenario_rows,ward_rows,field_rows)
 print(json.dumps({'scenarios':len(scenario_rows),'objects':len(records),'output':str(OUT),'validation':validate},ensure_ascii=False),flush=True)

def report(rows,wardrows,fields):
 esc=html.escape
 notes=['数据用途：三区已有风险情景的空间底图与后续模型对照；不是本次暴雨的预测。','更正：未发现独立 ifld，不等于没有内水资料；fld 中包含四类浸水予想区域图。一般包括内水与外水；江東内部河川情景因假定未发生河川溢水，显示的是下水道内水浸水（官方Q&A第2页）。','各来源、L1/L2及高潮分别保存；多来源面积不相加，不解释为同一事件。','浸水予想区域图采用10m网格；地盘高主要来自GSI航空激光5m DEM，原图一般显示≥0.1m浸水（官方Q&A）。网格大小不是高程精度。','LOD1 的 Z 是原浸水深＋原地盘高形成的浸水位面，不是浸水深。包内元数据警告：叠加不同地盘高可能造成显示深度不一致。','优先用来源解码的浸水深等级；没有精确 depth 时，不取等级中值，也不与新0.50 m DEM相减重建深度。','原图日期、设施年次、CityGML创建日期分开看；2026-03-13 创建不表示2026年设施或实测。','提取范围以三区包内区界裁剪；水域保留在面积分母/分子中。空白不证明未建模或安全；本步没有完整原计算域边界。','本批风险地物未提供可运行的模型、完整降雨时序、排水管网参数或事件流速时序。']
 text=['都心三区 PLATEAU 风险数据初览','']+notes+['','情景清单：']
 for r in rows:
  text.append(f"{r['scenario_id']} | {r['name']} | {r['hazard_type']} | 来源 {r['source_date']} | {r['scale']} | {r['rain_or_tide_condition']} | 三区内显示浸水面积 {r['inundation_footprint_km2']:.3f} km² | 对象 {r['retained_objects']} | 精确depth {r['precise_depth_present']} | duration {r['duration_present']}")
 text+=['','官方条件核对：',GENERAL,QA,'','文件说明：scenario_inventory.csv 情景及来源；ward_scenario_areas.csv 分区范围；attribute_availability.csv 属性缺失；hazard_source_features.geojson 保留来源对象；scenario_depth_class_footprints.geojson 同情景同等级合并。','方法：EPSG:6697中纬度、经度、Z，二维几何投影到EPSG:6677裁剪/求面积；输出GeoJSON经度、纬度。相同ZIP成员先CRC/大小筛选再SHA256确认去重；同情景同ID仅在裁剪几何及属性一致时去重。来源对象几何无简化保存；同情景同等级合并使用1mm数值精度网格，防止坐标往返引起的几何计算问题；图片只为显示简化0.5m。','下一步：先讨论哪些已有情景最接近研究目标；需要连续浸水深时，再取官方原始浸水深/地盘高CSV，不从可视化Z反推。']
 (OUT/'assessment.txt').write_text('\n'.join(text)+'\n')
 options=''.join(f'<option value="{r["scenario_id"]}">{esc(r["name"])} · {esc(r["hazard_type"])}</option>' for r in rows)
 trs=''.join('<tr>'+''.join(f'<td>{esc(str(r[k]))}</td>' for k in ('name','hazard_type','source_date','scale','rain_or_tide_condition','inundation_footprint_km2','retained_objects'))+'</tr>' for r in rows)
 wtrs=''.join('<tr>'+''.join(f'<td>{esc(str(r[k]))}</td>' for k in ('scenario_id','ward','scenario_footprint_area_km2','fraction_of_ward_boundary_percent'))+'</tr>' for r in wardrows)
 data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>都心三区浸水风险数据初览</title><style>body{margin:0;background:#f4f2ed;color:#24363c;font:16px/1.7 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.3}h2{font-size:22px}img{width:100%;height:auto;background:white;border-radius:12px}section{background:#fff;padding:22px;margin:20px 0;border-radius:12px}.note{border-left:4px solid #d8814c;padding:12px;background:#fff5e7}select{max-width:100%;width:100%;padding:12px;font:inherit}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ddd;padding:10px;text-align:left;vertical-align:top;overflow-wrap:anywhere}.tablewrap{overflow-x:auto}th{background:#f5f3ee}a{color:#22637a}.muted{color:#64777f}#detail{white-space:pre-line}#scenarioImage{max-width:760px;display:block;margin:auto}@media(max-width:600px){main{padding:14px}section{padding:14px}h1{font-size:25px}table{min-width:780px}}</style><main><p class="muted">数据初览 · 都心三区 · 保留各来源情景</p><h1>东京已有浸水风险数据，先看整体格局</h1><p class="note">PLATEAU 提供的是已有情景结果。四类“浸水予想区域图”已有内水信息；其中江東内部河川情景显示下水道内水，其余考虑内水与外水。河川洪水与高潮另列。图中空白不等于无风险。</p><img src="three_ward_hazard_overview.png" alt="三区风险空间概览"><section><h2>逐个来源查看</h2><label for="scenario">选择情景</label><select id="scenario">OPTIONS</select><p id="detail"></p><img id="scenarioImage" alt="选定来源情景地图"><p id="sourceLink"></p></section><section><h2>情景与来源</h2><p>面积是三区区界内本情景显示的浸水范围，包含水域；各行不能相加。原图日期不等于设施年次。</p><div class="tablewrap"><table><thead><tr><th>名称</th><th>类型</th><th>原图日期</th><th>规模</th><th>降雨/潮位条件</th><th>面积 km²</th><th>来源对象数</th></tr></thead><tbody>ROWS</tbody></table></div></section><section><h2>分区范围</h2><p>百分比的分母是包内整个行政区界面积，包含水域；它不是人口或建筑受灾比例，也不是完整计算域的覆盖率。</p><div class="tablewrap"><table><thead><tr><th>情景</th><th>行政区</th><th>显示浸水面积 km²</th><th>占区界面积 %</th></tr></thead><tbody>WARDROWS</tbody></table></div></section><section><h2>这些数据能回答什么</h2><p>能查看既有情景下风险分布、浸水深等级及其与水体的位置关系；可作为三区研究的参照底图。尚不能从这批文件重新计算一场指定暴雨的积水过程。</p><ul>NOTES</ul><p><a href="scenario_inventory.csv">情景清单 CSV</a> · <a href="ward_scenario_areas.csv">分区面积 CSV</a> · <a href="attribute_availability.csv">属性可用性 CSV</a> · <a href="assessment.txt">完整说明</a></p><p><a href="https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/shinsuiyosoukuikizuQ_A_R803">官方浸水予想图Q&A（2026-03）</a></p><p>来源：国土交通省 PLATEAU／東京都。原数据 CC BY 4.0；本页为研究数据整理。</p></section></main><script>const rows=DATA;const sel=document.querySelector('#scenario');function update(){const r=rows.find(x=>x.scenario_id===sel.value);document.querySelector('#scenarioImage').src=r.scenario_id+'.png';document.querySelector('#detail').textContent=r.hazard_type+'｜原图 '+r.source_date+'｜'+r.scale+'\\n'+r.rain_or_tide_condition+'\\n设施资料：'+r.facilities_epoch+'；CityGML创建：'+r.creation_dates.join(', ')+'\\n精确depth属性：'+r.precise_depth_present+'/'+r.retained_objects+'；浸水持续时间属性：'+r.duration_present+'/'+r.retained_objects;const a=document.createElement('a');a.href=r.source_url;a.textContent='查看官方来源说明';a.target='_blank';a.rel='noopener';document.querySelector('#sourceLink').replaceChildren(a)}sel.addEventListener('change',update);update();</script></html>'''
 page=page.replace('OPTIONS',options).replace('WARDROWS',wtrs).replace('ROWS',trs).replace('NOTES',''.join('<li>'+esc(n)+'</li>' for n in notes[2:])).replace('DATA',data)
 (OUT/'review.html').write_text(page)
if __name__=='__main__':main()
