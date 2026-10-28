"""Archive and assess official rainfall for 2024-08-21; no rainfall-field interpolation.
Source pages are cached without overwriting. Run from the project environment.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,timedelta,timezone
import csv,json,re,hashlib,shutil,subprocess,os
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='exp' else Path('/Volumes/Extension/Research/UDT01b')
RAW=moga_raw(ROOT, 'tokyo_rain_events/2024-08-21')
PRO=ROOT/'data/processed/tokyo_rain_events/2024-08-21'
OUT=ROOT/'data/exp/data-briefing/tokyo_rain_event_20240821'
OLD=moga_raw(ROOT, 'tokyo_inundation_reference/2026-10-13')
JMA='https://www.data.jma.go.jp/'
STATIONS=[('tokyo','東京（北の丸公園）','s','47662'),('setagaya','世田谷（気象庁）','a','0368'),('edogawa','江戸川臨海','a','0370'),('nerima','練馬（気象庁）','a','1002')]
class Table(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None;self.inside=False
 def handle_starttag(self,t,a):
  if t=='table' and dict(a).get('id')=='tablefix1':self.inside=True
  if self.inside and t=='tr':self.row=[]
  if self.inside and t in ('td','th'):self.cell=''
 def handle_data(self,d):
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if self.inside and t in ('td','th') and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
  if self.inside and t=='tr' and self.row is not None:self.rows.append(self.row);self.row=None
  if self.inside and t=='table':self.inside=False

def table(p):
 t=Table();t.feed(p.read_text());return t.rows

def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def writejson(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def value(s):
 if s=='--':return 0.,'no_precipitation'
 if s=='0.0':return 0.,'below_0.5mm_resolution'
 if re.fullmatch(r'\d+(\.\d+)?',s):return float(s),'normal'
 return None,'missing_or_flagged'

def main():
 for p in [RAW,PRO,OUT]:p.mkdir(parents=True,exist_ok=True)
 entries=[]
 for name,label,typ,block in STATIONS:
  for period in ['10min','hourly','daily']:
   entries.append((f'jma_{name}_{period}.html',f'{JMA}stats/etrn/view/{period}_{typ}1.php?prec_no=44&block_no={block}&year=2024&month=8&day=21&view='))
 entries += [('tokyo_event_report.pdf','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/000070667'),('jma_symbols.html',JMA+'stats/data/mdrr/man/remark.html'),('jma_sampling.html',JMA+'stats/data/mdrr/man/sampling.html'),('jma_time.html',JMA+'stats/data/mdrr/man/nikkai.html'),('jma_rain_intervals.html','https://www.jma.go.jp/jma/kishou/know/faq/faq1.html'),('minato_rain_portal.html','https://navi.city.minato.tokyo.jp/suii_uryo/list.html'),('jma_tokyo_relocation.html','https://www.jma.go.jp/jma/press/1410/03b/20141003_tokyo_rojo.html'),('jma_station_selector.html',JMA+'stats/etrn/select/prefecture.php?prec_no=44&block_no=&year=2024&month=8&day=21&view='),('tokyo_station_coordinates.csv','https://www.kasen-suibo.metro.tokyo.lg.jp/zahyo/zahyo.csv'),('tokyo_download_page.html','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/tsim0110g.html'),('nied_event_redirect.html','https://mizu.bosai.go.jp/key/20240821Tokyo')]
 redirect=Path('/private/tmp/nied_event.raw').read_text();nied_url=re.search(r'URL=([^"\s]+)',redirect)[1]
 entries.append(('nied_event_article.html',nied_url))
 caches={'jma_station_selector.html':'jma_station_selector.raw','tokyo_station_coordinates.csv':'tokyo_station_coordinates.raw','tokyo_download_page.html':'tokyo_history.html','nied_event_redirect.html':'nied_event.raw','nied_event_article.html':'nied_event_article.raw'}
 def fetch(ent):
  n,u=ent;p=RAW/n;tmp=Path('/private/tmp')/caches.get(n,n)
  if not p.exists():
   if tmp.exists():shutil.copyfile(tmp,p)
   else:
    r=requests.get(u,timeout=(15,40));r.raise_for_status();p.write_bytes(r.content)
  return {'filename':n,'url':u,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'retrieved_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}
 with ThreadPoolExecutor(max_workers=3) as pool:manifest=list(pool.map(fetch,entries))
 # A single documented-format historical URL was checked; the server returned 404.
 check=Path('/private/tmp/tokyo_archive_event_check.raw')
 if not (RAW/'tokyo_expired_event_404.html').exists():shutil.copyfile(check,RAW/'tokyo_expired_event_404.html')
 manifest.append({'filename':'tokyo_expired_event_404.html','url':'https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/Tokyo_Uryo10min_20240821.csv','http_status':404,'usable_data':False,'sha256':hashlib.sha256(check.read_bytes()).hexdigest(),'retrieved_utc':datetime.fromtimestamp(check.stat().st_mtime,timezone.utc).isoformat()})
 writejson(OUT/'acquisition_manifest.json',{'sources':manifest})
 ten=[];hours=[];summ=[];tests=[];jst=timezone(timedelta(hours=9));base=datetime(2024,8,21,tzinfo=jst)
 selector=(RAW/'jma_station_selector.html').read_text();coords={}
 for area in re.findall(r'<area[^>]+>',selector):
  if 'viewPoint(' not in area:continue
  args=re.findall(r"'([^']*)'",area.split('viewPoint(',1)[1].split(');',1)[0]);coords[args[1]]=(float(args[4])+float(args[5])/60,float(args[6])+float(args[7])/60)
 for name,label,typ,block in STATIONS:
  part=[]
  for r in table(RAW/f'jma_{name}_10min.html'):
   if not re.fullmatch(r'\d\d:\d\d',r[0]):continue
   h,m=map(int,r[0].split(':'));end=base+timedelta(hours=h,minutes=m);amount,q=value(r[3 if typ=='s' else 1])
   row={'station_id':block,'agency':'JMA','station_name':label,'interval_start_jst':(end-timedelta(minutes=10)).isoformat(),'interval_end_jst':end.isoformat(),'precip_mm':amount,'intensity_mm_per_hour':None if amount is None else amount*6,'raw_precip':r[3 if typ=='s' else 1],'quality':q};part.append(row);ten.append(row)
  assert len(part)==144 and len(set(x['interval_end_jst'] for x in part))==144
  vals=np.array([x['precip_mm'] if x['precip_mm'] is not None else np.nan for x in part]);assert np.isfinite(vals).all()
  hp=[]
  for r in table(RAW/f'jma_{name}_hourly.html'):
   if not re.fullmatch(r'\d{1,2}',r[0]):continue
   h=int(r[0]);v,q=value(r[3 if typ=='s' else 1]);hp.append(v);hours.append({'station_id':block,'station_name':label,'interval_end_jst':(base+timedelta(hours=h)).isoformat(),'precip_mm':v,'raw_precip':r[3 if typ=='s' else 1],'quality':q})
  d=next(r for r in table(RAW/f'jma_{name}_daily.html') if r[0]=='21');di=3 if typ=='s' else 1
  total,m60,m10=[value(d[i])[0] for i in range(di,di+3)]
  rolling=np.convolve(vals,np.ones(6),'valid');imax=int(np.argmax(rolling))+5
  tests.append({'station':label,'bins':len(part),'sum_10min_matches_daily':float(vals.sum())==total,'sum_hourly_matches_daily':sum(hp)==total,'every_hour_matches_six_10min':bool(np.allclose(vals.reshape(24,6).sum(axis=1),hp))})
  assert tests[-1]['sum_10min_matches_daily'] and tests[-1]['every_hour_matches_six_10min']
  lat,lon=coords[block]
  summ.append({'station_id':block,'agency':'JMA','station_name':label,'lat':lat,'lon':lon,'coordinate_note':'Official selector coordinates; rounded minutes, not surveyed precision; 2024 history not separately resolved except Tokyo relocation.','daily_total_mm':total,'official_daily_max60_mm':m60,'official_daily_max10_mm':m10,'aligned_10min_max_mm':float(vals.max()),'aligned_rolling60_max_mm':float(rolling.max()),'aligned_rolling60_peak_end_jst':part[imax]['interval_end_jst'] if rolling.max()>0 else None,'source_url':dict(entries)[f'jma_{name}_daily.html']})
 writecsv(PRO/'jma_rain_10min.csv',ten);writecsv(PRO/'jma_rain_hourly.csv',hours);writecsv(PRO/'jma_station_event_summary.csv',summ)
 # Original annual report, preserved; parse the three event-specific tables only.
 txt=OUT/'annual_2024_text.txt'
 subprocess.run(['/Users/lichao/opt/anaconda3/bin/pdftotext','-layout',str(OLD/'tokyo_flood_record_2024.pdf'),str(txt)],check=True)
 pages=txt.read_text().split('\f');event_pages=[p for p in pages if '災害発生年月日' in p and '8 月 21 日' in p and '雨量記録表' in p and '総雨量' in p];assert len(event_pages)==3
 coordrows=list(csv.reader((RAW/'tokyo_station_coordinates.csv').read_bytes().decode('cp932').splitlines()));tokyo_coords={r[0]:(int(r[2])+int(r[3])/60+float(r[4])/3600,int(r[5])+int(r[6])/60+float(r[7])/3600) for r in coordrows[1:] if r[1]=='雨量'}
 annual=[]
 for page in event_pages:
  printed=re.findall(r'-\s*(\d+)\s*-',page)[-1]
  for line in page.splitlines():
   m=re.match(r'^\s*(\d{1,3})\s+(\S+)\s+(.*)$',line)
   if not m:continue
   idx,n,rest=m.groups();parts=rest.split()
   if not parts or not re.fullmatch(r'\(?\d+\)?|-',parts[0]):continue
   row={'station_number':int(idx),'agency':'Tokyo flood-record gauge network','station_name':n,'event_start_jst':'2024-08-21T16:08:00+09:00','event_end_jst':'2024-08-21T21:34:00+09:00','event_total_mm':None,'daily_total_mm':None,'max60_mm':None,'max60_start_jst':None,'max60_end_jst':None,'max10_mm':None,'max10_start_jst':None,'max10_end_jst':None,'quality':'normal','lat':None,'lon':None,'coordinate_note':'Present-day Tokyo gauge catalog, name match only; event-era coordinates require verification.','report_printed_page':printed,'raw_table_line':line.strip()}
   def number(s):return float(s.strip('()')) if re.fullmatch(r'\(?\d+\)?',s) else None
   row['event_total_mm']=number(parts[0]);row['daily_total_mm']=number(parts[1])
   if parts[0]=='-':row['quality']='missing'
   elif '(' in rest:row['quality']='incomplete_reference'
   if len(parts)==4:row['max60_mm']=number(parts[2]);row['max10_mm']=number(parts[3])
   elif len(parts)==5 and row['quality']!='missing':
    row['max60_mm']=number(parts[3]);row['max10_mm']=number(parts[4])
   elif len(parts)==9:
    row['max60_mm']=number(parts[3]);row['max10_mm']=number(parts[6])
    for k,span in [('max60',parts[5]),('max10',parts[8])]:
     a,b=span.split('～');row[k+'_start_jst']='2024-08-21T'+a+':00+09:00';row[k+'_end_jst']='2024-08-21T'+b+':00+09:00'
   elif row['quality']!='missing':raise ValueError((n,parts))
   if n in tokyo_coords:row['lat'],row['lon']=tokyo_coords[n]
   annual.append(row)
 assert len(annual)==149 and len(set(r['station_number'] for r in annual))==149
 writecsv(PRO/'tokyo_gauge_event_statistics.csv',annual)
 towns=list(csv.DictReader((ROOT/'data/processed/tokyo_inundation_reference/historical_town_records_three_wards.csv').open(encoding='utf-8-sig')));matched=[r for r in towns if r['st']=='2024/8/21' and r['ed']=='2024/8/21'];assert len(matched)==10
 for r in matched:r.update({'rainfall_event_id':'2024-08-21','join_rule':'same event date only; no station/point/depth assignment'})
 writecsv(PRO/'matched_town_records.csv',matched)
 context={'event_id':'2024-08-21','assessment_date':'2026-10-14','event_timezone':'Asia/Tokyo','jma_summary':summ,'tokyo_gauges':{'statistics_rows':len(annual),'current_name_coordinate_matches':sum(r['lat'] is not None for r in annual),'time_series_available':False,'statistics_basis':'One-minute observations; arbitrary rolling 60/10-minute maxima, not fixed hourly bins.'},'minato_pdf':{'event_total_mm':127.5,'max60_mm':110.5,'rain_gauge_identity':None,'time_series_available':False,'underground_buildings':3,'road_ponding_reports':6,'above_floor_households':1,'below_floor_households':1,'source_url':'https://www.city.minato.tokyo.jp/documents/109794/20240821.pdf'},'river_response':{'river':'古川','gauge':'四ノ橋','reported_stage_rise_m_approx':4.3,'photo_time_jst':'2024-08-21T19:15:00+09:00','chart_datum':'天端下がり (distance below bank/crest), not DEM height or A.P./T.P. stage; cross-section reference survey required.','reservoir':'古川地下調節池','capacity_m3':135000,'stored_m3':42200,'reported_storage_percent':31,'inflow_start_jst':'2024-08-21T18:50:00+09:00','inflow_end_jst':'2024-08-21T19:34:00+09:00','source_url':dict(entries)['tokyo_event_report.pdf'],'usable_boundary_hydrograph':False},'matched_impact_records':{'town_rows':10,'reported_buildings':sum(int(r['ht']) for r in matched),'ward':'港区','other_two_wards':'No same-date town rows in this dataset; not evidence of zero flooding.','measured_surface_depth':False,'precise_surface_coordinates':False,'observed_peak_times':False},'availability':{'jma_10min_rows':len(ten),'tokyo_public_calendar':'2025-11 through 2026-10 at inspection','tokyo_event_csv_status':404,'minato_portal':'Public recent three-hour display; 2024 event series not found.','radar':'NIED XRAIN event analysis located; no numeric rainfall grid acquired; public video is qualitative context.'}}
 writejson(OUT/'event_assessment.json',context)
 writejson(OUT/'validation.json',{'checks':tests,'tokyo_statistics_rows':149,'matched_town_rows':10,'raw_originals_preserved':True,'rainfall_interpolation_performed':False,'flood_simulation_performed':False})
 plot(ten,summ,annual)
 deliver(context,annual)
 print(json.dumps({'ten_min_rows':len(ten),'hourly_rows':len(hours),'gauge_statistics':len(annual),'coordinate_matches':sum(r['lat'] is not None for r in annual),'jma':[(s['station_name'],s['daily_total_mm'],s['official_daily_max60_mm']) for s in summ],'output':str(OUT)},ensure_ascii=False),flush=True)

def plot(ten,summ,annual):
 for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
 os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 from matplotlib import rcParams
 from shapely.geometry import shape
 from pyproj import Transformer
 fp=FontProperties(fname='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc');rcParams['font.family']=fp.get_name();rcParams['axes.unicode_minus']=False
 fig,ax=plt.subplots(2,1,figsize=(11,7.8),sharex=True,gridspec_kw={'height_ratios':[1,1]});colors=['#2166ac','#d95f02','#109a82','#8b67b1']
 for s,c in zip(summ,colors):
  rows=[r for r in ten if r['station_id']==s['station_id']];hours=np.array([(datetime.fromisoformat(r['interval_end_jst'])-datetime(2024,8,21,tzinfo=timezone(timedelta(hours=9)))).total_seconds()/3600 for r in rows]);v=np.array([r['precip_mm'] for r in rows]);ax[0].step(hours,v,where='pre',color=c,label=s['station_name']);ax[1].plot(hours,np.cumsum(v),color=c,label=s['station_name'])
 ax[0].set_ylabel('10分間雨量 (mm)');ax[1].set_ylabel('日累積雨量 (mm)');ax[1].set_xlabel('2024-08-21 日本標準時／値は区間の終了時刻');ax[1].set_xlim(16,22);ax[1].set_xticks(range(16,23),[f'{x}:00' for x in range(16,23)]);ax[0].legend(loc='upper right',fontsize=10)
 for a in ax:a.grid(alpha=.2);a.set_ylim(bottom=0)
 fig.suptitle('公式観測の降雨時系列：JMA周辺4地点\n港区の暴雨核心を表す時系列は未取得',fontsize=14);fig.tight_layout();fig.savefig(OUT/'jma_rainfall_timeseries.png',dpi=180);plt.close(fig)
 trans=Transformer.from_crs(6668,6677,always_xy=True);bounds=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text());fig,ax=plt.subplots(figsize=(10,8.5));wn=['千代田区','中央区','港区']
 for f,label in zip(bounds['features'],wn):
  g=shape(f['geometry']);geoms=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
  for p in geoms:
   xy=np.asarray(p.exterior.coords);x,y=trans.transform(xy[:,0],xy[:,1]);ax.fill(np.asarray(x)/1000,np.asarray(y)/1000,color='#eef1f4',ec='#6b7788',lw=.8,zorder=1)
  p=g.representative_point();x,y=trans.transform(p.x,p.y);ax.annotate(label,(x/1000,y/1000),xytext=(-14,15),textcoords='offset points',color='#6b7788',fontsize=10,zorder=2)
 vals=[r for r in annual if r['lat'] is not None and 139.59<r['lon']<139.9 and 35.59<r['lat']<35.77 and r['max60_mm'] is not None];x,y=trans.transform([r['lon'] for r in vals],[r['lat'] for r in vals]);sc=ax.scatter(np.array(x)/1000,np.array(y)/1000,c=[r['max60_mm'] for r in vals],cmap='YlOrRd',vmin=0,vmax=110,s=54,edgecolors='#595959',lw=.6,zorder=3)
 for r in vals:
  if r['station_name'] in ['中央','高浜','品川','渋谷橋','新宿','亀島川']:
   x,y=trans.transform(r['lon'],r['lat']);ax.annotate(f"{r['station_name']} {r['max60_mm']:g}",(x/1000,y/1000),xytext={'中央':(-72,-30),'亀島川':(14,20),'渋谷橋':(-85,8)}.get(r['station_name'],(6,8)),textcoords='offset points',fontsize=10,zorder=4,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1),arrowprops=dict(arrowstyle='-',color='#6b7788',lw=.6))
 for s in summ:
  x,y=trans.transform(s['lon'],s['lat']);ax.scatter(x/1000,y/1000,marker='^',s=75,facecolor='#2477ad',edgecolor='white',zorder=5);ax.annotate(f"JMA {s['station_name'].split('（')[0]} {s['official_daily_max60_mm']:g}",(x/1000,y/1000),xytext=(-110,-10) if s['station_id']=='47662' else (6,-16),textcoords='offset points',fontsize=9,color='#12537d',zorder=6)
 fig.colorbar(sc,ax=ax,shrink=.75,label='東京都年報：任意60分最大雨量 (mm)');ax.set_aspect('equal');ax.set_xlabel('平面直角座標 IX 東西 (km)');ax.set_ylabel('南北 (km)');ax.grid(alpha=.12);ax.set_title('2024-08-21 三区と周辺の雨量差\n観測点の値のみ。面補間・三区一様化はしていない',fontsize=14);fig.text(.1,.018,'観測点位置：公式の現行一覧。測量精度・2024年当時の位置は別途確認。青△はJMA。',fontsize=9);fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(OUT/'rain_gauge_spatial_comparison.png',dpi=180);plt.close(fig)

def deliver(context,annual):
 text='''2024-08-21 真实暴雨数据匹配（三区尺度）
已归档：JMA四站全天10分钟/小时/日值；東京都149站事件统计；水防本部报告；当前雨量站目录；NIED XRAIN事件分析页面。
JMA全天10分钟576条、小时96条。已逐小时验证六个10分钟雨量之和，并与官方日总量核对一致。
时间为JST，10分钟雨量代表截至所列时刻之前10分钟；24:00写为次日00:00。强度=10分钟雨量×6，仅为区间平均强度。
-- 为无降水，0.0 为小于0.5mm，二者输出数值0但保留原值与质量标记；缺测符号不会自动补零。
正时小时量、固定10分钟窗口与任意连续60/10分钟最大量分别保存。东京固定10分钟最大6mm，而官方日最大10分钟10.5mm；时序粒度不可恢复任意起止峰值。

判断：选作港区/古川事件的优先验证候选。尚不能据此开展三区定量验证。
中央站总量11mm/最大60分钟11mm，高浜90/75，渋谷橋44/41，东京JMA21/20.5。
港区实绩图127.5/110.5的具体雨量站及统计窗口未注明，不赋予坐标，不把峰值拼成时序。
東京都年报总雨量窗口为16:08–21:34；日量为0–24时，峰值来自1分钟记录，保留实际起止时刻。已提取149站统计，站号属于年报，不能当门户实时编号。
现行东京站目录按名称匹配位置，2024年的站点迁移/测器变化尚需核对。JMA东京已核实2014年迁至北之丸公园；其他坐标来自官方选择页，分钟取整精度有限。
NIED确认港区18–19时附近约100mm，找到XRAIN事件分析；未获得数值网格，未从视频反推定量降雨场。

河道边界不能固定为自由排水：古川四ノ橋约19:15水位较平常上涨约4.3m；图示为天端下がり，不可直接作为DEM/T.P.水位。
古川地下调节池18:50–19:34流入，蓄水42200m³（容量135000，官方31%）；这是总蓄水量，不是泵站流量/逐时出流或地表排水能力。
年报另有A.P.最高水位表，仅列岸顶下1m以内站点；这不是所有河道水位数据。不可给四ノ橋套用该表基准或一律减某常数。

实绩匹配：按同一日期关联港区10条町丁目记录，共15栋（源ht字段），标为内水。保留原町名，没有伪造点位置/深度。
港区实绩图另记道路积水6件、地下3栋、床上/床下各1户；与町丁目表不同统计口径，不能合并计数。千代田/中央无该日期町记录，不等于没有积水。
水防本部报告“道路/河川/公园无损害”也不是港区道路积水为零的证明，机构范围与损害定义不同。

下一步优先顺序：
1. 获取这场事件17–21:30的港区九站及東京都中央、高浜、渋谷橋、新宿等原始1/10分钟雨量，明确127.5/110.5对应站；或取得同期XRAIN数值网格并用地面站校验。
2. 取得古川/渋谷川事件水位过程与水尺零点、岸顶标高换算；海侧至少有同期潮位边界。水位基准和DEM对接前不设全水域自由出口。
3. 并行搭建覆盖三区及跨区上游汇水范围的地表模型：先做统一地形/建筑阻水/河道海侧边界与分区雨强试算，输出积水位置与相对敏感性；这不是2024实况复现。排水作为分区损失/容量情景，不以完整管网为启动门槛。
4. 真实雨量场与水位边界补齐后回放2024-08-21，先核对港区报告积水街区是否覆盖；没有现场水深/峰值时刻就不声称深度或到峰时间准确。
公开东京历史CSV页目前仅2025-11–2026-10；按已公布格式检查2024-08-21链接返回404，未绕过限制。港区门户仅近期三小时，未找到2024下载接口。需使用官方档案申请/开放档案，尚未发送申请。
'''
 (OUT/'assessment.txt').write_text(text)
 readme='Rainfall amount: mm per preceding 10min; timezone JST. Interval-average intensity mm/h is amount*6. Preserve raw markers and quality. Never replace missing with zero. Station statistics are point observations, not areal rainfall. Annual rolling peaks use minute data and cannot reconstruct time series. Coordinate lists may reflect 2026, not event-era survey. Town join is date-only and supplies no exact flood position, time, or water depth. No synthetic rain field, flood simulation, hydrograph or global datum correction was generated.\n'
 (PRO/'README.txt').write_text(readme)
 rows=''.join(f'<tr><td>{s["station_name"]}</td><td>{s["daily_total_mm"]:g}</td><td>{s["official_daily_max60_mm"]:g}</td><td>完整10分钟序列</td></tr>' for s in context['jma_summary'])
 for name in ['中央','高浜','渋谷橋','新宿']:
  s=next(r for r in annual if r['station_name']==name);rows+=f'<tr><td>東京都 {name}</td><td>{s["event_total_mm"]:g}</td><td>{s["max60_mm"]:g}</td><td>仅事件统计；峰值起止已保存</td></tr>'
 (OUT/'review.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>2024-08-21 暴雨事件数据</title><style>body{{font:16px/1.7 system-ui;margin:30px auto;padding:0 18px;max-width:1050px;color:#203044}}h1{{font-size:26px}}.note{{background:#fff3db;padding:16px;border-radius:8px}}img{{max-width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dde3ea}}.scroll{{overflow-x:auto}}a{{color:#1267a0}}</style><h1>2024-08-21：先匹配真实暴雨</h1><p>已取得4站576条10分钟观测、149站事件统计，并按日期关联港区10条町丁目积水记录。</p><p class="note">这是港区／古川的优先验证候选。暴雨核心区的完整雨量序列、数值雷达网格和河道基准水位过程尚缺；目前不能把单站雨量均匀铺到三区开展定量验证。</p><div class="scroll"><table><thead><tr><th>观测站</th><th>总雨量 mm*</th><th>任意60分钟最大 mm</th><th>可用性</th></tr></thead><tbody>{rows}<tr><td>港区实绩图（站名待核）</td><td>127.5</td><td>110.5</td><td>仅汇总值；未赋坐标</td></tr></tbody></table></div><p>* JMA为日总量；東京都为16:08–21:34事件总量。峰值为任意连续60分钟量，并非正时小时值。</p><img src="rain_gauge_spatial_comparison.png" alt="三区周边站点雨量差"><img src="jma_rainfall_timeseries.png" alt="JMA四站雨量时序"><p>古川四ノ橋约19:15水位上涨约4.3m；地下调节池存水42,200m³。河道高水位期间的排水边界需要纳入模型，水体不能统一当作自由出口。</p><p><a href="assessment.txt">完整判断与下一步</a> · <a href="../../../processed/tokyo_rain_events/2024-08-21/jma_rain_10min.csv">10分钟雨量</a> · <a href="../../../processed/tokyo_rain_events/2024-08-21/tokyo_gauge_event_statistics.csv">149站事件统计</a> · <a href="../../../processed/tokyo_rain_events/2024-08-21/matched_town_records.csv">实绩日期关联</a> · <a href="../../../raw/tokyo_rain_events/2024-08-21/tokyo_event_report.pdf">水防本部报告</a></p><p>来源：<a href="https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=44&block_no=47662&year=2024&month=8&day=21&view=">气象厅</a> · <a href="https://www.suigai-risk.metro.tokyo.lg.jp/">東京都水害记录</a> · <a href="https://www.city.minato.tokyo.jp/documents/109794/20240821.pdf">港区实绩图</a> · <a href="https://mizu.bosai.go.jp/key/20240821Tokyo">NIED事件分析</a>。原始文件与哈希见 acquisition_manifest.json。</p></html>''')
if __name__=='__main__':main()
