"""Download official observations and provenance; never read simulation outputs."""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
import json,urllib.request,urllib.error,hashlib,datetime,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_official_observations_v2'
RAW=moga_raw(ROOT, 'official_observations/20261028');P=ROOT/'data/processed'/TAG
SOURCES=[
 ('O001','Tokyo flood response report 20250911','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/2025-09-12-160014-933','event_river_camera_and_gauge_report','E20250911'),
 ('O002','Tokyo flood response report 20240731','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/000070495','event_river_camera_and_gauge_report','E20240731'),
 ('O003','Tokyo flood response report 20250710','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/suibour7-3-2','event_river_camera_and_gauge_report','E20250710'),
 ('O004','Tokyo flood response report Typhoon10 20240829-0901','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/000071024','event_river_camera_and_gauge_report','E20240830'),
 ('O005','JMA Tokyo September11 event report','https://www.data.jma.go.jp/tokyo/shosai/chiiki/disaster/2025/20250911_sokuhou.pdf','meteorological_observations_not_road_labels','E20250911'),
 ('O006','JMA Tokyo July31 event report','https://www.jma-net.go.jp/tokyo/shosai/chiiki/disaster/2024/20240731_hm-air_tokyo.pdf','meteorological_observations_not_road_labels','E20240731'),
 ('O007','Tokyo 2024 disaster annual report','https://www.bousai.metro.tokyo.lg.jp/_res/projects/default_project/_page_/001/020/657/2024.pdf','reported_damage_statistics_not_road_labels','2024'),
 ('O008','Tokyo 2025 finalized disaster counts','https://www.bousai.metro.tokyo.lg.jp/taisaku/saigai/1020657/1030906.html','reported_damage_statistics_not_road_labels','2025'),
 ('O009','Flood-record publication timing FAQ','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/2024-12-02-130817-498','coverage_and_reporting_limit',''),
 ('O010','Official public flood-record book catalog','https://www.suigai-risk.metro.tokyo.lg.jp/api/tokyoflood/file/searchFloodRecordBook.py','official_historical_observation_catalog',''),
 ('O011','Official historical town-level incident table','https://www.suigai-risk.metro.tokyo.lg.jp/shiryoshu/shinsui-kensaku/js/data.json','official_town_level_incidents',''),
 ('O012','Minato flood-history catalog current snapshot','https://www.city.minato.tokyo.jp/dobokukeikaku/bosai-anzen/bosai/shinsui/suigaizissekizu.html','official_historical_observation_catalog',''),
 ('O013','Tokyo hydrological archive scope','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/tsim0107g.html','observational_download_coverage',''),
 ('O014','Tokyo hydrological past-data page','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/tsim0110g.html','observational_download_coverage',''),
 ('O015','Minato live rain and river levels','https://navi.city.minato.tokyo.jp/suii_uryo/list.html','live_observation_not_historical_event',''),
 ('O016','Minato August21 actual inundation map','https://www.city.minato.tokyo.jp/documents/109794/20240821.pdf','official_event_inundation_map_outside_original_five','E20240821'),
 ('O017','Tokyo August21 flood response report','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/000070667','event_river_camera_and_gauge_report','E20240821'),
 ('O018','R357 September11 road closure operator notice','https://www.ktr.mlit.go.jp/kisha/kyoku_s_kisha_00418.pdf','actual_flood_closure_external_tunnel','E20250911'),
 ('O019','Minato September11 resident account on official site','https://www.city.minato.tokyo.jp/kouchou/kuse/kocho/ikenshokai49/2816.html','resident_claim_not_municipal_measurement','E20250911'),
 ('O020','Chuo current actual inundation map','https://www.city.chuo.lg.jp/documents/2915/r8-sinsuijisseki.pdf','official_historical_inundation_map',''),
 ('O021','JMA rain symbol meanings','https://www.data.jma.go.jp/stats/data/mdrr/man/remark.html','rainfall_quality_metadata','')]
SOURCES.extend([('O022','Minato July2025 heavy-rain newsletter with August2024 field photos','https://www.city.minato.tokyo.jp/documents/168111/20250714144352.pdf','official_event_road_photographs_outside_original_five','E20240821'),('O023','JMA official Tokyo station coordinate selector','https://www.data.jma.go.jp/stats/etrn/select/prefecture.php?prec_no=44&block_no=&year=2024&month=8&day=21&view=','gauge_coordinate_metadata','')])
SOURCES.extend([('O024','Historical water-level CSV availability check September11','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/Tokyo_Suii_20250911.csv','historical_water_level_availability_check','E20250911'),('O025','Historical rainfall CSV availability check September11','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/Tokyo_Uryo10min_20250911.csv','historical_rainfall_availability_check','E20250911')])
DAYS=['2024-06-18','2024-07-31','2024-08-30','2025-07-10','2025-09-11']
for station,typ,block in [('tokyo','s','47662'),('setagaya','a','0368')]:
 for i,day in enumerate(DAYS):
  y,m,d=map(int,day.split('-'));sid=f'G{station[0].upper()}{i+1:02}'
  url=f'https://www.data.jma.go.jp/stats/etrn/view/10min_{typ}1.php?prec_no=44&block_no={block}&year={y}&month={m}&day={d}&view='
  SOURCES.append((sid,f'JMA {station} 10-minute precipitation {day}',url,'rain_gauge_10min','E'+day.replace('-','')))
def get_source(item):
 sid,title,url,role,event=item;stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();row={'source_id':sid,'title':title,'url':url,'role':role,'event_id':event,'retrieved_utc':stamp}
 prior=next(RAW.glob(sid+'.*'),None)
 if prior:
  previous=P/'acquisition_manifest.json'
  if previous.exists():
   old=next((x for x in json.loads(previous.read_text()) if x['source_id']==sid and x.get('sha256')==hashlib.sha256(prior.read_bytes()).hexdigest()),None)
   if old:return old
  data=prior.read_bytes();row.update(status='cached',path=moga_source(prior, ROOT),bytes=len(data),sha256=hashlib.sha256(data).hexdigest());return row
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (academic research; public observations)'})
  with urllib.request.urlopen(req,timeout=45) as response:
   data=response.read(48*1024**2+1)
   if len(data)>48*1024**2:raise ValueError('48 MiB per-source ceiling')
   kind=response.headers.get('Content-Type','');suffix='.pdf' if data.startswith(b'%PDF') else '.json' if data.lstrip().startswith((b'{',b'[')) else '.html'
   path=RAW/(sid+suffix);path.write_bytes(data)
   row.update(status='downloaded',path=moga_source(path, ROOT),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),http_status=response.status,content_type=kind,final_url=response.url)
 except Exception as e:row.update(status='unavailable',error=str(e))
 return row
def main():
 RAW.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 # Three bounded requests across providers; source failures remain explicit.
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  rows=[]
  for row in ex.map(get_source,SOURCES):
   rows.append(row);print(row['source_id'],row['status'],row.get('bytes',0),row.get('error',''),flush=True)
   (P/'acquisition_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 print('DOWNLOAD_COMPLETE',len(rows),sum(r.get('bytes',0) for r in rows),flush=True)
if __name__=='__main__':main()
