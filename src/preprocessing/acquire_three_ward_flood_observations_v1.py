"""Acquire public observational source documents only; never read model predictions."""
from pathlib import Path
import json,urllib.request,hashlib,datetime,concurrent.futures
ROOT=Path(__file__).resolve().parents[2]
TAG='three_ward_road_flood_observations_v1'
P=ROOT/'data/processed'/TAG
SOURCES=[
 ('S001','Minato September11 school-route account','https://www.city.minato.tokyo.jp/kouchou/kuse/kocho/ikenshokai49/2816.html','official_public_comment','E20250911'),
 ('S002','Chuo historical inundation catalog','https://www.city.chuo.lg.jp/a0011/bousaianzen/bousai/bousaitaisaku/suigaisonae/kouzuihazardmap/kouzui03.html','official_inventory',''),
 ('S003','Chuo August2026 inundation record map','https://www.city.chuo.lg.jp/documents/2915/r8-sinsuijisseki.pdf','official_map',''),
 ('S004','Chiyoda historical inundation catalog','https://www.city.chiyoda.lg.jp/koho/kurashi/bosai/bosai-taisaku/fusui-dosha-saigai/rirekizu.html','official_inventory',''),
 ('S005','Minato historical inundation catalog','https://www.city.minato.tokyo.jp/dobokukeikaku/bosai-anzen/bosai/shinsui/suigaizissekizu.html','official_inventory',''),
 ('S006','TokyoPort tunnel restriction event lead','https://trafficnews.jp/post/585855','secondary_operator_lead','E20250911'),
 ('S007','TokyoPort tunnel closure and reopening lead','https://www.chiba-tv.com/plus/detail/202509718476','secondary_closure_lead','E20250911'),
 ('S008','Weathernews July31 imagery','https://weathernews.jp/news/202407/310275/','primary_weather_media','E20240731'),
 ('S009','FNN September11 field coverage','https://www.youtube.com/watch?v=MTF_Wj4S-Qs','primary_news_video_metadata','E20250911'),
 ('S010','ANN September11 next-day field coverage','https://news.tv-asahi.co.jp/news_society/articles/900173303.html','primary_news_media','E20250911'),
 ('S011','Tokyo flood record catalog','https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/kako','official_inventory',''),
 ('S012','Shutoko August29 closure forecast','https://www.shutoko.co.jp/news/press-update/press/2024/08/29-typhoon10-04/','official_forecast_not_observation','E20240830'),
 ('S013','NEXCO June18 closure forecast','https://www.e-nexco.co.jp/en/news/important_info/2024/0618/00013865.html','official_forecast_not_observation','E20240618'),
 ('S014','TBS August30 field report','https://newsdig.tbs.co.jp/articles/-/1395180?display=1','primary_news_media','E20240830'),
 ('S015','Weathernews July31 retrospective','https://weathernews.jp/news/202407/300135/','primary_weather_media','E20240731'),
 ('S016','Minato August21 map outside selected events','https://www.city.minato.tokyo.jp/documents/109794/20240821.pdf','official_map_outside_event_scope',''),
 ('S017','Tokyo underpass susceptibility list June2025','https://www.ktr.mlit.go.jp/ktr_content/content/000752436.pdf','susceptibility_not_event_observation',''),
 ('S018','Chiyoda MyCityReport public-service page','https://www.city.chiyoda.lg.jp/koho/machizukuri/kotsu/sp-system.html','official_reporting_service',''),
 ('S019','R357 September11 actual closure notification','https://www.ktr.mlit.go.jp/kisha/kyoku_s_kisha_00418.pdf','official_actual_closure','E20250911'),
 ('S020','Minato July10 heavy-rain warning','https://www.city.minato.tokyo.jp/system/mobile/bousai/5452.html','official_warning_not_observation','E20250710'),
 ('S021','Mainichi July10 field video metadata','https://video.mainichi.jp/detail/video/6375495214112?autoStart=true','primary_news_video_metadata','E20250710'),
 ('S022','Weathernews August30 field report','https://weathernews.jp/news/202408/300165/','primary_weather_media','E20240830'),
 ('S023','June18 ANN rail service report','https://news.tv-asahi.co.jp/news_society/articles/000355087.html','primary_news_rail_not_road','E20240618'),
 ('S024','Operator public-post reproduction lead','https://blog.livedoor.jp/reer121/archives/59407467.html','secondary_post_locator','E20250911'),
 ('S025','Original Weathernews July31 report 87073572','https://wxrepo.weathernews.jp/report/detail/87073572','primary_user_weather_report','E20240731'),
]
def acquire(s):
 sid,title,url,kind,event=s;row=dict(source_id=sid,title=title,url=url,source_type=kind,event_id=event,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research source verification)'})
  with urllib.request.urlopen(request,timeout=35) as r:
   body=r.read(16*1024*1024+1);assert len(body)<=16*1024*1024
   suffix='.pdf' if body.startswith(b'%PDF') else '.html'
   path=P/'sources'/(sid+suffix);path.write_bytes(body)
   row.update(status='downloaded',http_status=r.status,final_url=r.url,local_path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))
 except Exception as e:row.update(status='unavailable',error=str(e))
 return row
def main():
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--ids',nargs='*')
 parser.add_argument('--refresh',action='store_true')
 args=parser.parse_args()
 (P/'sources').mkdir(parents=True,exist_ok=True)
 prior=json.loads((P/'source_acquisition.json').read_text()) if (P/'source_acquisition.json').exists() else []
 by_id={r['source_id']:r for r in prior}
 jobs=[s for s in SOURCES if (not args.ids or s[0] in args.ids) and (args.refresh or s[0] not in by_id)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(acquire,jobs))
 for r in rows:by_id[r['source_id']]=r
 (P/'source_acquisition.json').write_text(json.dumps(sorted(by_id.values(),key=lambda r:r['source_id']),indent=2,ensure_ascii=False)+'\n')
 for r in rows:print(r['source_id'],r['status'],r.get('bytes',0),r.get('error',''),flush=True)
if __name__=='__main__':main()
