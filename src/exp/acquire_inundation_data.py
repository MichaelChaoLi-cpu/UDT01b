"""Acquire explicitly selected public Tokyo scenario point data, preserving original bytes."""
from pathlib import Path
import requests,json,hashlib,csv,io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/tokyo_inundation_reference/2026-10-13'
OUT=ROOT/'data/exp/data-briefing/tokyo_inundation_reference'
BASE='https://www.opendata.metro.tokyo.lg.jp/gesui/R3/'
FILES=['kanda_jiban_shinsui_1.csv','kanda_jiban_shinsui_2.csv','jounan_jiban_shinsui_1.csv','jounan_jiban_shinsui_2.csv','koutou_jiban_shinsui_1.csv','sumida_shingashi_jiban_shinsui_1.csv','kanda_shinsui_zukaku.pdf','jounan_shinsui_zukaku.pdf','koutou_shinsui_zukaku.pdf','sumida_shingashi_shinsui_zukaku.pdf']
EXTRA=[('gesui_catalog.json','https://catalog.data.metro.tokyo.lg.jp/api/3/action/package_show?id=t000020d0000000043'),('kensetsu_catalog.json','https://catalog.data.metro.tokyo.lg.jp/api/3/action/package_show?id=t000014d0000000029'),('history_overview.html','https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/kako'),('town_history.json','https://www.suigai-risk.metro.tokyo.lg.jp/shiryoshu/shinsui-kensaku/js/data.json'),('town_history_city_codes.json','https://www.suigai-risk.metro.tokyo.lg.jp/shiryoshu/shinsui-kensaku/js/city.json'),('town_history_cause_codes.json','https://www.suigai-risk.metro.tokyo.lg.jp/shiryoshu/shinsui-kensaku/js/genin.json'),('town_history_interface.js','https://www.suigai-risk.metro.tokyo.lg.jp/shiryoshu/shinsui-kensaku/index.js'),('public_books_catalog.json','https://www.suigai-risk.metro.tokyo.lg.jp/api/tokyoflood/file/searchFloodRecordBook.py'),('history_map_layer.js','https://www.suigai-risk.metro.tokyo.lg.jp/shinsui/modules/map/layers/jisseki-layer.js'),('chiyoda_history_page.html','https://www.city.chiyoda.lg.jp/koho/kurashi/bosai/bosai-taisaku/fusui-dosha-saigai/rirekizu.html'),('chuo_history_page.html','https://d3ri4jk2y3zd2r.cloudfront.net/a0011/bousaianzen/bousai/bousaitaisaku/suigaisonae/kouzuihazardmap/kozui02.html'),('chuo_history_r8.pdf','https://d3ri4jk2y3zd2r.cloudfront.net/documents/2916/r8-sinsuijisseki.pdf'),('sumida_data_note.html','https://www.kensetsu.metro.tokyo.lg.jp/river/chusho_seibi/panhulink/menu/menu02-02-01'),('scenario_method_qa.pdf','https://www.kensetsu.metro.tokyo.lg.jp/documents/d/kensetsu/shinsuiyosoukuikizuQ_A_R803')]
def fetch(entry):
 name,url=entry;p=RAW/name;rec={'name':name,'url':url,'local_path':str(p.relative_to(ROOT)),'retrieved_utc':datetime.now(timezone.utc).isoformat()}
 try:
  if p.exists():rec['status']='existing_preserved'
  else:
   for attempt in range(2):
    temp=p.with_suffix(p.suffix+'.part')
    try:
     with requests.get(url,stream=True,timeout=(15,35)) as response:
      response.raise_for_status();rec['headers']={k:response.headers.get(k) for k in ('Content-Type','Content-Length','Last-Modified','ETag')};rec['resolved_url']=response.url
      with temp.open('wb') as f:
       for b in response.iter_content(1<<20):f.write(b)
     datahead=temp.open('rb').read(256)
     if name.endswith('.pdf') and not datahead.startswith(b'%PDF'):raise ValueError('Response is not PDF')
     if name.endswith('.json'):json.loads(temp.read_text(encoding='utf-8-sig'))
     if name.endswith('.csv'):
      head=datahead.decode('utf-8-sig',errors='replace')
      if not all(t in head for t in ['浸水','地盤','緯度','経度']):head=datahead.decode('cp932',errors='replace')
      if not all(s in head for s in ['浸水','地盤','緯度','経度']):raise ValueError('Unexpected CSV header: '+head[:100])
     temp.replace(p);rec['status']='downloaded';break
    except Exception:
     temp.unlink(missing_ok=True)
     if attempt==1:raise
  with p.open('rb') as f:rec['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
  rec['bytes']=p.stat().st_size
 except Exception as e:rec['status']='failed';rec['error']=str(e)
 print(name,rec['status'],rec.get('bytes',0),flush=True);return rec

def main():
 RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 entries=[(n,BASE+n) for n in FILES]+EXTRA
 with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,entries))
 (OUT/'acquisition_manifest.json').write_text(json.dumps({'sources':records,'purpose':'Official modeled depths/terrain and historical observation references; no new flood simulation.'},ensure_ascii=False,indent=2)+'\n')
 print('downloaded/preserved',sum(r['status']!='failed' for r in records),'of',len(records),flush=True)
if __name__=='__main__':main()
