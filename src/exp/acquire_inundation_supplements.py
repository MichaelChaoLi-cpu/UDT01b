"""Acquire observed official catalog resources and public historical PDFs/GeoJSON.
Run after acquire_inundation_data.py. Only enabled, public annual books are retrieved.
"""
from pathlib import Path
import re,json
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from acquire_inundation_data import fetch,RAW,OUT

def main():
 entries=[]
 j=json.loads((RAW/'kensetsu_catalog.json').read_text())['result']
 for r in j['resources']:
  if r['format']=='CSV' and any(s in r['name'] for s in ['神田','隅田','城南','江東']):entries.append((Path(r['url']).name,r['url']))
 entries.append(('historical_inundation_map.geojson','https://www.suigai-risk.metro.tokyo.lg.jp/api/tokyoflood/searchFloodRecordMap.py'))
 s=(RAW/'chiyoda_history_page.html').read_text()
 for x in sorted(set(re.findall(r'href="([^"]+\.pdf)"',s))):
  if 'shinsui' in x:entries.append((Path(x).name,urljoin('https://www.city.chiyoda.lg.jp',x)))
 r=fetch(('minato_history_page.html','https://www.city.minato.tokyo.jp/dobokukeikaku/bosai-anzen/bosai/shinsui/suigaizissekizu.html'))
 rs=[r]
 s=(RAW/'minato_history_page.html').read_text();links=sorted(set(re.findall(r'href="([^"]+\.pdf)"',s)))
 chosen=[x for x in links if Path(x).name in ('20240706.pdf','20240720_0721.pdf','20240821.pdf','all_h11h21.pdf','all_h22-r1.pdf','r01_10_11-13.pdf')]
 entries.extend([('minato_'+Path(x).name,urljoin('https://www.city.minato.tokyo.jp',x)) for x in chosen])
 books=json.loads((RAW/'public_books_catalog.json').read_text())
 for r in sorted([r for r in books if r['is_public'] and r['is_enable']],key=lambda r:r['year'],reverse=True)[:2]:
  u=f"https://www.suigai-risk.metro.tokyo.lg.jp/data/suigaikiroku/{r['year']}/{r['path']}";entries.append((f"tokyo_flood_record_{r['year']}.pdf",u))
 with ThreadPoolExecutor(max_workers=3) as pool:rs+=list(pool.map(fetch,entries))
 p=OUT/'acquisition_manifest.json';m=json.loads(p.read_text());old={r['url']:r for r in m['sources']};old.update({r['url']:r for r in rs});m['sources']=list(old.values());p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
