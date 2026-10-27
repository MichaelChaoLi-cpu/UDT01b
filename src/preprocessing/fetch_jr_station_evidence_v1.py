"""Acquire public JR station-name search and timetable indexes; preserve receipts."""
from pathlib import Path
import requests,re,json,hashlib,concurrent.futures,html,datetime
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_jr_toei_evidence_v1';S=P/'sources'
NAMES=['東京','品川','四ツ谷','市ケ谷','御茶ノ水','新日本橋','新橋','有楽町','水道橋','浜松町','田町','神田','秋葉原','飯田橋','馬喰町','高輪ゲートウェイ','八丁堀','新宿','渋谷','上野','御徒町','浅草橋','両国','錦糸町','大井町','目黒','代々木','信濃町','越中島','潮見','新木場']
def get(url,path,params=None):
 if path.exists():return path.read_bytes(),None
 r=requests.get(url,params=params,timeout=35);r.raise_for_status();path.write_bytes(r.content);return r.content,r.url

def one(name):
 p=S/('jr_search_'+name+'.html');raw,url=get('https://timetables.jreast.co.jp/cgi-bin/st_search.cgi',p,{'mode':0,'ekimei':name});txt=raw.decode('utf8');matches=[]
 for href,label in re.findall(r'<a[^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>',txt,re.S):
  m=re.search(r'/timetable/list(\d+)\.html',href);label=html.unescape(re.sub('<[^>]*>','',label)).strip();station=re.split('[（(]',label)[0]
  if m and station==name:matches.append((m[1],label))
 unique=list({sid:(sid,label) for sid,label in matches}.values());assert len(unique)==1,(name,unique)
 sid,label=unique[0];p=S/('jr_timetable_'+sid+'.html');raw2,url2=get('https://timetables.jreast.co.jp/timetable/list'+sid+'.html',p)
 assert re.search(r'<h1[^>]*>.*?'+re.escape(name),raw2.decode('utf8'),re.S),(name,'name_not_verified')
 return {'name':name,'station_id':str(int(sid)),'timetable_id':sid,'timetable_url':'https://timetables.jreast.co.jp/timetable/list'+sid+'.html','station_page':'https://www.jreast.co.jp/estation/stations/'+str(int(sid))+'.html','name_search_url':url or 'https://timetables.jreast.co.jp/cgi-bin/st_search.cgi?mode=0&ekimei='+requests.utils.quote(name),'name_match':'exact_official_station_search_label','retrieved_utc_date':datetime.date.today().isoformat(),'search_sha256':hashlib.sha256(raw).hexdigest(),'timetable_sha256':hashlib.sha256(raw2).hexdigest()}
def main():
 out=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for n,r in zip(NAMES,pool.map(one,NAMES)):out.append(r);print(n,r['station_id'],flush=True)
 (S/'jr_station_catalog_verified.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
