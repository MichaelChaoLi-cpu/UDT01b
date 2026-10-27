"""Fetch paired 2020 JGD2011 census population archives and definitions.
Existing originals are checksum-verified and never overwritten. No API key needed.
"""
from pathlib import Path
import urllib.request,json,hashlib,datetime,argparse
ITEMS=[('population_125m_5339.zip','https://www.e-stat.go.jp/gis/statmap-search/data?statsId=T001231&code=5339&downloadType=2'),('population_250m_5339.zip','https://www.e-stat.go.jp/gis/statmap-search/data?statsId=T001142&code=5339&downloadType=2'),('definition_125m.pdf','https://www.e-stat.go.jp/gis/statmap-search/data?datatype=1&statsId=T001231&downloadType=1'),('definition_250m.pdf','https://www.e-stat.go.jp/gis/statmap-search/data?datatype=1&statsId=T001142&downloadType=1'),('official_cautions_202512.pdf','https://www.stat.go.jp/data/mesh/pdf/ryuijikou.pdf'),('official_mesh_creation.pdf','https://www.stat.go.jp/data/mesh/pdf/gaiyo2.pdf')]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=ap.parse_args();root=args.root.resolve();dest=root/'data/raw/estat_population/2020_jgd2011';dest.mkdir(parents=True,exist_ok=True)
 mp=dest/'download_manifest.json';previous=json.loads(mp.read_text()) if mp.exists() else []
 manifest=[]
 for name,url in ITEMS:
  p=dest/name
  if p.exists():
   b=p.read_bytes();old=next((r for r in previous if r['file']==str(p.relative_to(root))),None)
   if old:assert hashlib.sha256(b).hexdigest()==old['sha256'],name
   else:raise RuntimeError('Existing file has no checksum manifest: '+str(p))
   manifest.append(old)
  else:
   with urllib.request.urlopen(url,timeout=90) as r:b=r.read();final=r.url;ct=r.headers.get('Content-Type')
   assert b.startswith(b'PK') or b.startswith(b'%PDF'),name
   with p.open('xb') as f:f.write(b)
   manifest.append({'file':str(p.relative_to(root)),'url':url,'final_url':final,'content_type':ct,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  print(name,len(b),flush=True)
 if not mp.exists():mp.write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
