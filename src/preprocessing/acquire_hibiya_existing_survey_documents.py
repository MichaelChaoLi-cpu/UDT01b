"""Preserve public official survey specifications; not delivered survey results."""
import hashlib,json,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/official_hibiya_documents'
OUT=ROOT/'data/exp/data-briefing/hibiya_priority_followup'
ENTRIES=[
('https://www.env.go.jp/garden/content/900457723.pdf','moe_hibiya_survey_2022_spec_and_drawings.pdf'),
('https://www.env.go.jp/garden/content/000324560.pdf','moe_hibiya_sakuradabori_2025_design_spec.pdf'),
]
def main():
 RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);sources=[]
 for url,name in ENTRIES:
  p=RAW/name
  if not p.exists():
   try:
    with urllib.request.urlopen(url,timeout=60) as response:p.write_bytes(response.read())
   except urllib.error.HTTPError as error:
    sources.append({'url':url,'path':None,'http_status':error.code,'retrieved':False,'document_role':'Indexed official specification; original download unavailable during this check'})
    continue
  if not p.read_bytes().startswith(b'%PDF'):raise ValueError('Not PDF')
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  sources.append({'url':url,'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha,'retrieved':True,'document_role':'Public tender specification / attached scope drawings; NOT final as-surveyed deliverable'})
 (OUT/'official_source_manifest.json').write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sources':sources},ensure_ascii=False,indent=2)+'\n')
 print('Public sources checked:',len(sources),'PDF files preserved:',sum(bool(s.get('retrieved')) for s in sources))
if __name__=='__main__':main()
