"""Decode authorized PT originals with existing local openpyxl; no value imputation.
Run with a local Python environment containing openpyxl (e.g. existing Anaconda).
"""
from pathlib import Path
import csv,json,zipfile,hashlib,datetime as dt
import openpyxl
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_full_pt_od_analysis_v1';S=P/'sources';O=P/'decoded_tables'
def main():
 O.mkdir(exist_ok=True);m=json.loads((S/'acquisition_manifest.json').read_text());out=[]
 for r in m['files']:
  f=S/r['name'];assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
  if f.suffix=='.xlsx':
   w=openpyxl.load_workbook(f,read_only=True,data_only=True)
   for sheet in w:
    dest=O/(f.stem+'.csv');count=0
    with dest.open('w',encoding='utf-8-sig',newline='')as h:
     cw=csv.writer(h)
     for row in sheet.values:cw.writerow(row);count+=1
    out.append({'source_file':r['name'],'decoded_file':dest.name,'source_sheet':sheet.title,'source_format':'XLSX','decode':'openpyxl read_only,data_only;native row order andblank cells retained','rows':count,'columns':sheet.max_column,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()});print(dest.name,count,flush=True)
   w.close()
  else:
   with zipfile.ZipFile(f)as z:
    assert len(z.namelist())==1;member=z.namelist()[0];raw=z.read(member);raw.decode('cp932');dest=O/'d-1.cp932.csv';dest.write_bytes(raw);out.append({'source_file':r['name'],'decoded_file':dest.name,'source_member':member,'source_format':'CSV inZIP','decode':'Original member bytes,CP932; title andthreeheaderrows retained','sha256':hashlib.sha256(raw).hexdigest()});print(dest.name,len(raw),flush=True)
 (P/'decode_manifest.json').write_text(json.dumps({'decoded_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'python_reader':'openpyxl '+openpyxl.__version__,'files':out},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
