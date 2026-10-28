"""Inventory top-level PLATEAU building attributes without changing raw archives."""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
import xml.etree.ElementTree as ET
import json,csv,zipfile,collections,hashlib,time
ROOT=Path('/Volumes/Extension/Research/UDT01b')
OUT=ROOT/'data/processed/three_ward_population_mesh_audit_v1'
OUT.mkdir(parents=True,exist_ok=True)
GML='{http://www.opengis.net/gml}'
def tag(e):return e.tag.rsplit('}',1)[-1]
def first(e,name):return next((x.text.strip() for x in e.iter() if tag(x)==name and x.text and x.text.strip()),'')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
rows={};duplicates=0;conflicts=0;files=[];codes={};sources=[];started=time.time()
for archive in sorted((moga_raw(ROOT, 'plateau/2025')).rglob('*citygml*.zip')):
 with zipfile.ZipFile(archive) as z:
  for n in ['Building_usage','BuildingDetailAttribute_detailedUsage']:
   e=ET.fromstring(z.read('codelists/'+n+'.xml'))
   codes[n]={first(x,'name'):first(x,'description') for x in e.iter() if tag(x)=='Definition'}
  for n in z.namelist():
   if n.startswith('metadata/') and n.endswith('_resource_op.csv'):
    text=z.read(n).decode('utf-8-sig');(OUT/(archive.parent.name+'_source_materials.csv')).write_text(text)
    import io
    for r in csv.DictReader(io.StringIO(text)):
     if r['feature']=='bldg:Building' and r['property'] in ['bldg:usage','bldg:storeysAboveGround','bldg:lod1Solid']:
      sources.append({'archive':moga_source(archive, ROOT),**r})
  members=[x for x in z.infolist() if x.filename.startswith('udx/bldg/') and x.filename.endswith('.gml')]
  for j,m in enumerate(members):
   count=0
   with z.open(m) as f:
    it=ET.iterparse(f,events=['start','end']);_,root=next(it)
    for event,e in it:
     if event!='end' or tag(e)!='cityObjectMember':continue
     b=next(iter(e),None)
     if b is not None and tag(b)=='Building':
      ident=b.get(GML+'id','');count+=1
      r={'gml_id':ident,'archive':moga_source(archive, ROOT),'member':m.filename}
      for name in ['usage','storeysAboveGround','storeysBelowGround','measuredHeight','detailedUsage','surveyYear','buildingRoofEdgeArea','buildingID','city','creationDate','yearOfConstruction','yearOfDemolition','totalFloorArea','numberOfHouseholds']:
       # Building standard attributes use direct children to avoid inheriting BuildingPart attributes.
       r[name]=next((x.text.strip() for x in b if tag(x)==name and x.text and x.text.strip()),'') if name in ['usage','storeysAboveGround','storeysBelowGround','measuredHeight','creationDate','yearOfConstruction','yearOfDemolition'] else first(b,name)
      r['building_part_count']=sum(tag(x)=='BuildingPart' for x in b.iter())
      for name in ['延べ面積換算係数','13+区市町村コード+大字・町コード+町・丁目コード']:
       v=next((first(x,'value') for x in b if x.get('name')==name),'');r['floor_area_conversion_factor' if name=='延べ面積換算係数' else 'administrative_town_code']=v
      if ident in rows:
       duplicates+=1
       if any(r[k]!=rows[ident][k] for k in ['usage','storeysAboveGround','detailedUsage','surveyYear']):conflicts+=1
      else:rows[ident]=r
     root.clear()
   files.append({'archive':moga_source(archive, ROOT),'member':m.filename,'bytes':m.file_size,'crc32':f'{m.CRC:08x}','building_objects':count})
   if j%5==0 or j==len(members)-1:print(archive.parent.name,j+1,len(members),'unique',len(rows),'seconds',round(time.time()-started),flush=True)
with (OUT/'plateau_building_attributes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(next(iter(rows.values()))));w.writeheader();w.writerows(rows.values())
save(OUT/'building_attribute_sources.json',{'files':files,'source_materials':sources,'codelists':codes,'unique_buildings':len(rows),'duplicate_occurrences':duplicates,'conflicting_duplicate_attributes':conflicts,'selection':'First occurrence in sorted archive and archive member order, matching existing geometry extractor.'})
print('COMPLETE',len(rows),'duplicates',duplicates,'conflicts',conflicts,flush=True)
