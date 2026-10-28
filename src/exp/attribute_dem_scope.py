"""Assign provisional surface candidates to official2024 administrative areas."""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
import csv,html,json,zipfile
from collections import defaultdict
from shapely.geometry import shape,mapping
from shapely.ops import transform,unary_union
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[2]
PROC=ROOT/'data/processed/three_ward_dem_domain'
OUT=ROOT/'data/exp/data-briefing/three_ward_dem_domain'
def main():
 t=Transformer.from_crs(6668,6677,always_xy=True);ll=Transformer.from_crs(6677,6668,always_xy=True)
 with zipfile.ZipFile(moga_raw(ROOT, 'ksj/N03-20240101_13_GML.zip')) as z:source=json.loads(z.read('N03-20240101_13.geojson'))
 external=unary_union([transform(t.transform,shape(x['geometry'])) for x in json.loads((PROC/'surface_contributing_candidates.geojson').read_text())['features']]);pieces=defaultdict(list)
 for f in source['features']:
  g=transform(t.transform,shape(f['geometry']))
  if g.intersects(external):
   x=g.intersection(external)
   if x.area:pieces[(f['properties']['N03_007'],f['properties']['N03_004'])].append(x)
 rows=[];features=[]
 for (code,name),xs in pieces.items():
  g=unary_union(xs)
  role=('ward-boundary pixel sliver, not external contribution' if code in ['13101','13102','13103'] else 'across-water candidate; exclude from baseline until physical surface link verified' if code=='13108' else 'potential surface contribution, sensitive to terrain/channel assumptions')
  row={'ward_code':code,'ward_name':name,'candidate_surface_area_km2':g.area/1e6,'hydraulic_connection_verified':False,'role':role};rows.append(row);features.append({'type':'Feature','geometry':mapping(transform(ll.transform,g)),'properties':row})
 rows.sort(key=lambda x:-x['candidate_surface_area_km2'])
 with (OUT/'external_candidates_by_ward.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (PROC/'external_candidates_by_ward.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False)+'\n')
 display=[r for r in rows if r['ward_code'] not in ['13101','13102','13103'] and r['candidate_surface_area_km2']>.01]
 info='；'.join(f"{r['ward_name']} {r['candidate_surface_area_km2']:.2f}km²" for r in display)
 page=OUT/'review.html';s=page.read_text();block='<section id="ward-attribution"><b>跨区候选的行政归属（不是已确认来水）</b><p>'+html.escape(info)+'</p><small>区名仅作数据管理。20m边界像元会产生少量三区内碎片；江東区隔水候选在确认真实地表连接前不纳入基线。文京、台東、北区等较远候选仍需核查是否通过河道或填洼假连通。</small></section>'
 if 'id="ward-attribution"' not in s:s=s.replace('<h2>实际有效高程覆盖</h2>',block+'<h2>实际有效高程覆盖</h2>');page.write_text(s)
 assessment=OUT/'assessment.txt';s=assessment.read_text()
 if '行政归属补充' not in s:s+='\n行政归属补充（官方N03 2024-01-01）：'+info+'。全部是地形筛查候选而非确认来水；江東隔水候选不直接纳入模型。三区内约0.03km²边界像元碎片不当外部贡献。\n';assessment.write_text(s)
 (OUT/'dem_gap_inquiry_draft_ja.txt').write_text('''未发送的官方DEM资料询问稿
宛先：東京都デジタルサービス局 デジタルサービス推進課（公開データの連絡先に確認）
件名：東京23区2024点群/0.50m DEMの欠測・ダウンロード不可図郭と測量成果について

公開の0.50m DEMを千代田・中央・港区及び隣接する地表流入候補域で研究利用予定です。公式索引に掲載される09LD1796、09LD1852、09LD1861、09LD1871のgrd50及びgrd25のZIPがHTTP403となります。現時点の公開状況、代替の正式入手方法があればご教示ください。
50図郭は原生0.50mのまま矩形範囲が通常の図郭より裁切され、皇居周辺等の非水域にもNoDataが確認されます。この欠測/裁切の意味と、同地域の利用可能な公式地表標高資料を確認したいです。欠測が水域/立入制約/計測未取得/別工程等のどれに当たるか、こちらでは推定していません。
データの測量時期（ファイル作成日や公開年度ではなく取得期間）、使用した標高基準と標高成果の版、T.P.との関係及び2025年以降の改定適用についても確認したいです。
申請者：[氏名・所属・連絡先を記入]
添付候補：取得図郭一覧、実際のGeoTIFF範囲、NoData分布、HTTP状態記録。
''')
 print(info)

if __name__=='__main__':main()
