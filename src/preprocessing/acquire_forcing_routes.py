"""Read-only evidence pack for obtaining2024Tokyo rainfall and river stages.
No messages, accounts, applications or purchases are submitted.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from html.parser import HTMLParser
import csv,io,json,hashlib,re
import requests
from pyproj import Transformer
from shapely.geometry import shape,Point,mapping
from shapely.ops import transform,unary_union

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/forcing_access_review/2026-10-14'
PROC=ROOT/'data/processed/three_ward_dem_domain'
OUT=ROOT/'data/exp/data-briefing/three_ward_dem_domain'
SOURCES=[
 ('tokyo_information','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/tsim0107g.html'),
 ('tokyo_history','https://www.kasen-suibo.metro.tokyo.lg.jp/im/other/tsim0110g.html'),
 ('tokyo_coordinates','https://www.kasen-suibo.metro.tokyo.lg.jp/zahyo/zahyo.csv'),
 ('minato_list','https://navi.city.minato.tokyo.jp/suii_uryo/list.html'),
 ('minato_contact','https://www.city.minato.tokyo.jp/bousai/kuse/kocho/faq/bosai/005.html'),
 ('dias_metadata','https://search.diasjp.net/ja/dataset/MLIT_XRAIN'),
 ('dias_original_app','https://diasjp.net/en/app_list/xrain-data/'),
 ('dias_original_portal','https://xrain.diasjp.net/original/'),
 ('dias_faq','https://diasjp.net/en/faq/'),
 ('fric_history','https://www.river.or.jp/koeki/opendata/offline.html'),
 ('fric_items','https://www.river.or.jp/koeki/opendata/komoku.html'),
 ('tokyo_disclosure_portal','https://www.kensetsu.metro.tokyo.lg.jp/about/info/jouhoukoukai-potal'),
 ('gsi_tiles','https://maps.gsi.go.jp/development/ichiran.html'),
 ('gsi_height_spec','https://maps.gsi.go.jp/development/demtile.html'),
 ('tokyo_dem_catalog','https://www.geospatial.jp/ckan/api/3/action/package_show?id=tokyopc-23ku-2024')]


def main():
 RAW.mkdir(parents=True,exist_ok=True)
 def job(entry):
  name,url=entry;p=RAW/(name+('.csv' if name=='tokyo_coordinates' else '.json' if name=='tokyo_dem_catalog' else '.html'))
  if p.exists():return json.loads((RAW/(name+'_access.json')).read_text())
  r=requests.get(url,timeout=(15,40));p.write_bytes(r.content)
  meta={'name':name,'url':url,'resolved_url':r.url,'status':r.status_code,'path':str(p.relative_to(ROOT)),'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'accessed_at_utc':datetime.now(timezone.utc).isoformat()}
  (RAW/(name+'_access.json')).write_text(json.dumps(meta,indent=2)+'\n');return meta
 with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(job,SOURCES))
 (OUT/'forcing_access_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 coord=RAW/'tokyo_coordinates.csv';rows=list(csv.DictReader(io.StringIO(coord.read_bytes().decode('cp932'))))
 t=Transformer.from_crs(6668,6677,always_xy=True);domain=unary_union([transform(t.transform,shape(f['geometry'])) for f in json.loads((PROC/'acquisition_domain.geojson').read_text())['features']])
 core_rain={'中央','高浜','渋谷橋','新宿','亀島川'}
 core_stage={'四ノ橋','渋谷橋','古川取水口上流','古川取水口下流','飯田橋','佃橋','小台'}
 selected=[];features=[];station_search=domain.buffer(2000)
 for row in rows:
  name=row['観測所名'];kind=row['観測区分']
  lat=float(row['緯度（度）'])+float(row['緯度（分）'])/60+float(row['緯度（秒）'])/3600;lon=float(row['経度（度）'])+float(row['経度（分）'])/60+float(row['経度（秒）'])/3600
  pt=Point(*t.transform(lon,lat));spatial=station_search.covers(pt)
  if (kind=='雨量' and (spatial or name in core_rain)) or (kind=='水位' and (name in core_stage or domain.covers(pt))):
   entry={'name':name,'kind':kind,'longitude':lon,'latitude':lat,'selection':'named_boundary_candidate' if kind=='水位' else 'core_event' if name in core_rain else 'within_acquisition_domain_plus2km_station_search','coordinates_source':'current official zahyo.csv','station_valid_in2024':'unverified','stage_datum_and_zero':'unverified' if kind=='水位' else 'not applicable','requested_period_jst':'2024-08-21 00:00 to2024-08-22 06:00','requested_sampling':'1–10 min original values','archive_status':'not yet obtained'}
   selected.append(entry);features.append({'type':'Feature','geometry':mapping(Point(lon,lat)),'properties':entry})
 with (PROC/'forcing_station_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
 (PROC/'forcing_station_candidates.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False)+'\n')
 routes=[
  dict(priority=1,product='东京都核心雨量和河段水位',provider='東京都建設局河川部防災課',route='先向数据主管询问历史原始数据及保存情况；如需正式手续再走公文书开示',contact='S0000386@section.metro.tokyo.jp',link=dict(SOURCES)['tokyo_information'],available='官网最多过去一年、00:00–23:50日CSV；2024事件已超窗口，历史保存/提供需主管确认',need='候选站1–10分钟雨量/H(t)，水尺零点、T.P./A.P.基准、站址和成果版本变更记录；古川调节池入流/蓄水及主要闸运行',status='request_prepared_not_sent'),
  dict(priority=1,product='港区9站雨量、2站古川水位',provider='港区防災危機管理室防災課防災係',route='向主管申请这场事件原始时序及观测定义；公开页面只有近3小时',contact='03-3578-2111 内線2541–2545',link=dict(SOURCES)['minato_contact'],available='当前列表每3分钟更新，雨量是前1小时滚动量；不是3分钟增量，水位基准仍待核实',need='港区役所、田町、表参道、品川、白金台、南麻布、麻布十番、白金3丁目、溜池山王；原始雨量区间、仪器/坐标/缺测；麻布十番/白金3丁目水位零点',status='request_prepared_not_sent'),
  dict(priority=2,product='2024XRAIN数值分布雨量',provider='MLIT/DIAS',route='询问MLIT/DIAS现行CXMP Composite档案获取方式；核实可用后再申请个人账号及研究许可',contact='hqt-dias@gxb.mlit.go.jp; dias-office@diasjp.net',link=dict(SOURCES)['dias_metadata'],available='元数据明确当前DIAS下载功能不可用；Original应用页仍说明账号+许可，门户本环境HTTP403，不能证明2024文件可直接下载',need='2024事件CXMP合成雨量，通常250m/1min；裁切本次域及相关河川上游，保留质量码/缺测/单位/时间说明；不是2020前旧XMP合成产品',status='availability_inquiry_prepared_not_sent'),
  dict(priority=3,product='历史雨量/水位和XRAIN备选',provider='河川情報センターFRIC',route='历史离线数据有偿包；只作为备选，先核实东京所需站和2024覆盖及可否事件子集报价',contact='官网数据提供窗口',link=dict(SOURCES)['fric_history'],available='页面列出雨量/水位2014-07起、XRAIN2017-07起；半年区域包，具体所需站/期间和条件需报价确认',need='2024下半年东京遥测站及关东XRAIN；不要因套餐声明就假定覆盖港区自设9站',status='not_contacted_not_purchased'),
  dict(priority=0,product='JMA雨量与晴海潮位',provider='JMA',route='已下载资料继续作为参考和海侧背景',contact='无需申请',link='https://www.data.jma.go.jp/kaiyou/db/tide/genbo/genbo.php?stn=TK&year=2024&month=08&G_HOUR=on&LV=TP',available='项目已有4站10分钟降雨和48条晴海小时平滑实测潮位',need='分钟河道/未滤波潮位另补；单站JMA东京不能代替核心空间暴雨，小时滤波潮位不能代替河段H(t)',status='reference_data_local_not_full_model_forcing')]
 with (OUT/'forcing_acquisition_routes.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(routes[0]));w.writeheader();w.writerows(routes)
 (OUT/'forcing_acquisition_routes.json').write_text(json.dumps(routes,ensure_ascii=False,indent=2)+'\n')
 request='''未发送的资料询问草稿（需要填写申请人所属及联系信息后才能发送）

東京都建設局河川部防災課 御中
件名：2024年8月21日 都心三区豪雨の雨量・河川水位観測データの提供方法について

千代田区・中央区・港区及び隣接する地表流入域を対象に、降雨による地表浸水と河川水位の影響を研究する予定です。公開サイトの過去データは最大1年との記載があるため、以下の過去観測資料の保存状況、提供可否と必要な手続きをご教示ください。

期間：2024年8月21日00:00〜8月22日06:00（日本標準時）。前後の湿潤条件・減水状況について、可能な追加期間も確認したいです。
雨量：中央、高浜、渋谷橋、新宿、亀島川、及び添付候補観測所一覧に含まれる局の1〜10分の原観測値。期間雨量・最大60分雨量の集計のみではなく、タイムスタンプと観測区間、欠測・品質符号、測器情報、2024年時点の座標を希望します。
河川：渋谷橋、四ノ橋、古川取水口上流/下流、飯田橋、佃橋、小台等の水位。日本橋川・亀島川に対応する観測局についてもご案内ください。2024年時点の観測局位置、量水標零点、T.P./A.P.等の基準、基準点標高、標高成果の版と適用日も確認したいです。可能であれば流量とその算定方法を希望します。
関連施設：古川地下調節池の取水・貯留・排水時系列、対象河段の主要水門等の当日運転記録と、利用可能な主要岸頂/河床断面資料。
流量・水位をモデル境界に使うため、相対的な増水幅、設計高水位及び動画だけでは標高基準を確定できません。電子形式（CSV等）が利用可能か、有償・申請・公文書開示等の条件もご教示いただけますと幸いです。
申請者：[氏名・所属・連絡先を記入]

港区向け補足：防災課防災係に9雨量局及び麻布十番・白金3丁目水位の原観測値と基準の保存状況を同じ期間で確認。公開3分更新の前1時間雨量から短時間雨量を逆算せず、原記録を希望。

DIAS/MLIT向け補足：CXMP Compositeの2024-08-21関東データの保存と利用手順を確認。メタデータにダウンロード機能停止の記載があるため、現在の提供窓口/申請入口を問い合わせる。研究者の所属、目的、研究グループ、期間は本人の情報で記入し、アカウント/利用許諾を代行提出しない。
'''
 (OUT/'archive_inquiry_draft_ja.txt').write_text(request)
 validation={'public_pages_checked':len(manifest),'selected_rain_stations':sum(x['kind']=='雨量' for x in selected),'selected_stage_stations':sum(x['kind']=='水位' for x in selected),'all_core_stage_names_found':core_stage.issubset({x['name'] for x in selected if x['kind']=='水位'}),'historical_event_series_newly_obtained':False,'outbound_requests_sent':False,'accounts_registered':False,'purchases':False,'route_constraints':['Tokyo public history limited1year','Minato rolling60min rainfall,3hour view','DIAS metadata currently disables download','FRIC historical package coverage and cost require confirmation']}
 (OUT/'forcing_route_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n');print(validation)


if __name__=='__main__':main()
