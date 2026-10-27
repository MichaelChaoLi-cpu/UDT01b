"""Preserve official public drainage sources and summarize a manual SEMIS audit.

Run locally from the project root:
  .venv/bin/python src/exp/audit_hibiya_drainage.py --fetch-sources
Static SEMIS pages are acquired without accepting terms or requesting hidden
facilities. Dynamic map observations were made through the official UI with
explicit human consent; they are dated evidence, not a reproducible live API.
No network geometry, elevations, DEM repair, Git or DVC changes are invented.
"""
import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SOURCES = [
    ('semis_entry','https://www.gesuijoho.metro.tokyo.lg.jp/semiswebsystem/','semis_entry.html'),
    ('semis_legend','https://www.gesuijoho.metro.tokyo.lg.jp/semiswebsystem/html/SemisWebSystem/web/info/index.html','semis_legend.html'),
    ('semis_help','https://www.gesuijoho.metro.tokyo.lg.jp/semiswebsystem/html/SemisWebSystem/web/help/index_help.html','semis_help.html'),
    ('semis_availability','https://www.gesui.metro.tokyo.lg.jp/contractor/daicyo','semis_availability.html'),
    ('annual_fy2024','https://www.gesui.metro.tokyo.lg.jp/documents/d/gesui/r6-2sho-pdf','tokyo_sewer_annual_fy2024_ch2.pdf'),
    ('historical_storage','https://www.gesui.metro.tokyo.lg.jp/documents/d/gesui/infn0787_pdf02','tokyo_historical_heavy_rain_plan.pdf'),
    ('height_revision','https://www.gsi.go.jp/WNEW/PRESS-RELEASE/keikaku61003_00001.html','gsi_height_revision.html'),
    ('chiyoda_road_ledger','https://www.city.chiyoda.lg.jp/koho/machizukuri/tochi/dorodaicho.html','chiyoda_road_ledger.html'),
]


def write_table(path, rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);parser.add_argument('--fetch-sources',action='store_true');args=parser.parse_args()
    root=(args.root or Path(__file__).resolve().parents[2]).resolve()
    if not (root/'AGENTS.md').exists():raise ValueError('Explicit research project root required')
    raw=root/'data/raw/drainage_official/2026-10-13';out=root/'data/exp/data-briefing/hibiya_drainage_audit'
    raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    def fetch(entry):
        ident,url,name=entry;path=raw/name
        status='existing_preserved' if path.exists() else 'not_acquired'
        if args.fetch_sources and not path.exists():
            try:
                with urllib.request.urlopen(url,timeout=45) as response:
                    payload=response.read()
                    if name.endswith('.pdf') and not payload.startswith(b'%PDF'):raise ValueError('Non-PDF response')
                    path.write_bytes(payload)
                status='downloaded'
            except Exception as e:status='fetch_failed: '+str(e)
        result={'id':ident,'url':url,'local_path':str(path.relative_to(root)),'status':status}
        if path.exists():
            with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            result.update(bytes=path.stat().st_size,sha256=sha)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(fetch,SOURCES))
    (out/'source_manifest.json').write_text(json.dumps({'audit_date_japan':'2026-10-13','manifest_recorded_utc':datetime.now(timezone.utc).isoformat(),'sources':manifest},indent=2)+'\n')
    observations={'observed_date_japan':'2026-10-13','method':'Official SEMIS browser UI; human explicitly authorized clicking Agree; no restricted facilities requested or bypassed.',
                  'map_url':'https://www.gesuijoho.metro.tokyo.lg.jp/semiswebsystem/SuperaPageWeb.aspx',
                  'views':[
                      {'address':'千代田区日比谷公園1付近','visible_extent_m':[279,152],'observed_sheet_label':'2319-2A','state':'yellow non-display area; no public facility information'},
                      {'address':'千代田区有楽町1-1付近','visible_extent_m':[209,114],'observed_sheet_label':'2319-1C','state':'yellow non-display area'},
                      {'address':'千代田区有楽町1-1付近','visible_extent_m':[697,380],'state':'mostly hidden; some neighbouring southern/eastern map portions publicly show lines'},
                      {'address':'千代田区内幸町1-1付近','visible_extent_m':[209,114],'observed_sheet_label':'2319-3B','state':'western part hidden; eastern neighbouring map portion unmasked'},
                      {'address':'千代田区内幸町1-1付近','visible_extent_m':[697,380],'state':'park/Hibiya Avenue side hidden; more distant eastern railway/street side partly public'},
                  ],'coverage_limit':'Address-centred UI observations and sheet labels, not a georeferenced mask polygon. No exact intersection-area fraction with the DEM is claimed.',
                  'source_measurement_date':'not disclosed in inspected views','source_feature_update_date':'not shown; retrieval date is not an update/survey date',
                  'facilities_extracted_from_pilot':0,'missingness_meaning':'Not publicly accessible in audited core views; not evidence that infrastructure is absent.'}
    (out/'semis_observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
    inventory=[
        dict(layer='Public road inlets / rainwater chambers',source='SEMIS',coverage='23-ward public roads generally; audited Hibiya core views hidden',public_format='Interactive ledger, printed 1:500 PDF where facilities are public',available_fields='Legend includes rainwater chambers and connecting pipes; actual pilot attributes unavailable',measurement_time='Not established',update_time='No feature timestamp in inspected views; construction changes may lag',vertical_reference='SEMIS ground/invert levels use TP; inlet lip elevation not established',dem_join='Not ready: retrieve permitted ledger/field inlet positions and levels',evidence='semis_entry.html; semis_legend.html; SEMIS screenshots'),
        dict(layer='Sewer pipes / manholes',source='SEMIS',coverage='Public roads; private-property/private-road drains outside advertised coverage; hidden blocks require ledger-room access',public_format='Interactive attributes and printable 1:500 PDF; bulk GIS/API not found in bounded catalog search',available_fields='System describes positions, depths, pipe size/type, flow direction, combined/separate system and ground/invert heights',measurement_time='Not established for Hibiya',update_time='No pilot record date obtained',vertical_reference='TP for ledger ground/invert, height-result revision status unknown',dem_join='Conditional: confirm datum version and permitted graph/levels; do not generate synthetic real pipes',evidence='semis_legend.html; semis_availability.html'),
        dict(layer='Pump station context',source='Tokyo sewer annual FY2024 chapter 2',coverage='23-ward station summary; Sakurabashi description names Yurakucho/Marunouchi; pilot connectivity unconfirmed',public_format='Official PDF tables, printed pages 153 and 156',available_fields='Location/address, purpose, service-area size, planned flow, existing capacity incl reserve pumps, commissioning dates',measurement_time='Operational summary as of FY2024 end, not an elevation survey',update_time='FY2024 end; acquisition 2026-10-13',vertical_reference='No pump sump/control/outfall elevations supplied in the reviewed tables',dem_join='Context only until connected drainage area, head-flow curve, controls and downstream water levels are verified',evidence='tokyo_sewer_annual_fy2024_ch2.pdf, PDF pages 60/63'),
        dict(layer='Hibiya intersection rainwater storage clue',source='Historical official heavy-rain plan',coverage='Facility named Hibiya intersection; footprint and pilot connection unverified',public_format='Historical PDF facility list/map',available_fields='Listed storage volume 2100 m3',measurement_time='Historical planning document; current facility condition unverified',update_time='Historical, not current operational metadata',vertical_reference='No inlet/outlet level or storage-level relation confirmed',dem_join='Request current status and connectivity; do not insert full storage volume into pilot model',evidence='tokyo_historical_heavy_rain_plan.pdf'),
        dict(layer='Chiyoda road ledger',source='Chiyoda district official site',coverage='District road names, widths and boundary information',public_format='PDF / GIS view with same road information',available_fields='Road identifiers and widths; not a demonstrated inlet/invert dataset',measurement_time='Not established for drainage',update_time='Web page updated 2025-12-01; not a drainage survey date',vertical_reference='Drainage elevation reference not established',dem_join='Useful registration/context only',evidence='chiyoda_road_ledger.html'),
        dict(layer='Native Tokyo 0.50 m DEM',source='Previously acquired Tokyo 09LD2812 ground raster',coverage='400 x 300 m, E -7200..-6800 / N -36600..-36300; EPSG6677',public_format='Original GeoTIFF',available_fields='Ground raster metres; source LAS consistency checked',measurement_time='Exact tile flight date unresolved; 2023 processing dates do not establish survey date',update_time='Original download product retained; no 2025 correction applied by this project',vertical_reference='Horizontal JGD2011 IX confirmed; named vertical datum and height-result version unresolved',dem_join='Hold absolute height merge until independent benchmark/document reconciliation',evidence='Previous Hibiya terrain and LAS metadata audits'),
    ]
    write_table(out/'availability_inventory.csv',inventory)
    pumps=[
        dict(name='桜橋',reported_as_of='FY2024 end',planned_rain_m3_s=17.783,existing_rain_capacity_m3_min=1320,reserve_included=True,address='中央区新富一丁目2番6号',service_description='Includes Yurakucho/Marunouchi and parts of Chuo in official summary',pilot_connection='Unconfirmed',interpretation='Not live capacity and not a pump curve'),
        dict(name='桜橋第二',reported_as_of='FY2024 end',planned_rain_m3_s=30.5,existing_rain_capacity_m3_min=2400,reserve_included=True,address='中央区湊一丁目1番2号',service_description='Parts of Chuo and Chiyoda',pilot_connection='Unconfirmed; distinct from Sakurabashi',interpretation='Not live capacity and not a pump curve'),
    ]
    write_table(out/'pump_context.csv',pumps)
    tasks=[
        ('P0','Ledger access','70 m road corridor + K1-K4 + known parking portal; extend through actual downstream connections','Permitted 1:500 ledger/as-built drawing; facility IDs, inlet/manhole positions, connected nodes, pipe section, length, upstream/downstream invert, ground level, combined/separate system','Official ledger room / sewer manager','Access and actual pilot connections documented; no restricted online scraping'),
        ('P0','Datum and version','DEM/LAS, ledger and new survey','Named vertical datum, adopted height-result revision, benchmark ID/value/date, survey dates, as-built/update dates and relevant later changes','Dataset producer and facility manager; independent benchmark survey','A documented common height reference; differences not corrected by an unexplained constant'),
        ('P0','External hydraulic boundary','Where the local network exits the DEM tile','Actual downstream route, contributing areas outside tile, tailwater/backwater, pump/catchment connection and storage interfaces','Sewer facility manager','DEM tile edge is not silently treated as a hydraulic catchment boundary'),
        ('P1','Surface microtopography','K1 west edge; K2 canopy-obscured verge; K3 east kerb; K4 median','One coordinated survey of hard-edge breaklines, curb crest/toe, gutter, inlet opening/lip, ground/check heights; retain photographs and point IDs','Targeted field survey tied to same benchmark','Resolve K2 feature meaning and capture controlling barriers/openings; preserve verified ordinary-road surface'),
        ('P1','Underground parking boundary','Verified descending parking ramp','Portal/sill/wall geometry, basement continuation, local drainage pump capacity and controls, any flood barrier operation','Parking operator / as-built drawings','Distinguish local parking pumps from municipal sewer pumps; do not close underground ramp as raster storage'),
        ('P1','Pump and storage','Only facilities proven connected','Operating pumps versus reserve, head-flow curves, start/stop levels, discharge elevations/water-level boundary, active storage-level-volume relation and inlet/outlet controls','Relevant pump/storage manager','Use actual connected operational parameters; reported station capacity is context only'),
        ('P2','Minimal local reconstruction','Existing pilot, extending graph only where needed','Surface terrain + measured breaklines/structures; inlets -> connection pipes -> manholes -> verified outlet; provenance and uncertainty for each parameter','Local reproducible scripts','No full-city solid pipe reconstruction required; unknown attributes remain explicit; no inferred pipe is labelled as observed'),
    ]
    write_table(out/'verification_and_reconstruction_plan.csv',[dict(priority=p,work=w,target=t,required_information=r,source=s,verification_gate=v) for p,w,t,r,s,v in tasks])
    urls={x[0]:x[1] for x in SOURCES}
    report=f'''日比谷排水数据核查与局部重建建议（探索性记录）
核查日期：2026-10-13（日本时间）
项目根目录：{root}
范围：既有 09LD2812 的 400 m × 300 m DEM；重点为已核验的 70 m 普通道路、K1-K4 和地下停车场坡道。

一、结论与实际覆盖
已按用户明确授权进入 SEMIS。日比谷公园1附近、有乐町1-1附近，以及内幸町1-1附近的西侧图面显示黄色“非表示エリア”。较大视图的周边有部分公开线条，但不能替代样区隐藏设施。观察到的图幅标签包括 2319-2A、2319-1C、2319-3B。
上述为按地址定位的 UI 观察，不是已配准的非显示范围矢量；尚未计算隐藏图幅与 DEM 的精确面积交集。核心访问结果足以说明，不能仅凭当前公开服务构建可信的完整样区排水网络。没有显示设施≠没有设施。本次未获取样区雨水口数量、管网图或真实设施高程。
官方页面对非显示区明确提供都厅第二厅舍27层台账阅览室途径，并要求身份证明；能否复制、提供数字格式以及研究使用范围应由窗口确认。未绕过限制，未联系任何机构或发送资料申请。
一般公开范围是23区公道，宅地内、私道等设施不在其宣称完整覆盖内。公园内部与停车场内部设施不能默认为都下水道台账已经包含。

二、高程、时间与格式
SEMIS 图例明确地盘和管底高程为 T.P.（东京湾平均海面）。它不是管道深度，也不能直接等同于雨水口格栅/堰口高程。还需辨别地盘高、井盖/入口高和管底高的字段含义。
本地 DEM 的水平 CRS 已确认 JGD2011 / 平面直角第9系 EPSG:6677，像元为0.50 m；这不能证明垂直基准一致。DEM/LAS 中命名垂直基准及标高成果版本未明确。
SEMIS 的 T.P. 标签也没有在所查页面说明是否重算到2025年实施的测地成果2024。GSI 于2025-04-01改定基准点标高，因此需要同时核对双方采用的成果版本及基准点，不应凭标签或近零道路差值自动消除偏差。本次未施加任何高程改正。
已查 SEMIS 视图没有逐设施测量时间或更新时间。官方提示施工后可能一段时间与现地不一致。访问日期、网页更新时间、图幅更新日期、竣工日期和实测日期应分开记录；没有统一可靠的测量日期可填写。
SEMIS 官方帮助说明可查看设施属性并打印1:500 PDF；未找到本样区可直接下载的公开管网 GIS/API。东京目录中的精确短语“雨水ます”“下水道台帳”各返回0条，“ポンプ所”返回1条防灾资料；泛词搜索会产生大量不相关记录。此为有界检索结果，不是全网不存在数据的证明。
千代田区道路台账提供道路名称、幅员和边界，不能替代雨水口与管底台账；网页更新2025-12-01并非排水设施测量时间。

三、泵站与蓄水资料
已下载并渲染核对令和6年度（FY2024）下水道年报区部章。纸面153页（PDF第60页）的数值为FY2024末，现有排水能力列包含备用泵。桜橋站计划雨水17.783 m³/s、现有雨水能力1320 m³/min；桜橋第二站相应为30.500 m³/s、2400 m³/min。它们是不同站，不能混同；计划量和现有能力的单位也不同。
纸面156页（PDF第63页）的桜橋服务说明明确包括有乐町、丸之内一带，故它是值得追查的下游候选。桜橋第二的服务范围说明为中央区及千代田区部分区域。但是，尚无资料证明样区的具体雨水口/管段接入其中任何一站。
公开表格未给出本样区所需的泵坑、启停或出口高程，也未建立逐泵 Q-H 曲线、运行台数/规则与样区连接。整站能力不能直接赋给400×300 m地块，不能按距离匹配泵站。还要核对外部汇水、潮水/回水及下游水位。
历史官方《豪雨対策下水道緊急プラン》列日比谷交差点雨水调节池2100 m³。仅作查询线索：当前存续、有效库容、连接对象、进出水高程/控制未核实，不能把2100 m³直接加入样区模型。
地下停车场排水泵属于另一类局部设施；市政泵站表不证明其排水能力。

四、高效方案（建议，尚未作为用户的最终建模设计）
1. 先拿连接资料：准备70 m道路、K1-K4与停车场洞口坐标/位置图，到台账窗口按图幅和道路索取允许提供的现况/竣工资料。提取雨水口—取付管—井—管—下游出口关系，仅随真实连接扩展；不用先收全区全部地下设施。验收：来源、允许取得范围、实际连接和管底字段可追溯。
2. 一次解决基准和时间：收集 DEM 与台账生产者的垂直基准、成果版本、基准点、测量/竣工/更新日期，并安排与同一基准点联测的检查高程。对同一可识别的硬质表面比较，分别记录坐标、时变改建和垂直偏差。验收：所有高程能追溯到同一基准；不凭一个经验常数改全图。
3. 一次集中补测关键点：沿已选道路同步测雨水口格栅/堰口、沟底、路缘上下边和缺口；优先 K1/K3 的窄边缘、K2 的树下地物身份、K4 的分隔带顶高及连续性，再查洞口门槛/挡墙/地下连通与局部泵。验收：控制水流的边界和开口可定位、可量高。普通道路已支持的连续表面保留，只更新测到的局部对象。
4. 建立最小水力对象：地表使用原 DEM 加有依据的局部边界；地下用节点—管段的网络关系及设施参数表达，不需要全城精细管道实体三维模型。雨水口位置、开口高程、管底高程和交换参数分开存。下游泵、蓄水设施只在连接与运行资料确认后纳入。验收：节点连接、管底坡向/高差、地表入口位置、单位、来源及外部边界明确；测量结果与假设分开。
5. 参数缺失时限定成果：可继续做地表形态/流向可行性分析；若将来选择假定排水情景，应标为情景而非现况复原。实际积水深度验证需排水参数和观测积水/水位等独立证据；目前不声称已具备验证条件。未开展新的积水模拟。
高效的关键是先确认高程基准、入口与下游连接，集中补测对水流有控制作用的少数位置；完整的三维外观不是本阶段的验收指标。

五、产物与复现
availability_inventory.csv：设施数据、覆盖、时间、垂直基准及对接状态。
pump_context.csv：明确单位和备用泵口径的候选站资料，不代表实际接入。
verification_and_reconstruction_plan.csv：优先顺序、所需资料及验收门槛。
semis_observations.json 与截图：手工 UI 核查证据，不是已提取的设施数据库。
source_manifest.json：官方 URL、下载状态及 SHA256；本脚本不调用地图内部接口或接受条款。
可在本地终端运行 .venv/bin/python src/exp/audit_hibiya_drainage.py；加 --fetch-sources 仅获取列出的公开静态说明/PDF，已有原始文件不覆盖。动态观察需从官方 UI 复核。原始 DEM 与点云不改动；未修改 AnaSOP，未执行 Git/DVC。
研究工作顺序已记录 MILI-D-20261013-004；项目移动是操作信息，未记为研究决定。

官方来源
SEMIS入口：{urls['semis_entry']}
图例与T.P.说明：{urls['semis_legend']}
操作及PDF说明：{urls['semis_help']}
一般覆盖说明：{urls['semis_availability']}
FY2024年报：{urls['annual_fy2024']}
历史蓄水线索：{urls['historical_storage']}
GSI标高改定：{urls['height_revision']}
千代田区道路台账：{urls['chiyoda_road_ledger']}
'''
    (out/'assessment.txt').write_text(report)
    print(json.dumps({'root':str(root),'source_status':manifest,'observations':len(observations['views']),'inventory_layers':len(inventory),'plan_tasks':len(tasks)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
