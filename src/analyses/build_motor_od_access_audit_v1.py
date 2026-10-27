"""Build an offline, evidence-bound audit. No application or API request is sent.

Use --stage-temp-downloads once to stage previously downloaded official materials;
subsequent runs use project-local caches only. No network code is included.
"""
from pathlib import Path
import argparse, csv, hashlib, html, io, json, shutil, zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

NAME = 'three_ward_motor_od_access_audit_v1'
ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'data/processed' / NAME
R = ROOT / 'data/exp/data-briefing' / NAME
S = P / 'sources'
SOURCES = [
 ('r3_vehicle_od_prefecture.xlsx', 'https://www.e-stat.go.jp/stat-search/file-download?statInfId=000040170611&fileKind=0', '/private/tmp/udt01b-r3_vehicle_od_prefecture.xlsx'),
 ('estat_table_page.html', 'https://www.e-stat.go.jp/stat-search/files?cycle=8&layout=datalist&page=1&stat_infid=000040170611&tclass1=000001216542&tclass2=000001216544&tclass3val=0&toukei=00600580&tstat=000001116435', '/private/tmp/udt01b-estat-od.html'),
 ('xroad_public_od_api.zip', 'https://www.xroad.mlit.go.jp/img/article/4a2b08d5e215ba4c13cade9fea833bba36cf18d7.zip', '/private/tmp/udt01b-xroad-api.zip'),
 ('transport_guide_detail_2022.pdf', 'https://www.mlit.go.jp/sogoseisaku/transport/content/001480548.pdf', '/private/tmp/udt01b-transport-guide3.pdf'),
 ('transport_guide_detail_2022.txt', 'https://www.mlit.go.jp/sogoseisaku/transport/content/001480548.pdf', '/private/tmp/udt01b-transport-guide3.txt'),
]
LINKS = {
 'survey': 'https://www.mlit.go.jp/statistics/details/t-other-2_tk_000384.html',
 'tables': SOURCES[1][1],
 'xroad': 'https://www.xroad.mlit.go.jp/database/232/',
 'viewer': 'https://view.xroad.mlit.go.jp/',
 'guide': SOURCES[3][1],
 'disclosure': 'https://www.mlit.go.jp/appli/file000011.html',
 'ktr': 'https://www.ktr.mlit.go.jp/sinsei/sinsei00000003.html',
}
def dump(path, obj):
 path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
def writecsv(path, rows):
 with path.open('w', newline='', encoding='utf-8-sig') as f:
  w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def read_xlsx(path):
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(path) as z:
  shared=[]
  if 'xl/sharedStrings.xml' in z.namelist():
   shared=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  rows=[]
  for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns):
   vals={}
   for c in row.findall('s:c',ns):
    v=c.find('s:v',ns)
    text=v.text if v is not None else ''
    if c.get('t')=='s': text=shared[int(text)]
    if c.get('t')=='inlineStr': text=''.join(c.find('s:is',ns).itertext())
    col=''.join(x for x in c.get('r') if x.isalpha()); vals[col]=text
   if vals.get('A') in ('1','2') and vals.get('C') in ('01','02','10') and vals.get('E'):
    rows.append([vals.get(c,'') for c in 'ABCDEFGHI'])
 return rows

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage-temp-downloads', action='store_true'); args=ap.parse_args()
 S.mkdir(parents=True,exist_ok=True); R.mkdir(parents=True,exist_ok=True)
 if args.stage_temp_downloads:
  for name,url,temp in SOURCES:
   target=S/name
   if target.exists(): assert sha(target)==sha(Path(temp)), name
   else: shutil.copyfile(temp,target)
 assert all((S/n).is_file() for n,_,_ in SOURCES)
 # Inspect unfilled official application form and public API definitions; no submission.
 with zipfile.ZipFile(S/'xroad_public_od_api.zip') as z:
  members=z.namelist()
  formname=next(n for n in members if n.endswith('.docx'))
  with zipfile.ZipFile(io.BytesIO(z.read(formname))) as doc:
   x=ET.fromstring(doc.read('word/document.xml'))
   formtext='\n'.join(''.join(p.itertext()) for p in x.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))
  (S/'xroad_api_blank_form_text.txt').write_text(formtext)
 rows=read_xlsx(S/'r3_vehicle_od_prefecture.xlsx')
 keys=[tuple(r[i] for i in (0,2,4,6)) for r in rows]
 real=[r for r in rows if r[4]!='100' and r[6]!='100']
 tokyo=[r for r in real if r[4]=='013' or r[6]=='013']
 header=['day_code','day_name','vehicle_code','vehicle_name','origin_code','origin_name','destination_code','destination_name','trips_per_day']
 writecsv(P/'tokyo_prefecture_reference_only.csv',[dict(zip(header,r)) for r in tokyo])
 inspection={
  'survey_year':2021,'public_release_at':'2024-03-28 14:00 (e-Stat page)',
  'unit':'vehicle trips/day; not persons, distinct vehicles or simultaneous traffic',
  'source_rows_including_national_totals':len(rows), 'prefecture_pair_rows_excluding_totals':len(real),
  'tokyo_related_rows_excluding_totals':len(tokyo),
  'day_types': sorted({r[1] for r in rows}), 'vehicle_types':sorted({r[3] for r in rows}),
  'joint_hourly_od':False,'ward_or_bzone_fields':False,
  'caution':'National totals and vehicle total rows must not be added to their components; Tokyo is not the three wards.'}
 dump(P/'public_table_inspection.json',inspection)
 findings=[
  dict(topic='Public 2021 vehicle OD',status='confirmed_downloaded',finding='e-Stat: prefecture OD x weekday/holiday x passenger/goods/total; trips/day; no B-zone or hourly OD.',source=LINKS['tables']),
  dict(topic='Public XROAD OD API',status='confirmed_documented',finding='Official public API v4.0 dated2026-06-10 lists prefecture layers and key authentication; form asks global IP and organization. Form is not proof of individual eligibility or B-zone access.',source=LINKS['xroad']),
  dict(topic='B-zone use in vehicle survey',status='confirmed_design_only',finding='Official survey design names B-zone generation margins for passenger/small goods/ordinary goods. It does not supply Tokyo B-zone boundaries or promise joint OD-cell accuracy.',source=LINKS['survey']),
  dict(topic='Detailed OD permission route',status='confirmed_generic_route',finding='Official2022 transport guide printedp90 says apply to regional development bureau for owner-master OD. It references2015 census; not an R3-specific personal eligibility approval.',source=LINKS['guide']),
  dict(topic='Individual disclosure requests',status='confirmed_right_not_release',finding='MLIT permits individuals to request existing administrative documents. This is distinct from approval for confidential statistical microdata and does not guarantee disclosure.',source=LINKS['disclosure']),
  dict(topic='R3 Tokyo B-zone deliverable',status='unresolved',finding='No confirmed publicly downloadable R3 Tokyo B-zone OD/geometry identified in audited official channels. Individual eligibility, exact zones, field intersections, fees and usage conditions need provider confirmation.',source=LINKS['ktr']),
 ]
 writecsv(P/'access_findings.csv',findings)
 request_scope={
  'preferred':'Existing aggregated actual2021 B-zone vehicle OD tables, not personal owner records or forecast OD',
  'geography':'Core three wards plus model gateway catchment; retain external-origin/destination aggregates and within-zone trips. Exact area follows provider zone geometry, not assumed PT-zone correspondence.',
  'minimum_fields':['origin B-zone','destination B-zone','expanded trips/day','weekday/holiday','vehicle class','within-zone trips','zone code/name/boundary/centroid','expansion, suppression and missing-value definitions'],
  'optional_if_existing':['OD x departure-time band','purpose','highway usage','sampling or reliability indicators'],
  'provider_questions':['Individual unaffiliated researcher eligibility','Correct procedure and contact','Actual2021 vs forecast data','Existing aggregation products and geometry availability','Goods classes and buses/taxis mapping','Coverage of external/through movements','License/publication/reproducibility rules','Fees and delivery time'],
  'fallback':'If fine-zone access is unavailable, retain2018 PT spatial priors and explicit vehicle/through-trip uncertainty; public prefecture OD supplies broad totals only. Do not fabricate detailed observed OD.',
  'action_state':'Prepared only; no inquiry sent, no application submitted, no key requested.'}
 dump(P/'proposed_request_scope.json',request_scope)
 inquiry='''件名：令和3年度自動車起終点調査のBゾーン別集計ODデータについて（個人研究者からの事前照会案・未送信）

道路経済調査室 ご担当者様

東京都千代田区・中央区・港区と周辺接続道路を対象に、極端降雨による道路通行支障と交通への影響を研究するため、令和3年度全国道路・街路交通情勢調査 自動車起終点調査の既存集計データの利用可否を確認したく存じます。

希望するのは、将来推計ではなく令和3年度現況のBゾーン別自動車OD集計表とゾーン定義資料です。個人を識別する所有者情報や未集計の調査票情報は希望していません。

1. 所属機関を持たない個人研究者が利用できる既存の集計表はありますか。利用許可申請・統計データ提供・行政文書開示等のうち、適切な手続と窓口をご教示ください。
2. 発着Bゾーン、平休日、車種別の拡大済みトリップ数（台トリップ/日）、ゾーン内々交通、および区域外との交通を含む表は提供可能ですか。乗用車、小型貨物車、普通貨物車の区分、タクシー・バス等の扱いも確認したく存じます。
3. 対象地域と必要な周辺・外部ゾーンを特定するため、東京都及び関東地方のBゾーンコード表、境界図/GIS、代表点及び対応表を閲覧又は取得できますか。既存PTゾーンと同一とは仮定していません。
4. 時間帯別OD、目的別OD、高速道路利用別ODが既存の集計表として存在する場合、その区分と提供可否をご教示ください。必要な交差集計が存在しない場合、新規集計の作成を前提とはしません。
5. オーナーマスター由来とOD集計用マスター由来の違い、外部交通・通過交通・拡大係数、秘匿/欠測/ゼロ値の定義について、説明資料はありますか。
6. 利用条件、成果公表・派生データ共有の制限、申請様式、費用、提供時期をご教示ください。

所管が異なる場合は、関東地方整備局等の担当窓口をご案内いただけますと幸いです。

［送信時に本人が氏名・連絡先を記入。未署名・未送信。］
'''
 (P/'inquiry_draft_ja_unsent.txt').write_text(inquiry)
 manifest=[]
 for n,u,_ in SOURCES:
  f=S/n;manifest.append(dict(file=n,url=u,bytes=f.stat().st_size,sha256=sha(f)))
 dump(P/'source_manifest.json',dict(checked_date='2026-10-18 JST',sources=manifest,
  transport_note='Public downloads used requests verify=False due local CA failure; hashes bind local bytes, not server authenticity. Official source content independently read via web tool. Derived text uses pdftotext -layout.',
  no_credentials=True, no_personal_information_submitted=True))
 checks={
  'complete_2_days_3_classes_48_by_48_including_national':len(rows)==2*3*48*48,
  'unique_source_keys':len(keys)==len(set(keys)),
  'complete_47_by_47_excluding_totals':len(real)==2*3*47*47,
  'all_trips_nonnegative_numeric':all(float(r[8])>=0 for r in rows),
  'tokyo_extract_rows':len(tokyo)==2*3*(47*2-1),
  'api_form_requires_global_ip':'グローバルIP' in formtext,
  'api_form_unfilled_template':'申請者氏名：XX' in formtext,
  'permission_route_evidence_present':'許可申請を提出し' in (S/'transport_guide_detail_2022.txt').read_text(),
 }
 assert all(checks.values()), checks
 dump(R/'validation.json',dict(checks=checks,all_passed=True,meaning='Evidence and download integrity only; not traffic baseline acceptance or detailed OD access approval.'))
 link=lambda k,t:f'<a href="{html.escape(LINKS[k])}">{t}</a>'
 tab=''.join('<tr><td>'+html.escape(f['topic'])+'</td><td>'+html.escape(f['status'])+'</td><td>'+html.escape(f['finding'])+'</td></tr>' for f in findings)
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2021机动车细分区OD：获取渠道核查</title><style>body{{font:16px/1.8 system-ui,sans-serif;color:#233e4b;background:#edf2f5;margin:0}}main{{max-width:1080px;margin:28px auto;padding:30px;background:white;border-radius:14px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:28px}}.lead{{background:#fff0dd;padding:20px;border-left:5px solid #b9833a}}a{{color:#146587}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{main{{padding:18px;margin:0}}table{{min-width:750px}}}}</style><main>
<p>2026-10-18 JST · 都心三区机动车需求数据 · MILI-D-20261018-008</p>
<h1>细分区机动车OD有申请线索，个人获取资格尚未确认</h1>
<div class="lead">已下载并检查2021年公开机动车OD表和XROAD公开API资料。它们的空间单位都是都道府县，不能直接补齐三区内部OD。官方手引提供详细OD申请路径，但本次未获得R3东京Bゾーン数据，也没有收到个人研究资格批准。</div>
<h2>已下载的表实际有什么</h2>
<p>e-Stat表1：{len(rows):,}条含全国合计的记录，剔除全国合计后{len(real):,}条。维度为平日／休日 × 乘用车／货车／合计 × 47个出发都道府县 × 47个到达都道府县。单位是车辆出行次数／日，既不是人数，也不是独立车辆数或路上同时车辆数。东京相关参考提取{len(tokyo)}行，仍含车型合计，不能重复求和。没有区级、Bゾーン或小时OD字段。{link('tables','官方表页')}</p>
<p>XROAD公开API说明第4.0版（2026-06-10）列出的也是都道府县层，另分乘用、小型货物、普通货物和合计。API采用密钥认证，申请表要求预计公网IP与所属信息；这不证明个人无单位一定可申请，更不代表密钥可取得Bゾーン。公开表1的货车合并口径与API三车型口径需区分。未申请密钥、未调用需授权接口。{link('xroad','官方目录及API包')}</p>
<h2>可以确认的申请路径，以及不能确认的部分</h2>
<p>官方2022年《地域公共交通計画等の作成と運用の手引き》详细编印刷页90说明，获取オーナーマスターOD需向地方整备局提交许可申请。该段以2015年道路普查为背景，能够证明一般渠道，不能据此确认2021年数据、个人资格、具体字段、费用或授权结果。{link('guide','官方手引PDF')}</p>
<p>国交省信息公开制度允许个人请求已有行政文书，关东地方整备局有对应流程；这与统计微观数据许可不同，也不意味着必然公开。正式开示请求通过邮寄、现场或在线提交，不能以普通邮件或FAX代替。应先咨询已有Bゾーン汇总表的正式名称和保管部门，再确定手续。{link('disclosure','个人请求资格说明')} · {link('ktr','关东地方整备局流程')}</p>
<p>Bゾーン出现在官方抽样精度设计中，但本次未取得东京区划文件，不能声称它与PT基本区相同或保证细到街区。95%／相对误差20%的设计描述针对有关发生成分，不是每个细OD单元的精度承诺。{link('survey','调查概要')}</p>
<h2>建议先咨询这份最小数据清单</h2>
<p>优先请求<strong>现成的2021现况Bゾーン汇总机动车OD和区划说明</strong>：发着区、拡大后台次／日、平休日、车型、区内出行、外部交通；附代码、边界GIS或图、代表点、秘匿和扩样规则。若已有OD×时段、目的或高速使用表，再确认可否提供，不能把独立小时边际表当成小时OD。</p>
<p>三区之外要覆盖模型道路入口所需周边，并保留其他外部来源／去向；外部—外部OD仍需路径判断才知道是否穿越研究区。详细车辆OD有助于弥补货运和人次转车次的问题，但没有真实行车路线时，仍不能消除路网／路径支持的结构性误差。2021年与暴雨事件年份不同，需保留需求情景差异。</p>
<p>官方道路经济调查室：03-5253-8487；XROAD目录提供咨询邮箱 hqt-xroad-viewer-contact1@gxb.mlit.go.jp，可先询问负责详细OD的数据窗口。已准备<strong>未发送、未签名</strong>日文咨询稿。API申请表仅保存在官方压缩包中，未填个人资料、未接受申请条款。{link('xroad','联系来源')}</p>
<h2>证据状态明细</h2><div class="scroll"><table><tr><th>主题</th><th>状态</th><th>证据边界</th></tr>{tab}</table></div>
<h2>对当前实验的含义</h2>
<p>当前应先确认数据资格和交付粒度；若不能取得细区OD，继续用已有2018 PT空间先验并显式保留货车、外部穿越和时段假设。公开都道府县表只能作宏观总量参考。上轮正常交通准入失败结论不变；本次没有重新拟合负荷、运行交通或水动力模拟，也没有计算受影响人数。</p>
<p><a href="../../../processed/{NAME}/inquiry_draft_ja_unsent.txt">日文咨询草稿（未发送）</a> · <a href="../../../processed/{NAME}/proposed_request_scope.json">建议数据范围</a> · <a href="../../../processed/{NAME}/sources/r3_vehicle_od_prefecture.xlsx">官方公开OD原表</a> · <a href="../../../processed/{NAME}/public_table_inspection.json">字段核验</a> · <a href="../../../processed/{NAME}/source_manifest.json">下载来源与哈希</a> · <a href="validation.json">核验结果</a></p>
<p>本报告将官方事实、拟申请范围和待确认资格分开；咨询草稿为研究团队建议，不是主管机构的提供承诺。脚本仅离线复现报告与表检查，不包含联网提交操作。</p></main></html>'''
 (R/'review.html').write_text(body)
 dump(R/'execution_state.json',dict(state='audit_complete_access_pending',checked_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),application_submitted=False,inquiry_sent=False,restricted_api_called=False,new_simulations=0,affected_people_calculated=False,decision_id='MILI-D-20261018-008'))
 print(json.dumps(dict(report=str(R/'review.html'),checks=checks,table=inspection),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
