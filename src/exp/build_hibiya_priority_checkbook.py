"""First-priority source-epoch verification worksheets; no measurements invented."""
import json,csv
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/exp/data-briefing/hibiya_priority_followup'
OLD=ROOT/'data/exp/data-briefing/hibiya_water_corridor'
OUTPUT=ROOT/'output/pdf/hibiya_first_priority_checkbook.pdf'
pdfmetrics.registerFont(TTFont('STSong-Light','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
pdfmetrics.registerFontFamily('STSong-Light',normal='STSong-Light',bold='STSong-Light',italic='STSong-Light',boldItalic='STSong-Light')
W,H=595.2756,841.8898
STYLE=ParagraphStyle('body',fontName='STSong-Light',fontSize=10.5,leading=15,textColor=colors.HexColor('#253345'))
SMALL=ParagraphStyle('small',parent=STYLE,fontSize=9,leading=12)

def para(c,text,x,y,width,style=STYLE):
 p=Paragraph(text,style);_,h=p.wrap(width,2000);p.drawOn(c,x,y-h);return y-h
def title(c,text,subtitle):
 c.setFillColor(colors.HexColor('#123d57'));c.setFont('STSong-Light',20);c.drawString(40,H-54,text)
 para(c,subtitle,40,H-75,W-80,SMALL)
 c.setStrokeColor(colors.HexColor('#7eabbf'));c.line(40,H-103,W-40,H-103)
def footer(c,page):
 c.setFillColor(colors.HexColor('#526575'));c.setFont('STSong-Light',8)
 c.drawString(40,24,'UDT01b | 2026-10-13 | 待核验记录表，空白项不是已测值')
 c.setFont('STSong-Light',8);c.drawRightString(W-40,24,f'{page} / 4')
def field(c,name,x,y,width,height=18):
 c.acroForm.textfield(name=name,tooltip=name,x=x,y=y,width=width,height=height,borderWidth=.5,borderColor=colors.HexColor('#7994a6'),fillColor=colors.HexColor('#f7fafc'),fontName='Helvetica',fontSize=10,value='')
def blank(c,label,x,y,width):
 c.setFillColor(colors.HexColor('#253345'));c.setFont('STSong-Light',10);c.drawString(x,y,label);c.setStrokeColor(colors.HexColor('#8196a5'));c.line(x+pdfmetrics.stringWidth(label,'STSong-Light',10)+8,y-2,x+width,y-2)

def main():
 OUTPUT.parent.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((OLD/'priority_survey_list.csv').open()));packs=[r for r in rows if r['priority']=='1']
 c=canvas.Canvas(str(OUTPUT),pagesize=(W,H));c.setTitle('日比谷第一优先核查表：历史资料与现场缺口');c.setAuthor('UDT01b')
 title(c,'日比谷第一优先核查表','测绘时期地貌为基准 | S1 道路过渡、S2 心字池南岸、S3 日比谷濠南端')
 y=H-122
 y=para(c,'本阶段先取得已有测量图纸，再补充决定连通性的缺口。候选路径已完成资料筛查，三个位置均未确认街道至水体的真实入口。',40,y,W-80)
 y-=13
 c.setFillColor(colors.HexColor('#edf5f8'));c.roundRect(40,y-88,W-80,88,5,stroke=0,fill=1)
 para(c,'<b>新增线索：日比谷濠历史测量</b><br/>环境省2022年规格要求测石垣天端及道路侧步道，交付平面图、纵横断面和DWG。附图包含南端。现取得的是任务规格书，不是完成的测量成果；具体高程和入口仍未知。',51,y-10,W-102,SMALL)
 y-=106
 image=ImageReader(str(OLD/'corridor_reviewed_overview.png'));iw,ih=image.getSize();height=(W-80)*ih/iw
 c.drawImage(image,40,y-height,width=W-80,height=height,mask='auto')
 y-=height+14
 y=para(c,'办理顺序：先查图纸和控制点资料；确认缺口后补测；取得证据后局部重建断线。历史期与改造后现状分别保存，改造后的高程不能直接覆盖历史DEM。',40,y,W-80,SMALL)
 y-=10
 para(c,'来源：环境省《令和４年度皇居外苑日比谷濠測量業務》规格书（正文1-2页、PDF第6页范围图）；东京官方2024公开DEM/LAS/正射及本项目路径核查。',40,y,W-80,SMALL)
 c.linkURL('https://www.env.go.jp/garden/content/900457723.pdf',(40,49,W-40,70),relative=0,thickness=0)
 footer(c,1);c.showPage()
 expected=[]
 for page,r in enumerate(packs,2):
  ident=r['work_package_id'];ids=r['site_ids'].split('|');image_id=ids[0]
  title(c,ident+' | '+r['location'],'导航锚点，不是已确认入口或最低岸顶；需要核查整个连接走廊')
  para(c,f"经度 {float(r['anchor_longitude']):.6f}°E　纬度 {float(r['anchor_latitude']):.6f}°N<br/>JGD2011 IX：E={float(r['anchor_easting_m']):.2f} m，N={float(r['anchor_northing_m']):.2f} m",40,H-122,W-80)
  y=H-164
  y=para(c,'范围：'+r['suggested_extent'],40,y,W-80,SMALL)
  y-=7
  y=para(c,'已有证据：'+r['existing_evidence'],40,y,W-80,SMALL)
  # Fixed image slot avoids pushing field tables past bottom margin.
  im=ImageReader(str(OLD/f'{image_id}_ortho_ground_profile.png'));iw,ih=im.getSize();ihout=(W-80)*ih/iw
  c.drawImage(im,40,409,width=W-80,height=ihout,mask='auto')
  para(c,'图示红线是源数据剖面方向；现场须延伸到实际车道、水缘及控制地物。水域内回波不作河床或水位。',40,395,W-80,SMALL)
  c.setFont('STSong-Light',10);c.setFillColor(colors.HexColor('#253345'))
  c.drawString(40,350,'观测日期');field(c,ident+'_date',94,345,100)
  c.drawString(211,350,'水位时间');field(c,ident+'_water_time',265,345,120)
  c.drawString(40,322,'控制点/资料号');field(c,ident+'_control_id',123,317,150)
  c.drawString(288,322,'成果版本/日期');field(c,ident+'_datum_epoch',377,317,178)
  expected.extend([ident+'_date',ident+'_water_time',ident+'_control_id',ident+'_datum_epoch'])
  # Blank numeric forms; datum/source notes are written separately below.
  top=297;col1=40;col2=248;col3=365
  c.setFillColor(colors.HexColor('#eaf2f6'));c.rect(40,top-24,W-80,24,fill=1,stroke=0)
  c.setFillColor(colors.HexColor('#253345'));c.drawString(48,top-16,'需记录的地物/参数');c.drawString(col2+7,top-16,'实测值 (m)');c.drawString(col3+7,top-16,'照片/断面/资料编号')
  measure=[('R：车道/沟槽高程','road_z'),('K：路缘上缘高程','kerb_z'),('B：控制岸顶/障碍高程','crest_z'),('I：入口底高程（若存在）','inlet_z'),('W：受纳水位及时间','water_z'),('入口/控制宽度（若存在）','opening_width')]
  for i,(label,key) in enumerate(measure):
   yy=top-24-(i+1)*24;c.setStrokeColor(colors.HexColor('#b7c8d3'));c.line(40,yy,W-40,yy)
   c.setFont('STSong-Light',9.5);c.drawString(48,yy+8,label);field(c,ident+'_'+key,col2+6,yy+3,103,18);expected.append(ident+'_'+key)
   c.line(col3+8,yy+7,W-48,yy+7)
  cy=112;c.setFont('STSong-Light',9);c.drawString(40,cy,'现场识别（可多选，空白表示未记录）：')
  for i,(key,label) in enumerate([('toe','低岸脚'),('crest','岸顶候选'),('inlet','实际入口'),('barrier','阻挡地物'),('unknown','无法判断')]):
   x=40+i*102;c.acroForm.checkbox(name=ident+'_'+key,tooltip=label,x=x,y=86,size=11,checked=False,borderWidth=.6)
   c.setFont('STSong-Light',9);c.drawString(x+16,88,label);expected.append(ident+'_'+key)
  blank(c,'连通证据/高程基准说明：',40,59,W-80)
  blank(c,'观察者/资料日期/改造前后：',40,39,W-80)
  footer(c,page);c.showPage()
 c.save()
 reader=PdfReader(str(OUTPUT));fields=reader.get_fields() or {}
 assert len(reader.pages)==4 and set(expected)==set(fields)
 widgets=[]
 for page in reader.pages:
  for ref in page.get('/Annots',[]):
   a=ref.get_object()
   if a.get('/Subtype')=='/Widget':
    assert a.get('/AP',{}).get('/N') is not None
    assert a.get('/V','') in ['', '/Off']
    widgets.append(str(a.get('/T')))
 assert set(widgets)==set(expected)
 qa={'pages':4,'fields':len(fields),'widgets':len(widgets),'all_fields_blank_or_off':True,'all_widgets_have_normal_appearance':True,'measurements_entered':False,'output':str(OUTPUT.relative_to(ROOT)),'visual_review_pending':True}
 (DATA/'pdf_form_validation.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(qa,ensure_ascii=False))

if __name__=='__main__':main()
