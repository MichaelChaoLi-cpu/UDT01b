"""Group reviewed source-epoch sites into non-duplicated survey work packages."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_hibiya_water_corridor import ROOT,OUT,PROCESSED,projected,feature,savegeo,writecsv
from shapely.geometry import shape

PACKAGES=[
 ('S1',1,['A10'],'共用道路—公园过渡','以A10为锚点约20–40m范围，延伸到真实车行道和公园地面','连续横断面；路缘顶/底、沟槽、人行道、围栏/绿篱底；降缘/开口及宽度；能否绕过宽面抬升','宽面约0.7–0.9m高差可见，但没有证明连通的开口'),
 ('S2',1,['A2'],'心字池南岸','从实际道路侧至水缘的完整走廊；至少一条完整横断面，必要时沿最低岸顶延长','最低连续岸顶、地面隆起、台阶、入口底/顶高程、开口宽度、池水位及其基准','向池边降低，但局部有约2.25m隆起，树冠遮挡岸缘，入口未证实'),
 ('S3',1,['A3','A8'],'日比谷濠南端角部','合并约30m岸缘，至少两条不同方向横断面；A3/A8合并一次核查','实际岸顶断线和道路路缘；若有口门/管口则测底/顶、宽度、连接和控制；濠水位','道路侧低值不能代替岸顶，岸缘有约2.1–2.2m隆起；自动1.95m漏掉边缘高点'),
 ('S4',2,['A9'],'心字池东侧陡坎','约40m横断面，必须跨越外侧高陡坎，不能在近岸低台地结束','近岸低台地与外侧陡坎/石垣的连续断线；是否存在贯通开口及其阈值','近岸1.35m不能作街道入口，外侧仍有约5.8–6.0m陡坎'),
 ('S5',2,['A7','A6'],'雲形池东南岸及路径高点','沿岸顶候选测连续高程，并在A6复核路径高点','岸顶/低口断线、真实入口及宽度、路径高点、池水位；与约2.58m道路源点的关系','约4.20m岸缘和约4.07m路径高点使其成为较低优先级受纳候选'),
 ('S6',3,['A1'],'P3树下道路过渡','约30–40m道路到公园横断面，按实际铺装边缘调整','路缘顶/底、围栏/绿篱底、真实开口；与附近地下出入口区分','树冠遮挡，不能由裸地DEM较平就确认可通过'),
]

def main():
    sites={f['properties']['candidate_id']:f for f in json.loads((PROCESSED/'reviewed_priority_sites.geojson').read_text())['features']}
    rows=[];features=[]
    for ident,priority,ids,name,extent,params,evidence in PACKAGES:
        main=sites[ids[0]];p=main['properties'];g=projected(shape(main['geometry']))
        r={'work_package_id':ident,'priority':priority,'site_ids':'|'.join(ids),'location':name,'anchor_longitude':p['longitude'],'anchor_latitude':p['latitude'],'anchor_easting_m':p['easting_m'],'anchor_northing_m':p['northing_m'],'suggested_extent':extent,'measure_or_verify':params,'existing_evidence':evidence,'confirmed_hydraulic_inlet':False,'height_reference_requirement':'明确当地T.P.控制点及高程成果版本；记录测量/水位日期和局部重复或闭合检查','historical_epoch_requirement':'历史重建优先找改造前测量、竣工/施工图及照片；改造后现场补测独立用于现状，不直接覆盖历史DEM','evidence_images':'|'.join(str((OUT/f'{i}_ortho_ground_profile.png').relative_to(ROOT)) for i in ids),'anchor_note':'导航核查锚点，不是已测入口或确定的最低岸顶'}
        rows.append(r);features.append(feature(g,r))
    writecsv(OUT/'priority_survey_list.csv',rows);savegeo(PROCESSED/'priority_survey_list.geojson',features)
    (OUT/'priority_survey_list.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print('Saved',len(rows),'survey work packages; first priority',sum(r['priority']==1 for r in rows))

if __name__=='__main__':main()
