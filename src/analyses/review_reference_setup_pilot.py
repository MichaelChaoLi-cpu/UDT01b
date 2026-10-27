"""Report the bounded primary-grid pilot and current execution index without training release."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,csv,html,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_setup_pilot import OUT,REPORT,CONTROLS,PILOT,BASE
from prepare_sfincs_cpu_build import dump
from load_three_ward_minimum_scenarios import find_case,load_forcing
from audit_three_ward_negative_volumes import readnp,csvwrite
from compare_adaptive_positivity import wquantile
from advance_reference_setup_pilot import update_overlay

def table(rows):
 if not rows:return ''
 keys=list(rows[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in keys)+'</tr>'for r in rows)+'</table></div>'
def main():
 out=ROOT/OUT;report=ROOT/REPORT;update_overlay();plan=json.loads((out/'authorized_plan.json').read_text());overlay=json.loads((out/'execution_overlay.json').read_text());rows=[];details={};coverage=[];inventory=[];variables=[]
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];ward=support['Ward Code'];scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3)}
 for phase,cases in [('qualification',CONTROLS),('pilot',PILOT)]:
  for c in cases:
   a=json.loads((report/phase/c/'audit.json').read_text());f=load_forcing(c);case=find_case(c);s=a['summary'];details[c]=a
   with(report/phase/c/'label_support.csv').open()as stream:cov=list(csv.DictReader(stream));coverage+=cov
   tw=next(r for r in cov if r['scope']=='Three wards');tot=float(np.trapezoid(f['Physical Rainfall Rate'],f['Event Time'])/3600)
   rows.append({'phase':phase,'case':c,'split':a['split'],'shape':case['parameters']['shape_class'],'rain_peak_mm_h':float(np.max(f['Physical Rainfall Rate'])),'rain_total_mm':tot,'stage_max_m':float(np.max(f['River Stage Increment By Group'])),'all_checks_passed':a['all_checks_passed'],'original_policy_state':a['admission']['state'],'negative_cells':s['negative_cells'],'rain_error_m3':s['rain_difference_m3'],'max_budget_residual_m3':s['max_budget_residual_m3'],'three_ward_certain_wet_km2':float(tw['certain_peak_wet_gt0_1m_km2']),'three_ward_admitted_ground_km2':float(tw['Admitted_km2']),'three_ward_quarantine_km2':float(tw['Quarantine_km2']),'runtime_minutes':s['runtime_s']/60})
   labels=readnp(out/phase/(c+'_candidate_reference.npz'))
   for cv in cov:
    wet=scopes[cv['scope']]&(labels['Peak Mean Depth Lower']>.1);wetarea=float(area[wet].sum());qarea=float(area[wet&labels['Quarantine Mask']].sum());widearea=float(area[wet&labels['Wide Interval Mask']].sum());okarea=float(area[wet&labels['Admitted Mask']].sum());cv.update({'quarantined_certain_wet_km2':qarea/1e6,'wide_certain_wet_km2':widearea/1e6,'admitted_certain_wet_km2':okarea/1e6,'quarantined_certain_wet_pct':qarea/wetarea*100 if wetarea else None})
   rows[-1]['three_ward_quarantined_certain_wet_pct']=tw['quarantined_certain_wet_pct'];rows[-1]['three_ward_admitted_certain_wet_km2']=tw['admitted_certain_wet_km2']
   for name,v in [('Peak Mean Depth Lower',labels['Peak Mean Depth Lower']),('Peak Mean Depth Upper',labels['Peak Mean Depth Upper']),('Peak Interval Width',labels['Peak Mean Depth Upper']-labels['Peak Mean Depth Lower'])]:
    good=valid&np.isfinite(v);variables.append({'Case':c,'Readable Variable':name,'Eligible Count':int(valid.sum()),'Nonmissing Count':int(good.sum()),'Minimum':float(np.min(v[good])),'Area Weighted Median':wquantile(v[good],area[good],.5),'Area Weighted 95th Percentile':wquantile(v[good],area[good],.95),'Maximum':float(np.max(v[good])),'Final Variable Status':'no'})
   inventory.append({'phase':phase,'case':c,'eligible_cells':int(valid.sum()),'finite_lower':int(np.isfinite(labels['Peak Mean Depth Lower'][valid]).sum()),'finite_upper':int(np.isfinite(labels['Peak Mean Depth Upper'][valid]).sum()),'snapshots':s['map_records'],'grid_m':10,'actual_clocks':True,'training_release':False})
 csvwrite(report/'case_summary.csv',rows);csvwrite(report/'label_support.csv',coverage);csvwrite(report/'response_inventory.csv',inventory)
 brief=ROOT/'data/exp/data-briefing/reference_setup_pilot';brief.mkdir(parents=True,exist_ok=True);csvwrite(brief/'response_inventory.csv',inventory);csvwrite(brief/'variable_summary.csv',variables);csvwrite(brief/'skipped_files.csv',[{'scope':'Previously briefed raw and shared input sources','reason':'Unchanged sources were hash-verified; this briefing inspects only the newly authorized response packages.'}])
 primary=json.loads((report/'primary_grid_qualification.json').read_text());mesh=[]
 with(report/'mesh_comparison/matched_extent.csv').open()as f:
  for r in csv.DictReader(f):
   if r['scope']=='Three wards'and r['support']=='all_eligible_ground'and float(r['threshold_m'])==.1:mesh.append(r)
 independent={find_case(c)['parent_id']for c in PILOT if find_case(c)['independent_parent_candidate']};eligible=[c for c in PILOT if details[c]['all_checks_passed']and details[c]['admission']['technical_reference']];held=[c for c in PILOT if c not in eligible]
 summary={'primary_grid_qualification_passed':primary['all_checks_passed'],'qualification_controls':CONTROLS,'pilot_cases':PILOT,'new_pilot_responses_completed':8,'new_independent_parent_candidates':len(independent),'new_open_control_responses':2,'numerically_eligible_pilot_cases':eligible,'held_pilot_cases':held,'remaining_prospective_runs':50,'remaining_regular_runs':42,'remaining_adaptive_runs':8,'sealed_test_and_challenge_responses_generated':0,'training':0,'formal_reference_release':False,'real_world_accuracy_verified':False,'formal_grid_convergence_verified':False,'matched_extent':mesh,'case_summaries':rows};dump(report/'summary.json',summary);dump(report/'reference_admission_assessment.json',{c:{'required_gates':a['required_gates'],'preview':a['admission'],'all_checks_passed':a['all_checks_passed']}for c,a in details.items()})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(2,4,figsize=(16,10),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32]
 for ax,c in zip(axs.ravel(),PILOT):
  d=readnp(out/'pilot'/(c+'_candidate_reference.npz'));lo=d['Peak Mean Depth Lower'];ax.imshow(np.where(valid,1,np.nan),extent=extent,cmap=ListedColormap(['#e7ecee']),interpolation='none');im=ax.imshow(np.where(valid&(lo>.01),lo,np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1.5,interpolation='none');ax.set_title(c,fontsize=10)
  for code in [1,2,3]:ax.contour(-11.36+(np.arange(770)+.5)*.01,-32.32-(np.arange(1012)+.5)*.01,(ward==code).astype('f4'),levels=[.5],colors='#6c8088',linewidths=.4)
  ax.set_aspect('equal');ax.set_xlabel('Easting (km)');ax.set_ylabel('Northing (km)')
 fig.colorbar(im,ax=list(axs.ravel()),label='Common10m peak mean depth lower bound (m)',shrink=.65,extend='max');fig.savefig(report/'pilot_whole_domain.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(8);allcov=[]
 for c in PILOT:
  d=readnp(out/'pilot'/(c+'_candidate_reference.npz'));sel=valid&(ward>0);v=d['Peak Mean Depth Lower'][sel];w=area[sel];order=np.argsort(v);axs[0].plot(v[order],np.cumsum(w[order])/w.sum(),label=c,lw=1);tw=next(r for r in coverage if r['case']==c and r['scope']=='Three wards');allcov.append(tw)
 axs[0].set_xlim(0,1.5);axs[0].set_xlabel('Peak mean depth lower bound (m)');axs[0].set_ylabel('Three-ward eligible-ground area fraction');axs[0].legend(fontsize=7);bottom=np.zeros(8)
 for name,col in [('Exact_Label','#357c88'),('Interval_Label','#9dcbd0'),('Wide_Interval','#e8b77b'),('Quarantine','#b17e8a')]:
  values=np.array([float(r[name+'_km2'])for r in allcov]);axs[1].bar(x,values,bottom=bottom,label=name.replace('_',' '),color=col);bottom+=values
 axs[1].set_ylim(0,float(np.max(bottom))*1.08);axs[1].set_xticks(x,[c.replace('_current_geometry','')for c in PILOT],rotation=60,ha='right',fontsize=8);axs[1].set_ylabel('Three-ward eligible ground (km2)');axs[1].legend(fontsize=8);fig.savefig(report/'pilot_depth_and_support.png',dpi=150);fig.savefig(brief/'pilot_depth_and_support.png',dpi=150);plt.close(fig)
 notes='本轮只运行2份既有全域10 m控制及8份预选的规则10 m试运行。八份新响应包含6个新父事件候选与2个同属既有控制谱系的控制，不是8个独立暴雨。没有生成试运行的加密伴随组，也没有生成封存测试或挑战响应。原58份计划保留为历史快照，当前执行状态见新的覆盖索引；剩余42份规则网格与8份加密响应未运行。'
 limits='沿用原事件门槛及精确、窄区间、宽区间、建筑接触和负体积隔离规则；保留上下界，不用中点或零值填补。数值准入仅支持模型条件下的候选主参考标签，默认注册表、训练入口和学习归一化继续关闭。六个父事件不足以建立泛化结论；水位为给定情景，非完整河道水动力。峰值图不是同时水面或动态排水路径，尚未验证实测精度及正式网格收敛。'
 (report/'README.md').write_text('# 全域10 m资格核验与八份参考库试运行\n\n'+notes+'\n\n两组主网格资格检查：'+str(primary['all_checks_passed'])+'。八份试运行数值通过：'+str(len(eligible))+'；保留待审：'+str(held)+'。\n\n'+limits+'\n\nFramelog: MILI-D-20261015-017.\n')
 (brief/'README.md').write_text('# Reference response pilot inventory\n\nInspected ten new versioned response packages: two legacy-control replays and eight preselected pilot runs. Each package retains 49 full-field snapshots, actual-step and output clocks, native online peaks and common10m candidate depth bounds with original quality masks. Six new parent candidates and two open controls form the pilot; no sealed or challenge responses, learned scaling or training are generated. The distribution figure summarizes three-ward eligible ground only, not observations or generalization performance. Raw sources and the frozen input catalog are unchanged. No variables are selected or renamed.\n')
 display=[{'阶段':r['phase'],'输入':r['case'],'分区':r['split'],'峰值雨强 mm/h':round(r['rain_peak_mm_h'],2),'水位抬升 m':round(r['stage_max_m'],3),'数值检查':r['all_checks_passed'],'原政策预览':r['original_policy_state'],'三区 >0.10 m km²':round(r['three_ward_certain_wet_km2'],4),'三区可用地面 km²':round(r['three_ward_admitted_ground_km2'],4),'积水支持隔离 %':round(r['three_ward_quarantined_certain_wet_pct'],2)if r['three_ward_quarantined_certain_wet_pct']is not None else 'N/A'}for r in rows]
 (report/'execution_overlay.json').write_bytes((out/'execution_overlay.json').read_bytes())
 links=['provenance.json','execution_overlay.json','summary.json','case_summary.csv','label_support.csv','reference_admission_assessment.json','response_inventory.csv','primary_grid_qualification.json','mesh_comparison/matched_extent.csv','mesh_comparison/boundary_group_comparison.csv','README.md']
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区主参考模型与小规模情景试运行</title><style>body{font:16px/1.8 system-ui;background:#edf2f4;color:#203c48;margin:0}main{max-width:1350px;margin:auto;padding:28px}section{background:white;border-radius:12px;padding:22px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dfe6e9;text-align:left}img{width:100%}a{color:#146c81}</style><main><p>Mili · Whole-three-ward reference setup</p><h1>全域10 m主参考核验与八份情景试运行</h1><p>'+notes+'</p><section><h2>输入、数值结果与标签准入</h2>'+table(display)+'</section><section><h2>同一修正求解器的10 m与10/5 m对照</h2>'+table(mesh)+'<img src="mesh_comparison/whole_domain_grid_comparison.png" alt="两种网格的全域对照"><p>只复用既有修正加密结果；网格变体不增加父事件。范围相近不等于局部水深或动态路径收敛。</p></section><section><h2>八份试运行的全域峰值</h2><img src="pilot_whole_domain.png" alt="八份预选输入的全域峰值深度下界"></section><section><h2>三区深度分布与标签支持</h2><img src="pilot_depth_and_support.png" alt="三区面积加权深度分布及原质量掩膜覆盖"><p>'+limits+'</p></section><section><h2>复核材料</h2><p>'+' · '.join('<a href="'+p+'">'+p+'</a>'for p in links)+'</p><p>Framelog MILI-D-20261015-017</p></section></main></html>'
 (report/'review.html').write_text(page);print(json.dumps({'qualified_pilot':len(eligible),'held':held,'remaining':50,'training':0}))
if __name__=='__main__':main()
