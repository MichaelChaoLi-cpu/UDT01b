from run_surface_trial import *
from rasterio.transform import Affine
import hashlib
config=json.loads((OUT/'model_assumptions.json').read_text());validation=json.loads((OUT/'numerical_validation.json').read_text());audit=json.loads((OUT/'input_audit.json').read_text());a=dict(np.load(PRO/'surface_inputs.npz'));tf=Affine(*audit['transform']);outputs={}
for key in ['A','B','C','D']:
 r=np.load(PRO/f'{key}_result.npz');outputs[key]={'peak':r['peak'],'final':r['final'],'flow_east':r['flow_east_m3'],'flow_north':r['flow_north_m3']}
checks=numerical_checks();validation['additional_checks']=checks
fine=np.load(PRO/'C_timestep_check_1_25s.npz')['peak'];delta=np.abs(outputs['C']['peak']-fine);sensitive=np.isfinite(delta)&(delta>.1);validation['timestep_comparison']['cells_with_peak_difference_over0_10m']=int(sensitive.sum());validation['timestep_comparison']['local_depth_convergence_claim']=False;jswrite(OUT/'numerical_validation.json',validation)
with rasterio.open(PRO/'terrain_20m.tif') as d:
 meta=d.meta.copy();meta.update(dtype='uint8',nodata=0)
 for name,v in [('C_timestep_sensitive',sensitive),('deep_model_suspect',np.any(np.array([outputs[k]['peak']>2 for k in outputs]),axis=0))]:
  with rasterio.open(PRO/f'{name}.tif','w',**meta) as ds:ds.write(v.astype('uint8'),1)
summ=[]
for key in outputs:
 peak=outputs[key]['peak']
 for w,name in [(1,'千代田区'),(2,'中央区'),(3,'港区')]:
  for interior in [False,True]:
   k=a['land']&(a['ward']==w)&(~a['edge_uncertain'] if interior else True);summ.append({'scenario':key,'ward':name,'edge_zone_excluded':interior,'land_area_km2':float(k.sum()*DX*DX/1e6),'screen_area_peak_ge_0_10m_km2':float(np.sum(k&(peak>=.1))*DX*DX/1e6),'screen_area_peak_ge_0_30m_km2':float(np.sum(k&(peak>=.3))*DX*DX/1e6),'p95_peak_depth_m':float(np.quantile(peak[k],.95)) if k.any() else None,'max_peak_depth_m':float(peak[k].max()) if k.any() else None})
csvwrite(OUT/'ward_scenario_summary.csv',summ);hydros=list(csv.DictReader((OUT/'volume_balance_timeseries.csv').open(encoding='utf-8-sig')))
for r in hydros:
 for key in r:
  if key!='scenario':r[key]=float(r[key])
hotspots=list(csv.DictReader((OUT/'screening_hotspot_candidates.csv').open(encoding='utf-8-sig')))
for r in hotspots:
 rr,cc=rasterio.transform.rowcol(tf,float(r['representative_x']),float(r['representative_y']));r['C_timestep_sensitive_at_representative']=bool(sensitive[rr,cc]) if r['scenario']=='C' else 'not_checked';r['depth_status']='not_validated'
csvwrite(OUT/'screening_hotspot_candidates.csv',hotspots)
plots(outputs,summ,hydros,a,audit,tf);report(config,validation,summ,audit,hotspots)
# Label evidence quality explicitly in the entry page.
p=OUT/'review.html';s=p.read_text().replace('黑格代表>2m的模型疑义结果，不能读作真实洪深。','黑格代表>2m的模型疑义结果，不能读作真实洪深。C情景减半时间步后，>=0.30m面积变化约2%，但局部峰值差仍可达1.66m；局部深度尚未收敛。');s=s.replace('<a href="numerical_validation.json">数值验证</a>','<a href="numerical_validation.json">数值验证</a> · <a href="input_coverage_validation.json">地形覆盖检查</a> · <a href="../../../processed/three_ward_surface_trial/C_timestep_sensitive.tif">C时间步敏感格</a>');p.write_text(s)
# Archive compact diagnostics, with no claims of surveyed flood locations.
files=[PRO/'surface_inputs.npz',PRO/'building_roof_proxy_5m.tif',PRO/'road_mask_5m.tif',ROOT/'src/exp/prepare_surface_trial.py',ROOT/'src/exp/run_surface_trial.py'];jswrite(OUT/'reproducibility_manifest.json',[{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]);print('FINALIZED',validation['timestep_comparison'],flush=True)
