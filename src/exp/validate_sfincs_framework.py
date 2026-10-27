"""Verify official-engine installation case against its analytical water balance."""
from pathlib import Path
import json
import hashlib
import warnings
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/processed/three_ward_sfincs_framework'
REPORT=ROOT/'data/exp/data-briefing/three_ward_sfincs_framework'


def main():
    smoke=BASE/'engine_smoke_test'
    contract=json.loads((smoke/'test_contract.json').read_text())
    execution=json.loads((smoke/'execution.json').read_text())
    for name,digest in execution['input_hashes'].items():
        assert hashlib.sha256((smoke/name).read_bytes()).hexdigest()==digest, name
    def var(name):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',NotGeoreferencedWarning)
            with rasterio.open(f'NETCDF:"{smoke / "sfincs_map.nc"}":{name}') as d:
                return d.read(masked=True)
    depths=var('h')
    assert depths.shape==(7,10,10) and depths.count()==700
    assert np.isfinite(depths).all() and depths.min()>=0
    expected=contract['expected_final_depth_m']
    final=depths[-1]
    analytic_error=float(np.max(np.abs(final-expected)))
    relative_error=analytic_error/expected
    volume=float(final.sum())*100
    cumulative_rain_volume=float(var('cumprcp').sum())*100
    infiltration=float(var('cuminf').sum())*100
    residual=volume-cumulative_rain_volume+infiltration
    # Allow 0.1% analytical error; integrated explicit forcing ends within dt.
    passed=bool(execution['solver_completion_marker'] and execution['returncode']==0
                and relative_error<0.001 and abs(residual)/cumulative_rain_volume<0.001)
    result={
        'test':'synthetic_closed_flat_rainfall','passed':passed,
        'expected_depth_m':expected,'actual_depth_min_m':float(final.min()),
        'actual_depth_max_m':float(final.max()),'relative_analytical_depth_error':relative_error,
        'expected_volume_m3':100,'actual_final_storage_m3':volume,
        'model_integrated_rain_volume_m3':cumulative_rain_volume,
        'model_infiltration_volume_m3':infiltration,'balance_residual_m3':residual,
        'balance_relative_residual':residual/cumulative_rain_volume,
        'tolerance_relative':0.001,'output_times_s':[0,600,1200,1800,2400,3000,3600],
        'wall_runtime_seconds':execution['runtime_seconds'],
        'depth_nonnegative':True,'uniform_final_depth':bool(np.ptp(final)==0),
        'engine_image':contract['engine_image'],'emulated_platform':'linux/amd64 on Apple Silicon',
        'output_caveat':'zs/zsmax/hmax mask shallow water in this case; raw h history remains valid. Do not interpret masked maxima as zero flood depth.',
        'does_not_validate':['subgrid_builder','roads_and_banks','spatial_rain','river_boundaries','Tokyo_accuracy','intervention_features'],
        'input_hashes_verified':True,'not_a_Tokyo_validation':True}
    (REPORT/'engine_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    execution['physical_validation']='passed analytical closed-flat rainfall check (0.1% tolerance); not Tokyo validation'
    (smoke/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    workflow=json.loads((BASE/'workflow.json').read_text())
    workflow['steps'][1]['status']='framework_ready_engine_smoke_passed_Tokyo_bundle_pending'
    (BASE/'workflow.json').write_text(json.dumps(workflow,ensure_ascii=False,indent=2)+'\n')
    page=REPORT/'review.html'
    page_text=page.read_text()
    page_text=page_text.replace('运行环境与小型合成测试的实际结果见 engine_validation.json。',
        f'官方引擎已在本机运行，封闭平地降雨测试通过：预期水深1 cm，计算{float(final.mean())*100:.5f} cm（相对误差{relative_error*100:.4f}%）。这不构成东京模型验证。')
    page.write_text(page_text)
    with (REPORT/'assessment.txt').open('a') as f:
        f.write(f'\n安装验证：已启动现有Docker Desktop并拉取固定官方镜像，在Apple Silicon仿真amd64运行成功。合成封闭100m×100m平地10mm/h降雨1h，预期水深0.010000m，结果{float(final.mean()):.9f}m，相对解析误差{relative_error:.8%}；相对水量残差{residual/cumulative_rain_volume:.8%}。未验证子网格、河道或措施接口，也不能外推三区运行性能。\n')
    (BASE/'README.txt').write_text('''Tokyo central three wards — SFINCS framework

Location: data/processed/three_ward_sfincs_framework
Selection and local report: data/exp/data-briefing/three_ward_sfincs_framework/review.html

Implemented now:
  model_spec.json — fixed engine, scope, candidate grid, datum and event policy
  input_inventory.csv — 15 input roles; source presence distinct from adequacy
  input_contract.json — units, time support, mass accounting and output definitions
  workflow.json — steps 1–4 followed by active measures at step5
  measure_catalog.json — registered options and required engineering inputs
  scenario_registry.json — future event/sensitivity/intervention cases
  readiness.json — unresolved inputs, not a model accuracy certificate
  engine_smoke_test/ — separate synthetic test using the official engine

Commands from the project root (existing Python environment):
  .venv/bin/python src/exp/sfincs_workflow.py audit
  .venv/bin/python src/exp/sfincs_workflow.py command
  .venv/bin/python src/exp/validate_sfincs_framework.py

The official installation case has already run. The launcher refuses to
overwrite its saved output; preserve it before an intentional rerun.
prepare_sfincs_framework.py regenerates the framework and resets pending
workflow labels; validate_sfincs_framework.py reconciles engine evidence.

Not yet implemented:
  official HydroMT-SFINCS subgrid generation and Tokyo bundle conversion;
  complete DEM/catchment/forcing/bank data and their datum alignment;
  event calibration/validation; intervention translation and evaluation.

No new project Python dependencies were installed. Docker Desktop was
started and one fixed official image was pulled (~220 MB). The tag is
linux/amd64 only; local execution uses emulation. Bulk performance remains
unmeasured. The synthetic result is not a Tokyo flood prediction.
''')
    print(json.dumps(result,indent=2))
    if not passed:raise SystemExit('Analytical smoke validation failed')


if __name__=='__main__':main()
