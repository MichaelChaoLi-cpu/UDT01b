"""Read-only progress and numerical sentinel, safe while CSV output is growing."""
from pathlib import Path
import datetime as dt,json,time
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
results=[]
for case in ['radar_low','radar_raised']:
 p=P/'runs'/case;row={'case':case,'phase':'not_started'}
 if (p/'launch.json').exists():
  launch=json.loads((p/'launch.json').read_text());elapsed=time.time()-dt.datetime.fromisoformat(launch['start_utc'].replace('Z','+00:00')).timestamp();row.update(phase='running',elapsed_wall_s=elapsed)
  f=p/'udt_step_budget.csv'
  if f.exists():
   with f.open('rb') as h:
    h.seek(max(0,f.stat().st_size-16384));lines=h.read().decode(errors='replace').splitlines()
   candidates=[s for s in lines if len(s.split(','))==20 and s.split(',')[0].strip().isdigit()]
   if candidates:
    values=[float(x) for x in candidates[-1].split(',')];row.update(step=int(values[0]),actual_time_s=values[1],completion_fraction=values[1]/32400,model_time_jst=(dt.datetime(2024,8,21,15)+dt.timedelta(seconds=values[1])).isoformat(),storage_m3=values[3],rain_applied_m3=values[4],out_m3=values[5],in_m3=values[6],negative_cells=int(values[10]),budget_residual_m3=values[14])
 if (p/'execution.json').exists():
  ex=json.loads((p/'execution.json').read_text());row.update(phase='completed' if ex['returncode']==0 and ex['completion_marker'] else 'failed',runtime_s=ex['runtime_seconds'])
 results.append(row)
print(json.dumps(results,ensure_ascii=False,indent=2))
