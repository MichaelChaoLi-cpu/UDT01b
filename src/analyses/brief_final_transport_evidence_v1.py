#!/usr/bin/env python3
"""Scoped data-briefing adapter; boolean fields remain categorical audit metadata."""
from pathlib import Path
import importlib.util,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
SOURCES=[ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1'/n for n in ['three_parent_road_comparison.csv','three_parent_Q5_vs_Q2p5_population_summary.csv','three_parent_population_totals_all_arms.csv','three_parent_head_transfer_comparison.csv']]+[ROOT/'data/processed/three_ward_s1r_september_traffic_phases_v1'/n for n in ['all_times_traffic_outcomes.csv','all_times_paired_traffic_effects.csv','fixed_all_times_cohort_costs.csv']]
spec=importlib.util.spec_from_file_location('mili_brief',ROOT/'.codex/skills/data-briefing/scripts/create_data_briefing.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=m.summarize_dataset
def summarize(name,path,frame):
    # NumPy quantiles do not subtract booleans. Profile audit flags as categories;
    # this local view never writes or transforms source result tables.
    view=frame.copy()
    for c in view:
        if pd.api.types.is_bool_dtype(view[c]):view[c]=view[c].astype('category')
    return original(name,path,view)
m.summarize_dataset=summarize
if __name__=='__main__':
    sys.argv=[str(Path(m.__file__)),'--root',str(ROOT),'--output','data/exp/data-briefing/three_ward_final_evidence_v1','--inputs']+[str(p.relative_to(ROOT))for p in SOURCES]+['--include-time-series','no','--max-distribution-plots','12','--pretty']
    raise SystemExit(m.main())
