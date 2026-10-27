"""Analyze completed controls only; does not launch a hydrodynamic solver."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_rain_step_integral import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main(case,phase):
 if phase=='audit':import audit_adaptive_positivity_controls as m
 elif phase=='reconstruct':import reconstruct_adaptive_positivity_labels as m
 elif phase=='exchange':import compare_adaptive_boundary_exchange as m
 else:raise ValueError(phase)
 m.OUT=OUT;m.REPORT=REPORT
 m.main(only=case)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','reconstruct','exchange']);p.add_argument('--only',choices=CASES);a=p.parse_args();main(a.only,a.phase)
