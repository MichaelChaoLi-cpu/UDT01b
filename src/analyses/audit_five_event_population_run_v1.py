"""Metadata-only adapter: original numerical auditor executes unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import audit_cross_event_protection_run_v1 as original
original.OUT='data/processed/three_ward_five_event_population_validation_v1'
original.REPORT='data/exp/estimation/three_ward_five_event_population_validation_v1'
if __name__=='__main__':
 assert len(sys.argv)==3 and sys.argv[1]=='execution';original.main(sys.argv[1],sys.argv[2])
