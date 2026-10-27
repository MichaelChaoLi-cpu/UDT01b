"""Run only the two byte-identical adaptive controls in the versioned rainfall-integral image."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_rain_step_integral import OUT,REPORT
import run_adaptive_positivity_controls as qualified_runner
# Reuse its bounded, serial, no-retry execution guard without editing frozen code.
qualified_runner.OUT=OUT;qualified_runner.REPORT=REPORT
if __name__=='__main__':qualified_runner.main()
