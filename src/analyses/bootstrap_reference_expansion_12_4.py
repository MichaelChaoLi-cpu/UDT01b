"""Create the host audit output directory, then run or resume the unchanged bounded controller.

Completed hydraulic executions are retained by the bound controller. This bootstrap
changes no forcing, numerical setting, numerical policy or bound execution source.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_expansion_12_4 import OUT
from run_reference_expansion_12_4 import main
if __name__=='__main__':
 (ROOT/OUT/'expansion').mkdir(parents=True,exist_ok=True)
 main()
