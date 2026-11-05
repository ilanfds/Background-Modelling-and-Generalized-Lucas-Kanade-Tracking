# Tutors: Stephanie Käs and Ilya Fradlin
"""Run the exercise independently of the working directory."""
from pathlib import Path
import argparse, runpy, sys
p=argparse.ArgumentParser(); p.add_argument('variant', choices=['exercise']); p.add_argument('part', choices=['background','tracking','flow']); a, rest=p.parse_known_args()
r=Path(__file__).resolve().parent
sub='background_modelling' if a.part=='background' else 'generalized_lucas_kanade'
f={'background':'main.py','tracking':'generalized_lucas_kanade_tracking.py','flow':'optical_flow.py'}[a.part]
import os
os.chdir(r/a.variant/sub); sys.path.insert(0,str(Path.cwd())); sys.argv=[f]+rest
runpy.run_path(f,run_name='__main__')
