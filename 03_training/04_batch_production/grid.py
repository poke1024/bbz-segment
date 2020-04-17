#!/usr/bin/python

import json
import os
import subprocess
import functools
import sys

from pathlib import Path
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

assert "BBZ_SEGMENT_RUNS_PATH" in os.environ
assert len(sys.argv) == 2

script_dir = Path(os.path.realpath(__file__)).parent

with open(script_dir / sys.argv[1], "r") as f:
	param_grid = json.loads(f.read())

n = functools.reduce(lambda x, y: x * y,
	map(len, param_grid.values()))

for p in tqdm(ParameterGrid(param_grid), total=n):
	args = ["%s=%s" % (k, v) for k, v in p.items()]
	cmd = ["python", str(script_dir / "train.py"), "with", *args]
	subprocess.run(cmd)
