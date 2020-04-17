from pathlib import Path
import collections
import os

data_path = Path(os.path.realpath(__file__)).parent

data = dict()
for p in (data_path / "evaluation").iterdir():
	if p.name.endswith(".csv"):
		with open(p, "r") as f:
			for i, line in enumerate(f):
				if i > 0:  # not header?
					mode, metric, value, n = line.split(";")
					data[(p.stem, mode, metric)] = float(value)

columns = ("pixel_accuracy", "mean_accuracy", "mean_IU", "frequency_weighted_IU", "matthews")

with open(data_path / "table.tex", "w") as f:
	f.write(" & ".join(("",) + columns) + "\\\\" + "\n")

	for mode in ("sep", "blk", "blkx"):
		for fold in range(1, 6):
			row = ["%s/fold-%s" % (mode, str(fold))]
			for c in columns:
				key = ("fold-%d" % fold, mode, c)
				if key in data:
					row.append("%.2f" % (100 * data[key]))
			f.write(" & ".join(row) + "\\\\" + "\n")
