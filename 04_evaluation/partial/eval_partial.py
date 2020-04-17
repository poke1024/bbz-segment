import json
import numpy as np
import collections
import re
import os
import sys
import matplotlib.pyplot as plt
import tikzplotlib
import math

from pathlib import Path


#plt.style.use("ggplot")


Configuration = collections.namedtuple('Configuration', [
	'tiling', 'backbone', 'model', 'labels', 'fold', 'partial'])


Tiling = collections.namedtuple('Tiling', [
	'full_size', 'tile_size'])


Result = collections.namedtuple('Result', [
	'best', 'items'])


Value = collections.namedtuple('Value', [
	'x', 'err', 'n'])


def gather_data(path, data):
	if (path / "config.json").exists():
		with open(path / "config.json", "r") as f:
			config = json.loads(f.read())

		match = re.match(
			r"layout_(\d+)x(\d+)(_T(\d+)x(\d+))?",
			config["data_name"])
		assert match

		# for the moment, only use fold 1. don't mix with other folds.
		if int(config["fold"]) != 1:
			return

		if config.get("illustrations", False):
			config["labels"] = "blkx"
		elif config["labels"] == "txt":
			config["labels"] = "blk"

		key = Configuration(
			Tiling((int(match.group(1)), int(match.group(2))),
				(int(match.group(4)), int(match.group(5))) if match.group(3) else None),
			config["backbone"],
			config["model"],
			config["labels"],
			int(config["fold"]),
			float(config["partial"]))
		n_epochs = config["epochs"]

		with open(path / "metrics.json", "r") as f:
			metrics = json.loads(f.read())

		decider_metric = "val_matthews"
		val_metric_prefix = "val_"

		if decider_metric in metrics:
			decider_metric_values = metrics[decider_metric]["values"]
		else:
			decider_metric_values = []

		if len(decider_metric_values) != n_epochs:
			print("skipped incomplete %s" % str(path))
			return

		best_model_epoch = np.argmax(np.array(decider_metric_values))

		best_model_metrics = dict()
		for k, v in metrics.items():
			if k.startswith(val_metric_prefix):
				best_model_metrics[k] = v["values"][best_model_epoch]

		data[key] = best_model_metrics

	else:
		for p in path.iterdir():
			if p.is_dir():
				gather_data(p, data)


script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

n_training_pages = 104 - 21

for runs_path in (script_dir / "data").iterdir():

	if not runs_path.is_dir():
		continue

	data = dict()
	gather_data(runs_path, data)

	default_key = list(data.keys())[0]

	for mode in ('blkx', 'blk', 'sep'):

		table = []

		for i_partial in range(0, 11):
			partial = i_partial / 10

			key = Configuration(
				default_key.tiling,
				default_key.backbone,
				default_key.model,
				mode,
				int(1),
				partial)

			if key in data:
				table.append((partial, data[key]["val_matthews"]))

		if table:
			if False:
				print("%s: {" % mode)
				for k, v in table:
					print("{%.2f, %.2f}," % (k, v * 100))
				print("}")

			t = []
			y = []
			for k, v in table:
				t.append(math.floor(k * n_training_pages))
				y.append(v * 100)
			plt.plot(t, y, "+-", lw=4.1, label=mode)

plt.xlabel("Number of training pages")
plt.ylabel("Quality (MCC)")
plt.legend(loc="lower right")
#plt.title("")

tikzplotlib.save(script_dir / "partial.tex")



