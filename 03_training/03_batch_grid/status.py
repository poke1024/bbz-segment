#!/usr/bin/python3

import json
import collections
import humanize
import os

from pathlib import Path

import datetime
import dateutil.parser

# what is everyone working on currently?

ignore_elapsed_threshold = 12 * 60 * 60  # 12 hours

parameters = [
	"backbone",
	"model",
	"labels",
	"data_name",
	"ngpus",
	"batchsize",
	"fold",
	"illustrations"]

Status = collections.namedtuple('Status', ['parameters', 'meta'])

def gather_status(path, summary):
	if (path / "config.json").exists():

		with open(path / "run.json", "r") as f:
			run_data = json.loads(f.read())

		status = run_data["status"]

		with open(path / "config.json", "r") as f:
			config_data = json.loads(f.read())

		if status != "RUNNING":
			return

		elapsed = (datetime.datetime.now() - dateutil.parser.parse(run_data["start_time"]))
		if elapsed.total_seconds() > ignore_elapsed_threshold:
			return

		work = dict(zip(parameters, [config_data[p] for p in parameters]))

		meta = collections.OrderedDict()
		meta['running for'] = humanize.naturaldelta(elapsed)

		n_epochs = config_data["epochs"]
		if (path / "metrics.json").exists():
			with open(path / "metrics.json", "r") as f:
				metrics_data = json.loads(f.read())
		else:
			metrics_data = None

		if not metrics_data:
			n_epochs_done = 0
		else:
			first_metric = list(metrics_data.keys())[0]
			n_epochs_done = len(metrics_data[first_metric]["values"])
			meta['epochs'] = "%d/%d" % (n_epochs_done, n_epochs)

			seconds_per_epoch = elapsed.total_seconds() / n_epochs_done
			meta['time/epoch'] = humanize.naturaldelta(seconds_per_epoch)

			meta['finished in'] = humanize.naturaldelta((n_epochs - n_epochs_done) * seconds_per_epoch)

		meta['hostname'] = run_data["host"]["hostname"]
		meta['run id'] = path.stem

		gpu_name = run_data["host"]["gpus"]["gpus"][0]["model"]
		summary[gpu_name].append(Status(work, meta))

	else:
		for p in path.iterdir():
			if p.is_dir():
				gather_status(p, summary)

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
summary = collections.defaultdict(list)
# Path("/Users/offline/Documents/runs")
gather_status(script_dir / "runs", summary)

for gpu, tasks in summary.items():
	print("-" * 80)
	print("%s (%s tasks)" % (gpu, len(tasks)))
	print("-" * 80)
	print("")

	for i, task in enumerate(tasks):
		print("    task %d" % (1 + i))

		for k, v in task.meta.items():
			print("        %s: %s" % (k, v))

		for k, v in task.parameters.items():
			print("    %s: %s" % (k, v))
		print("")


