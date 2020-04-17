from pathlib import Path
import json
import dateutil.parser
import collections
import humanize
import datetime
import numpy as np

runtimes = collections.defaultdict(int)
all_runtimes = []

backbones = set([
	"resnet34",
	"resnet50",
	#"seresnet18", 
	"seresnet34",
	"seresnet50",
	"vgg16",
	"inceptionv3",
	#"efficientnetb0",
	"efficientnetb1",
	"efficientnetb2",
	"seresnext50",
	"inceptionresnetv2"])
	#"seresnext50",
	#"densenet121"])

seen = set()

for p in Path("/Users/arbeit/Documents/runs").iterdir():
	if (p / "run.json").exists():
		with open(p / "run.json", "r") as f:
			data = json.loads(f.read())

		if data["status"] != "COMPLETED":
			continue

		with open(p / "config.json", "r") as f:
			config = json.loads(f.read())

		if config["backbone"] not in backbones:
			continue  # ignore

		key = (config["model"], config["backbone"], config["data_name"], config["labels"], config["illustrations"])
		if key in seen:
			continue
		seen.add(key)

		#print(config["model"], config["backbone"])

		start_time = dateutil.parser.isoparse(data["start_time"])
		stop_time = dateutil.parser.isoparse(data["stop_time"])
		runtime = stop_time - start_time

		gpus = data["host"]["gpus"]["gpus"]
		gpu = gpus[0]["model"]
		n_gpus = len(gpus)

		runtimes[gpu] += runtime.total_seconds() * n_gpus
		all_runtimes.append(runtime.total_seconds() * n_gpus)

for k, v in runtimes.items():
	print(k, humanize.naturaldelta(datetime.timedelta(seconds=v)))

print(len(all_runtimes), "mean runtime", humanize.naturaldelta(datetime.timedelta(seconds=np.mean(all_runtimes))))