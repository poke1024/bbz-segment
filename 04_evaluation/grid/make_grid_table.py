import json
import numpy as np
import collections
import re
import os
import sys
import json

from pathlib import Path


Configuration = collections.namedtuple('Configuration', [
	'tiling', 'backbone', 'model', 'labels', 'fold'])


Tiling = collections.namedtuple('Tiling', [
	'full_size', 'tile_size'])


Result = collections.namedtuple('Result', [
	'best', 'items'])


Value = collections.namedtuple('Value', [
	'x', 'err', 'n'])


def layout_name(tiling):
	n = ['layout_', '%dx%d' % tiling.full_size]
	if tiling.tile_size:
		n.append("_T%dx%d" % tiling.tile_size)
	return "".join(n)


class Measurement:
	def __init__(self, values):
		self._values = values

	@property
	def size(self):
		return len(self._values.get(None, []))

	@property
	def value(self):
		return self._values[None].x

	def __str__(self):
		r = dict()
		for k, v in self._values.items():
			r[k] = "%.2f %s" % (
				v.x * 100, "" if v.err is None else ("±%.2f" % (v.err * 100)))
		if len(r) and None in r:
			return r[None]
		else:
			return str(r)

	def latex(self, ablations=[None]):
		r = []
		for ablation in ablations:
			v = self._values.get(ablation, None)
			if v is None:
				r.append(("", ""))
			else:
				x = ("%.2f" % (v.x * 100))
				if v.err is None:
					r.append((x, ""))
				else:
					r.append((x, (("$\\pm$%.2f" % (v.err * 100)))))
		return r

	def __equals__(self, other):
		return self._values[None] == other._values[None]

	@staticmethod
	def select_best(measurements):
		if all(None in m._values for m in measurements):
			return sorted(measurements, key=lambda m: m._values[None].x)[-1]
		else:
			print("no default ablation. cannot sort, picking random value.")
			return measurements[0]


class DataPoint:
	def __init__(self):
		self._samples = collections.defaultdict(list)

	def add_sample(self, metrics, ablation):
		self._samples[ablation].append(metrics)

	@property
	def n_samples(self):
		return len(self._samples[None])

	@property
	def metrics(self):
		values = collections.defaultdict(lambda: collections.defaultdict(list))
		for ablation, s_all in self._samples.items():
			for s in s_all:
				for k in s.keys():
					values[k][ablation].append(s[k])

		result = dict()
		for k, v0 in values.items():
			ablation_values = dict()
			for ablation, v in v0.items():
				mean = np.mean(v)
				if len(v) > 1:
					err = np.max(np.abs([x - mean for x in v]))
				else:
					err = None
				ablation_values[ablation] = Value(mean, err, len(v))
			result[k] = Measurement(ablation_values)

		return result


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

		key = Configuration(
			Tiling((int(match.group(1)), int(match.group(2))),
				(int(match.group(4)), int(match.group(5))) if match.group(3) else None),
			config["backbone"],
			config["model"],
			config["labels"],
			int(config["fold"]))
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

		ablation = None
		data[key].add_sample(best_model_metrics, ablation)

	else:
		for p in path.iterdir():
			if p.is_dir():
				gather_data(p, data)


def by_backbone(metrics):
	backbones = {"resnet18": "resnet", "resnet34": "resnet",
		"resnet50": "resnet", "seresnet18": "seresnet", "seresnet34": "seresnet",
		"seresnet50": "seresnet", "efficientnetb0": "efficientnet",
		"efficientnetb1": "efficientnet", "efficientnetb2": "efficientnet",
		"inceptionv3": "inceptionv3", "vgg16": "vgg16"}

	values = collections.defaultdict(list)
	for k, v in metrics.items():
		values[backbones.get(k.backbone, "other")].append(v)

	agg = dict((k, Measurement.select_best(v)) for k, v in values.items())

	best = Measurement.select_best(list(agg.values()))
	items = sorted(list(agg.items()), key=lambda x: x[0])

	return Result(best, items)


def by_model(metrics, ablations):
	# "\\textbf{%s} & %s" % 

	values = collections.defaultdict(dict)
	for k, v in metrics.items():
		values[k.model][k] = v

	agg = dict((k, by_backbone(v)) for k, v in values.items())

	best = Measurement.select_best([r.best for r in agg.values()])
	for j, (model, r) in enumerate(sorted(list(agg.items()), key=lambda x: x[0])):

		yield "%s\\\\\n" % model

		if j > 0:
			yield ""  # vertical space

		for i, (backbone, value) in enumerate(r.items):
			def make_line():
				if i == 0:
					if best == r.best:
						yield "\\textbf{"
					yield model
					if best == r.best:
						yield"}"

				yield " & "

				if value == best:
					yield "\\textbf{"
				yield backbone
				if value == best:
					yield "}"
				yield " & "

				for ia, a in enumerate(value.latex(ablations)):
					if ia > 0:
						yield " & "
					for k, s in enumerate(a):
						if k > 0:
							yield " & "
						if value == best and k == 0:
							yield "\\textbf{"
						yield s
						if value == best and k == 0:
							yield "}"

			yield "".join(make_line())


def tiling_latex(t):
	name = "$%d \\times %d$" % t.full_size
	if t.tile_size is not None:
		name += ", tiled at $%d \\times %d$" % t.tile_size		
	return name


def tiling_pixels(t):
	return t.full_size[0] * t.full_size[1]


def build_tables(metrics, metric_name, mode, fold=1, output="backbone"):

	backbones = [
		"vgg16",
		#"resnet18",
		"resnet34",
		"resnet50",
		#"seresnet18",
		"seresnet34",
		"seresnet50",
		"seresnext50",
		#"efficientnetb0",
		"efficientnetb1",
		"efficientnetb2",
		"inceptionv3",
		"inceptionresnetv2"]

	official_backbone_names = dict(
		vgg16="VGG16",
		resnet34="ResNet-34",
		resnet50="ResNet-50",
		seresnet34="SE-ResNet-34",
		seresnet50="SE-ResNet-50",
		seresnext50="SE-ResNeXt-50",
		efficientnetb1="EfficientNet-B1",
		efficientnetb2="EfficientNet-B2",
		inceptionv3="Inception-v3",
		inceptionresnetv2="Inception-ResNet-v2"
	)

	tilings = [
		Tiling(full_size=(512, 768), tile_size=None),
		Tiling(full_size=(640, 1024), tile_size=(384, 1024)),  # sm[h]
		Tiling(full_size=(768, 1280), tile_size=(768, 512)),  # sm[v]

		Tiling(full_size=(896, 1280), tile_size=(256, 1280)),  # m[h]
		Tiling(full_size=(896, 1280), tile_size=(896, 384)),  # m[v]
		Tiling(full_size=(896, 1280), tile_size=(512, 768)),   # m[hv]
		Tiling(full_size=(896, 1280), tile_size=None),  # m
		
		Tiling(full_size=(1280, 2400), tile_size=(1280, 896)), 
		Tiling(full_size=(1640, 2400), tile_size=(896, 1280))]

	values = collections.defaultdict(dict)
	for k, v in metrics.items():
		values[k.labels][k] = v

	for i, (label, cells) in enumerate(sorted(list(values.items()))):
		if output.endswith(".tex"):
			yield """
		\\begin{table*}\\centering
		\\ra{1.0}
		\\begin{tabular}{@{}lccccccccc@{}}\\toprule

		& sm & sm[h] & sm[v] & m[h] & m[v] & m[hv] & m & lg[v] & lg[hv] \\\\ \\midrule
			"""

		# compute best value for each tiling.
		best_values = dict()
		for tiling in tilings:
			best_value = 0
			for model in ('fpn', 'unet'):
				for backbone in backbones:
					c = Configuration(tiling, backbone, model, label, fold)
					if c in cells:
						best_value = max(best_value, cells[c].value)

			d = dhs_data[mode].get(layout_name(tiling), None)
			if d:
				best_value = max(best_value, float(d["matthews"]))

			best_values[tiling] = best_value


		# now generate table.
		if False:
			for j, model in enumerate(('fpn', 'unet')):

				if j > 0:
					yield "\\\\\n"

				yield "%s\\\\\n" % model

				for backbone in backbones:
					r = [backbone]

					for tiling in tilings:
						c = Configuration(tiling, backbone, model, label, fold)
						if c in cells:
							if cells[c].value == best_values[tiling]:
								r.append("\\textbf{%s}" % cells[c].latex()[0][0])
							else:
								r.append(cells[c].latex()[0][0])
						else:
							r.append("")

					yield " & ".join(r)

					yield "\\\\\n"

		elif output == "backbone.tex":
			for backbone in backbones:
				r = [official_backbone_names[backbone]]

				for tiling in tilings:
					c_best = None
					for model in ('fpn', 'unet'):
						c = Configuration(tiling, backbone, model, label, fold)
						if c_best is None:
							if c in cells:
								c_best = c
						elif c in cells and cells[c].value > cells[c_best].value:
							c_best = c

					if c_best in cells:
						if cells[c_best].value == best_values[tiling]:
							r.append("\\textbf{%s}" % cells[c_best].latex()[0][0])
						else:
							r.append(cells[c_best].latex()[0][0])
					else:
						r.append("")

				yield " & ".join(r)
				yield "\\\\\n"

			r = ["dhSegment"]
			for tiling in tilings:
				d = dhs_data[mode].get(layout_name(tiling), None)
				if d:
					if float(d["matthews"]) == best_values[tiling]:
						r.append("\\textbf{%.2f}" % (float(d["matthews"]) * 100))
					else:
						r.append("%.2f" % (float(d["matthews"]) * 100))
				else:
					r.append("")

			yield " & ".join(r)
			yield "\\\\\n"

		elif output == "model.tex":
			for backbone in backbones:
				r = [backbone]

				for tiling in tilings:
					c_best = None
					c_best_model = None

					for model in ('fpn', 'unet'):
						c = Configuration(tiling, backbone, model, label, fold)
						if c_best is None:
							if c in cells:
								c_best = c
								c_best_model = model
						elif c in cells and cells[c].value > cells[c_best].value:
							c_best = c
							c_best_model = model

					if c_best in cells:
						if cells[c_best].value == best_values[tiling]:
							r.append("\\textbf{%s}" % c_best_model)
						else:
							r.append(c_best_model)
					else:
						r.append("")

				yield " & ".join(r)

				yield "\\\\\n"


		elif output == "model.csv":
			for backbone in backbones:
				r = [backbone]

				for tiling in tilings:
					c_best = None
					for model in ('fpn', 'unet'):
						c = Configuration(tiling, backbone, model, label, fold)
						if c_best is None:
							if c in cells:
								c_best = c
						elif c in cells and cells[c].value > cells[c_best].value:
							c_best = c

					if c_best in cells:
						r.append(cells[c_best].latex()[0][0])
					else:
						r.append("")

				yield ", ".join(r)
				yield "\n"

			r = ["dhSegment"]
			for tiling in tilings:
				d = dhs_data[mode].get(layout_name(tiling), None)
				if d:
					r.append("%.2f" % (float(d["matthews"]) * 100))
				else:
					r.append("")

			yield ", ".join(r)
			yield "\n"

		#for k, v in cells.items():

		if output.endswith(".tex"):
			yield """
	\\bottomrule
	\\end{tabular}
	\\caption{%s-%s}
	\\end{table*}
			""" % (label, metric_name)


data = collections.defaultdict(DataPoint)

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

runs_path = script_dir / "runs"
assert runs_path.is_dir()
gather_data(runs_path, data)

dhs_path = script_dir / "dhs.json"
with open(dhs_path, "r") as f:
	dhs_data = json.loads(f.read())

#print(list(data.keys()))
#sys.exit(0)

if False:
	tilings = set()
	for c, dp in data.items():
		tilings.add(c.tiling)
	tilings = list(tilings)
	tilings = sorted(tilings, key=lambda t: t.full_size[0] * t.full_size[1])
	print(tilings)

if False:  # show specific configuration results
	#m = data[Configuration(Tiling((896, 1280), None), 'seresnet50', 'fpn', 'txt', 1)].metrics
	for c, dp in data.items():
		print("%s:" % str(c))
		print("-" * 40)
		for k, v in dp.metrics.items():
			print("%s: %s" % (k, str(v)))
		print("")
	sys.exit(0)

for output in ("backbone.tex", "model.tex", "model.csv"):
	with open(script_dir / ("grid-table-%s" % output), "w") as f:
		if output.endswith(".tex"):
			f.write("\\newcommand{\\ra}[1]{\\renewcommand{\\arraystretch}{#1}}\n")

		for m in ('matthews', 'precision', 'recall'):

			if not output.endswith(".tex"):
				f.write("# sep-%s\n" % m)

			f.write("".join(build_tables(
				dict((k, v.metrics["val_%s" % m]) for k, v in data.items() if k.labels == "sep"), m, output=output, mode="sep")
			))

			if not output.endswith(".tex"):
				f.write("# blk-%s\n" % m)

			f.write("".join(build_tables(
				dict((k, v.metrics["val_%s" % m]) for k, v in data.items() if k.labels == "txt"), m, output=output, mode="blk")
			))

			if not output.endswith(".tex"):
				f.write("# blkx-%s\n" % m)

			f.write("".join(build_tables(
				dict((k, v.metrics["val_%s" % m]) for k, v in data.items() if k.labels == "blkx"), m, output=output, mode="blkx")
			))
