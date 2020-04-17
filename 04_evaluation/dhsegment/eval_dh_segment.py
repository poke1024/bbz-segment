import os
import cv2
from glob import glob
import numpy as np
import random
import tensorflow as tf
#from imageio import imread, imsave
from pathlib import Path
import PIL.Image
import matplotlib.pyplot as plt
import collections
import statistics
import json
import itertools
import math
import re
import argparse
import traceback

from functools import lru_cache
from fractions import Fraction
from collections import namedtuple
from tqdm import tqdm

import decimal
from decimal import Decimal

import sklearn.metrics

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization


parser = argparse.ArgumentParser()
parser.add_argument(
	'--force', help='force recomputation of all evaluations', action='store_true')
args = parser.parse_args()


class Tile:
	def __init__(self, outer, inner):
		self._outer = outer
		self._inner = inner

	@property
	def pos(self):
		return tuple(self._outer[:2])

	@property
	def outer(self):
		return self._outer

	@property
	def inner(self):
		return self._inner

	def read_outer(self, pixels):
		x0, y0, x1, y1 = self._outer
		#print("read_outer", x0, y0, x1, y1, pixels.shape)
		return pixels[y0:y1, x0:x1]

	def write_inner(self, labels, data):
		x0, y0, x1, y1 = self._inner
		dx, dy = np.array(self._inner[:2]) - np.array(self._outer[:2])
		#print("x", x0, x1, "y", y0, y1, "dxdy", dx, dy)
		#print(labels.shape, "data.shape", data.shape)
		labels[y0:y1, x0:x1] = data[dy:dy + (y1 - y0), dx:dx + (x1 - x0)]


class Tiles:
	def __init__(self, tile_size, beta0=50):
		self._tile_size = tile_size
		assert all(beta0 < s for s in tile_size)
		self._beta0 = beta0

	def _tiles_1(self, full_size, tile_size):
		if tile_size == full_size:
			yield (0, full_size), (0, full_size)
		else:
			n_steps = math.ceil(full_size / tile_size)

			while True:
				r = (full_size - tile_size) / ((n_steps - 1) * tile_size)

				if tile_size * (1 - r) > self._beta0:
					break

				n_steps += 1

			x0 = []
			x1 = []
			for i in range(n_steps):
				if True:  # old tile generation code
					overlap = int(((n_steps * tile_size) - full_size) / (n_steps - 1))
					x = i * (tile_size - overlap)
				else:
					x = round(i * tile_size * r)
				
				x -= max(0, x + tile_size - full_size)

				x0.append(x)
				x1.append(x + tile_size)

			for i in range(n_steps):
				if i > 0:
					x0_inner = (x1[i - 1] + x0[i]) // 2
				else:
					x0_inner = 0

				if i < n_steps - 1:
					x1_inner = (x1[i] + x0[i + 1]) // 2
				else:
					x1_inner = full_size

				yield (x0[i], x1[i]), (x0_inner, x1_inner)

	def __call__(self, full_size):
		p = [list(self._tiles_1(f, t)) for f, t in zip(full_size, self._tile_size)]
		for x, y in itertools.product(*p):
			(x0, x1), (xi0, xi1) = x
			(y0, y1), (yi0, yi1) = y
			yield Tile((x0, y0, x1, y1), (xi0, yi0, xi1, yi1))


class TileLoader:
	def __init__(self, model, classes):
		self._model = model
		self._classes = classes

	def load_y_pred(self, im_path):
		prediction_outputs = self._model.predict(str(im_path))
		probs = prediction_outputs['probs'][0]

		n_classes = len(self._classes)
		# probs = probs[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
		probs = probs[:, :, :n_classes]
		y_pred = np.argmax(probs, axis=2)

		return y_pred.astype(np.uint8)

	def load_y_true(self, gt_path):
		# load y_true and map y_true indices to classes.
		im = PIL.Image.open(gt_path)
		y_true = np.array(im)

		pal = np.reshape(np.array(im.getpalette()), (256, 3))
		lut = np.zeros((256, ), dtype=np.uint8)
		for j, class_color in enumerate(self._classes):
			found = False
			for i, color in enumerate(pal):
				if tuple(color) == class_color:
					lut[i] = j
					found = True
					break
			assert found
		y_true = lut[y_true]

		return y_true


def load_document(full_size, tiles_data, load_tile):
	y = np.zeros(tuple(reversed(full_size)), dtype=np.uint8)

	for tile, im_path, gt_path in tiles_data:
		y_tile = load_tile(im_path, gt_path)
		if tile:
			#print("y shape", y.shape, flush=True)
			#print("y_tile shape", y_tile.shape, flush=True)
			#print("tile outer", tile.outer, flush=True)
			tile.write_inner(y, y_tile)
		else:
			y[:, :] = y_tile[:, :]

	return y


def parse_layout_name(layout_name):
	lexer = re.compile(r"^layout_([0-9]+)x([0-9]+)(_T([0-9]+)x([0-9]+))?$")
	m = lexer.match(layout_name)
	assert m

	full_size = (int(m.group(1)), int(m.group(2)))
	if m.group(3):
		tile_size = (int(m.group(4)), int(m.group(5)))

		tiles_gen = Tiles(tile_size)
		tiles = list(tiles_gen(full_size))
	else:
		tile_size = None
		tiles = None

	return full_size, tile_size, tiles


def load_documents(layout_name, data, tile_loader):
	full_size, tile_size, tiles = parse_layout_name(layout_name)

	lexer = re.compile(r"^(.*?)(-T-([0-9]+)-([0-9]+))?$")

	documents_data = collections.defaultdict(list)
	for im_path, gt_path in data:
		m = lexer.match(im_path.stem)
		assert m

		name = m.group(1)

		tile = None
		if m.group(2):
			pos = (int(m.group(3)), int(m.group(4)))
			for t in tiles:
				if t.pos == pos:
					tile = t
					break

			if tile is None:
				raise ValueError("tile position mismatch in filename")

		documents_data[name].append((tile, im_path, gt_path))

	results = []
	for name, v in tqdm(documents_data.items(), desc="computing inference"):
		y_pred = load_document(full_size, v,
			lambda im_path, gt_path: tile_loader.load_y_pred(im_path))
		y_true = load_document(full_size, v,
			lambda im_path, gt_path: tile_loader.load_y_true(gt_path))
		results.append((name, y_pred, y_true))

	return results


def evaluate(layout_name, model_dir, data_dir):

	evaluation_path = data_dir / "evaluation.json"

	if (not args.force) and evaluation_path.exists():
		return evaluation_path

	n_classes = 0
	classes = []
	with open(data_dir / "classes.txt") as f:
		classes = [line.strip() for line in f.readlines()]
		classes = [tuple(map(int, line.split())) for line in classes if line]
		n_classes = len(classes)

	data = []
	with open(data_dir / "val.csv", "r") as f:
		for line in f.readlines():
			if line.strip():
				im, gt = line.strip().split(",")
				data.append((Path(im), Path(gt)))

	def colorize(pixels):
		colors = plt.get_cmap("tab10").colors
		colors = (np.array(colors).flatten() * 255).astype(np.uint8)
		im = PIL.Image.fromarray(pixels, "P")
		palette = np.zeros((768,), dtype=np.uint8)
		palette[:len(colors)] = colors
		im.putpalette(palette)
		return np.array(im.convert("RGB"))


	LongShelhamerDarrellMetrics = namedtuple(
		'LongShelhamerDarrellMetrics',
		['pixel_accuracy', 'mean_accuracy', 'mean_IU', 'frequency_weighted_IU'])


	def lsd_metrics(
		prediction: np.ndarray,
		truth: np.ndarray,
		n_classes: int) -> LongShelhamerDarrellMetrics:

		"""This computes the evaluation metrics given for semantic segmentation given in:
		[1] J. Long, E. Shelhamer, and T. Darrell, "Fully Convolutional Networks for
		Semantic Segmentation", 2014. (available at https://arxiv.org/abs/1411.4038).

		Note:
			Modified to exclude empty classes.

		Args:
			prediction: integer array of predicted classes for each pixel.
			truth: integer array of ground truth for each pixel.
			n_classes: defines the pixel classes [0, 1, ..., n_classes - 1].

		Returns:
			LongShelhamerDarrellMetrics: The computed metrics.
		"""

		def _check_array(name, a):
			if not np.issubdtype(a.dtype, np.integer):
				raise ValueError("given %s-array must be of type integer" % name)

			if not (0 <= np.min(a) < n_classes and 0 <= np.max(a) < n_classes):
				raise ValueError("non-class values in given %s-array" % name)

		_check_array('prediction', prediction)
		_check_array('truth', truth)

		classes = list(range(n_classes))

		@lru_cache(maxsize=None)
		def n(i: int, j: int) -> Fraction:
			# n(i, j) is "the number of pixels of class i predicted to belong to
			# class j", see [1].
			return Fraction(int(np.sum(np.logical_and(
				truth == i, prediction == j).astype(np.uint8), dtype=np.uint64)))

		@lru_cache(maxsize=None)
		def t(i: int) -> Fraction:
			# t(i) is "the total number of pixels of class i", see [1].
			return sum(n(j, i) for j in classes)

		non_empty_classes = [i for i in classes if t(i) > 0]

		return LongShelhamerDarrellMetrics(
			pixel_accuracy=sum(n(i, i) for i in classes) / sum(t(i) for i in classes),

			mean_accuracy=(Fraction(1) / len(non_empty_classes)) * sum(
				(n(i, i) / t(i)) for i in non_empty_classes),

			mean_IU=(Fraction(1) / len(non_empty_classes)) * sum(
				(
					n(i, i) / (
						t(i) + sum(n(j, i) for j in non_empty_classes) - n(i, i))
				) for i in non_empty_classes),

			frequency_weighted_IU=(Fraction(1) / sum(t(k) for k in non_empty_classes)) * sum(
				(
					(t(i) * n(i, i)) / (
						t(i) + sum(n(j, i) for j in non_empty_classes) - n(i, i))
				) for i in non_empty_classes)
		)


	decimal.getcontext().prec = 30

	tf.reset_default_graph()
	session = tf.InteractiveSession()

	# see dhSegment/dh_segment/inference/loader.py
	model = LoadedModel(model_dir, predict_mode='filename_original_shape')
	tile_loader = TileLoader(model, classes)
	documents = load_documents(layout_name, data, tile_loader)

	print("found %d documents." % len(documents), flush=True)

	images_path = data_dir / "inference"
	images_path.mkdir(exist_ok=True)

	metrics_results = collections.defaultdict(list)
	for name, y_pred, y_true in tqdm(documents, desc="computing metrics"):
		
		#print("y_pred", y_pred.shape, y_pred.dtype)
		#print("y_true", y_true.shape, y_true.dtype)
		#print("original shape", prediction_outputs['original_shape'])

		#original_shape = prediction_outputs['original_shape']

		lsdm = lsd_metrics(y_pred, y_true, n_classes)
		for k in lsdm._fields:
			x = getattr(lsdm, k)
			metrics_results[k].append(Decimal(x.numerator) / Decimal(x.denominator))


		y_true_flat = y_true.reshape((y_true.size, ))
		y_pred_flat = y_pred.reshape((y_pred.size, ))


		for k in ('micro', 'macro', 'weighted'):
			metrics_results['precision-%s' % k].append(
				Decimal(sklearn.metrics.precision_score(y_true_flat, y_pred_flat, average=k)))
			metrics_results['recall-%s' % k].append(
				Decimal(sklearn.metrics.recall_score(y_true_flat, y_pred_flat, average=k)))
			metrics_results['jaccard-%s' % k].append(
				Decimal(sklearn.metrics.jaccard_score(y_true_flat, y_pred_flat, average=k)))

		metrics_results['matthews'].append(
			Decimal(sklearn.metrics.matthews_corrcoef(y_true_flat, y_pred_flat)))

		PIL.Image.fromarray(colorize(y_pred), "RGB").save(
			images_path / ("%s-pred.png" % name))
		PIL.Image.fromarray(colorize(y_true), "RGB").save(
			images_path / ("%s-true.png" % name))

	with open(evaluation_path, "w") as result_file:
		result_data = dict()
		for k, v in metrics_results.items():
			result_data[k] = str(statistics.mean(v).quantize(Decimal('.0001'), rounding=decimal.ROUND_DOWN))
		result_file.write(json.dumps(result_data))

		#result_file.write("metric;value;number of samples\n")
		#for k, v in metrics_results.items():
		#	result_file.write("%s;%s;%d\n" % (
		#		k, str(statistics.mean(v).quantize(Decimal('.0001'), rounding=decimal.ROUND_DOWN)), len(v)))

	session.close()

	return evaluation_path

preprocessed_dir = Path("/home/sc.uni-leipzig.de/bo140rasi/data/preprocessed/")
models_dir = Path("/home/sc.uni-leipzig.de/bo140rasi/dhsegment")
all_data = dict(blk=dict(), blkx=dict(), sep=dict())

for p in preprocessed_dir.iterdir():
	if not p.is_dir():
		continue

	for mode in ("blk", "blkx", "sep"):
		name = "%s_%s" % (p.stem, mode)
		model_dir = models_dir / name / "export"
		if model_dir.exists():
			try:
				print("==================== evaluating %s..." % name, flush=True)
				eval_path = evaluate(p.stem, model_dir, p / "dhsegment" / mode)

				with open(eval_path, "r") as f:
					all_data[mode][p.stem] = json.loads(f.read())
			except Exception as e:
				print("ERROR: ignored %s due to: %s" % (name, e))
				traceback.print_exc()

with open("dhs.json", "w") as f:
	f.write(json.dumps(all_data))

