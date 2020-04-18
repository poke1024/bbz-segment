import segmentation_models
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

import json
import cv2
import math
import re
import os
import itertools
import enum
import scipy
import PIL.Image
import collections
import logging

from pathlib import Path
from tqdm import tqdm


def category_colors(n):
	colors = plt.get_cmap("tab10").colors
	return np.array(list(colors)).flatten() * 255


def colorize(labels):
	n_labels = np.max(labels) + 1
	colors = category_colors(n_labels)

	im = PIL.Image.fromarray(labels, "P")
	pil_pal = np.zeros((768,), dtype=np.uint8)
	pil_pal[:len(colors)] = colors
	im.putpalette(pil_pal)

	return im


class Tile:
	def __init__(self, outer, inner):
		self._outer = outer
		self._inner = inner

	@property
	def outer(self):
		return self._outer

	@property
	def inner(self):
		return self._inner

	def read_outer(self, pixels):
		x0, y0, x1, y1 = self._outer
		return pixels[y0:y1, x0:x1]

	def write_inner(self, labels, data):
		x0, y0, x1, y1 = self._inner
		dx, dy = np.array(self._inner[:2]) - np.array(self._outer[:2])
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


def load(what, **kwargs):
	loaded = dict()
	for c, name in tqdm(what, desc="loading models"):
		loaded[name] = c(name, **kwargs)
	return loaded


class Predictor:
	pass


class NetPredictor(Predictor):
	def __init__(self, name, models_path, use_page_cache=False):
		if not re.match(r"^[a-z0-9/]+$", name):
			raise ValueError("illegal model name '%s'" % name)

		self._name = name
		self._page_cache = None

		models_path = Path(models_path)
		network_path = models_path / name

		for filename in ("meta.json", "model.h5"):
			asset_path = network_path / filename
			if not asset_path.exists():
				raise RuntimeError("no model file found at %s" % asset_path)

		with open(network_path / "meta.json", "r") as f:
			meta = json.loads(f.read())
		classes = meta["classes"]

		if False:
			model = getattr(segmentation_models, meta["model"])(
				meta["backbone"],
				classes=len(classes),
				activation="softmax")
			logging.info("loading model at %s" % str(network_path / "model.h5"))
			model.load_weights(str(network_path / "model.h5"))
		else:
			# see https://github.com/qubvel/segmentation_models/issues/153
			# see https://stackoverflow.com/questions/54835331/how-do-i-load-a-keras-saved-model-with-custom-optimizer
			model = load_model(str(network_path / "model.h5"), compile=False)

		self._preprocess = segmentation_models.get_preprocessing(
			meta["backbone"])

		self._model = model
		self._full_size = tuple(meta["full_size"])
		self._full_shape = tuple(reversed(self._full_size))
		self._tile_size = tuple(meta["tile_size"])

		self._tiles = list(Tiles(
			self._tile_size,
			beta0=meta["tile_beta"])(meta["full_size"]))

		self._classes = enum.Enum(
			meta["type"] + "Label", dict((v, i) for i, v in enumerate(classes)))
		self._type = meta["type"]

	@property
	def type(self):
		return self._type

	@property
	def classes(self):
		return self._classes

	@property
	def size(self):
		return self._full_size

	@property
	def tile_size(self):
		return self._tile_size

	def close(self):
		if self._cache:
			self._cache.close()

	def _predict(self, page, labels=None, verbose=False):
		if labels is None:
			net_input = cv2.resize(page.pixels, self._full_size, interpolation=cv2.INTER_AREA)
			net_input = cv2.cvtColor(net_input, cv2.COLOR_BGR2RGB)
			net_input = self._preprocess(net_input)

			labels = np.empty(self._full_shape, dtype=np.uint8)

			if verbose:
				tiles = tqdm(self._tiles, desc="prediction")
			else:
				tiles = self._tiles

			for tile in tiles:
				tile_pixels = tile.read_outer(net_input)
				tile_pixels = np.expand_dims(tile_pixels, axis=0)
				pr_mask = self._model.predict(tile_pixels)
				tile_labels = np.argmax(pr_mask.squeeze(), axis=-1).astype(np.uint8)
				tile.write_inner(labels, tile_labels)

		return Prediction(
			page,
			labels,
			self._classes)

	@property
	def background(self):
		return self._classes["BACKGROUND"]

	def __call__(self, page, use_cache=True):
		labels = None
		cache_key = self._name + "/" + page.key

		if use_cache and self._page_cache is not None:
			if cache_key in self._page_cache:
				labels = self._page_cache[cache_key]

		prediction = self._predict(page, labels)

		if use_cache and labels is None and self._page_cache is not None:
			self._page_cache.set(cache_key, prediction.labels)

		return prediction


def _majority_vote(data, undecided=0):
	data = np.array(data, dtype=data[0].dtype)
	n_labels = np.max(data) + 1

	counts = np.zeros(
		(n_labels,) + data[0].shape, dtype=np.int32)
	for label in range(n_labels):
		for pr in data:
			counts[label][pr == label] += 1

	counts = np.dstack(counts)

	order = np.argsort(counts)
	candidates_count = np.take_along_axis(counts, order[:, :, -2:], axis=-1)
	tie = np.logical_not(candidates_count[:, :, 0] < candidates_count[:, :, 1])

	most_freq = np.argmax(counts, axis=-1).astype(data.dtype)
	most_freq[tie] = undecided

	return most_freq


class VotingPredictor(Predictor):
	def __init__(self, *predictors):
		if not all(p.type == predictors[0].type for p in predictors):
			raise RuntimeError("predictor need to have same pr types")
		self._predictors = predictors
		self._undecided = predictors[0].background.value

	def __call__(self, pixels):
		predictions = [p(pixels) for p in self._predictors]
		return Prediction(
			predictions[0].page,
			_majority_vote([p.labels for p in predictions], self._undecided),
			self._predictors[0].classes)


class Page:
	def __init__(self, path=None, pixels=None):
		if path is None:
			self._key = None
			self._pixels = pixels
		else:
			path = Path(path)

			self._key = str(path.absolute())

			if not path.is_file():
				raise FileNotFoundError(path)

			im = PIL.Image.open(str(path.absolute())).convert("RGB")
			self._pixels = np.array(im)

			#self._pixels = cv2.imread(
			#	str(path.absolute()), cv2.IMREAD_COLOR)

			if self._pixels is None:
				raise ValueError("issue in loading image %s" % path)

	@property
	def key(self):
		return self._key

	@property
	def pixels(self):
		return self._pixels

	@property
	def shape(self):
		return self._pixels.shape[:2]

	@property
	def size(self):
		return tuple(reversed(self.shape))

	@property
	def extent(self):
		return np.linalg.norm(np.array(self.size))

	@property
	def width(self):
		return self.size[0]

	@property
	def height(self):
		return self.size[1]

	def annotate(self, **kwargs):
		import annotations
		return annotations.Annotate().page(self, **kwargs)


class Prediction:
	def __init__(self, page, labels, classes):
		self._page = page
		self._labels = labels
		self._classes = classes
		self._blocks = None
		self._separators = None
		self._background = self._classes["BACKGROUND"]

	@property
	def background_label(self):
		return self._background

	@property
	def page(self):
		return self._page

	@property
	def labels(self):
		return self._labels

	@property
	def classes(self):
		return self._classes

	def save(self, path):
		colorize(self._labels).save(path)
