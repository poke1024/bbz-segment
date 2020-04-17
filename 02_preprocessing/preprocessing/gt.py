import numpy
import PIL.Image

from psd_tools import PSDImage
from psd_tools.constants import BlendMode

from .labels import Label, Annotations
from. utils.transform import Resize


class GroundTruth:
	_label_weights = dict((
		(Label.BACKGROUND, 2),
		(Label.FRAKTUR_BG, 1),
		(Label.FRAKTUR_SM, 1),
		(Label.ANTIQUA_BG, 1),
		(Label.ANTIQUA_SM, 1),
		(Label.TABTXT, 1),
		(Label.BORDER, 1),
		(Label.TABCOL, 3),
		(Label.H, 4),
		(Label.V, 5),
		(Label.H_SM, 1)
	))

	def __init__(self, ref, unbinarized, binarized, master):
		self._ref = ref

		assert binarized is not None
		assert master is not None

		self._binarized = binarized
		self._labels = dict(master=master)
		self._unbinarized = unbinarized

		images = [unbinarized, binarized, *self._labels.values()]
		assert all(im.shape[:2] == images[0].shape[:2] for im in images)

	def add_labels(self, name, labels):
		assert(labels.shape[:2] == self._unbinarized.shape[:2])
		self._labels[name] = labels

	def asset_path(self, *args, **kwargs):
		return self._ref.asset_path(*args, **kwargs)

	def transform(self, f):
		images = [self._unbinarized, self._binarized]
		images = list(map(lambda im: f("image", im), images))

		labels = dict((k, f("labels", l)) for k, l in self._labels.items())

		gt = GroundTruth(self._ref, images[0], images[1], labels["master"])
		for k, v in labels.items():
			if k != "master":
				gt.add_labels(k, v)
		return gt

	@property
	def path(self):
		return self._ref.path

	@property
	def shape(self):
		return self._unbinarized.shape

	@property
	def unbinarized(self):
		return self._unbinarized

	@property
	def binarized(self):
		return self._binarized

	@property
	def labels(self):
		return self._labels["master"]

	@staticmethod
	def label_weights():
		n_labels = len([l for l in Label])
		label_weights = numpy.empty(shape=(n_labels, ), dtype=numpy.float32)
		label_weights.fill(1)
		for label, w in GroundTruth._label_weights.items():
			label_weights[int(label)] = w
		return label_weights

	def _resize_bin(self, image, shape):
		if tuple(image.shape) == tuple(shape):
			return image

		resize = Resize(from_size=reversed(image.shape), to_size=reversed(shape))
		return resize.mask(image > 0).astype(numpy.uint8)

	def _resize_labels(self, image, shape):
		if tuple(image.shape) == tuple(shape):
			return image

		resize = Resize(from_size=reversed(image.shape), to_size=reversed(shape))
		return resize.labels(image, weights=GroundTruth.label_weights())

	def has_annotations(self, kind):
		return kind in self._labels

	def annotations(self, kind="master", shape=None, img_path=None):
		labels = self._labels[kind]

		if shape is None:
			shape = labels.shape

		# resize to prediction size.
		labels = self._resize_labels(labels, shape)

		if kind == "master":
			# base layer (binarization).

			# note: as soon as one pixel is marked in binarized as "interesting"
			# in the larger image, we want to have that pixel in the smaller
			# image as "interesting" as well. otherwise we would throw away
			# labels that might be very important (e.g. thin separators).

			binarized = self._resize_bin(self._binarized > 0, shape)  # white?

			# mask out non-binarized parts.
			labels[numpy.logical_not(binarized)] = int(Label.BACKGROUND)

		return Annotations(labels, img_path)


class Loader:
	def __init__(self):
		self._palette_image = PIL.Image.new('P', (16, 16))
		self._palette_image.putpalette(Annotations.palette())

	def _rgb2labels(self, pixels, bin_data=None, logger=None):
		ann_data = pixels.quantize(method=1, palette=self._palette_image)
		ann_data = numpy.array(ann_data, dtype=numpy.uint8)

		ann_rgb = PIL.Image.fromarray(ann_data, "P")
		ann_rgb.putpalette(self._palette_image.getpalette())
		ann_rgb = numpy.array(ann_rgb.convert("RGB", dither=PIL.Image.NONE))
		ignore = numpy.all(ann_rgb != numpy.array(pixels), axis=-1)

		if logger:
			if bin_data is not None:
				n_ignore = numpy.sum(numpy.logical_and(ignore, bin_data).astype(numpy.uint8))
			else:
				n_ignore = numpy.sum(ignore.astype(numpy.uint8))

			ignore_ratio = n_ignore / ann_rgb.size

			if n_ignore == 0:
				logger.info("no pixels ignored")
			elif ignore_ratio < 1 / 1000:
				logger.warning("ignored < 1%%%% pixels (%d)" % n_ignore)
			else:
				logger.warning("ignored > 1%%%% pixels (%d)" % n_ignore)

		if bin_data is not None:
			ignore = numpy.logical_or(ignore, numpy.logical_not(bin_data))

			# tmp.save(self._out_path / "images" / (psd_path.stem + ".debug.png"))
			# tmp.convert("RGB", dither=PIL.Image.NONE).save(self._out_path / "images" / (psd_path.stem + ".debug.png"))

			ann_data[ignore] = int(Label.BACKGROUND)

		return ann_data

	def _generate_regions(self, ground_truth):
		annotations = ground_truth.annotations()

		try:
			regions = annotations.regions()
			regions.hmerge(ground_truth.unbinarized)
		except:
			print("error on generating data for ", ground_truth.path)
			raise

		ann = regions.to_annotations()
		im = ann.image

		im.save(ground_truth.asset_path(
			"seg", prefix="regions.", ext=".png"))

		debug_im = PIL.Image.blend(
			PIL.Image.fromarray(ground_truth.unbinarized).convert("RGB"),
			im.convert("RGB"),
			0.75)
		debug_im.save(ground_truth.asset_path(
			"seg", prefix="debug.regions.", ext=".jpg"), "JPEG", quality=75)

		return ann.labels

	def __call__(self, ground_truth_ref, psd_path, img_path, logger=None, generate=True):
		bin_data = None
		ann_data = None
		image_size = 0

		unbinarized = numpy.array(PIL.Image.open(img_path).convert('L'))

		psd = PSDImage.open(str(psd_path))

		for layer in psd:
			if layer.blend_mode == BlendMode.NORMAL:
				# assert layer.offset == (0, 0)

				layer_image = layer.topil().convert('L')
				image_size = layer.size
				bin_data = numpy.array(layer_image)

			elif layer.blend_mode == BlendMode.MULTIPLY:
				layer_image = layer.topil()

				alpha_mask = PIL.Image.fromarray((numpy.array(layer_image)[:, :, 3] > 128).astype(numpy.uint8) * 255)
				layer_image = layer_image.convert("RGB")  # remove alpha channel

				ann_rgb_image = PIL.Image.new("RGB", image_size, (255, 255, 255))
				ann_rgb_image.paste(layer_image, layer.offset, alpha_mask)

				# now reduce to palette
				# tmp = rgb_image.im.convert("P", 0, self._palette_image.im)
				# tmp = rgb_image._new(tmp)

				ann_data = self._rgb2labels(ann_rgb_image, bin_data, logger)

		gt = GroundTruth(ground_truth_ref, unbinarized, bin_data, ann_data)

		asset_path = gt.asset_path("seg", prefix="regions.", ext=".png")
		if asset_path.is_file():
			gt.add_labels("regions", numpy.array(PIL.Image.open(asset_path)))
		else:
			gt.add_labels("regions", self._generate_regions(gt))

		return gt


class GroundTruthRef:
	_loader = Loader()

	def __init__(self, psd_path, image_ext):
		self._path = psd_path
		self._image_ext = image_ext

	@property
	def path(self):
		return self._path

	def load(self, logger=None):
		return GroundTruthRef._loader(self, self.annotated_path, self.document_path, logger)

	@property
	def annotated_path(self):
		return self._path

	@property
	def document_path(self):
		return self.asset_path("img", ext=self._image_ext)

	def asset_path(self, kind, prefix="", ext=".png"):
		container = self._path.parent.parent / kind
		container.mkdir(exist_ok=True)
		return container / (prefix + self._path.stem + ext)


def collect_ground_truth(corpus_path):
	def iter_int_dir(p):
		return filter(lambda d: d.stem.isdigit(), p.iterdir())

	def gather_files(ann_path):
		for ann_file in ann_path.iterdir():
			if ann_file.is_file() and ann_file.suffix == ".psd":
				yield ann_file

	# scanning.

	inputs = []

	for year_path in iter_int_dir(corpus_path):
		for month_path in iter_int_dir(year_path):
			for day_path in iter_int_dir(month_path):
				ann_path = day_path / "ann"
				if ann_path.is_dir():
					for ann_file in gather_files(ann_path):
						inputs.append(GroundTruthRef(ann_file, ".png"))

	ann_path = corpus_path / "0000" / "ann"
	if ann_path.is_dir():
		for ann_file in gather_files(ann_path):
			inputs.append(GroundTruthRef(ann_file, ".jpg"))

	return inputs
