import enum
import numpy
import math
import PIL.Image
import cv2
import itertools
import collections
import skimage.morphology
import shapely.geometry
import shapely.ops
import intervaltree
import tesserocr

from shapely.strtree import STRtree


class Label(enum.IntEnum):
	BACKGROUND = 0
	FRAKTUR_BG = 1
	FRAKTUR_SM = 2
	ANTIQUA_BG = 3
	ANTIQUA_SM = 4
	TABTXT = 5
	BORDER = 6
	TABCOL = 7
	H = 8
	V = 9
	H_SM = 10
	ILLUSTRATION = 11
	LINE = 12  # computed


def _gen_pts(segments):
	for i, s in enumerate(segments):
		a, b = s.endpoints
		if any(a != b):
			yield (i, a)
			yield (i, b)


def _estimate_skew(segments):
	total_length = 0
	sum_of_angles = 0

	if any(s.dominant_label in (Label.V, Label.TABCOL) for s in segments):
		selected_labels = (Label.V, Label.TABCOL)
	else:
		selected_labels = (Label.H, )

	from .utils.geometry import estimate_angle

	for s in segments:
		if s.dominant_label in selected_labels:

			phi = estimate_angle(s.path.coords, s.dominant_label == Label.H)

			if phi is False:
				continue

			if s.dominant_label == Label.TABCOL and abs(phi) > 15:
				# there are such things as tables rotated by 90 degrees. so
				# be careful to not use these for skew estimation here.
				continue

			# print("!", s.dominant_label, phi, s.length, vx, vy)

			l = s.length

			total_length += l
			sum_of_angles += phi * l

	return sum_of_angles / total_length


class AnnotationsGenerator:
	def __init__(self, data, segments):
		from .utils import transform

		self._data = data
		self._segments = segments
		self._master = self._generate(segment_thickness_scale=2)

		self._skew = _estimate_skew(segments)
		self._deskewing = transform.Rotate(reversed(data.shape), -self._skew)

		self._master = self._master.transform(self._deskewing)
		self._stops = self._generate(
			segment_thickness_scale=2,
			segment_extend_amount=dict((
				(Label.V, 0.02),  # fixes Handelsblatt sep in SNP2436020X-19100601-0-0-0-0.12
				(Label.H, 0.2)))).transform(self._deskewing)

		self._deskewed_segments = [s.transform(self._deskewing) for s in self._segments]

	@property
	def deskewing(self):
		return self._deskewing

	@property
	def master(self):
		return self._master

	@property
	def stops(self):
		return self._stops

	@property
	def segments(self):
		return self._deskewed_segments

	def _generate(self, segment_thickness_scale=1, segment_extend_amount=None):
		data = self._data.copy()

		line_shift = 0
		line_scale = 1 << line_shift

		def draw_polyline(pts, label, thickness):
			for a, b in zip(pts, pts[1:]):
				a = numpy.round(a * line_scale).astype(numpy.int32)
				b = numpy.round(b * line_scale).astype(numpy.int32)
				cv2.line(data, tuple(a), tuple(b), label, thickness=thickness, shift=line_shift)

		for i, s in enumerate(self._segments):

			if segment_extend_amount and s.dominant_label in segment_extend_amount:
				s = s.extend_by(segment_extend_amount[s.dominant_label])

				# clip this against all non-extended labels.
				for j, t in enumerate(self._segments):
					if i != j:
						try:
							shapes = shapely.ops.split(s.path, t.path)
							if len(shapes.geoms) == 2:
								k = numpy.argmax([geom.length for geom in shapes.geoms])
								s._path = shapes.geoms[k]  # HACK
						except ValueError:  # error in split on full overlap
							pass  # ignore

			thickness = max(2, int(math.floor(s.thickness * segment_thickness_scale)))
			draw_polyline(s.path.coords, s.dominant_label, thickness)

		return Annotations(data)


class Merger:
	default_pipeline = (
		lambda m: m.merge_by_endpoints(),
		lambda m: m.merge_parallel(),
		lambda m: m.filter_by_length(),
		lambda m: m.filter_by_quality()
	)

	def __init__(self, labels, segments):
		self._labels = labels.copy()  # will get modified
		self._segments = list(segments)
		self._join_regions = numpy.zeros(labels.shape, dtype=numpy.uint8)

	@property
	def segments(self):
		return self._segments

	def filter_by_quality(self, max_error=0.1, min_length=1e-3):
		scale = min(*self._labels.shape)
		self._segments = [s for s in self._segments if s.error < max_error and s.length >= min_length * scale]

	def _debug_join_region(self, dominant_label, mask):
		# for debugging
		if self._join_regions is not None:
			self._join_regions[mask] = dominant_label

	@property
	def join_regions(self):
		return Annotations(self._join_regions)

	def merge_by_endpoints(self):
		# try to merge by nearest endpoints.

		from .segments import Segment, JoinResult

		max_distance = dict((
			(Label.H, 500),
			(Label.V, 500),
			(Label.TABCOL, 500)))

		# 10 max distance for TABCOL fixes table spilling over
		# "Erze." text in SNP2436020X-19100601-1-0-0-0.03

		pts = list(_gen_pts(self._segments))

		candidates = []
		for i, (ia, a) in enumerate(pts):
			for j, (ib, b) in enumerate(pts[:i]):
				if ia == ib:
					continue
				seg_a = self._segments[ia]
				seg_b = self._segments[ib]
				if seg_a.dominant_label != seg_b.dominant_label:
					continue
				d = numpy.linalg.norm(a - b)
				# print("dist", ia, ib, d)
				if d < max_distance[seg_a.dominant_label]:
					candidates.append((d, ia, a, ib, b))
		candidates = sorted(candidates, key=lambda c: c[0])

		for _, ia, a, ib, b in candidates:
			seg_a = self._segments[ia]
			seg_b = self._segments[ib]

			if seg_a is seg_b:
				continue

			# print("checking", a, b)

			ea = seg_a.endpoint_i(a)
			if ea is None:  # already merged?
				continue

			eb = seg_b.endpoint_i(b)
			if eb is None:  # already merged?
				continue

			if ea == eb:
				# never merge two start points or two end points. we assume
				# our segment endpoint indices are always ordered from left
				# to right (for H) or top to bottom (for V, TABCOL).
				continue

			# if ea == eb:
			#    print("NOPE", a, b)
			# assert ea != eb

			# print("!", a.name, b.name)
			result, s = Segment.join(
				self._labels, seg_a, seg_b, (ea, eb), self._debug_join_region)

			if result == JoinResult.OK and s is not None:
				# patch labels.
				self._labels[s.mask] = s.dominant_label

				self._segments[ia] = s
				self._segments[ib] = s

				assert all(seg_a.endpoints[1 - ea] == s.endpoints[1 - ea])
				assert all(seg_b.endpoints[1 - eb] == s.endpoints[1 - eb])

				assert s.endpoint_i(a) is None
				assert s.endpoint_i(b) is None

				# print("merged", ia, ib, s.endpoints)

		segs = dict()
		for s in self._segments:
			segs[id(s)] = s
		self._segments = list(segs.values())

	def merge_parallel(self, overlap_buffer=1, close_distance=5):
		# now merge parallel segments that overlap or are very near
		# (see overlap_buffer).

		from .segments import Segment

		while True:
			segs = collections.defaultdict(list)
			for s in self._segments:
				segs[s.path.bounds].append(s)

			tree = STRtree([s.path for s in self._segments])
			merged = set()
			added = []

			for s in self._segments:
				if s.name in merged:
					continue
				for other_path in tree.query(s.path.buffer(overlap_buffer)):
					if not other_path.equals(s.path):
						for t in segs[other_path.bounds]:
							if t.name in merged:
								continue
							if s.dominant_label != t.dominant_label:
								continue
							if Segment.parallel_and_close(s, t, close_distance):
								added.append(Segment.from_mask(
									self._labels,
									numpy.logical_or(s.mask, t.mask),
									"%s+%s" % (s.name, t.name)))
								merged.add(s.name)
								merged.add(t.name)

			old_n = len(self._segments)
			self._segments = [s for s in self._segments if s.name not in merged] + added
			if old_n == len(self._segments):
				break

	def filter_by_length(self):
		# get rid of remaining noise.
		min_segment_lengths = dict((
			(Label.H, 10),
			(Label.V, 10),
			(Label.TABCOL, 0)))

		self._segments = [s for s in self._segments if s.length > min_segment_lengths[s.dominant_label]]

	def filter_by_region(self, region):
		self._segments = list(filter(lambda s: not s.path.disjoint(region), self._segments))


class Annotations:
	colors = dict((
		(Label.BACKGROUND, (255, 255, 255)),
		(Label.FRAKTUR_BG, (0, 249, 0)),
		(Label.FRAKTUR_SM, (255, 64, 255)),
		(Label.ANTIQUA_BG, (255, 251, 0)),
		(Label.ANTIQUA_SM, (170, 121, 66)),
		(Label.TABTXT, (255, 147, 0)),
		(Label.BORDER, (168, 211, 170)),
		(Label.TABCOL, (242, 148, 138)),
		(Label.H, (255, 38, 0)),
		(Label.V, (0, 48, 255)),
		(Label.H_SM, (0, 253, 255)),
		(Label.ILLUSTRATION, (163, 215, 223)),
		(Label.LINE, (80, 200, 20))
	))

	def __init__(self, labels, img_path=None):
		self._labels = labels
		self._img_path = img_path
		self._skew = None

	def apply_lut(self, lut):
		return Annotations(lut[self._labels], self._img_path)

	@property
	def shape(self):
		return self._labels.shape

	@property
	def stats(self):
		xx = numpy.bincount(self._labels.flatten())
		print("H", xx[int(Label.H)])
		print("V", xx[int(Label.V)])
		print("TABCOL", xx[int(Label.TABCOL)])

	@staticmethod
	def palette():
		palette = numpy.zeros((3 * 256,), dtype=numpy.uint8)
		for label, (r, g, b) in Annotations.colors.items():
			i = int(label) * 3
			palette[i:i + 3] = (r, g, b)
		return palette

	@property
	def image(self):
		im = PIL.Image.fromarray(self._labels, "P")
		im.putpalette(self.palette())
		return im

	def c_image(self, *colors):
		im = PIL.Image.fromarray(self._labels, "P")
		pal = self.palette().copy()
		for k, v in colors:
			pal[int(k) * 3:int(k) * 3 + 3] = v
		im.putpalette(pal)
		return im

	@property
	def path(self):
		return self._img_path

	@property
	def document_image(self):
		if self._img_path:
			return PIL.Image.open(self._img_path).convert('L')
		else:
			return None

	@property
	def labels(self):
		return self._labels

	@property
	def scale(self):
		return min(*self._labels.shape)

	def text_mask(self):
		return self.mask(
			Label.FRAKTUR_BG, Label.FRAKTUR_SM,
			Label.ANTIQUA_BG, Label.ANTIQUA_SM,
			Label.TABTXT)

	def mask(self, *labels):
		n_labels = len([l for l in Label])
		lut = numpy.zeros((n_labels, ), dtype=numpy.bool)
		for l in labels:
			lut[int(l)] = True
		return lut[self._labels]

		#return numpy.logical_or.reduce(
		#	list(self._labels == int(l) for l in labels))

	def _find_segment_components(self, *labels):
		mask = self.mask(*labels)

		'''
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(kernel_size * numpy.array(axis)))
		closed_mask = cv2.morphologyEx(
			mask.astype(numpy.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
		reduced_text_mask = cv2.morphologyEx(
			self.text_mask().astype(numpy.uint8), cv2.MORPH_OPEN, kernel, iterations=5)

		closed_mask = numpy.logical_and(
			closed_mask.astype(numpy.bool),
			numpy.logical_not(reduced_text_mask.astype(numpy.bool))).astype(numpy.uint8)
		'''

		n, c = cv2.connectedComponents(mask.astype(numpy.uint8))
		for i in range(1, n + 1):
			yield c == i

	@property
	def unprocessed_segments(self, min_pts=5):
		# detect segments based on label features.

		# min_pts of 5 works good to remove noise from annotations
		# that splitters seg lines. noise often has 3 or 4 pixels.

		from .segments import Segment

		index = 1
		for mask in itertools.chain(
				self._find_segment_components(Label.H),
				self._find_segment_components(Label.TABCOL, Label.V)):

			if numpy.sum(mask.astype(numpy.uint8)) < min_pts:
				continue  # not enough points

			s = Segment.from_mask(self._labels, mask, str(index))

			if s is not None:
				yield s
				index += 1

	def merger(self, pipeline=None):
		if pipeline is None:
			pipeline = Merger.default_pipeline

		merger = Merger(
			self._labels, list(self.unprocessed_segments))

		for p in pipeline:
			p(merger)

		return merger

	@property
	def clabels(self):
		h, w = self.shape
		labels = numpy.empty((h, w), dtype=numpy.uint8)
		labels.fill(int(Label.BACKGROUND))

		m = self.mask(*[label for label in Label if label not in (Label.H, Label.V, Label.TABCOL)])
		labels[m] = self._labels[m]
		return labels

	def repaired_segments(self, pipeline=None):
		return self.merger(pipeline).segments

	def regions(self, pipeline=None):
		segments = self.repaired_segments(pipeline)
		return Regions(self.clabels, segments)

	def transform(self, t):
		return Annotations(t.labels(self._labels))


def _add_hull(hulls, mask):
	if mask.any():
		return numpy.logical_or(hulls, skimage.morphology.convex_hull_image(mask))
	else:
		return hulls


def _regions_to_convex_hull(mask):
	from .utils import mask_to_polygons, polygons_to_mask
	polygons = mask_to_polygons(mask, convex_hulls=True)
	polygons = [polygon for polygon in polygons if polygon.area > 100]
	# polygons = [polygon.exterior.convex_hull for polygon in polygons]
	return polygons_to_mask(mask.shape, polygons)

	'''
	labels, num = skimage.morphology.label(mask.astype(numpy.bool), return_num=True)
	labels = skimage.morphology.remove_small_objects(labels, 100)
	
	hulls = numpy.zeros(mask.shape, dtype=numpy.bool)
	for i in range(1, num + 1):
		hulls = _add_hull(hulls, labels == i)
	return hulls
	'''


'''
def _merge_convex(region, mask):
	if (numpy.logical_and(region, mask)).any():
		labels, num = skimage.morphology.label(region, return_num=True)
		for i in range(1, num + 1):
			candidate = (labels == i)
			if (numpy.logical_and(candidate, mask)).any():
				mask = skimage.morphology.convex_hull_image(numpy.logical_or(candidate, mask))
				break

	return numpy.logical_or(region, mask)
'''


class PolygonV:
	def __init__(self, polygons):
		self._tree = intervaltree.IntervalTree()

		self._polygons = polygons

		for i, polygon in enumerate(polygons):
			_, miny, _, maxy = polygon.bounds
			self._tree[miny:maxy + 1] = i

	def query(self, polygon):
		_, miny, _, maxy = polygon.bounds
		min_y_overlap = 3
		# set min_y_overlap to not merge "tüchtiger Beamter" with top text on SNP2436020X-19100601-0-0-0-0.10

		ivs = sorted(
			self._tree.overlap(miny + min_y_overlap, maxy + 1 - min_y_overlap),
			key=lambda iv: self._polygons[iv.data].bounds[0])

		return list(map(lambda iv: iv.data, ivs))

	def remove(self, index):
		_, miny, _, maxy = self._polygons[index].bounds
		self._tree.removei(miny, maxy + 1, index)


class HMerger:
	def __init__(self, polygons, stoppers):
		self._polygons = polygons
		self._stop = STRtree(stoppers)

		# we sometimes touch text regions which prevent merging, that's
		# why we erode a bit here to allow merging to allow touching
		# text regions.
		# -20 derived from tests on SNP2436020X-19100601-0-0-0-0.09.
		self._erosion = 20

	@property
	def polygons(self):
		return self._polygons

	def _removed_polygon(self, polygon):
		pass

	def _added_polygon(self, polygon):
		pass

	def _can_merge(self, polygon1, polygon2):
		centroid1 = polygon1.centroid.coords[0]
		centroid2 = polygon2.centroid.coords[0]
		connection = shapely.geometry.LineString([centroid1, centroid2])

		if self._stop.query(connection):
			return False
		else:
			return True

	def _can_merge_into(self, polygon):
		if self._stop.query(polygon):
			return False
		else:
			return True

	def _sort_by_x(self):
		self._polygons = sorted(self._polygons, key=lambda p: p.bounds[0])

	def _begin_merge_step(self):
		self._sort_by_x()

	def _merge_step(self):
		self._begin_merge_step()

		tree = PolygonV(self._polygons)

		was_merged = numpy.zeros((len(self._polygons),), dtype=numpy.bool)
		merged_polygons = []

		for i, polygon1 in enumerate(self._polygons):
			if was_merged[i]:
				continue

			tree.remove(i)

			for j in tree.query(polygon1):
				assert j != i

				polygon2 = self._polygons[j]

				if not self._can_merge(polygon1, polygon2):
					continue

				union = shapely.ops.cascaded_union([polygon1, polygon2]).convex_hull

				if not self._can_merge_into(union.buffer(-self._erosion)):
					continue

				# good to go.
				was_merged[j] = True
				tree.remove(j)

				self._removed_polygon(polygon1)
				self._removed_polygon(polygon2)
				self._added_polygon(union)

				# update polygon1.
				polygon1 = union

			merged_polygons.append(polygon1)

		success = len(merged_polygons) < len(self._polygons)
		self._polygons = merged_polygons
		return success

	def merge(self):
		while self._merge_step():
			pass

		return self._polygons


class HTextMerger(HMerger):
	def __init__(self, unbinarized, polygons, stoppers):
		super().__init__(polygons, stoppers)
		self._unbinarized = unbinarized
		self._ocr = dict()
		self._unmerged = None
		self._debug = False

	def _line_height(self, polygon):
		key = tuple(polygon.centroid.coords[0])
		if key not in self._ocr:
			from .utils import polygons_to_mask
			mask = polygons_to_mask(self._unbinarized.shape, [polygon])

			minx, miny, maxx, maxy = polygon.bounds
			minx, miny = numpy.floor(numpy.array([minx, miny])).astype(numpy.int32)
			maxx, maxy = numpy.ceil(numpy.array([maxx, maxy])).astype(numpy.int32)

			pixels = self._unbinarized[miny:maxy, minx:maxx]
			mask = mask[miny:maxy, minx:maxx]
			pixels[numpy.logical_not(mask)] = 255

			with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK) as api:
				api.SetImage(PIL.Image.fromarray(pixels, "L"))

				heights = []
				for i, data in enumerate(api.GetTextlines()):
					bbox = data[1]
					heights.append(bbox["h"])

				if heights:
					n_lines = len(heights)
					lh = numpy.min(heights)
				else:
					lh = maxy - miny
					n_lines = 1

				if self._debug:
					api.Recognize()

					ri = api.GetIterator()
					level = tesserocr.RIL.TEXTLINE

					text = ""
					#lines = []
					for r in tesserocr.iterate_level(ri, level):
						#baseline = r.Baseline(level)
						#if baseline:
						#	p1, p2 = baseline
						#	lines.append(shapely.geometry.LineString([p1, p2]))

						try:
							text += r.GetUTF8Text(level) + " "
						except RuntimeError:
							pass

					#print("txt", text.strip(), "lh", lh, "#", n_lines)
				else:
					text = ""

			self._ocr[key] = (n_lines, lh, text)

		return self._ocr[key]

	def _begin_merge_step(self):
		super()._begin_merge_step()
		self._unmerged = STRtree(self._polygons)

	def _removed_polygon(self, polygon):
		super()._removed_polygon(polygon)
		key = tuple(polygon.centroid.coords[0])
		if key in self._ocr:
			del self._ocr[key]

	def _can_merge(self, polygon1, polygon2):
		if not super()._can_merge(polygon1, polygon2):
			return False

		n_lines1, lh1, text1 = self._line_height(polygon1)
		n_lines2, lh2, text2 = self._line_height(polygon2)

		if lh1 is None or lh2 is None:
			# never merge if no text is detected.
			return False

		if n_lines1 == 1 and n_lines2 == 1:
			y1 = polygon1.centroid.coords[0][1]
			y2 = polygon2.centroid.coords[0][1]
			if abs(y1 - y2) > min(lh1, lh2) * 0.5:
				return False

		minx, miny, maxx, maxy = polygon1.bounds
		dx1 = maxx - minx

		minx, miny, maxx, maxy = polygon2.bounds
		dx2 = maxx - minx

		# 1.5 based on SNP2436020X-19100601-0-0-0-0.10 and SNP2436020X-19100601-0-0-0-0.11.
		# and SNP2436020X-19100601-1-0-0-0.03.
		max_dy_ratio = 1.5
		dy_ratio = max(lh1 / lh2, lh2 / lh1)

		if dy_ratio > max_dy_ratio:
			if self._debug:
				print("reject by max_dy_ratio", dy_ratio, lh1, lh2, text1, text2)
			return False

		max_dx = 0.25

		if polygon1.distance(polygon2) > max_dx * max(dx1, dx2):
			if self._debug:
				print("reject by distance", text1, text2)
			return False

		return True

	def _can_merge_into(self, polygon):
		if not super()._can_merge_into(polygon):
			return False

		# if merging merges more than two unmerged segments,
		# something went wrong. we might have skipped an
		# intermediate polygon that now gets merged.
		if len(self._unmerged.query(polygon)) > 2:
			return False

		return True


def _merge_convex_all(polygons):
	polygons = [polygon for polygon in polygons if not polygon.is_empty]

	import rtree
	idx = rtree.index.Index()

	for i, polygon in enumerate(polygons):
		try:
			idx.insert(i, polygon.bounds)
		except:
			print("rtree error with", polygon.bounds, polygon, polygon.geom_type)
			raise

	remaining = list(range(len(polygons)))
	next_polygon_id = len(polygons)

	candidates = dict((i, polygon) for i, polygon in enumerate(polygons))

	while remaining:
		i = remaining.pop()

		if i not in candidates:
			continue
		indices = [i]
		polygon1 = candidates[i]

		for j in list(idx.intersection(polygon1.bounds)):
			if j != i:
				polygon2 = candidates[j]
				if polygon2.intersects(polygon1):
					indices.append(j)

		if len(indices) > 1:
			merged = shapely.ops.cascaded_union(
				[candidates[k] for k in indices]).convex_hull

			for k in indices:
				idx.delete(k, candidates[k].bounds)
				del candidates[k]

			candidates[next_polygon_id] = merged
			remaining.append(next_polygon_id)
			idx.insert(next_polygon_id, merged.bounds)

			next_polygon_id += 1

	return list(candidates.values())


class Regions:
	def __init__(self, clabels, segments):
		morpholizer = Morpholizer(clabels, segments)
		self._morph = morpholizer

		self._tables = morpholizer.table_polygons()
		self._text = morpholizer.text_polygons()
		self._slabels = morpholizer.slabels
		self._segments = morpholizer.segments

		self._shape = self._slabels.shape
		self._deskewing = morpholizer.deskewing

		self._clabels = clabels

	@property
	def deskewing(self):
		return self._deskewing

	@property
	def input(self):
		return self._morph.input

	@property
	def to_deskewed_segments(self):  # for debugging
		pixels = self._morph.input.labels.copy()

		def draw_polyline(pts, label, thickness):
			for a, b in zip(pts, pts[1:]):
				a = numpy.round(a ).astype(numpy.int32)
				b = numpy.round(b).astype(numpy.int32)
				cv2.line(pixels, tuple(a), tuple(b), label, thickness=thickness, shift=0)

		for s in self._segments:
			draw_polyline(list(s.path.coords), s.dominant_label, 16)

		return Annotations(pixels)

	def to_text_boxes(self):
		mask = numpy.zeros(self._shape, dtype=numpy.uint8)
		mask.fill(int(Label.BACKGROUND))

		for i, polygon in enumerate(self._text):
			minx, miny, maxx, maxy = polygon.bounds

			p1 = numpy.array([minx, miny]).astype(numpy.int32)
			p2 = numpy.array([maxx, maxy]).astype(numpy.int32)
			cv2.rectangle(mask, tuple(p1), tuple(p2), int(Label.ANTIQUA_SM), thickness=2)

			cv2.putText(
				mask, "(%d,%d)" % (minx, miny), (int(minx), int((miny + maxy) / 2)),
				cv2.FONT_HERSHEY_SIMPLEX, 1, int(Label.ANTIQUA_SM), 2)

		return Annotations(mask).transform(self.deskewing.inverse)

	def hmerge(self, unbinarized):
		from .utils import polygons_to_mask

		unbinarized = cv2.warpAffine(
			unbinarized,
			self.deskewing.matrix,
			self.deskewing.target_size,
			flags=cv2.INTER_AREA)

		merger = HMerger(
			self._tables,
			self._text + [s.path for s in self._segments if s.dominant_label != Label.TABCOL])
		self._tables = merger.merge()

		merger = HTextMerger(
			unbinarized,
			self._text,
			self._tables + [s.path for s in self._segments])
		self._text = merger.merge()

	def to_annotations(self):
		from .utils import polygons_to_mask

		optimized_mask = numpy.zeros(self._shape, dtype=numpy.uint8)
		optimized_mask.fill(int(Label.BACKGROUND))

		text_mask = polygons_to_mask(self._shape, self._text)
		tables_mask = polygons_to_mask(self._shape, self._tables)

		optimized_mask[tables_mask] = int(Label.TABTXT)
		optimized_mask[text_mask] = int(Label.ANTIQUA_SM)

		slabels_mask = self._slabels != int(Label.BACKGROUND)
		optimized_mask[slabels_mask] = self._slabels[slabels_mask]

		ann = Annotations(optimized_mask).transform(self._deskewing.inverse)
		ann._labels[self._clabels == int(Label.BORDER)] = int(Label.BORDER)
		ann._labels[self._clabels == int(Label.ILLUSTRATION)] = int(Label.ILLUSTRATION)
		return ann


class Morpholizer:
	def __init__(self, labels, segments):
		gen = AnnotationsGenerator(labels, segments)

		self._annotations = gen.master
		self._stops = gen.stops
		self._deskewing = gen.deskewing
		self._segments = gen.segments

	@property
	def input(self):
		return self._annotations

	@property
	def shape(self):
		return self._annotations.shape

	@property
	def segments(self):  # deskewed
		return self._segments

	@property
	def deskewing(self):
		return self._deskewing

	@property
	def slabels(self):
		h, w = self._annotations.shape

		labels = numpy.empty((h, w), dtype=numpy.uint8)
		labels.fill(int(Label.BACKGROUND))

		m = self._annotations.mask(
			Label.H, Label.V, Label.TABCOL)
		labels[m] = self._annotations.labels[m]

		return labels

	def v_separator_stopper(self, kernel=(10, 30), tabcol=True):
		return self._stops.mask(
			*([Label.V] + ([Label.TABCOL] if tabcol else []))).astype(numpy.uint8)

		'''        
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)

		no_text_mask = self._annotations.mask(
			*([Label.V] + ([Label.TABCOL] if tabcol else []))).astype(numpy.uint8)

		no_text_mask = cv2.morphologyEx(
			no_text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

		return cv2.dilate(no_text_mask, kernel, iterations=1)
		'''

	def separator_stopper(self, kernel=(10, 10), tabcol=True):
		return numpy.logical_or(
			self._stops.mask(Label.H, Label.BORDER, Label.H_SM),
			self.v_separator_stopper(tabcol=tabcol)).astype(numpy.uint8)

		'''
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)

		no_text_mask = self._annotations.mask(
			Label.H, Label.BORDER, Label.H_SM).astype(numpy.uint8)

		no_text_mask = cv2.morphologyEx(
			no_text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

		return numpy.logical_or(cv2.dilate(
			no_text_mask, kernel, iterations=1), self.v_separator_stopper(tabcol=tabcol)).astype(numpy.uint8)
		'''

	def table_stopper(self, kernel=(5, 3), iterations=3):
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)

		stopper = self._annotations.mask(Label.TABTXT).astype(numpy.uint8)

		for _ in range(iterations):
			stopper = cv2.morphologyEx(
				stopper, cv2.MORPH_CLOSE, kernel, iterations=1)

			stopper = cv2.dilate(stopper, kernel, iterations=1)

		return stopper

	def text_stopper(self, kernel=(7, 5), iterations=3):
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)

		stopper = self._annotations.mask(
			Label.FRAKTUR_BG, Label.FRAKTUR_SM, Label.ANTIQUA_BG, Label.ANTIQUA_SM).astype(numpy.uint8)

		for _ in range(iterations):
			stopper = cv2.morphologyEx(
				stopper, cv2.MORPH_CLOSE, kernel, iterations=1)

			stopper = cv2.dilate(stopper, kernel, iterations=1)

		return stopper

	def _text_region(self, text_label, kernel=(10, 3), iterations=2, close_iterations=1):
		despeckle_tol = 15

		no_text_mask = numpy.logical_or(self.separator_stopper(), self.table_stopper())

		# text region.
		text_mask = self._annotations.mask(text_label)
		text_mask = skimage.morphology.remove_small_objects(text_mask.astype(numpy.bool), despeckle_tol)
		text_mask = text_mask.astype(numpy.uint8)

		extend_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
		stopper = numpy.logical_not(no_text_mask)

		for _ in range(iterations):
			text_mask = cv2.dilate(
				text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel), iterations=1)

			text_mask = cv2.morphologyEx(
				text_mask, cv2.MORPH_CLOSE, extend_kernel, iterations=close_iterations)
			text_mask = numpy.logical_and(text_mask, stopper).astype(numpy.uint8)

			text_mask = numpy.logical_and(text_mask, stopper).astype(numpy.uint8)

		text_mask = numpy.logical_and(text_mask, stopper)

		return text_mask.astype(numpy.uint8)

	@property
	def text_macro(self):
		return self._text_region((8, 3), 3, close_iterations=3)

	@property
	def text_background(self):
		background0 = self._annotations.mask(Label.ILLUSTRATION, Label.TABTXT)
		background0 = cv2.dilate(
			background0.astype(numpy.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)
		# kernel, see SNP2436020X-19200601-1-0-0-0.03.
		return numpy.logical_and(background0, numpy.logical_not(self.text_macro))

	def illustrations(self, macro_regions):
		background = self._annotations.mask(Label.ILLUSTRATION)
		background = cv2.dilate(
			background.astype(numpy.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)
		# kernel, see SNP2436020X-19200601-1-0-0-0.03.
		background = numpy.logical_and(background, numpy.logical_not(macro_regions))
		from .utils import mask_to_polygons
		return shapely.ops.cascaded_union(mask_to_polygons(background, convex_hulls=True))

	def _erode(self, h):
		return h.buffer(-2)

	def text_polygons(self):
		text_labels = (Label.FRAKTUR_BG, Label.FRAKTUR_SM, Label.ANTIQUA_BG, Label.ANTIQUA_SM)

		results = []
		for text_label in text_labels:
			other_text_labels = [l for l in text_labels if l != text_label]

			background0 = self._annotations.mask(Label.ILLUSTRATION, Label.TABTXT, *other_text_labels)
			background0 = cv2.dilate(
				background0.astype(numpy.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)), iterations=1)

			mini_regions = self._text_region(text_label, (8, 2), 1)
			micro_regions = self._text_region(text_label, (8, 3), 1)
			macro_regions = self._text_region(text_label, (8, 3), 3, close_iterations=3)
			# (8, 3), 3 ensures that "Handelsnachrichten." head stays separated on SNP2436020X-19100601-0-0-0-0.12
			# 8 (< 9!) ensures that "Berliner Börsenzeitung" stays horizontally separated, SNP2436020X-19100601-1-0-0-0.00
			# FIXME: (9, 3) would connect "Ein Sommernachtstraum" in SNP2436020X-19000601-0-0-0-0.13.

			# kernel, see SNP2436020X-19200601-1-0-0-0.03.
			macro_regions = numpy.logical_and(macro_regions, numpy.logical_not(background0))

			labels, num = skimage.morphology.label(macro_regions.astype(numpy.bool), return_num=True)
			#region = numpy.zeros(self._annotations.shape, dtype=numpy.bool)
			hulls = []

			from .utils.geometry import contours, convex_contours, Simplifier, convex_hull
			simplify = Simplifier(simplify=0)

			# detect border cases like "Schiller-Theater" in SNP2436020X-19100601-0-0-0-0.11.
			tree = STRtree([s.path for s in self.segments if s.dominant_label == Label.V])

			for i in range(1, num + 1):
				mask = numpy.logical_and(labels == i, micro_regions)
				if not mask.any():
					continue

				hull = convex_hull(simplify(contours(mask)))

				segment_by = None
				for s in tree.query(hull):
					if hull.buffer(-5).contains(shapely.geometry.Point(*s.centroid.coords)):
						segment_by = s
						break

				if segment_by:
					micros = list(simplify(convex_contours(
						numpy.logical_and(labels == i, mini_regions))))

					region = collections.defaultdict(list)
					coords = numpy.array(segment_by.coords)
					x0 = numpy.median(coords[:, 0])
					y0 = numpy.min(coords[:, 1])
					for m in micros:
						x, y = m.centroid.coords[0]
						if y < y0:
							region["top"].append(m)
						elif x < x0:
							region["left"].append(m)
						else:
							region["right"].append(m)
					for k, v in region.items():
						hulls.append(shapely.ops.cascaded_union(v).convex_hull)
				else:
					hulls.append(hull)

			hulls = _merge_convex_all(hulls)
			hulls = [self._erode(h) for h in hulls]
			results.extend(hulls)

		#region = polygons_to_mask(mask.shape, hulls)

		illustrations = self.illustrations(macro_regions)
		#region = numpy.logical_and(region, numpy.logical_not(illustrations))

		# remove_small_objects fixes noise in SNP2436020X-19200601-1-0-0-0.03.
		#region = skimage.morphology.remove_small_objects(region, 100)

		#return region.astype(numpy.uint8)

		result = shapely.ops.cascaded_union(results).difference(illustrations)
		if result.geom_type == "Polygon":
			return [result]
		else:
			polygons = list(result.geoms)
			return [p for p in polygons if p.geom_type == "Polygon" and p.area > 100]

	def _table_regions_at_iterations(self, kernel=(10, 3), iterations=(3, 5)):
		despeckle_tol = 15

		table_mask = self._annotations.mask(Label.TABTXT, Label.TABCOL)
		table_mask = skimage.morphology.remove_small_objects(table_mask.astype(numpy.bool), despeckle_tol)
		table_mask = table_mask.astype(numpy.uint8)

		stopper = numpy.logical_not(numpy.logical_or(self.separator_stopper(tabcol=False), self.text_stopper()))
		results = []

		extend_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
		for i in range(max(iterations)):
			table_mask = cv2.morphologyEx(
				table_mask, cv2.MORPH_CLOSE, extend_kernel, iterations=1)
			table_mask = numpy.logical_and(table_mask, stopper).astype(numpy.uint8)

			table_mask = cv2.dilate(table_mask, extend_kernel, iterations=1)
			table_mask = numpy.logical_and(table_mask, stopper).astype(numpy.uint8)

			if (i + 1) in iterations:
				results.append(_regions_to_convex_hull(table_mask).astype(numpy.uint8))

		return tuple(results)

	def table_polygons(self, kernel=(6, 5)):
		# kernel = (6, 5) fixes missed left bottom " in SNP2436020X-19100601-0-0-0-0.13
		# and fixes non-contiguous table regions in SNP2436020X-19100601-1-0-0-0.03
		# and avoids spilling over v sep in SNP2436020X-19000601-0-0-0-0.09

		#mini_regions = self._table_regions_at_iterations((3, 3), (1,))[0]

		micro_regions, macro_regions = self._table_regions_at_iterations(kernel, (2, 5))
		# don't do more than 5 iterations here, otherwise tables will merge over text in
		# SNP2436020X-19100601-0-0-0-0.14

		background0 = self.text_stopper(kernel=(7, 3), iterations=1)
		# iterations=1 tuned via SNP2436020X-19100601-0-0-0-0.13.
		# kernel, see SNP2436020X-19200601-1-0-0-0.03.

		labels, num = skimage.morphology.label(
			macro_regions.astype(numpy.bool), return_num=True)

		from .utils.geometry import contours, Simplifier, convex_hull, convex_contours
		simplify = Simplifier(simplify=0)

		# detect border cases like bottom table in 2436020X_1925-02-27_70_98_006.
		tree = STRtree([s.path for s in self.segments if s.dominant_label == Label.V])

		hulls = []

		#tables = numpy.zeros(self._annotations.shape, dtype=numpy.bool)
		for i in range(1, num + 1):
			added = numpy.logical_and(labels == i, micro_regions)

			if not added.any():
				continue

			added = skimage.morphology.convex_hull_image(added)
			added = numpy.logical_and(added, numpy.logical_not(background0))
			hull = convex_hull(simplify(contours(added)))

			segment_by = None
			for s in tree.query(hull):
				if hull.buffer(-5).contains(shapely.geometry.Point(*s.centroid.coords)):
					segment_by = s
					break

			if segment_by:
				added = numpy.logical_and(labels == i, micro_regions)
				added = numpy.logical_and(added, numpy.logical_not(background0))
				for c in convex_contours(added):
					hulls.append(c)
			else:
				hulls.append(hull)

		return _merge_convex_all(hulls)

	'''
	def generate(self):
		optimized_mask = numpy.zeros(self._annotations.shape, dtype=numpy.uint8)

		optimized_mask[self.table_region() > 0] = int(Label.TABTXT)
		optimized_mask[self.text_region() > 0] = int(Label.ANTIQUA_SM)

		for label in (Label.V, Label.TABCOL):
			sep_mask = self._annotations.mask(label).astype(numpy.uint8)
			#sep_mask = cv2.dilate(sep_mask, (5, 1), iterations=3)
			optimized_mask[sep_mask > 0] = int(label)

		for label in (Label.H, ):
			sep_mask = self._annotations.mask(label).astype(numpy.uint8)
			#sep_mask = cv2.dilate(sep_mask, (1, 5), iterations=1)
			optimized_mask[sep_mask > 0] = int(label)

		return Annotations(optimized_mask).rotate(self._skew)
	'''
