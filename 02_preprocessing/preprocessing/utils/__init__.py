import numpy
import math
import cv2
import PIL.Image
import shapely.geometry
import cairo
import time

from ..labels import Label


def colorized(pixels, colors):
	im = PIL.Image.fromarray(pixels, "P")
	palette = numpy.zeros((768,), dtype=numpy.uint8)
	for i, c in enumerate(colors):
		palette[i * 3 + 0:i * 3 + 3] = c[:3]
	im.putpalette(palette)
	return im.convert("RGB")


def binarized(pixels, mask, alpha):
	background_mask = numpy.logical_not(mask)
	pixels_f32 = pixels.astype(numpy.float32)
	pixels_f32[background_mask] = pixels_f32[background_mask] * alpha + (1 - alpha) * 255
	return numpy.clip(pixels_f32, 0, 255).astype(numpy.uint8)


def mask_to_contours(mask, eps_area=100, simplify=3, convex_hulls=True, cls=shapely.geometry.LinearRing):
	mask = mask.astype(numpy.uint8)

	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	#gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

	contours = cv2.findContours(
		mask,  #gradient,
		mode=cv2.RETR_EXTERNAL,
		method=cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[1]

	if False:
		image = numpy.zeros((*mask.shape, 1), dtype=numpy.uint8)
		image[mask > 0] = 255
		# image = numpy.broadcast_to(image, [*mask.shape, 3])
		image = numpy.tile(image, (1, 1, 3))

	# image = numpy.array(image)
	# print(image.shape, image.dtype)

	polylines = []
	for c in contours:
		if len(c) < 3:
			continue

		if convex_hulls:
			hull = cv2.convexHull(c, returnPoints=False)
			# hull = numpy.array(hull, dtype=numpy.int32)
			hull = hull.reshape((hull.shape[0],))
			pts = c.reshape(c.shape[0], 2)[hull]
		else:
			pts = c.reshape(c.shape[0], 2)

		polyline = cls(pts)

		minx, miny, maxx, maxy = polyline.bounds
		if (maxx - minx) * (maxy - miny) < eps_area:
			continue
		area = shapely.geometry.Polygon(pts).area
		#print(area)
		if area < eps_area:
			continue

		polyline = polyline.simplify(simplify, preserve_topology=False)

		if not polyline.is_empty:
			polylines.append(polyline)

		if False:
			pts = polyline.coords
			for a, b in zip(pts, pts[1:]):
				a = numpy.array(a, dtype=numpy.int32)
				b = numpy.array(b, dtype=numpy.int32)
				cv2.line(image, tuple(a), tuple(b), (200, 0, 0), thickness=4)

	return polylines


def mask_to_polygons(mask, **kwargs):
	return mask_to_contours(mask, cls=shapely.geometry.Polygon, **kwargs)


def polygons_to_mask(shape, polygons):
	assert type(polygons) is list

	h, w = shape

	with cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h) as surface:
		ctx = cairo.Context(surface)

		for polygon in polygons:
			assert polygon.geom_type == 'Polygon'
			pts = list(polygon.exterior.coords)
			ctx.move_to(*pts[0])
			for p in pts[1:]:
				ctx.line_to(*p)
			ctx.set_source_rgb(1, 1, 1)
			ctx.fill()

		surface.finish()

		data = numpy.ndarray(
			shape=(surface.get_height(), surface.get_stride() // 4),
			dtype=numpy.uint32,
			buffer=surface.get_data())

		data = numpy.right_shift(data, 24).astype(numpy.uint8)

	return data > 0


def polygon_mask(polygon):
	assert polygon.geom_type == 'Polygon'

	minx, miny, maxx, maxy = polygon.bounds
	minx, miny = numpy.floor([minx, miny]).astype(numpy.int32)
	maxx, maxy = numpy.ceil([maxx, maxy]).astype(numpy.int32)

	w = int(maxx - minx)
	h = int(maxy - miny)

	pts = list(polygon.exterior.coords) - numpy.array([minx, miny])

	with cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h) as surface:
		ctx = cairo.Context(surface)
		ctx.move_to(*pts[0])
		for p in pts[1:]:
			ctx.line_to(*p)
		ctx.set_source_rgb(1, 1, 1)
		ctx.fill()

		surface.finish()

		data = numpy.ndarray(
			shape=(surface.get_height(), surface.get_stride() // 4),
			dtype=numpy.uint32,
			buffer=surface.get_data())

		data = numpy.right_shift(data, 24).astype(numpy.uint8)

	return data > 0, (minx, miny, w, h)


def extract_polygon_mask(labels, polygon):
	mask, (x, y, w, h) = polygon_mask(polygon)
	return labels[y:y + h, x:x + w][mask]


def _normalized(v):
	return v / numpy.linalg.norm(v)


def _running_mean(x, N):
	cumsum = numpy.cumsum(numpy.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / float(N)


def fast_snake(mask, label):
	import skimage.morphology
	skel = skimage.morphology.skeletonize(mask)

	dsize = 256
	scale = numpy.array([mask.shape[0] / dsize, mask.shape[1] / dsize])

	skel = cv2.resize(skel.astype(numpy.float32), (dsize, dsize), interpolation=cv2.INTER_AREA) > 0
	skel0 = skel

	pts = numpy.argwhere(skel)
	pts = numpy.flip(pts, -1)



	if label == Label.H:
		x0 = numpy.min(pts[:, 0])
		x1 = numpy.min(pts[:, 0])
		y0 = numpy.median(pts[pts[:, 0] == x0][:, 1])
		y1 = numpy.median(pts[pts[:, 0] == x1][:, 1])
	else:
		y0 = numpy.min(pts[:, 1])
		y1 = numpy.min(pts[:, 1])
		x0 = numpy.median(pts[pts[:, 1] == y0][:, 0])
		x1 = numpy.median(pts[pts[:, 1] == y1][:, 0])



	skel = skimage.filters.gaussian(skel, 3)

	n_snake_pts = dsize // 4

	import skimage.segmentation
	import skimage.filters

	r = numpy.linspace(y0, y1, n_snake_pts)
	c = numpy.linspace(x0, x1, n_snake_pts)
	init = numpy.array([r, c]).T

	snake = skimage.segmentation.active_contour(
		skel, init, boundary_condition='fixed',
		alpha=0.01, beta=0.1, w_line=20, w_edge=2, gamma=0.1, convergence=0.5,
		coordinates='rc')
	#print(len(snake))

	import skimage.measure
	snake = skimage.measure.approximate_polygon(snake, tolerance=0.1)
	#print(len(snake))

	'''
	snake = numpy.flip(snake, -1)
	pixels = numpy.array(PIL.Image.fromarray(skel0).convert("RGB"))
	for i, (a, b) in enumerate(zip(snake, snake[1:])):
		a = numpy.array(a).astype(numpy.int32)
		b = numpy.array(b).astype(numpy.int32)
		cv2.line(pixels, tuple(a), tuple(b), (255, 0, 0), thickness=1)
	return PIL.Image.fromarray(pixels)
	'''

	if False:
		pixels = numpy.array(PIL.Image.fromarray(mask).convert("RGB"))
		snake = numpy.array([p * scale for p in snake])
		f_snake = numpy.flip(snake, -1)
		for i, (a, b) in enumerate(zip(f_snake, f_snake[1:])):
			a = (numpy.array(a)).astype(numpy.int32)
			b = (numpy.array(b)).astype(numpy.int32)
			# cv2.line(pixels, tuple(a), tuple(b), (255, 0, 0), thickness=1)
			cv2.circle(pixels, tuple(a), 2, (0, 255, 0), thickness=cv2.FILLED)
		return PIL.Image.fromarray(pixels)

	return [p * scale for p in snake]


def mask_to_polyline_hq(mask, label):
	direction = numpy.array([1, 0]) if label == Label.H else numpy.array([0, 1])

	import skimage.morphology
	skel = skimage.morphology.skeletonize(mask)

	pts = numpy.argwhere(skel)
	pts = numpy.flip(pts, -1)

	if label == Label.H:
		x0 = numpy.min(pts[:, 0])
		x1 = numpy.min(pts[:, 0])
		y0 = numpy.median(pts[pts[:, 0] == x0][:, 1])
		y1 = numpy.median(pts[pts[:, 0] == x1][:, 1])
	else:
		y0 = numpy.min(pts[:, 1])
		y1 = numpy.min(pts[:, 1])
		x0 = numpy.median(pts[pts[:, 1] == y0][:, 0])
		x1 = numpy.median(pts[pts[:, 1] == y1][:, 0])

	#pts_v = numpy.dot(pts, direction)


	#y0 = numpy.min(pts[:, 1])
	#y1 = numpy.max(pts[:, 1])

	#x0 = pts[pts[:, 1] == y0][0][0]
	#x1 = pts[pts[:, 1] == y1][0][0]

	'''
	bx0, by0, bx1, by1 = cv2.boundingRect(pts)
	n_snake_pts = max(abs(by1 - by0), abs(bx1 - bx0)) // 5

	import skimage.segmentation
	import skimage.filters

	r = numpy.linspace(y0, y1, n_snake_pts)
	c = numpy.linspace(x0, x1, n_snake_pts)
	init = numpy.array([r, c]).T

	t0 = time.time()
	snake = skimage.segmentation.active_contour(
		skimage.filters.gaussian(skel, 3), init, boundary_condition='fixed',
		alpha=0.05, beta=0.1, w_line=10, w_edge=0, gamma=0.1, convergence=0.1,
		coordinates='rc')
	print("snake size", len(snake), "found in", time.time() - t0)
	'''

	snake = fast_snake(mask, label)

	# import skimage.measure
	# snake = skimage.measure.approximate_polygon(snake, tolerance=1)
	# print(len(snake))

	rtangents = []
	for p, q in zip(snake, snake[1:]):
		rtangents.append(numpy.flip(_normalized(q - p), -1))

	tangents = []
	tangents.append(rtangents[0])
	for p, q in zip(rtangents, rtangents[1:]):
		tangents.append(_normalized(p + q))
	tangents.append(rtangents[-1])

	markers = numpy.zeros(mask.shape, dtype=numpy.int32)
	for i, p in enumerate(snake):
		p = p.astype(numpy.int32)
		markers[p[0], p[1]] = i + 1

	shed = skimage.segmentation.watershed(mask, markers=markers, mask=mask)

	n_parts = numpy.max(shed)
	line = []
	widths = []

	for i, v in zip(range(1, 1 + n_parts), tangents):
		pts = numpy.argwhere(shed == i)

		if len(pts) == 0:
			# print("!", i, n_parts)
			continue

		pts = numpy.flip(pts, -1)

		cx, cy = numpy.median(pts, axis=0)
		vx, vy = v

		x, y = numpy.transpose(pts, axes=(1, 0))

		qx = x - cx
		qy = y - cy

		sv = qx * vx + qy * vy

		ux = -vy
		uy = vx

		su = qx * ux + qy * uy

		tv = (numpy.min(sv), numpy.max(sv))
		tu = (numpy.min(su), numpy.max(su))

		p = numpy.array([cx, cy]).flatten()
		v = numpy.array([vx, vy]).flatten()
		u = numpy.array([-vy, vx]).flatten()

		a = tv[0] * v + p
		b = tv[1] * v + p

		widths.append(numpy.median(numpy.abs(su)))

		line.extend([a, b])

	if not line:
		polyline = shapely.geometry.LineString()
		return polyline, 0  # FIXME

	new_line = []
	new_line.append(line[0])
	for a, b in zip(line[1::2], line[2::2]):
		new_line.append((a + b) / 2)
	new_line.append(line[-1])

	if numpy.dot(_normalized(new_line[-1] - new_line[0]), direction) < 0:
		new_line = list(reversed(new_line))

	thickness = 2 * numpy.median(widths)

	print("line done.")

	polyline = shapely.geometry.LineString(new_line)
	polyline = polyline.simplify(0.5, preserve_topology=False)

	return polyline, thickness, 0


def mask_to_polyline_robust(mask, label, accuracy=5):
	pts = numpy.argwhere(mask).astype(numpy.float32)
	pts = numpy.flip(pts, -1)

	vx, vy, cx, cy = cv2.fitLine(
		pts, cv2.DIST_L2, 0, 0.01, 0.01)

	if label == Label.H:
		if vx < 0:  # always point right
			vy = -vy
			vx = -vx
	else:
		if vy < 0:  # always point down
			vy = -vy
			vx = -vx

	x, y = numpy.transpose(pts, axes=(1, 0))

	qx = x - cx
	qy = y - cy

	sv = qx * vx + qy * vy

	ux = -vy
	uy = vx

	su = qx * ux + qy * uy

	tv = (numpy.min(sv), numpy.max(sv))
	tu = (numpy.min(su), numpy.max(su))

	num = max(math.ceil((tv[1] - tv[0]) / accuracy), 3)

	p = numpy.array([cx, cy]).flatten()
	v = numpy.array([vx, vy]).flatten()
	u = numpy.array([-vy, vx]).flatten()

	r = []
	widths = []

	t = numpy.linspace(tv[0], tv[1], num=num)
	for t0, t1 in zip(t, t[1:]):
		mask = numpy.logical_and(sv >= t0, sv <= t1)
		if numpy.sum(mask.astype(numpy.uint8)) > 0:
			part_sv = numpy.expand_dims(sv[mask], axis=0).T
			part_su = numpy.expand_dims(su[mask], axis=0).T
			pts = p + v * part_sv + u * part_su
			r.append(numpy.median(pts, axis=0))
			widths.append(numpy.median(abs(part_su)))

	if True and len(r) > 5:
		r = numpy.array(r)
		x = _running_mean(r[:, 0], 5)
		y = _running_mean(r[:, 1], 5)

		x = list(r[:2, 0]) + list(x) + list(r[-2:, 0])
		y = list(r[:2, 1]) + list(y) + list(r[-2:, 1])
		r = numpy.array([x, y]).T

		'''
		r = numpy.array(r)
		x = _running_mean(r[:, 0], 3)
		y = _running_mean(r[:, 1], 3)

		x = [r[0, 0]] + list(x) + [r[-1, 0]]
		y = [r[0, 1]] + list(y) + [r[-1, 1]]
		r = numpy.array([x, y]).T
		'''

	lines = shapely.geometry.LineString(r)
	lines = lines.simplify(0.5, preserve_topology=False)

	thickness = numpy.median(widths)
	err = thickness / (tv[1] - tv[0])

	return lines, max(1, thickness), err


def mask_to_polyline(mask, label):
	return mask_to_polyline_robust(mask, label)


def smoothened_at(pts, i):
	if i < 3 or len(pts) < 3 + i:
		return pts

	pts = numpy.array(pts.copy())

	x = _running_mean(pts[:, 0], 5)
	y = _running_mean(pts[:, 1], 5)

	k = i - 2 - 2
	if k < 0:
		return pts

	n = len(x[k:k + 5])
	n = min(n, len(pts[i - 2:i - 2 + n, 0]))

	if n < 1:
		return pts

	try:
		pts[i - 2:i - 2 + n, 0] = x[k:k + n]
		pts[i - 2:i - 2 + n, 1] = y[k:k + n]
	except ValueError:
		print(len(pts), len(x), i, len(x[k:k + 5]))
		raise

	return pts
