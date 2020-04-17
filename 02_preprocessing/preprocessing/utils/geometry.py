import cv2
import numpy
import math
import shapely
import skimage.measure


def contours(mask, convex=False):
	contours = cv2.findContours(
		mask.astype(numpy.uint8),
		mode=cv2.RETR_EXTERNAL,
		method=cv2.CHAIN_APPROX_SIMPLE)[1]

	for c in contours:
		if len(c) < 3:
			continue

		if convex:
			hull = cv2.convexHull(c, returnPoints=False)
			hull = hull.reshape((hull.shape[0],))
			pts = c.reshape(c.shape[0], 2)[hull]
		else:
			pts = c.reshape(c.shape[0], 2)

		polygon = shapely.geometry.Polygon(pts)

		if not polygon.is_empty:
			yield polygon


def convex_contours(mask):
	return contours(mask, convex=True)


class Simplifier:
	def __init__(self, simplify=3, eps_area=100):
		self._simplify = simplify
		self._eps_area = eps_area

	def __call__(self, polygons):
		for polygon in polygons:
			if self._simplify is not None:
				polygon = polygon.simplify(
					self._simplify, preserve_topology=False)

				if polygon.is_empty:
					continue

			if self._eps_area is not None:
				minx, miny, maxx, maxy = polygon.bounds
				if (maxx - minx) * (maxy - miny) < self._eps_area:
					continue

				area = polygon.area
				if area < self._eps_area:
					continue

			yield polygon


def convex_hull(polygons):
	return shapely.ops.cascaded_union(list(polygons)).convex_hull


def estimate_angle(coords, orthogonal=False):
	coords = numpy.array(coords)

	if len(coords) < 3:
		return False

	x0 = coords[0, 0]
	x1 = coords[-1, 0]
	y0 = coords[0, 1]
	y1 = coords[-1, 1]

	try:
		if abs(x1 - x0) > abs(y1 - y0):
			model, _ = skimage.measure.ransac(
				coords,
				skimage.measure.LineModelND,
				min_samples=2,
				residual_threshold=1,
				max_trials=1000)

			y0, y1 = model.predict_y([x0, x1])
			vy = y1 - y0
			vx = x1 - x0
			phi = math.pi / 2 - math.atan2(vy, vx)
		else:
			model, _ = skimage.measure.ransac(
				numpy.flip(coords, -1),
				skimage.measure.LineModelND,
				min_samples=2,
				residual_threshold=1,
				max_trials=1000)

			x0, x1 = model.predict_y([y0, y1])
			vy = y1 - y0
			vx = x1 - x0
			phi = math.pi / 2 + math.atan2(vy, vx)

	except ValueError:
		return False

	if orthogonal:
		phi -= math.pi / 2

	phi = math.asin(math.sin(phi))  # limit to -pi/2, pi/2
	phi = numpy.degrees(phi)

	return phi
