# utilities for working with 2x3 matrices, as used by OpenCV.

import numpy


def p(m, x, y):
	return m.dot(numpy.array([x, y, 1]))


def v(m, x, y):
	return m.dot(numpy.array([x, y, 0]))


def mul(a, b):
	return numpy.vstack([a, (0, 0, 1)]).dot(numpy.vstack([b, (0, 0, 1)]))[:2]


def inv(a):
	# note: might also use cv2.invertAffineTransform().
	return numpy.linalg.inv(numpy.vstack([a, (0, 0, 1)]))[:2]


def to_shapely(m):
	matrix = numpy.zeros((12, ), dtype=m.dtype)

	matrix[0:2] = m[0, 0:2]
	matrix[3:5] = m[1, 0:2]

	matrix[8] = 1

	matrix[9] = m[0, 2]
	matrix[10] = m[1, 2]

	return matrix
