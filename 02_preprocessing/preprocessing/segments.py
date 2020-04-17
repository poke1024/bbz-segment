import numpy
import scipy
import math
import cv2
import shapely.geometry
import enum


from .labels import Label
from .utils import mask_to_polyline, smoothened_at


class JoinResult(enum.Enum):
    OK = 0
    LABEL_FAIL = 1
    COLLAPSE_FAIL = 2
    PARALLEL_FAIL = 3
    DISTANCE_FAIL = 4
    MASK_FAIL = 5
    DIRECTION_FAIL = 6


class Segment:
    def __init__(self, segment=None):
        self._label = segment._label if segment else None
        self._mask = None
        self._path = None
        self._thickness = None
        self._err = None
        self._name = None
        self._xpath = None

    @property
    def path(self):
        return self._path

    @property
    def centre(self):
        return self._path.centroid.coords[0]

    @property
    def name(self):
        return self._name

    @property
    def thickness(self):
        return self._thickness

    @staticmethod
    def from_mask(labels, mask, name):
        segment = Segment()

        segment._label = Label(numpy.argmax(numpy.bincount(labels[mask])))
        segment._mask = mask
        segment._path, segment._thickness, segment._err = mask_to_polyline(mask, segment._label)
        segment._name = name

        return segment

    @staticmethod
    def _are_vectors_parallel(a, b):
        la, lb = numpy.linalg.norm(a), numpy.linalg.norm(b)
        length = min(la, lb)

        if length < 3:
            return True  # no reliable direction here.
        elif length < 50:
            # SNP2436020X-19200601-0-0-0-0.02 top right segment patch.
            phi_threshold = 30
        else:
            phi_threshold = 15

        try:
            dot = numpy.dot(a / la, b / lb)
            phi = numpy.degrees(abs(math.acos(dot)))
        except ValueError:
            return False

        if phi > 90:
            phi = 180 - phi
        if phi > phi_threshold:
            return False

        return True

    @staticmethod
    def parallel_and_close(a, b, distance):
        if not Segment._are_vectors_parallel(a.v, b.v):
            return False
        elif a.path.distance(b.path) > distance:
            return False
        else:
            return True

    @staticmethod
    def join_mask_counts(dominant_label, labels, a, b, thickness, debug=None):
        join_mask = numpy.zeros(labels.shape, dtype=numpy.uint8)
        cv2.line(
            join_mask,
            tuple(a.astype(numpy.int32)),
            tuple(b.astype(numpy.int32)),
            1,
            int(max(1, math.ceil(thickness))))
        join_mask = join_mask > 0

        if debug:
            debug(dominant_label, join_mask)

        n_labels = len([x for x in Label])
        counts = numpy.bincount(labels[join_mask], minlength=n_labels)
        counts = counts.astype(numpy.float32) / float(numpy.sum(counts))
        return counts, join_mask

    @staticmethod
    def check_join_mask(dominant_label, labels, a, b, thickness, ignore=[], debug=None):
        counts, join_mask = Segment.join_mask_counts(dominant_label, labels, a, b, thickness, debug)
        # print(counts)

        for label in ignore:
            counts[int(label)] = 0

        if dominant_label == Label.TABCOL:
            if counts[int(Label.TABCOL)] > counts[int(Label.TABTXT)]:
                counts[int(Label.TABTXT)] = 0
        elif dominant_label == Label.H:
            if counts[int(Label.H)] > counts[int(Label.TABTXT)]:
                counts[int(Label.TABTXT)] = 0
        elif dominant_label == Label.V:
            # allow to fix nasty broken line in SNP2436020X-18720201-1-0-0-0.05
            if counts[int(Label.V)] > counts[int(Label.ANTIQUA_SM)]:
                counts[int(Label.ANTIQUA_SM)] = 0

        counts[int(Label.BACKGROUND)] = 0
        counts[int(dominant_label)] = 0
        counts[int(Label.BORDER)] = 0

        # print(counts)
        if numpy.sum(counts) > 0:
            return False, join_mask
        else:
            return True, join_mask

    @staticmethod
    def join(labels, a, b, indices=None, debug=None):
        if a.dominant_label != b.dominant_label:
            return JoinResult.LABEL_FAIL, None

        a_pts = a.endpoints
        b_pts = b.endpoints

        if all(a_pts[0] == b_pts[0]) or all(a_pts[1] == b_pts[1]):
            return JoinResult.COLLAPSE_FAIL, None

        if indices is None:
            distances = [(numpy.linalg.norm(p1 - p2), i1, i2)
                         for i1, p1 in enumerate(a_pts)
                         for i2, p2 in enumerate(b_pts)]

            indices = sorted(distances, key=lambda d: d[0])[0][1:]

        # note: indices might not always be the best distance here.
        # we might have checked the shortest distance earlier and
        # already rejected it.

        a_i, b_i = indices

        vv = b_pts[b_i] - a_pts[a_i]
        #vv /= numpy.linalg.norm(vv)

        if a.dominant_label == Label.H:
            dominant_dir = numpy.float32([1, 0])
        else:
            dominant_dir = numpy.float32([0, 1])
        tangent = vv / numpy.linalg.norm(vv)
        # endpoints are ordered in dominant_dir.
        if a_i == 1:
            # b must lie in direction of dominant_dir.
            if tangent.dot(dominant_dir) < 0:
                return JoinResult.DIRECTION_FAIL, False
        else:
            if tangent.dot(dominant_dir) > 0:
                return JoinResult.DIRECTION_FAIL, False

        v0 = a.v if a.length > b.length else b.v
        #v0 /= numpy.linalg.norm(v0)

        #if a.dominant_label == Label.H:
        #    print("!", vv, v0)

        #if (vv / numpy.linalg.norm(vv)).dot(v0 / numpy.linalg.norm(v0)) < 0:
        #    return JoinResult.PARALLEL_FAIL, None

        if not Segment._are_vectors_parallel(v0, vv):
            return JoinResult.PARALLEL_FAIL, None

        v0 /= numpy.linalg.norm(v0)
        orth_dist = abs(numpy.dot(a_pts[0] - b_pts[0], numpy.array([-v0[1], v0[0]])))
        # SNP2436020X-19200601-0-0-0-0.02 top right segment patch.
        if orth_dist > 25:
            return JoinResult.DISTANCE_FAIL, None

        thickness = (a.thickness * a.length + b.thickness * b.length) / (a.length + b.length)

        ok, join_mask = Segment.check_join_mask(
            a.dominant_label, labels,
            a_pts[a_i], b_pts[b_i],
            thickness, debug=debug)
        if not ok:
            return JoinResult.MASK_FAIL, None

        if a.dominant_label == Label.TABCOL:
            # prevent table columns to extend through text blocks,
            # see e.g. spilling over "Erze." in SNP2436020X-19200601-1-0-0-0.03
            # or "Noten." in 2436020X_1925-02-27_70_98_006

            test_tt = 10

            for test_t in (100,):  # 500, 400, 300, 200, 100, 50):
                counts, _ = Segment.join_mask_counts(
                    a.dominant_label, labels, a_pts[a_i], b_pts[b_i], test_t)
                if counts[int(Label.V)] == 0.:
                    test_tt = test_t
                    break

            ok, _ = Segment.check_join_mask(
                a.dominant_label, labels,
                a_pts[a_i], b_pts[b_i],
                test_tt, ignore=[Label.TABTXT, Label.H], debug=debug)
            if not ok:
                return JoinResult.MASK_FAIL, None

        a_coords = list(a.path.coords)
        b_coords = list(b.path.coords)
        if a_i == 1 and b_i == 0:
            j_coords = a_coords + b_coords
            j_coords = smoothened_at(j_coords, len(a_coords))
        elif b_i == 1 and a_i == 0:
            j_coords = b_coords + a_coords
            j_coords = smoothened_at(j_coords, len(b_coords))
        else:  # should not happen
            print("WARN", "illegal order", a_i, b_i)
            if a_i == 0:
                a_coords = list(reversed(a_coords))
            if b_i == 1:
                b_coords = list(reversed(b_coords))
            j_coords = a_coords + b_coords
            j_coords = smoothened_at(j_coords, len(a_coords))

        try:
            assert any(j_coords[0] != a_pts[a_i])
            assert any(j_coords[-1] != a_pts[a_i])
            assert any(j_coords[0] != b_pts[b_i])
            assert any(j_coords[-1] != b_pts[b_i])
        except:
            print("ERR", "j", j_coords[0], j_coords[-1], "a_i", a_i, "a_pts", a_pts)
            print("a_coords", a_coords)
            print("b_coords", b_coords)
            raise

        joined = Segment(a)
        joined._mask = numpy.logical_or.reduce([a._mask, b._mask, join_mask])
        joined._path = shapely.geometry.LineString(j_coords)
        joined._thickness = thickness
        joined._err = min(a._err, b._err)
        joined._name = "%s-%s" % (a._name, b._name)

        return JoinResult.OK, joined

    def transform(self, t):
        segment = Segment(self)

        if self._mask is not None:
            segment._mask = t.mask(self._mask)

        segment._path = t.geometry(self._path)

        # we assume that m is only a rotation.

        segment._thickness = self._thickness
        segment._err = self._err
        segment._name = self._name

        return segment

    @property
    def p(self):
        return numpy.array(self._path.coords[0])

    @property
    def v(self):
        v = numpy.array(self._path.coords[-1]) - numpy.array(self._path.coords[0])
        length = numpy.linalg.norm(v)
        if length == 0:
            return numpy.array([0, 0])
        else:
            return v / length

    @property
    def u(self):
        v = self.v
        return numpy.array([-v[1], v[0]])

    @property
    def angle(self):
        vx, vy = self.v
        return numpy.degrees(math.atan2(vy, vx))

    @property
    def error(self):
        return self._err

    @property
    def length(self):
        return self._path.length

    @property
    def mask(self):
        return self._mask

    @property
    def endpoints(self):
        a = self._path.coords[0]
        b = self._path.coords[-1]
        return numpy.array(a), numpy.array(b)

    def endpoint_i(self, p):
        for i, q in enumerate(self.endpoints):
            if all(p == q):
                return i
        return None

    @property
    def dominant_label(self):
        return self._label

    @property
    def dewarped_v(self):
        if self.dominant_label == Label.H:
            v = [numpy.sign(self._v[0]), 0]
        else:
            v = [0, numpy.sign(self._v[1])]

        return numpy.array(v, dtype=numpy.float32).flatten()

    @property
    def dewarped(self):
        line = (self._p, self.dewarped_v)

        segment = Segment(self)

        segment._p, segment._v = self._p, self.dewarped_v

        segment._tv = self._tv
        segment._tu = self._tu

        segment._len = self._len
        segment._err = self._err

        segment._sv = None
        segment._su = None

        return segment

    def extend_by(self, amount):
        if amount == 1:
            return self
        else:
            spl, (t0, t1) = self.spline

            length = t1 - t0
            t0 -= length * amount / 2
            t1 += length * amount / 2

            t = numpy.linspace(t0, t1, math.ceil(len(self.path.coords) * (1 + amount)))
            s = scipy.interpolate.splev(t, spl)

            vv = self.v * numpy.array([t]).T
            uu = self.u * numpy.array([s]).T

            segment = Segment()

            segment._label = self._label
            segment._mask = self._mask  # not extended
            segment._path = shapely.geometry.LineString(self.p + vv + uu)
            segment._thickness = self._thickness
            segment._err = self._err
            segment._name = self._name

            return segment

    def _extrapolate(self, spl, shape, num=10):
        p = self.p
        v = self.v
        u = self.u

        dx = abs(v[0])
        dy = abs(v[1])

        if self.dominant_label == Label.H:
            t0 = -p[0] / dx
            t1 = (shape[1] - p[0]) / dx
        else:
            t0 = -p[1] / dy
            t1 = (shape[0] - p[1]) / dy

        t = numpy.linspace(t0, t1, num)
        s = scipy.interpolate.splev(t, spl)

        vv = v * numpy.array([t]).T
        uu = u * numpy.array([s]).T

        return shapely.geometry.LineString(self.p + vv + uu)

    def _estimate_thickness(self):
        spl = self.spline
        self._mask

    @property
    def spline(self):
        #spl = scipy.interpolate.UnivariateSpline(pts[0], pts[1], k=2)
        #spl.set_smoothing_factor(0.5)

        pts = list(self._path.simplify(10, preserve_topology=False).coords)
        pts = numpy.array(pts)

        p = self.p
        v = self.v
        u = self.u

        lpts = pts - p

        x = lpts.dot(numpy.array([v]).T).flatten()
        y = lpts.dot(numpy.array([u]).T).flatten()

        i = numpy.argsort(x)
        x = x[i]
        y = y[i]

        m = lpts.shape[0]

        if m < 10:
            k = 1
        else:
            k = 2

        try:
            spl = scipy.interpolate.splrep(
               x, y, k=k, s=m + math.sqrt(2 * m))
        except ValueError:
            print("error in splrep", x, y, m)
            return None

        return spl, (numpy.min(x), numpy.max(x))

    def compute_xpath(self, shape):
        spl, _ = self.spline
        self._xpath = self._extrapolate(spl, shape)

    @property
    def xpath(self):
        return self._xpath
