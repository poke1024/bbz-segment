import numpy
import math
import cv2
import PIL.Image

from .labels import Label
from .utils import polygon_mask

# -------------------------------------------------------------------------------------------------------------------------


class Baseline:
    def __init__(self, p1, p2):
        self._p1 = p1
        self._p2 = p2

    @property
    def p1(self):  # label space
        return self._p1

    @property
    def p2(self):  # label space
        return self._p2

    @property
    def v(self):
        return (self._p2 - self._p1) / self.length

    def translate(self, p):
        return Baseline(self._p1 + p, self._p2 + p)

    @property
    def length(self):
        return numpy.linalg.norm(self._p2 - self._p1)

    @property
    def centre(self):
        return (self._p1 + self._p2) / 2

    def transform(self, m):
        p1 = numpy.append(self._p1, 1)
        p2 = numpy.append(self._p2, 1)
        return Baseline(m.dot(p1), m.dot(p2))

    @property
    def dewarped(self):
        p1, p2 = self._p1, self._p2
        p = (p1 + p2) / 2
        l = numpy.linalg.norm(p2 - p1)
        v = numpy.array([l / 2, 0])
        return Baseline(p - v, p + v)

# -------------------------------------------------------------------------------------------------------------------------


class Line:  # i.e. line of text
    def __init__(self, block, polygon, baseline, debug_text):
        self._block = block
        self._polygon = polygon
        self._confident_baseline = baseline
        self._text = debug_text  # from tesseract

        mask, (x, y, w, h) = polygon_mask(polygon)
        self._bbox = (x, y, w, h)

        # weights = collections.defaultdict(lambda: 1)
        # weights[Label.TABTXT] = 2

        # determine label.
        n_labels = len([x for x in Label])
        labels = block.layout.labels
        counts = numpy.bincount(labels[y:y + h, x:x + w][mask], minlength=n_labels)
        counts[int(Label.BACKGROUND)] = 0
        counts[int(Label.TABTXT)] *= 2
        counts[int(Label.FRAKTUR_BG)] *= 2
        counts[int(Label.ANTIQUA_BG)] *= 2
        self._label = Label(numpy.argmax(counts))

        # extend baseline.
        '''
        pts = numpy.argwhere(mask).astype(numpy.int32)
        pts = numpy.flip(pts, -1)
        d = (pts - baseline.p1).dot(numpy.array([baseline.v]).T).flatten()
        t0, t1 = numpy.min(d), numpy.max(d)
        self._length = t1 - t0

        p1 = baseline.p1 + t0 * baseline.v
        p2 = baseline.p1 + t1 * baseline.v
        self._baseline = Baseline(p1, p2)
        '''
        self._baseline = baseline
        self._length = baseline.length

    @property
    def polygon(self):
        return self._polygon

    @property
    def image_space_polygon(self):
        return self._block.layout.page.polygon_to_image_space(self._polygon)

    def view(self, labels):
        x, y, w, h = self._bbox
        return labels[y:y + h, x:x + w]

    def mark(self, mask):
        # x, y, w, h = self._bbox
        line_mask, (x, y, w, h) = polygon_mask(self._polygon)
        mask[y:y + h, x:x + w] = numpy.logical_or(line_mask, mask[y:y + h, x:x + w])

    def binarized(self, alpha=0):
        pixels = self.pixels.copy()

        mask = self._mask.astype(numpy.float32)
        mask = cv2.resize(mask, tuple(reversed(pixels.shape)), interpolation=cv2.INTER_LINEAR)
        mask = mask > 0

        pixels = _binarized(pixels, mask, alpha)
        return PIL.Image.fromarray(pixels, "L")

    @property
    def labels(self):
        layout = self._block.layout
        x, y, w, h = self._bbox
        labels = layout.labels[y:y + h, x:x + w]

        im = PIL.Image.fromarray(labels, "P")
        im.putpalette(layout._palette())
        return im

    @property
    def pixels(self):
        x, y, w, h = self._bbox
        page = self._block.layout.page

        page_x0, page_y0 = page._label_to_image_space(x, y)
        page_x1, page_y1 = page._label_to_image_space(x + w, y + h)
        page_x0, page_y0 = math.floor(page_x0), math.floor(page_y0)
        page_x1, page_y1 = math.ceil(page_x1), math.ceil(page_y1)

        return page.pixels[page_y0:page_y1, page_x0:page_x1]

    @property
    def image(self):
        mask, (x, y, w, h) = polygon_mask(self.image_space_polygon)
        pixels = self._block.layout.page.pixels[y:y + h, x:x + w].copy()
        pixels[numpy.logical_not(mask)] = 255
        return PIL.Image.fromarray(pixels)

    @property
    def dominant_label(self):
        return self._label

    @property
    def mask(self):
        return self._mask

    @property
    def relative_pos(self):
        return self._pos

    @property
    def absolute_pos(self):
        return self._pos + self._block.pos

    @property
    def baseline(self):
        return self._baseline

    @property
    def length(self):
        return self._length

    @property
    def centre(self):
        return self._baseline.centre + self.absolute_pos

    @property
    def text(self):
        return self._text
