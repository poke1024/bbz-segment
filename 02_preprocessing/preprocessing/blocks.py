import numpy
import math
import cv2
import PIL.Image
import shapely


from .labels import Label
from .lines import Line, Baseline
from .utils import extract_polygon_mask, polygon_mask


class Block:
    def __init__(self, layout, polygon, label, lines=None):
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny = numpy.floor([minx, miny])
        maxx, maxy = numpy.floor([maxx, maxy])
        self._bbox = numpy.array([minx, miny, maxx - minx, maxy - miny]).astype(numpy.int32)

        self._polygon = polygon

        self._layout = layout
        self._label = label
        self._centre = numpy.array(polygon.centroid.coords[0])

        self._lines = lines if lines else []

    @property
    def polygon(self):
        return self._polygon

    @property
    def image_space_polygon(self):
        return self._layout.page.polygon_to_image_space(self._polygon)

    @property
    def page(self):
        return self._layout.page

    @property
    def fingerprint(self):
        n_labels = len([x for x in Label])
        counts = numpy.bincount(self.view(self._layout.labels)[self._mask], minlength=n_labels)
        return counts.astype(numpy.float32) / numpy.sum(counts)

    @staticmethod
    def from_polygon(layout, polygon):
        counts = numpy.bincount(extract_polygon_mask(layout.labels, polygon))
        counts[int(Label.BACKGROUND)] = 0
        label = Label(numpy.argmax(counts))

        block = Block(layout, polygon, label)

        # find baselines.
        # note: using PSM.SINGLE_BLOCK here (instead of SINGLE_COLUMN) is important
        # as it makes the warped left column in SNP2436020X-18781028-0-2-0-0 work.

        import tesserocr

        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK) as api:

            # don't force parallel baselines (defaults to 1)
            api.SetVariable("textord_parallel_baselines", "0")

            # api.SetVariable("textord_linespace_iqrlimit", "0.1")  # default 0.2

            # note: block.binarized works much worse for detecting correct baselines,
            # see SNP2436020X-18720630-0-1-0-0 "Telegraphische Depeschen" top left.
            api.SetImage(block.image)

            api.Recognize()
            # api.AnalyseLayout()  # should be sufficient but generates None iterators.

            ri = api.GetIterator()

            # if ri is None:
            #    return block

            level = tesserocr.RIL.TEXTLINE

            rest_shape = polygon

            baselines = [r.Baseline(level) for r in tesserocr.iterate_level(ri, level)]
            baselines = [l for l in baselines if l]

            for i, line in enumerate(baselines):
                # debug_text = r.GetUTF8Text(level)

                p1, p2 = line  # in image space
                p1 = numpy.array(layout.page._image_to_label_space(*p1), dtype=numpy.float32)
                p2 = numpy.array(layout.page._image_to_label_space(*p2), dtype=numpy.float32)

                # Line(local_pts, p1, p2)

                # print("y", ((p1 + p2) / 2)[1])

                v = (p2 - p1) / numpy.linalg.norm(p2 - p1)
                u = numpy.array([-v[1], v[0]])
                p = (p1 + p2) / 2 + block.pos
                splitter = shapely.geometry.LineString([p - v * 10000, p + v * 10000])

                shapes = shapely.ops.split(rest_shape, splitter)

                if len(shapes.geoms) == 1 and i == len(baselines) - 1:
                    block._lines.append(Line(
                        block, rest_shape, Baseline(p1, p2).translate(block.pos), ""))
                    continue

                if len(shapes.geoms) < 2:
                    continue

                assert len(shapes.geoms) == 2

                d = numpy.array([s.centroid.coords[0] for s in shapes.geoms]).dot(numpy.array([u]).T).flatten()
                i = numpy.argmin(d)
                line_shape = shapes.geoms[i]
                rest_shape = shapes.geoms[1 - i]

                block._lines.append(Line(
                    block, line_shape, Baseline(p1, p2).translate(block.pos), ""))

            # pixels = numpy.array(image.convert("RGB"))
            # for l in baselines:
            #    cv2.line(pixels, l[0], l[1], color=(255, 0, 0))

        #print("found %d lines for block" % len(block._lines))

        return block

    def regrouped(self, segments):

        groups = [[]]

        split = [a.dominant_label != b.dominant_label
                 for a, b in zip(self._lines, self._lines[1:])] + [False]

        for l, split_here in zip(self._lines, split):
            groups[-1].append(l)
            if split_here:
                groups.append([])

        for lines in groups:
            if not lines:
                continue

            polygon = shapely.ops.cascaded_union([l.polygon for l in lines]).convex_hull

            label = lines[0].dominant_label
            if label != Label.TABTXT:
                yield Block(self._layout, polygon, label, lines)
            else:
                yield Table(self._layout, polygon, lines, segments)

    @property
    def layout(self):
        return self._layout

    @property
    def dominant_label(self):
        return self._label

    def view(self, labels):
        x0, y0, w, h = self._bbox
        return labels[y0:y0 + h, x0:x0 + w]

    @property
    def mask(self):
        return self._mask

    @property
    def lines(self):
        return self._lines

    @property
    def pos(self):
        return self._bbox[:2]

    @property
    def bbox(self):
        return self._bbox

    @property
    def centre(self):
        return self._centre

    @property
    def width(self):
        if self._lines:
            return max(l.length for l in self._lines)
        else:
            return 0

    @property
    def derotated(self):
        pass  # only as flag

    @property
    def pixels(self):
        x0, y0, w, h = self._bbox
        page = self._layout.page

        page_x0, page_y0 = page._label_to_image_space(x0, y0)
        page_x1, page_y1 = page._label_to_image_space(x0 + w, y0 + h)
        page_x0, page_y0 = math.floor(page_x0), math.floor(page_y0)
        page_x1, page_y1 = math.ceil(page_x1), math.ceil(page_y1)

        return page.pixels[page_y0:page_y1, page_x0:page_x1]

    @property
    def image(self):
        mask, (x, y, w, h) = polygon_mask(self.image_space_polygon)
        pixels = self._layout.page.pixels[y:y + h, x:x + w].copy()
        pixels[numpy.logical_not(mask)] = 255
        return PIL.Image.fromarray(pixels)

    def binarized(self, alpha=0):
        return self.image

        '''
        pixels = self.pixels.copy()

        mask = self._mask.astype(numpy.float32)
        mask = cv2.resize(mask, tuple(reversed(pixels.shape)), interpolation=cv2.INTER_LINEAR)
        mask = mask > 0

        pixels = _binarized(pixels, mask, alpha)
        return PIL.Image.fromarray(pixels, "L")
        '''

    def annotate_baselines(self):
        pass  # using tesserocr with self.binarized

    def annotate_lines(self, pixels, colors):
        for i, line in enumerate(self.lines):
            color = colors[self.dominant_label]
            brightness = [0.75, 1.25][i % 2]
            color = numpy.clip(numpy.array(color) * brightness, 0, 255).astype(numpy.int32)
            mask, (x, y, w, h) = polygon_mask(line.image_space_polygon)
            pixels[y:y + h, x:x + w, :][mask] = color


class Cell:
    def __init__(self, row, polygon):
        self._row = row
        self._polygon = polygon

    @property
    def page(self):
        return self._row.page

    @property
    def image_space_polygon(self):
        return self.page.polygon_to_image_space(self._polygon)

    @property
    def image(self):
        mask, (x, y, w, h) = polygon_mask(self.image_space_polygon)
        pixels = self.page.pixels[y:y + h, x:x + w].copy()
        pixels[numpy.logical_not(mask)] = 255
        return PIL.Image.fromarray(pixels)


class Row:
    def __init__(self, table, line, segments):
        self._table = table
        self._line = line
        self._cell_shapes = []

        segments = [s for s in segments if s.dominant_label in (Label.TABCOL, Label.V)]
        segments = sorted(segments, key=lambda s: s.centre[0])

        if not segments:
            self._cell_shapes.append(line.polygon)
        else:
            rest_shape = line.polygon

            for s in segments:
                shapes = shapely.ops.split(rest_shape, s.xpath)

                if len(shapes.geoms) < 2:
                    continue

                d = numpy.array([s.centroid.coords[0][0] for s in shapes.geoms]).flatten()
                i = numpy.argmin(d)
                cell_shape = shapes.geoms[i]
                rest_shape = shapes.geoms[1 - i]

                self._cell_shapes.append(cell_shape)

            self._cell_shapes.append(rest_shape)

    @property
    def page(self):
        return self._table.page

    @property
    def cells(self):
        return [Cell(self, polygon) for polygon in self._cell_shapes]


class Table(Block):
    def __init__(self, layout, polygon, lines, segments):
        super(Table, self).__init__(layout, polygon, Label.TABTXT, lines)
        self._segments = list(filter(lambda s: not s.path.disjoint(polygon), segments))

    @property
    def rows(self):
        return [Row(self, line, self._segments) for line in self._lines]

    def annotate_lines(self, pixels, colors):
        base_color = numpy.array(colors[self.dominant_label])

        for i, row in enumerate(self.rows):
            row_brightness = [0.75, 1.25][i % 2]

            for j, cell in enumerate(row.cells):
                brightness = [0.75, 1.25][j % 2] * row_brightness
                color = numpy.clip(base_color * brightness, 0, 255).astype(numpy.int32)
                mask, (x, y, w, h) = polygon_mask(cell.image_space_polygon)
                pixels[y:y + h, x:x + w, :][mask] = color
