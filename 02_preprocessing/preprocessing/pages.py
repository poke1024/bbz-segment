import numpy
import math
import cv2
import itertools
import PIL.Image
import collections
import shapely.geometry
import shapely.ops

from tqdm import tqdm

from .labels import Label, Merger, Annotations
from .blocks import Block, Table
from .utils import mask_to_contours, mask_to_polygons, polygon_mask


def centroid_coordinate(axis):
    return lambda g: g.centroid.coords[0][axis]


def _polygons(geometry):
    if geometry.geom_type == 'MultiPolygon':
        return geometry.geoms
    else:
        return [geometry]


def pick_first(segments, axis):
    i = numpy.argmin(list(map(lambda s: s.path.centroid.coords[0][axis], segments)))
    return segments[i]


def best_split(content, segments):
    candidates = collections.defaultdict(list)
    for s in segments:
        if content.disjoint(s.xpath):
            candidates[s.dominant_label].append(s)

    if Label.H in candidates:
        return pick_first(candidates[Label.H], 1)  # topmost
    elif Label.V in candidates:
        return pick_first(candidates[Label.V], 0)  # leftmost
    else:
        candidates = collections.defaultdict(list)
        for s in segments:
            candidates[s.dominant_label].append(s)

        # FIXME. this might need to be more complex: v splitting needs
        # to account for top, bottom, right and left regions.

        if Label.H in candidates:
            return pick_first(candidates[Label.H], 1)  # topmost
        elif Label.V in candidates:
            return pick_first(candidates[Label.V], 0)  # leftmost
        else:
            return None


class LayoutNode:
    def __init__(self, page, frame, content, segments):
        self._page = page
        self._frame = frame
        self._content = content
        self._segments = segments

    @property
    def tables(self):
        return [block for block in self.blocks if type(block) == Table]

    def determine_blocks(self):
        leaves = list(self.gather_leaves())
        for leaf in tqdm(leaves, "determining blocks"):
            leaf._find_and_add_blocks()

    def determine_order(self, index):
        pass

    def _annotate_lines(self, pixels):
        pass

    def _annotate_blocks(self, pixels):
        pass

    def _annotate_layout(self, pixels, level, descend):
        pass

    def annotate(self, mode="blocks"):
        image = self._page.image

        pixels = numpy.zeros(shape=(image.size[1], image.size[0], 3), dtype=numpy.uint8)
        pixels.fill(255)
        if mode == "blocks":
            self._annotate_blocks(pixels)
        elif mode == "lines":
            self._annotate_lines(pixels)
        background = numpy.array(PIL.Image.blend(PIL.Image.fromarray(pixels, "RGB"), image, 0.75))

        self._annotate_layout(background, 1, True)
        return PIL.Image.fromarray(background)

    def _polyline(self, pixels, pts, color, thickness=4):
        pts = pts + [pts[0]]
        for a, b in zip(pts, pts[1:]):
            a = self._page._label_to_image_space(*a)
            b = self._page._label_to_image_space(*b)
            a = numpy.array(a, dtype=numpy.int32)
            b = numpy.array(b, dtype=numpy.int32)
            cv2.line(pixels, tuple(a), tuple(b), color, thickness=thickness)


class SplitNode(LayoutNode):
    def __init__(self, page, frame, content, segments, children):
        super(SplitNode, self).__init__(page, frame, content, segments)
        self._children = children

    def __getitem__(self, key):
        return self._children[key]

    @property
    def blocks(self):
        return list(itertools.chain(*[c.blocks for c in self._children]))

    def gather_leaves(self):
        return itertools.chain(*[c.gather_leaves() for c in self._children])

    def determine_order(self, index):
        for c in self._children:
            index = c.determine_order(index)
        return index

    def _annotate_lines(self, pixels):
        for c in self._children:
            c._annotate_lines(pixels)

    def _annotate_blocks(self, pixels):
        for c in self._children:
            c._annotate_blocks(pixels)

    def _annotate_layout(self, pixels, level, descend):
        if descend:
            for c in self._children:
                c._annotate_layout(pixels, level + 1, descend)

        if level == 1:
            self._polyline(pixels, list(self._frame.exterior.coords), (200, 0, 0))

        for s in self._segments:
            if s.dominant_label in (Label.H, Label.V):
                self._polyline(pixels, list(s.path.coords), (0, 200, 0))


class LeafNode(LayoutNode):
    def __init__(self, page, frame, content, segments):
        super(LeafNode, self).__init__(page, frame, content, segments)
        self._index = None
        self._blocks = None

    @property
    def blocks(self):
        return self._blocks

    def gather_leaves(self):
        return [self]

    def _find_and_add_blocks(self):
        blocks = []

        for polygon in _polygons(self._content):
            blocks.append(Block.from_polygon(self._page.layout, polygon))

        blocks = sorted(blocks, key=lambda block: block.centre[1])

        self._blocks = list(itertools.chain(*[block.regrouped(self._segments) for block in blocks]))

    @property
    def blocks(self):
        return self._blocks

    def _annotate_lines(self, pixels):
        for block in self._blocks:
            block.annotate_lines(pixels, Layout.colors)

    def _annotate_blocks(self, pixels):
        for block in self._blocks:
            # block boundary.
            color = Annotations.colors[block.dominant_label]
            mask, (x, y, w, h) = polygon_mask(block.image_space_polygon)
            pixels[y:y + h, x:x + w, :][mask] = color

            # draw baselines.
            for line in block.lines:
                baseline = line.baseline

                p1 = numpy.array(self._page._label_to_image_space(*baseline.p1), dtype=numpy.int32)
                p2 = numpy.array(self._page._label_to_image_space(*baseline.p2), dtype=numpy.int32)

                cv2.line(pixels, tuple(p1), tuple(p2), color=(80, 200, 20), thickness=2)

    def _annotate_layout(self, pixels, level, descend):
        if self._index is not None:
            # table column separators.
            if any(b.dominant_label == Label.TABTXT for b in self._blocks):
                for s in self._segments:
                    if s.dominant_label == Label.TABCOL:
                        self._polyline(pixels, list(s.path.coords), (0, 0, 200))

            # leaf node/reading order index.
            a = self._frame.centroid.coords[0]
            a = self._page._label_to_image_space(*a)
            a = numpy.array(a, dtype=numpy.int32)
            cv2.putText(
                pixels, str(self._index), tuple(a),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 0, 0), 8)

    def determine_order(self, index):
        self._index = index
        return index + 1

    def split(self):
        candidate_segments = [s for s in self._segments if s.dominant_label in (Label.H, Label.V)]

        split_segment = best_split(self._content, candidate_segments)
        if split_segment is None:
            if candidate_segments:
                print("did not find any good splits among %s." % candidate_segments)
            return self

        split_frame = shapely.ops.split(self._frame, split_segment.xpath)
        assert split_frame.geom_type == "GeometryCollection"
        geoms = list(split_frame.geoms)
        if len(geoms) != 2:
            print("number of geometries was %d." % len(geoms))
            return self

        child_frame = geoms

        if split_segment.dominant_label == Label.H:
            child_frame = sorted(child_frame, key=centroid_coordinate(1))
        else:
            child_frame = sorted(child_frame, key=centroid_coordinate(0))

        child_content = []
        for f in child_frame:
            child_content.append(f.intersection(self._content))

        child_segments = [[] for _ in child_frame]

        for s in self._segments:
            if s is split_segment:
                continue
            j = numpy.argmax([f.intersection(s.path).length for f in child_frame])
            child_segments[j].append(s)

        children = []
        for f, c, s in zip(child_frame, child_content, child_segments):
            children.append(LeafNode(self._page, f, c, s).split())

        return SplitNode(self._page, self._frame, self._content, self._segments, children)

# -------------------------------------------------------------------------------------------------------------------------


class Layout:
    block_threshold = 20  # minimum number of pixels

    def __init__(self, page, annotations, segments=None):
        self._page = page
        self._annotations = annotations
        self._skew = None

        self._area_mask = None
        self._noise_margins = None
        self._frame = None

        self.determine_rough_area_mask()

        if segments is None:
            self._segments = annotations.unprocessed_segments
        else:
            self._segments = segments

    def simplify_segments(self, pipeline):
        merger = Merger(self._annotations.labels, self._segments)

        for p in pipeline:
            p(merger)

        self._segments =  merger.segments

    @property
    def scale(self):
        return self._annotations.scale

    @property
    def page(self):
        return self._page

    @property
    def labels(self):
        return self._annotations.labels

    def determine_rough_area_mask(self):
        text_mask = self._annotations.text_mask().astype(numpy.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        closed_mask = cv2.morphologyEx(
            text_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        sep_mask = self._annotations.mask(
            Label.BORDER, Label.TABCOL, Label.H, Label.V)

        closed_mask = numpy.logical_and(closed_mask, numpy.logical_not(sep_mask))

        self._area_mask = closed_mask

    def determine_frame(self):
        _, page_width = self.labels.shape
        width_threshold = page_width * 0.1

        def n_noise(polygons, where):
            for i, polygon in tqdm(enumerate(polygons), "determine %s margin" % where):
                block = Block.from_polygon(self, polygon)
                if block.width > width_threshold:
                    return i
            return len(polygons)

        polygons = mask_to_polygons(
            self._area_mask.astype(numpy.uint8),
            simplify=5,
            eps_area=0)

        polygons = [p for p in polygons if p.bounds]

        good = []
        self._noise_margins = dict()

        for where in ("left", "right"):
            if where == "left":
                candidates = sorted(polygons, key=lambda p: p.bounds[0])
            else:
                candidates = sorted(polygons, key=lambda p: p.bounds[2], reverse=True)

            i = n_noise(candidates, where)

            if i > 0:
                area = shapely.ops.cascaded_union(candidates[:i]).convex_hull
                if not area.is_empty:
                    self._noise_margins[where] = area

            good.append(shapely.ops.cascaded_union(candidates[i:]).convex_hull)

        # compute frame.
        self._frame = (good[0].intersection(good[1])).convex_hull

    @property
    def content_mask(self):
        # call this after self.merge_segments() to get high quality results.

        text_mask = self._annotations.text_mask().astype(numpy.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

        closed_mask = cv2.morphologyEx(
            text_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        sep_mask = numpy.zeros(self.labels.shape, dtype=numpy.uint8)

        def draw_polyline(pts, thickness):
            for a, b in zip(pts, pts[1:]):
                a = numpy.round(a).astype(numpy.int32)
                b = numpy.round(b).astype(numpy.int32)
                cv2.line(sep_mask, tuple(a), tuple(b), (1,), thickness=thickness)

        t_scale = 3

        for s in self._segments:
            if s.dominant_label in (Label.H, Label.V):
                draw_polyline(s.path.coords, t_scale * max(2, int(math.ceil(s.thickness))))

        return numpy.logical_and(closed_mask, numpy.logical_not(sep_mask > 0))

    @property
    def content(self):
        # call this after self.merge_segments() to get high quality results.

        areas = mask_to_contours(self.content_mask)
        return shapely.ops.cascaded_union([a.convex_hull for a in areas])

    @property
    def image(self):
        return self._annotations.image

    @property
    def skew(self):
        return self._annotations.skew

    def rotate(self, phi):
        size = tuple(reversed(self.labels.shape))
        cx, cy = numpy.array(size, dtype=numpy.float64) / 2
        m = cv2.getRotationMatrix2D((cx, cy), phi, 1.0)
        return Layout(self._page, Annotations(cv2.warpAffine(
            self.labels,
            m,
            dsize=size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=int(Label.BACKGROUND))))

    def _overlay(self, background, overlay, labels, alpha=0.5):
        overlay_p = PIL.Image.fromarray(overlay, "P")

        palette = numpy.empty((3 * 256, ), dtype=numpy.uint8)
        palette.fill(255)
        for i, k in enumerate(labels):
            palette[3 + i * 3:6 + i * 3] = Annotations.colors[k]
        overlay_p.putpalette(palette)

        return PIL.Image.blend(
            background.convert("L").convert("RGB"),
            overlay_p.convert("RGB").resize(background.size),
            alpha)

    def annotate_content(self, background):
        overlay = numpy.array(background).copy()

        def polyline(pts, color):
            pts = pts + [pts[0]]
            for a, b in zip(pts, pts[1:]):
                a = self._page._label_to_image_space(*a)
                b = self._page._label_to_image_space(*b)
                a = numpy.array(a, dtype=numpy.int32)
                b = numpy.array(b, dtype=numpy.int32)
                cv2.line(overlay, tuple(a), tuple(b), color, thickness=4)

        for r in self.content.geoms:
            if r.geom_type == 'Polygon':
                polyline(list(r.exterior.coords), (200, 0, 0))

        # frame.
        polyline(list(self._frame.exterior.coords), (0, 200, 0))

        # noise areas.
        for area in self._noise_margins.values():
            polyline(list(area.exterior.coords), (0, 0, 200))

        return PIL.Image.fromarray(overlay, "RGB")

    def annotate_content_mask(self, background):
        h, w = self.labels.shape
        overlay = numpy.zeros(shape=(h, w), dtype=numpy.uint8)
        kinds = [Label.LINE]
        overlay[self.content_mask] = 1
        return self._overlay(background, overlay, kinds, 0.75)

    def annotate_segments_join(self, background, i, j):
        im = self.annotate_segments(background)

        a = self.segments[i].spline()
        b = self.segments[j].spline()

        mask = _colorized(
            a.join_mask(b).astype(numpy.uint8),
            [(255, 255, 255), (200, 0, 200)])

        return PIL.Image.blend(
            im,
            mask.resize(background.size),
            0.5)

    def annotate_segments(self, background, *kinds, mode='mask', show_names=False, accuracy=50, **kwargs):
        if not kinds:
            kinds = [Label.H, Label.V, Label.TABCOL]

        h, w = self.labels.shape  # small layout image size
        overlay = numpy.zeros(shape=(h, w), dtype=numpy.uint8)

        # segments = [s.dewarped for s in self._segments]  # DEBUG
        segments = self._segments

        k2i = dict((k, 1 + i) for i, k in enumerate(kinds))
        s_labels = [k2i.get(s.dominant_label, 0) for s in segments]

        # annotate names.
        for i, (s, label) in enumerate(zip(segments, s_labels)):
            if label < 1:
                continue

            # annotate with unique segment index.
            a, _ = s.endpoints
            a = numpy.round(a).astype(numpy.int32)
            cv2.putText(
                overlay, s.name if show_names else str(i), tuple(a),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (label, ), 4)

        def draw_polyline(pts, label, thickness):
            for a, b in zip(pts, pts[1:]):
                a = numpy.round(a).astype(numpy.int32)
                b = numpy.round(b).astype(numpy.int32)
                cv2.line(overlay, tuple(a), tuple(b), (label,), thickness=thickness)

        # now annotate segments themselves.
        if mode == "mask":
            for s, label in zip(segments, s_labels):
                if label < 1:
                    continue

                overlay[s.mask] = label

        elif mode == "line":
            for i, (s, label) in enumerate(zip(segments, s_labels)):
                if label < 1:
                    continue

                thickness = max(2, int(math.ceil(s.thickness)))
                draw_polyline(s.path.coords, label, thickness)

        elif mode == "xtra":
            for i, (s, label) in enumerate(zip(segments, s_labels)):
                if label < 1:
                    continue

                thickness = max(2, int(math.ceil(s.thickness)))
                draw_polyline(s.path.coords, label, thickness)

                xpath = s.xpath
                if xpath is None:
                    continue

                draw_polyline(xpath.coords, label, thickness // 2)

        else:
            raise RuntimeError("illegal mode %s" % mode)

        return self._overlay(background, overlay, kinds, 0.75)

    def annotate_blocks(self, background):
        h, w = self.labels.shape  # small layout image size
        overlay = numpy.zeros(shape=(h, w), dtype=numpy.uint8)

        def draw_polyline(pts, label, thickness):
            for a, b in zip(pts, pts[1:]):
                a = numpy.round(a).astype(numpy.int32)
                b = numpy.round(b).astype(numpy.int32)
                cv2.line(overlay, tuple(a), tuple(b), (label,), thickness=thickness)

        for block in self.blocks:
            color = int(1 + block.dominant_label)

            # block mask.
            #block.view(overlay)[block.mask] = color

            # block baselines.
            for line in block.lines:
                baseline = line.baseline

                p1 = baseline.p1.astype(numpy.int32)
                p2 = baseline.p2.astype(numpy.int32)

                cv2.line(overlay, tuple(p1), tuple(p2), color=int(1 + Label.LINE), thickness=2)

            # block bbox.
            draw_polyline(list(block.polygon.exterior.coords), color, 5)
            '''
            p1 = tuple(map(int, block.pos))

            p2 = numpy.array(block.bbox[2:]) + numpy.array(p1)
            p2 = tuple(map(int, p2))

            cv2.rectangle(overlay, p1, p2, color=color, thickness=5)
            '''

            # block centre.
            p = block.centre.astype(numpy.int32)
            cv2.circle(overlay, tuple(p), 10, color=int(1 + Label.LINE), thickness=cv2.FILLED)

        return self._overlay(background, overlay, [l.value for l in Label])

    def annotate_lines(self, background):
        h, w = self.labels.shape  # small layout image size
        overlay = numpy.zeros(shape=(h, w), dtype=numpy.uint8)

        for block in self.blocks:
            for line in block.lines:
                color = int(1 + line.dominant_label)
                line.view(overlay)[line.mask] = color

            # block bbox.
            p1 = tuple(map(int, block.pos))

            p2 = numpy.array(block.bbox[2:]) + numpy.array(p1)
            p2 = tuple(map(int, p2))

            cv2.rectangle(overlay, p1, p2, color=int(1 + block.dominant_label), thickness=5)

        return self._overlay(background, overlay, [l.value for l in Label])

    @property
    def segments(self):
        return self._segments

    @property
    def deskew_transform(self):  # in labels space
        phi = self.skew
        h, w = self.labels.shape
        return cv2.getRotationMatrix2D((w / 2, h / 2), -phi, 1.0)

    @property
    def deskewed_segments(self):
        m = self.deskew_transform
        return [s.transform(m) for s in self._segments]

    @property
    def dewarped_segments(self):
        return [s.dewarped for s in self.deskewed_segments]

    @property
    def baselines(self):
        return list(itertools.chain(*[b.baselines for b in self.blocks]))

    @property
    def deskewed_baselines(self):
        m = self.deskew_transform
        return [l.transform(m) for l in self.baselines]

    @property
    def dewarped_baselines(self):
        return [l.dewarped for l in self.deskewed_baselines]

    def dewarping_projections(self, kinds=[Label.H, Label.V, Label.TABCOL], accuracy=None):
        if accuracy is None:
            accuracy = 50

        length_threshold = self.scale // 10

        orig_s = []
        proj_s = []

        orig_p = []
        proj_p = []

        if True:  # segment dewarping
            for s, t in zip(self._segments, self.dewarped_segments):
                assert s.dominant_label == t.dominant_label

                if s.length > length_threshold and s.dominant_label in kinds:

                    a = s.markers_from_mask(accuracy)
                    b = t.markers_from_regression(accuracy)
                    assert len(a) == len(b)

                    orig_s.append(s)
                    proj_s.append(t)

                    orig_p.append(a)
                    proj_p.append(b)

        if True:  # baseline dewarping
            for s, t in zip(self.baselines, self.dewarped_baselines):
                orig_s.append(None)
                proj_s.append(None)

                orig_p.append(numpy.array([s.p1, s.p2]))
                proj_p.append(numpy.array([t.p1, t.p2]))

        return orig_s, proj_s, orig_p, proj_p


# -------------------------------------------------------------------------------------------------------------------------

class Page:
    def __init__(self):
        self._image = None
        self._pixels = None
        self._layout = None

    def preprocess(self):
        self._layout.determine_frame()
        self._layout.simplify_segments(
            [lambda m: m.filter_by_region(self._layout._frame)] +
            list(Merger.default_pipeline)
        )

    @property
    def pixels(self):  # no transforms here.
        if self._pixels is None:
            self._pixels = numpy.array(self._image.convert("L"))
        return self._pixels

    @property
    def image(self):
        return self._image

    @property
    def foreground(self):
        mask = (self._layout.labels != int(Label.BACKGROUND)).astype(numpy.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        pixels = self.pixels.copy()
        mask = cv2.resize(mask, tuple(reversed(pixels.shape)), interpolation=cv2.INTER_LINEAR)

        pixels[mask == 0] = 255
        return PIL.Image.fromarray(pixels, "L")

    @property
    def layout(self):
        return self._layout

    @property
    def frame(self):
        import tesserocr

        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.AUTO_ONLY) as api:
            api.SetImage(self.foreground)

            pts = []
            for _, bbox in api.GetConnectedComponents():
                x = bbox["x"]
                y = bbox["y"]
                w = bbox["w"]
                h = bbox["h"]
                pts.append((x, y))
                pts.append((x + w, y + h))

        pts = numpy.array(pts, dtype=numpy.int32)
        return cv2.boundingRect(pts)

    @staticmethod
    def from_path(bnet, path):
        page = Page()
        page._image = PIL.Image.open(path)
        page._layout = bnet(path, page)
        return page

    def polygon_to_image_space(self, polygon):
        pts = list(polygon.exterior.coords)
        pts = [self._label_to_image_space(*p) for p in pts]
        return shapely.geometry.Polygon(pts)

    def _label_to_image_space(self, x, y):
        iw, ih = self._image.size
        lh, lw = self._layout.labels.shape
        return x * (iw / lw), y * (ih / lh)

    def _image_to_label_space(self, x, y):
        iw, ih = self._image.size
        lh, lw = self._layout.labels.shape
        return x * (lw / iw), y * (lh / ih)

    @property
    def blocks(self):
        return self._layout.blocks

    @property
    def skew(self):
        return self._layout.skew

    def rotate(self, phi):
        w, h = self._image.size
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -phi, 1.0)
        image = cv2.warpAffine(numpy.array(self._image), m, (w, h), flags=cv2.INTER_AREA)

        page = Page()
        page._image = PIL.Image.fromarray(image)
        page._layout = self._layout.rotate(phi)

        return page

    @property
    def deskewed(self):
        return self.rotate(-self.skew)

    def dewarping_projections(self, accuracy=None):
        orig_s, proj_s, orig_p, proj_p = self._layout.dewarping_projections(accuracy=None)

        for p in itertools.chain(orig_p, proj_p):
            p[:, 0], p[:, 1] = self._label_to_image_space(p[:, 0], p[:, 1])

        return orig_s, proj_s, orig_p, proj_p

    @property
    def dewarped(self):
        _, proj_s, orig_p, proj_p = self.dewarping_projections()

        '''
        mapper = Mapper(self.rotation, orig_p, proj_p)

        labels = mapper(self._layout._labels, interpolation=cv2.INTER_NEAREST, border=Label.BACKGROUND)

        if False:
            image = self._image
        else:
            image = self.annotate_with_segments(Label.H, mode="markers")
        '''

        orig_p = list(itertools.chain(*orig_p))
        proj_p = list(itertools.chain(*proj_p))

        w, h = self._image.size
        for p in ((0, 0), (w, 0), (w, h), (0, h)):
            orig_p.append(p)
            proj_p.append(p)

        orig_p = numpy.array(orig_p)
        proj_p = numpy.array(proj_p)

        orig_p = orig_p.reshape((-1, 2))
        proj_p = proj_p.reshape((-1, 2))

        image = numpy.array(self._image)

        tform = skimage.transform.PiecewiseAffineTransform()
        tform.estimate(proj_p, orig_p)
        out = skimage.transform.warp(image, tform, output_shape=image.shape)

        out = (out * 255).astype(numpy.uint8)
        # print(out.shape)

        page = Page()
        page._image = PIL.Image.fromarray(out)  # mapper(numpy.array(image)))
        page._layout = self._layout  # Layout(labels, proj_s)
        return page

    def annotate_segments(self, *labels, **kwargs):
        return self._layout.annotate_segments(self._image, *labels, **kwargs)

    def annotate_segments_join(self, i, j):
        return self._layout.annotate_segments_join(self._image, i, j)

    def annotate_content_mask(self, *labels, **kwargs):
        return self._layout.annotate_content_mask(self._image, *labels, **kwargs)

    def annotate_content(self, *labels, **kwargs):
        return self._layout.annotate_content(self._image, *labels, **kwargs)

    def annotate_blocks(self, *labels, **kwargs):
        return self._layout.annotate_blocks(self._image, *labels, **kwargs)

    def annotate_lines(self, *labels, **kwargs):
        return self._layout.annotate_lines(self._image, *labels, **kwargs)

    def annotate_frame(self):
        im = numpy.array(self._image.convert("L").convert("RGB"))
        x, y, w, h = self.frame
        cv2.rectangle(im, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
        return PIL.Image.fromarray(im)

    def annotate_dewarping_projections(self, accuracy=None):
        colors = {
            Label.H: (255, 0, 0),
            Label.V: (0, 0, 255),
            Label.TABCOL: (0, 255, 0)}

        im = numpy.array(self._image.convert("L").convert("RGB"))

        segments, _, orig_p, proj_p = self.dewarping_projections(accuracy=accuracy)

        for s, orig_span, proj_span in zip(segments, orig_p, proj_p):
            for p, q in zip(orig_span, proj_span):
                px, py = map(int, p)
                qx, qy = map(int, q)

                if s:
                    color = colors[s.dominant_label]
                else:
                    color = (128, 255, 64)  # baseline

                if numpy.linalg.norm(p - q) < 2:
                    cv2.circle(im, (px, py), 5, color, thickness=2)
                else:
                    cv2.arrowedLine(
                        im,
                        (px, py), (qx, qy),
                        color=color,
                        thickness=2, tipLength=0.3)

        return PIL.Image.fromarray(im)

    def annotate_rotation(self):
        rho, theta = self._layout.rho_theta

        a = numpy.cos(theta)
        b = numpy.sin(theta)
        d = max(*self._layout.labels.shape)

        x0, y0 = self._label_to_image_space(a * rho + d * -b, b * rho + d * a)
        x1, y1 = self._label_to_image_space(a * rho - d * -b, b * rho - d * a)

        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)

        im = numpy.array(self._image.convert("L").convert("RGB"))
        cv2.line(im, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)

        return PIL.Image.fromarray(im)

    def split(self):
        for s in self.layout.segments:
            s.compute_xpath(self.layout.labels.shape)
        node = LeafNode(self, self.layout._frame, self.layout.content, self.layout.segments).split()
        node.determine_order(1)
        node.determine_blocks()
        return node
