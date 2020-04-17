import numpy
import PIL.Image
from tqdm import tqdm

from .labels import Annotations
from .pages import Layout


def acc_layout(input, target):
    return None


def acc_text_1(input, target):
    return None


def acc_text_2(input, target):
    return None


class BNet:
    def __init__(self, bbz_path, model_name="bnet20191028"):
        self._bbz_path = bbz_path
        model_path = bbz_path / "models" / model_name

        from fastai.basic_train import load_learner

        # see https://docs.fast.ai/tutorial.inference.html
        self._learner = load_learner(model_path)

    def _predict_labels(self, im):
        import fastai.vision

        # see fastai.vision.open_image() on how to build an image for
        # learner.predict().
        im = fastai.vision.pil2tensor(im, numpy.float32)
        im.div_(255)

        # note: if inference crashes, use "conda install nomkl", see
        # https://github.com/openai/spinningup/issues/16

        import warnings
        with warnings.catch_warnings():
            from torch.serialization import SourceChangeWarning
            warnings.filterwarnings("ignore", category=SourceChangeWarning)

            pclass, plabel, pprob = self._learner.predict(fastai.vision.Image(im))
            labels = numpy.array(pclass.data).squeeze(0).astype(numpy.uint8)

        return labels

    def __call__(self, path, page):
        cache_path = self._bbz_path / "cache" / "layout" / (path.stem + ".png")
        if cache_path.is_file():
            return Layout(page, Annotations(numpy.array(PIL.Image.open(cache_path))))

        im = PIL.Image.open(path).convert("RGB")

        # target input image height for network is always 3200.
        image_h = 3200
        image_w = 2400

        # squish to label image size. important, since otherwise
        # our scaling later will be totally off.
        im = im.resize((image_w, image_h), PIL.Image.LANCZOS)

        # find crops.
        infer_h = 500
        n_steps = 6

        overlap = int(((n_steps * infer_h) - image_h) / (n_steps - 1))
        crops = []

        for i in range(0, n_steps):
            y = i * (infer_h - overlap)
            crops.append((0, y, 2400, y + infer_h))

        # predict with sliding window.
        labels = numpy.empty((image_h, image_w), dtype=numpy.uint8)

        for i, box in tqdm(enumerate(crops), "predicting layout for %s" % path.name):
            box_labels = self._predict_labels(im.crop(box))

            _, y0, _, y1 = box

            border_y0 = overlap // 2 if i > 0 else 0
            border_y1 = overlap // 2 if i < n_steps - 1 else 0

            out_slice = slice(y0 + border_y0, y1 - border_y1)
            in_slice = slice(0 + border_y0, infer_h - border_y1)

            labels[out_slice, :] = box_labels[in_slice, :]

        return Layout(page, labels)
