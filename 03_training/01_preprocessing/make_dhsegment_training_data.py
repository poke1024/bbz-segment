#!/usr/bin/python3

from pathlib import Path
from string import Template

import sys
import os
import json
import re
import PIL.Image
import numpy as np

# see https://dhsegment.readthedocs.io/en/latest/start/training.html

dhs_config = Template("""
{
  "training_params" : {
      "learning_rate": 5e-5,
      "batch_size": 1,
      "make_patches": false,
      "training_margin" : 0,
      "n_epochs": 50,
      "data_augmentation" : true,
      "data_augmentation_max_rotation" : 0.2,
      "data_augmentation_max_scaling" : 0.2,
      "data_augmentation_flip_lr": false,
      "data_augmentation_flip_ud": false,
      "data_augmentation_color": false,
      "evaluate_every_epoch" : 10
  },
  "pretrained_model_name" : "resnet50",
  "prediction_type": "CLASSIFICATION",
  "train_data" : "$train_csv",
  "eval_data" : "$val_csv",
  "classes_file" : "$classes_txt",
  "model_output_dir" : "$model",
  "gpu" : "0"
}
""")

def find_tile_suffixes(regions_path):
	tile_lexer = re.compile(r".*-(T-[0-9]+-[0-9]+)")
	tile_suffixes = set()

	for p in regions_path.iterdir():
		m = tile_lexer.match(p.stem)
		if m:
			tile_suffixes.add(m.group(1))

	return tile_suffixes


def validation_set_for_tiles(path, fold=1):
	tile_suffixes = find_tile_suffixes(path / "regions")
	
	with open(path.parent / ("valid%d.txt" % fold), "r") as f:
		valid = [s.strip() for s in f.readlines()]

	print("found %s items in validation set for fold %d." % (len(valid), fold))
	assert len(valid) > 0

	if tile_suffixes:
		fixed_valid = []
		for valid_item in valid:
			p = Path(valid_item)
			for suffix in tile_suffixes:
				fixed_valid.append(p.stem + "-" + suffix + p.suffix)
	else:
		fixed_valid = valid
	
	augmented_valid = set()
	for p in (path / "images").iterdir():
		if any(p.name.endswith(x) for x in fixed_valid):
			augmented_valid.add(p.stem)
			
	return augmented_valid

all_dhs_codes = dict(
	blk=["background", "text", "tabelleninhalt"],
	blkx=["background", "text", "tabelleninhalt", "illustration"],
	sep=["background", "tab", "h", "v"]
)

dhs_models_dir = Path(os.environ["HOME"]) / "dhsegment"
dhs_models_dir.mkdir(exist_ok=True)

def gen_dhs_data(path):
	print("processing %s..." % str(path))

	dhsegment_path = path / "dhsegment"
	dhsegment_path.mkdir(exist_ok=True)

	with open(path / "codes.json", "r") as f:
		global_codes = json.loads(f.read())

	image_name_lexer = re.compile(r"^(.+?)(-T-[0-9]+-[0-9]+)?$")
	pairs = []

	for image_path in (path / "images").iterdir():
		m = image_name_lexer.match(image_path.stem)
		assert m is not None
		tile_suffix = m.group(2)
		if tile_suffix is None:
			tile_suffix = ""
		labels_path = path / "regions" / (m.group(1) + "_C" + tile_suffix + ".png")
		pairs.append((image_path, labels_path))

	val_stems = validation_set_for_tiles(path)

	for labels_mode in ('blk', 'blkx', 'sep'):
		dhs_model_dir = dhs_models_dir / ("%s_%s" % (path.stem, labels_mode))
		#dhs_model_dir.mkdir(exist_ok=True)

		dhs_data_dir = dhsegment_path / labels_mode
		dhs_data_dir.mkdir(exist_ok=True)

		dhs_labels_dir = dhs_data_dir / "labels"
		dhs_labels_dir.mkdir(exist_ok=True)

		dhs_codes = all_dhs_codes[labels_mode]
		dhs_pairs = dict(train=[], val=[])

		for image_path, labels_path in pairs:
			im = PIL.Image.open(labels_path)
			palette = im.getpalette()

			pixels = np.array(im)
			dhs_pixels = np.empty(pixels.shape, dtype=np.uint8)
			dhs_pixels.fill(global_codes.index("background"))

			for i, code in enumerate(global_codes):
				if code in dhs_codes:
					j = i
				else:
					j = global_codes.index("background")
				dhs_pixels[pixels == i] = j

			im = PIL.Image.fromarray(dhs_pixels, "P")
			im.putpalette(palette)
			im.save(dhs_labels_dir / labels_path.name, "PNG")

			t = 'val' if image_path.stem in val_stems else 'train'
			dhs_pairs[t].append((image_path, dhs_labels_dir / labels_path.name))

		for t in dhs_pairs.keys():
			with open(dhs_data_dir / ("%s.csv" % t), "w") as f:
				for image_path, labels_path in dhs_pairs[t]:
					f.write("%s,%s\n" % (str(image_path), str(labels_path)))

		im = PIL.Image.open(dhs_pairs["train"][0][1])
		pal = np.array(im.getpalette(), dtype=np.int32).reshape(256, 3)

		with open(dhs_data_dir / "classes.txt", "w") as f:
			for c in dhs_codes:
				f.write("%d %d %d\n" % tuple(pal[global_codes.index(c)].tolist()))

		subs = dict(
			train_csv=dhs_data_dir / "train.csv",
			val_csv=dhs_data_dir / "val.csv",
			classes_txt=dhs_data_dir / "classes.txt",
			model=dhs_model_dir)


		with open(dhs_data_dir / "config.json", "w") as f:
			f.write(dhs_config.substitute(subs))

	print("done.")

basepath = Path(sys.argv[1])  # e.g. /home/sc.uni-leipzig.de/bo140rasi/data/preprocessed
assert basepath.is_dir()
assert (basepath / "valid1.txt").exists()

for p in basepath.iterdir():
	if p.is_dir() and (p / "codes.json").exists():
		gen_dhs_data(p)

