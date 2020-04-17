#!/usr/bin/python3

import shutil
import json
import re
import os

from pathlib import Path
import sys


class Generator:
	def __init__(self, input_path):
		self._input_path = input_path

		self._output_path = self._input_path.parent.parent / "training"
		if not self._output_path.exists():
			self._output_path.mkdir()

		n = self._count_folder("images")
		self._check_counts("pixels", n * 2)  # _C and _P
		self._check_counts("regions", n * 2)  # _C and _P

		print("found %s documents in %s." % (n, input_path))

	@property
	def _tile_suffixes(self):
		tile_lexer = re.compile(r".*-(T-[0-9]+-[0-9]+)")
		tile_suffixes = set()

		labels_path = self._input_path / "regions"
		for p in labels_path.iterdir():
			m = tile_lexer.match(p.stem)
			if m:
				tile_suffixes.add(m.group(1))

		return tile_suffixes


	def _validation_set_for_tiles(self, fold):
		tile_suffixes = self._tile_suffixes
		
		with open(self._input_path.parent / ("valid%d.txt" % fold), "r") as f:
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
		for p in (self._input_path / "images").iterdir():
			if any(p.name.endswith(x) for x in fixed_valid):
				augmented_valid.add(p.stem)
				
		return augmented_valid

	def _check_counts(self, a, n):
		n_a = self._count_folder(a)
		if n_a != n:
			raise RuntimeError(
				"found %s documents in '%s'." % (n_a, a))

	def _count_folder(self, name):
		count = 0
		for p in (self._input_path / name).iterdir():
			if p.name.endswith(".png") and not p.name.startswith("WARP-"):
				count += 1
		return count // max(1, len(self._tile_suffixes))
		   
	# ---------------------------------------------------------------------------------

	def _copy_image(self, src, dst):
		shutil.copyfile(src, dst)


	def _copy_labels(self, src, dst):
		shutil.copyfile(src, dst)


	def _gen_fold(self, fold):
		valid = self._validation_set_for_tiles(fold)

		data_path = self._output_path / (self._input_path.name + ("_FOLD%d" % fold))

		if data_path.exists():
			raise RuntimeError("%s already exists. please delete it first to recreate." % data_path)

		data_path.mkdir(exist_ok=False)
		(data_path / "test_images").mkdir(exist_ok=False)
		(data_path / "train_images").mkdir(exist_ok=False)
		(data_path / "val_images").mkdir(exist_ok=False)
		(data_path / "test_labels").mkdir(exist_ok=False)
		(data_path / "train_labels").mkdir(exist_ok=False)
		(data_path / "val_labels").mkdir(exist_ok=False)

		(data_path / "meta").mkdir(exist_ok=False)
		shutil.copyfile(self._input_path / "codes.json", data_path / "meta" / "codes.json")
		shutil.copyfile(self._input_path / "regions" / "weights.json", data_path / "meta" / "weights.json")

		
		tile_name_lexer = re.compile(r"(.*)-(T-[0-9]+-[0-9]+)")
		
		for image_p in (self._input_path / "images").iterdir():        
			if image_p.name.startswith(".") or not image_p.name.endswith(".png"):
				continue
				
			m = tile_name_lexer.match(image_p.stem)
			if m:
				label_p = self._input_path / "regions" / (m.group(1) + "_P-" + m.group(2) + ".png")
			else:
				label_p = self._input_path / "regions" / (image_p.stem  + "_P.png")
				
			assert label_p.is_file()

			if image_p.stem in valid:
				self._copy_image(image_p, data_path / "val_images" / image_p.name)
				self._copy_labels(label_p, data_path / "val_labels" / image_p.name)

				self._copy_image(image_p, data_path / "test_images" / image_p.name)
				self._copy_labels(label_p, data_path / "test_labels" / image_p.name)
			else:
				self._copy_image(image_p, data_path / "train_images" / image_p.name)
				self._copy_labels(label_p, data_path / "train_labels" / image_p.name)


	def generate_all_folds(self):
		for fold in (1, 2, 3, 4, 5):
			self._gen_fold(fold)

		print("done processing %s." % self._input_path)


if __name__ == "__main__":
	if len(sys.argv) > 2:
		raise ValueError()
	elif len(sys.argv) == 2:
		bbz_path = Path(sys.argv[1])
	else:
		script_dir = Path(os.path.realpath(__file__)).parent
		if (script_dir / "data").exists():
			bbz_path = script_dir / "data"
		else:
			bbz_path = script_dir.parent / "00_demo_data"

	for training_data_path in (bbz_path/"preprocessed").iterdir():
		if training_data_path.is_dir():
			g = Generator(training_data_path)
			g.generate_all_folds()
