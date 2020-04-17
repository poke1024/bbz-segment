#!/usr/bin/python3

import sqlite3
import json
import collections
import os

import subprocess
import sys
import argparse
import logging

from pathlib import Path


TIMEOUT = 60  # 24 * 60 * 60  # 24 hours


def _config_logging():
	root = logging.getLogger()
	root.setLevel(logging.DEBUG)

	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)
	FORMAT = "[%(asctime)s %(filename)s:%(lineno)s:%(funcName)s] %(levelname)s: %(message)s"
	formatter = logging.Formatter(FORMAT)
	ch.setFormatter(formatter)
	root.addHandler(ch)


_config_logging()


class GridMate:
	def __init__(self, parameters, sample_count=1):
		self._script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
		self._parameters = tuple(parameters)
		self._sample_count = sample_count

	@property
	def parameters(self):
		return self._parameters

	def _run(self, params):
		raise NotImplementedError()		

	@property
	def _db_path(self):
		return Path(self._script_dir / 'gridmate.db')

	def _connect(self):
		return sqlite3.connect(str(self._db_path))

	def _tasks_data(self, histogram):
		tasks_data = []

		for params, gpu in self._combinations():
			assert set(params.keys()) == set(self._parameters)
			key = tuple([params[p] for p in self._parameters])

			tasks_data.append((json.dumps(key), gpu, histogram[key]))

		return tasks_data

	def generate_db(self, histogram):
		tasks_data = self._tasks_data(histogram)

		if not self._db_path.exists():
			conn = self._connect()

			try:
				with conn:
					conn.execute('CREATE TABLE tasks(params text, gpu varchar(255), n int)')
					conn.execute('CREATE UNIQUE INDEX idx_tasks_params ON tasks(params)')
					conn.execute('CREATE INDEX idx_tasks_n ON tasks(n)')

					conn.executemany(
						'INSERT INTO tasks VALUES (?,?,?)', tasks_data)

					conn.commit()

			finally:
				conn.close()

			print("database created with %s items." % len(tasks_data))
		else:
			conn = self._connect()

			try:
				with conn:
					conn.execute('begin exclusive') 
					conn.execute('DELETE FROM tasks')
					conn.executemany(
						'INSERT INTO tasks VALUES (?,?,?)', tasks_data)

					conn.commit()

			finally:
				conn.close()

			print("database updated to %s items." % len(tasks_data))

	def _fetch_task(self, gpu):
		conn = self._connect()

		try:
			with conn:
				conn.execute('begin exclusive') 

				for row in conn.execute(
					'SELECT params, n FROM tasks WHERE gpu=? ORDER BY n LIMIT 1', (gpu, )):

					task_id, n = row
					if n >= self._sample_count:
						return None  # we're done.
					params = dict(zip(self._parameters, json.loads(task_id)))

					conn.execute('UPDATE tasks SET n=n+1 WHERE params=?', (task_id, ))
					conn.commit()

					return params

				return None

		finally:
			conn.close()

	def run_one_task(self, gpu):
		params = self._fetch_task(gpu)
		if params is None:
			return False

		self._run_task(params)
		return True

	def run_until_done(self, gpu):
		while self.run_one_task(gpu):
			pass

		print("all work done for gpu %s." % gpu)



def _determine_task_gpu(params):
	if params["backbone"] == "seresnext50" and params["data_name"] in ("layout_896x1280_T512x768", "layout_768x1280_T768x512"):
		return "v100"
	elif params["data_name"] in ("layout_896x1280", "layout_1280x2400_T1280x896", "layout_1640x2400_T896x1280"):
		return "v100"
	elif params["backbone"] in ("seresnet50", "efficientnetb2"):
		return "v100"
	elif params["backbone"] in ("efficientnetb1", "seresnext50", "inceptionresnetv2") and params["model"] == "fpn":
		return "v100"
	else:
		return "rtx2080ti"


def _patch_configuration(params):
	if params["backbone"] == "seresnext50" and params["model"] == "fpn" and params["data_name"] in (
		"layout_896x1280",
		"layout_1280x2400_T1280x896",
		"layout_1640x2400_T896x1280"):
		params["batchsize"] = params["batchsize"] // 3


class OrigamiGridMate(GridMate):
	def __init__(self):
		super().__init__([
			"backbone",
			"model",
			"labels",
			"data_name",
			"ngpus",
			"batchsize",
			"fold",
			"illustrations"
		])

	def run(self):
		parser = argparse.ArgumentParser()
		subparsers = parser.add_subparsers(dest='command')

		parser_init = subparsers.add_parser(
			'init', help='initialize gridmate')
		parser_init.add_argument(
			'runspath', type=str, help='path of sacred runs data')

		parser_work = subparsers.add_parser(
			'work', help='run gridmate tasks')
		parser_work.add_argument(
			'gpu', type=str, help='gpu to work on')

		args = parser.parse_args()

		if args.command == 'init':
			assert Path(args.runspath).is_dir()
			histogram = collections.defaultdict(int)
			self._gather_counts(Path(args.runspath), histogram)
			self.generate_db(histogram)
		elif args.command == 'work':
			self.run_until_done(args.gpu)
		else:
			parser.print_help(sys.stderr)
			sys.exit(1)

	def _gather_counts(self, path, counts):
		if (path / "config.json").exists():

			with open(path / "run.json", "r") as f:
				run_data = json.loads(f.read())

			status = run_data["status"]

			with open(path / "config.json", "r") as f:
				config_data = json.loads(f.read())

			if status == "RUNNING":
				import datetime
				import dateutil.parser

				elapsed = (datetime.datetime.now() - dateutil.parser.parse(run_data["start_time"])).total_seconds()
				if elapsed > TIMEOUT:
					return

			elif status == "COMPLETED":
				# fix config_data.
				if 'illustrations' not in config_data:
					config_data['illustrations'] = False

				# for the moment, only use fold 1. don't mix with other folds.
				if int(config_data["fold"]) != 1:
					return

				# check metrics data for completed runs.
				n_epochs = config_data["epochs"]

				with open(path / "metrics.json", "r") as f:
					metrics_data = json.loads(f.read())

				# check counts of metrics values.
				decider_metric = "val_matthews"
				val_metric_prefix = "val_"

				if decider_metric in metrics_data:
					decider_metric_values = metrics_data[decider_metric]["values"]
				else:
					decider_metric_values = []

				if len(decider_metric_values) != n_epochs:
					return
			else:
				# status is neither RUNNING nor COMPLETED.
				return

			# gather count.
			counts[tuple([config_data[p] for p in self.parameters])] += 1

		else:
			for p in path.iterdir():
				if p.is_dir():
					self._gather_counts(p, counts)

	def _combinations(self):
		from sklearn.model_selection import ParameterGrid

		param_grid = {
			"backbone": [
				#"resnet18", 
				"resnet34",
				"resnet50",
				#"seresnet18", 
				"seresnet34",
				"seresnet50",
				"vgg16", "inceptionv3",
				#"efficientnetb0",
				"efficientnetb1",
				"efficientnetb2",
				"seresnext50",
				"inceptionresnetv2"],
				#"seresnext50",
				#"densenet121"],
			"model": ["fpn", "unet"],
			"labels": ["sep", "txt"],
			"data_name": [
				"layout_512x768",
				"layout_768x1280_T768x512",
				"layout_896x1280_T512x768",
				"layout_640x1024_T384x1024",
				"layout_896x1280_T256x1280",
				"layout_896x1280_T896x384",
				"layout_896x1280",
				"layout_1280x2400_T1280x896",
				"layout_1640x2400_T896x1280"],
			"ngpus": [4],
			"batchsize": [12],
			"fold": [1]
		}

		for p in ParameterGrid(param_grid):
			if p["labels"] == "txt":
				for illustrations in (False, True):
					q = p.copy()
					q["illustrations"] = illustrations
					_patch_configuration(q)
					yield q, _determine_task_gpu(q)
			else:
				q = p.copy()
				q["illustrations"] = False
				_patch_configuration(q)
				yield q, _determine_task_gpu(q)

	def _run_task(self, params):
		args = ["%s=%s" % (k, v) for k, v in params.items()]
		train_path = str(self._script_dir / "train.py")
		cmd = ["python", train_path, "with", *args]
		logging.info("calling subprocess %s" % cmd)
		p = subprocess.run(cmd)
		logging.info("subprocess returned %d." % p.returncode)


gm = OrigamiGridMate()
gm.run()
