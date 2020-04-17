#!/usr/local/bin/python

# note this does currently not work with Tensorflow 2.

# issue with tf >= 2.1
# https://github.com/tensorflow/tensorflow/issues/35925

# might work with tf 2.0, issue with installing this with drivers.

# best solution right now: use tf 1.15.2-gpu

from sacred import Experiment
from sacred.observers import FileStorageObserver

import os
import cv2
import math
import json

#see https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

print("running on tensorflow %s" % tf.__version__)
assert tf.test.is_gpu_available()
#assert len(tf.config.list_physical_devices('GPU')) > 0

# see https://github.com/keras-team/keras/issues/4161
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import PIL.Image


from keras import backend as K
from keras.optimizers import Optimizer, SGD, Adam, Nadam

import numpy as np
from pathlib import Path

from functools import lru_cache
from fractions import Fraction
from collections import namedtuple, defaultdict

import segmentation_models as sm
from keras_lookahead import Lookahead


sacred_runs_path = Path(os.environ["BBZ_SEGMENT_RUNS_PATH"])
sacred_runs_path.mkdir(exist_ok=True)
assert sacred_runs_path.is_dir()

ex = Experiment('bbz_segment')
ex.observers.append(FileStorageObserver(sacred_runs_path))

# -----------------------------------------------------------------------------

LongShelhamerDarrellMetrics = namedtuple(
	'LongShelhamerDarrellMetrics',
	['pixel_accuracy', 'mean_accuracy', 'mean_IU', 'frequency_weighted_IU'])


def lsd_metrics(
	prediction: np.ndarray,
	truth: np.ndarray,
	n_classes: int) -> LongShelhamerDarrellMetrics:

	"""This computes the evaluation metrics given for semantic segmentation given in:
	[1] J. Long, E. Shelhamer, and T. Darrell, "Fully Convolutional Networks for
	Semantic Segmentation", 2014. (available at https://arxiv.org/abs/1411.4038).
	Note:
		Modified to exclude empty classes.
	Args:
		prediction: integer array of predicted classes for each pixel.
		truth: integer array of ground truth for each pixel.
		n_classes: defines the pixel classes [0, 1, ..., n_classes - 1].
	Returns:
		LongShelhamerDarrellMetrics: The computed metrics.
	"""

	def _check_array(name, a):
		if not np.issubdtype(a.dtype, np.integer):
			raise ValueError("given %s-array must be of type integer" % name)

		if not (0 <= np.min(a) < n_classes and 0 <= np.max(a) < n_classes):
			raise ValueError("non-class values in given %s-array" % name)

	_check_array('prediction', prediction)
	_check_array('truth', truth)

	classes = list(range(n_classes))

	@lru_cache(maxsize=None)
	def n(i: int, j: int) -> Fraction:
		# n(i, j) is "the number of pixels of class i predicted to belong to
		# class j", see [1].
		return Fraction(int(np.sum(np.logical_and(
			truth == i, prediction == j).astype(np.uint8), dtype=np.uint64)))

	@lru_cache(maxsize=None)
	def t(i: int) -> Fraction:
		# t(i) is "the total number of pixels of class i", see [1].
		return sum(n(j, i) for j in classes)

	non_empty_classes = [i for i in classes if t(i) > 0]

	return LongShelhamerDarrellMetrics(
		pixel_accuracy=sum(n(i, i) for i in classes) / sum(t(i) for i in classes),

		mean_accuracy=(Fraction(1) / len(non_empty_classes)) * sum(
			(n(i, i) / t(i)) for i in non_empty_classes),

		mean_IU=(Fraction(1) / len(non_empty_classes)) * sum(
			(
				n(i, i) / (
					t(i) + sum(n(j, i) for j in non_empty_classes) - n(i, i))
			) for i in non_empty_classes),

		frequency_weighted_IU=(Fraction(1) / sum(t(k) for k in non_empty_classes)) * sum(
			(
				(t(i) * n(i, i)) / (
					t(i) + sum(n(j, i) for j in non_empty_classes) - n(i, i))
			) for i in non_empty_classes)
	)


if False:
	class LsdMetrics(sm.base.Metric):
		def __init__(self, name=None):
			name = name or 'lsd_metrics'
			super().__init__(name=name)

		def __call__(self, gt, pr):
			# see https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/base/functional.py:
			# gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
			# pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)

			# tensorflow.python.framework.ops.Tensor
			gt = gt.eval()

			backend = kwargs['backend']
			if backend.image_data_format() == 'channels_last':
				x = backend.permute_dimensions(x, (3, 0, 1, 2))
			else:
				x = backend.permute_dimensions(x, (1, 0, 2, 3))

			print("!!", type(gt))
			print("!", gt.shape)
			print(gt[:, 0, 0, 0])  # (B, H, W)

			#n_classes = gt.shape[-1]

			#gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

			#lsd_metrics()

# -----------------------------------------------------------------------------

# https://github.com/OverLordGoldDragon/keras-adamw

from keras.legacy import interfaces
import numpy as np
#from .utils import _apply_weight_decays, _compute_eta_t
#from .utils import _apply_lr_multiplier, _check_args, K_eval

import tensorflow as tf
import keras.backend as K
import numpy as np
import random
from termcolor import colored
'''Helper methods for optimizers
'''


def warn_str():
	return colored('WARNING: ', 'red')


def get_weight_decays(model, verbose=1):
	wd_dict = {}
	for layer in model.layers:
		layer_l2regs = _get_layer_l2regs(layer)
		if layer_l2regs:
			for layer_l2 in layer_l2regs:
				weight_name, weight_l2 = layer_l2
				wd_dict.update({weight_name: weight_l2})
				if weight_l2 != 0 and verbose:
					print((warn_str() + "{} l2-regularization = {} - should be "
						  "set 0 before compiling model").format(
								  weight_name, weight_l2))
	return wd_dict


def fill_dict_in_order(_dict, _list_of_vals):
	for idx, key in enumerate(_dict.keys()):
		_dict[key] = _list_of_vals[idx]
	return _dict


def _get_layer_l2regs(layer):
	if hasattr(layer, 'cell') or \
	  (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
		return _rnn_l2regs(layer)
	elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
		layer = layer.layer
	l2_lambda_kb = []
	for weight_name in ['kernel', 'bias']:
		_lambda = getattr(layer, weight_name + '_regularizer', None)
		if _lambda is not None:
			l2_lambda_kb.append([getattr(layer, weight_name).name,
								 float(_lambda.l2)])
	return l2_lambda_kb


def _rnn_l2regs(layer):
	l2_lambda_krb = []
	if hasattr(layer, 'backward_layer'):
		for layer in [layer.forward_layer, layer.backward_layer]:
			l2_lambda_krb += _cell_l2regs(layer.cell)
		return l2_lambda_krb
	else:
		return _cell_l2regs(layer.cell)


def _cell_l2regs(rnn_cell):
	cell = rnn_cell
	l2_lambda_krb = []  # kernel-recurrent-bias

	for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
		_lambda = getattr(cell, weight_type + '_regularizer', None)
		if _lambda is not None:
			weight_name = cell.weights[weight_idx].name
			l2_lambda_krb.append([weight_name, float(_lambda.l2)])
	return l2_lambda_krb


def _apply_weight_decays(cls, var, var_t):
	wd = cls.weight_decays[var.name]
	wd_normalized = wd * K.cast(
			K.sqrt(cls.batch_size / cls.total_iterations_wd), 'float32')
	var_t = var_t - cls.eta_t * wd_normalized * var

	if cls.init_verbose and not cls._init_notified:
		print('{} weight decay set for {}'.format(
				K_eval(wd_normalized), var.name))
	return var_t


def _compute_eta_t(cls):
	PI = 3.141592653589793
	t_frac = K.cast(cls.t_cur / cls.total_iterations, 'float32')
	eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * \
		(1 + K.cos(PI * t_frac))
	return eta_t


def _apply_lr_multiplier(cls, lr_t, var):
	multiplier_name = [mult_name for mult_name in cls.lr_multipliers
					   if mult_name in var.name]
	if multiplier_name != []:
		lr_mult = cls.lr_multipliers[multiplier_name[0]]
	else:
		lr_mult = 1
	lr_t = lr_t * lr_mult

	if cls.init_verbose and not cls._init_notified:
		if lr_mult != 1:
			print('{} init learning rate set for {} -- {}'.format(
			   '%.e' % K_eval(lr_t), var.name, lr_t))
		else:
			print('No change in learning rate {} -- {}'.format(
											  var.name, K_eval(lr_t)))
	return lr_t


def _check_args(total_iterations, use_cosine_annealing, weight_decays):
	if use_cosine_annealing and total_iterations != 0:
		print('Using cosine annealing learning rates')
	elif (use_cosine_annealing or weight_decays != {}) and total_iterations == 0:
		print(warn_str() + "'total_iterations'==0, must be !=0 to use "
			  + "cosine annealing and/or weight decays; "
			  + "proceeding without either")


def reset_seeds(reset_graph_with_backend=None, verbose=1):
	if reset_graph_with_backend is not None:
		K = reset_graph_with_backend
		K.clear_session()
		tf.compat.v1.reset_default_graph()
		if verbose:
			print("KERAS AND TENSORFLOW GRAPHS RESET")

	np.random.seed(1)
	random.seed(2)
	if tf.__version__[0] == '2':
		tf.random.set_seed(3)
	else:
		tf.set_random_seed(3)
	if verbose:
		print("RANDOM SEEDS RESET")


def K_eval(x, backend=K):
	K = backend
	try:
		return K.get_value(K.to_dense(x))
	except Exception as e:
		try:
			eval_fn = K.function([], [x])
			return eval_fn([])[0]
		except Exception as e:
			return K.eager(K.eval)(x)



class AdamW(Optimizer):
	"""AdamW optimizer.
	Default parameters follow those provided in the original paper.
	# Arguments
		learning_rate: float >= 0. Learning rate.
		beta_1: float, 0 < beta < 1. Generally close to 1.
		beta_2: float, 0 < beta < 1. Generally close to 1.
		amsgrad: boolean. Whether to apply the AMSGrad variant of this
			algorithm from the paper "On the Convergence of Adam and Beyond".
		batch_size:       int >= 1. Train input batch size; used for normalization
		total_iterations: int >= 0. Total expected iterations / weight updates
						  throughout training, used for normalization; <1>
		weight_decays:    dict / None. Name-value pairs specifying weight decays,
						  as {<weight matrix name>:<weight decay value>}; <2>
		lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
						  multipliers, as {<layer name>:<multiplier value>}; <2>
		use_cosine_annealing: bool. If True, multiplies lr each train iteration
							  as a function of eta_min, eta_max, total_iterations,
							  and t_cur (current); [2]-Appendix, 2
		eta_min, eta_max: int, int. Min & max values of cosine annealing
						  lr multiplier; [2]-Appendix, 2
		t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
			   To be used together with use_cosine_annealing==True
		total_iterations_wd: int / None. If not None, weight_decays will be
					 applied according to total_iterations_wd instead of
					 total_iterations, contrary to authors' scheme. Set to
					 sum(total_iterations) over all restarts to normalize over
					 all epochs. May yield improvement over `None`.
		init_verbose: bool. If True, print weight-name--weight-decay, and
					  lr-multiplier--layer-name value pairs set during
					  optimizer initialization (recommended)
	# <1> - if using 'warm restarts', then refers to total expected iterations
			for a given restart; can be an estimate, and training won't stop
			at iterations == total_iterations. [2]-Appendix, pg 1
	# <2> - [AdamW Keras Implementation - Github repository]
			(https://github.com/OverLordGoldDragon/keras_adamw)
	# References
		- [1][Adam - A Method for Stochastic Optimization]
			 (http://arxiv.org/abs/1412.6980v8)
		- [2][Fixing Weight Decay Regularization in Adam]
			 (https://arxiv.org/abs/1711.05101)
	"""

	def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
				 amsgrad=False, batch_size=32, total_iterations=0,
				 total_iterations_wd=None, use_cosine_annealing=False,
				 weight_decays=None, lr_multipliers=None, init_verbose=True,
				 eta_min=0, eta_max=1, t_cur=0, **kwargs):
		self.initial_decay = kwargs.pop('decay', 0.0)
		self.epsilon = kwargs.pop('epsilon', K.epsilon())
		learning_rate = kwargs.pop('lr', learning_rate)
		eta_t = kwargs.pop('eta_t', 1.)
		super(AdamW, self).__init__(**kwargs)

		with K.name_scope(self.__class__.__name__):
			self.iterations = K.variable(0, dtype='int64', name='iterations')
			self.learning_rate = K.variable(learning_rate, name='learning_rate')
			self.beta_1 = K.variable(beta_1, name='beta_1')
			self.beta_2 = K.variable(beta_2, name='beta_2')
			self.decay = K.variable(self.initial_decay, name='decay')
			self.batch_size = K.variable(batch_size, dtype='int64',
										 name='batch_size')
			self.eta_min = K.constant(eta_min, name='eta_min')
			self.eta_max = K.constant(eta_max, name='eta_max')
			self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
			self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

		self.total_iterations = total_iterations
		self.total_iterations_wd = total_iterations_wd or total_iterations
		self.amsgrad = amsgrad
		self.lr_multipliers = lr_multipliers
		self.weight_decays = weight_decays or {}
		self.init_verbose = init_verbose
		self.use_cosine_annealing = use_cosine_annealing

		self._init_notified = False
		_check_args(total_iterations, use_cosine_annealing, self.weight_decays)

	@interfaces.legacy_get_updates_support
	@K.symbolic
	def get_updates(self, loss, params):
		grads = self.get_gradients(loss, params)
		self.updates = [K.update_add(self.iterations, 1)]
		self.updates.append(K.update_add(self.t_cur, 1))

		lr = self.learning_rate
		if self.initial_decay > 0:
			lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
													  K.dtype(self.decay))))

		t = K.cast(self.iterations, K.floatx()) + 1
		lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
					 (1. - K.pow(self.beta_1, t)))

		ms = [K.zeros(K.int_shape(p),
			  dtype=K.dtype(p),
			  name='m_' + str(i))
			  for (i, p) in enumerate(params)]
		vs = [K.zeros(K.int_shape(p),
			  dtype=K.dtype(p),
			  name='v_' + str(i))
			  for (i, p) in enumerate(params)]

		if self.amsgrad:
			vhats = [K.zeros(K.int_shape(p),
					 dtype=K.dtype(p),
					 name='vhat_' + str(i))
					 for (i, p) in enumerate(params)]
		else:
			vhats = [K.zeros(1, name='vhat_' + str(i))
					 for i in range(len(params))]
		self.weights = [self.iterations] + ms + vs + vhats

		total_iterations = self.total_iterations
		# Cosine annealing
		if self.use_cosine_annealing and total_iterations != 0:
			self.eta_t = _compute_eta_t(self)
		self.lr_t = lr_t * self.eta_t  # for external tracking

		for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
			# Learning rate multipliers
			if self.lr_multipliers is not None:
				lr_t = _apply_lr_multiplier(self, lr_t, p)

			m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
			v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
			if self.amsgrad:
				vhat_t = K.maximum(vhat, v_t)
				p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
				self.updates.append(K.update(vhat, vhat_t))
			else:
				p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

			self.updates.append(K.update(m, m_t))
			self.updates.append(K.update(v, v_t))

			# Weight decays
			if p.name in self.weight_decays.keys() and total_iterations != 0:
				p_t = _apply_weight_decays(self, p, p_t)
			new_p = p_t

			# Apply constraints.
			if getattr(p, 'constraint', None) is not None:
				new_p = p.constraint(new_p)

			self.updates.append(K.update(p, new_p))

		self._init_notified = True
		return self.updates

	def get_config(self):
		config = {
			'learning_rate': float(K_eval(self.learning_rate)),
			'beta_1': float(K_eval(self.beta_1)),
			'beta_2': float(K_eval(self.beta_2)),
			'decay': float(K_eval(self.decay)),
			'batch_size': int(K_eval(self.batch_size)),
			'total_iterations': int(self.total_iterations),
			'weight_decays': self.weight_decays,
			'lr_multipliers': self.lr_multipliers,
			'use_cosine_annealing': self.use_cosine_annealing,
			't_cur': int(K_eval(self.t_cur)),
			'eta_t': int(K_eval(self.eta_t)),
			'eta_min': int(K_eval(self.eta_min)),
			'eta_max': int(K_eval(self.eta_max)),
			'init_verbose': self.init_verbose,
			'epsilon': self.epsilon,
			'amsgrad': self.amsgrad
		}
		base_config = super(AdamW, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

# -----------------------------------------------------------------------------

# see https://stackoverflow.com/questions/50659482/why-cant-i-get-reproducible-results-in-keras-even-though-i-set-the-random-seeds

#import os
#os.environ['PYTHONHASHSEED'] = str(5825482)

import random
random.seed(5825482)

import numpy.random
numpy.random.seed(5825482)

#tf.random.set_seed(5825482)
tf.set_random_seed(5825482)

# -----------------------------------------------------------------------------

@ex.config
def cfg():
	ngpus = 1

	backbone = "resnet18"
	model = "unet"
	labels = "txt"
	epochs = 50
	batchsize = 5 * ngpus
	fold = 1
	partial = 1
	augment = 0
	lr = 1e-3 * ((batchsize / ngpus) / 4.0)
	save_model = False
	illustrations = False

	script_dir = Path(os.path.realpath(__file__)).parent
	if (script_dir / "data").exists():
		data_path = script_dir / "data" / "training"

# -----------------------------------------------------------------------------

if False:
	def colorize(pixels):
		colors = np.array(list(map(colormap.hex2rgb, Category20[20]))).flatten()
		im = PIL.Image.fromarray(pixels, "P")
		palette = np.zeros((768,), dtype=np.uint8)
		palette[:len(colors)] = colors
		im.putpalette(palette)
		return np.array(im.convert("RGB"))

	def visualize(**images):
		"""Plot images in one row."""
		n = len(images)
		plt.figure(figsize=(16, 5))
		for i, (name, image) in enumerate(images.items()):
			plt.subplot(1, n, i + 1)
			plt.xticks([])
			plt.yticks([])
			plt.title(' '.join(name.split('_')).title())
			if name.endswith("_mask") and len(image.shape) == 3:
				image = colorize(np.argmax(image, axis=-1).astype(np.uint8))
			plt.imshow(image, interpolation="bessel")
		return plt
	
	# helper function for data visualization    
	def denormalize(x):
		"""Scale image to range 0..1 for correct plot"""
		x_max = np.percentile(x, 98)
		x_min = np.percentile(x, 2)    
		x = (x - x_min) / (x_max - x_min)
		x = x.clip(0, 1)
		return x
		
# classes for data loading and preprocessing
class Dataset:
	"""CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
	
	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline 
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing 
			(e.g. noralization, shape manipulation, etc.)
	
	"""
	
	CLASSES = ['text', 'tabelleninhalt', 'illustration', 'tab', 'h', 'v', 'unlabelled']
	
	def __init__(
			self, 
			images_dir, 
			masks_dir, 
			classes=None, 
			augmentation=None, 
			preprocessing=None,
			partial=None,
	):
		self.ids = sorted(os.listdir(images_dir))

		if partial is not None:
			# partial data.
			n = len(self.ids)
			k = math.floor(n * partial)
			self.ids = random.sample(self.ids, k)

		# images and masks.
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
		self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
		
		# convert str names to class values on masks
		self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
		
		self.augmentation = augmentation
		self.preprocessing = preprocessing
		
		# preprocessing. check that all images load properly.
		for i in range(len(self.images_fps)):
			#print("loading image", self.images_fps[i])
			image = cv2.imread(self.images_fps[i])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	
	def __getitem__(self, i):
		
		# read data
		image = cv2.imread(self.images_fps[i])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.masks_fps[i], 0)
		
		# extract certain classes from mask (e.g. cars)
		masks = [(mask == 1 + v) for v in self.class_values]
		mask = np.stack(masks, axis=-1).astype('float')
				
		# apply augmentations
		if self.augmentation:
			sample = self.augmentation(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
		
		# apply preprocessing
		if self.preprocessing:
			sample = self.preprocessing(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
			
		# add background if mask is not binary
		if mask.shape[-1] != 1:
			background = 1 - mask.sum(axis=-1, keepdims=True)
			mask = np.concatenate((mask, background), axis=-1)
		
		return image, mask
		
	def __len__(self):
		return len(self.ids)
	
	
class Dataloader(keras.utils.Sequence):
	"""Load data from dataset and form batches
	
	Args:
		dataset: instance of Dataset class for image loading and preprocessing.
		batch_size: Integet number of images in batch.
		shuffle: Boolean, if `True` shuffle image indexes each epoch.
	"""
	
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(dataset))

		self.on_epoch_end()

	def __getitem__(self, i):
		
		# collect batch data
		start = i * self.batch_size
		stop = (i + 1) * self.batch_size
		data = []
		for j in range(start, stop):
			data.append(self.dataset[j])
		
		# transpose list of lists
		batch = [np.stack(samples, axis=0) for samples in zip(*data)]
		
		return batch
	
	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return len(self.indexes) // self.batch_size
	
	def on_epoch_end(self):
		"""Callback function to shuffle indexes each epoch"""
		if self.shuffle:
			self.indexes = np.random.permutation(self.indexes)
			
# -----------------------------------------------------------------------------

import albumentations as A

# see https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

def round_clip_0_1(x, **kwargs):
	x = x.round().clip(0, 1)
	#print("!", x.shape)
	
	#e = np.logical_and.reduce(x[:, :, :-1] < 1, axis=-1)
	#x[:, :, -1][e] = 1
	
	return x

# define heavy augmentations
@ex.capture
def get_training_augmentation(augment):
	if augment > 0:
		train_transform = [A.GridDistortion(
			distort_limit=args.augment, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=(240, 240, 240), mask_value=0)]
	else:
		train_transform = []
		
	train_transform += [
		
		#A.ShiftScaleRotate(rotate_limit=10, scale_limit=0, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=1),

		#A.ElasticTransform(
		#    alpha=0.4, alpha_affine=0.3, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=(240, 240, 240), mask_value=0),
		
		A.Rotate(
			limit=10, interpolation=cv2.INTER_AREA,
			border_mode=cv2.BORDER_CONSTANT, value=(240, 240, 240), mask_value=0),

		#A.IAAAdditiveGaussianNoise(p=0.1),

		#A.OneOf(
		#    [
		#        A.CLAHE(p=1),
		#        A.RandomBrightness(p=1),
		#        A.RandomGamma(p=1),
		#    ],
		#    p=0.9,
		#),

		#A.OneOf(
		#    [
		#        A.IAASharpen(p=1),
		#        A.Blur(blur_limit=3, p=1),
		#    ],
		#    p=0.9,
		#),

		A.OneOf(
			[
				A.RandomContrast(p=1),
				A.HueSaturationValue(p=1),
			],
			p=0.9,
		),        

		A.Lambda(mask=round_clip_0_1)
	]

	return A.Compose(train_transform)


def get_validation_augmentation():
	"""Add paddings to make image shape divisible by 32"""
	test_transform = [
	#    A.PadIfNeeded(512, 512)
	]
	return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
	"""Construct preprocessing transform
	
	Args:
		preprocessing_fn (callbale): data normalization function 
			(can be specific for each pretrained neural network)
	Return:
		transform: albumentations.Compose
	
	"""
	
	_transform = [
		A.Lambda(image=preprocessing_fn),
	]
	return A.Compose(_transform)

# -----------------------------------------------------------------------------

@ex.capture
def log_std_metrics(_run, logs):
	for k in ('loss', 'iou_score', 'precision', 'recall'):
		if logs.get(k) is not None:
			_run.log_scalar(k, float(logs.get(k)))
		if logs.get('val_' + k) is not None:
			_run.log_scalar("val_" + k, float(logs.get('val_' + k)))


class LogMetricsToSacred(keras.callbacks.Callback):
	def on_epoch_end(self, _, logs={}):
		log_std_metrics(logs=logs)


@ex.capture
def log_custom_metric(_run, key, value):
	_run.log_scalar(key, value)


class EvaluateMoreMetrics(keras.callbacks.Callback):
	def __init__(self, valid_dataset, n_classes, save_path=None):
		self._valid_dataset = valid_dataset
		self._n_classes = n_classes
		self._save_path = save_path
		self._best = -np.inf

	def on_epoch_end(self, _, logs={}):
		metrics_values = defaultdict(list)

		for i in range(len(self._valid_dataset)):
			
			image, gt_mask = self._valid_dataset[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = self.model.predict(image)

			y_pred = np.argmax(pr_mask.squeeze(), axis=-1).astype(np.uint8)
			y_true = np.argmax(gt_mask.squeeze(), axis=-1).astype(np.uint8)
					
			metrics = lsd_metrics(
				y_pred,
				y_true,
				self._n_classes)

			for k in metrics._fields:
				metrics_values[k].append(float(getattr(metrics, k)))

			y_true_flat = y_true.reshape((y_true.size, ))
			y_pred_flat = y_pred.reshape((y_pred.size, ))

			if False:
				for k in ('micro', 'macro', 'weighted'):
					metrics_values['precision-%s' % k].append(
						sklearn.metrics.precision_score(y_true_flat, y_pred_flat, average=k))
					metrics_values['recall-%s' % k].append(
						sklearn.metrics.recall_score(y_true_flat, y_pred_flat, average=k))
					metrics_values['jaccard-%s' % k].append(
						sklearn.metrics.jaccard_score(y_true_flat, y_pred_flat, average=k))

			metrics_values['matthews'].append(
				sklearn.metrics.matthews_corrcoef(y_true_flat, y_pred_flat))


		for k, v in metrics_values.items():
			log_custom_metric(key="val_" + k, value=np.mean(np.array(v)))

		if self._save_path is not None:
			current = np.mean(np.array(metrics_values['matthews']))
			if current > self._best:
				self.model.save(str(self._save_path), overwrite=True)
				self._best = current


@ex.capture
def get_tmp_model_path(_run):
	(sacred_runs_path / "tmp").mkdir(exist_ok=True)
	return sacred_runs_path / "tmp" / ("model-%s.h5" % str(_run._id))


@ex.automain
def run(model, backbone, batchsize, lr, epochs, data_path, data_name, fold, labels, partial, illustrations, ngpus, save_model):
	BACKBONE = backbone
	BATCH_SIZE = batchsize
	LR = lr
	EPOCHS = epochs

	DATA_DIR = Path(data_path) / (data_name + ("_FOLD%d" % fold))

	x_train_dir = os.path.join(DATA_DIR, 'train_images')
	y_train_dir = os.path.join(DATA_DIR, 'train_labels')

	x_valid_dir = os.path.join(DATA_DIR, 'val_images')
	y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

	x_test_dir = os.path.join(DATA_DIR, 'test_images')
	y_test_dir = os.path.join(DATA_DIR, 'test_labels')

	preprocess_input = sm.get_preprocessing(BACKBONE)

	all_classes = json.loads(open(DATA_DIR/"meta"/"codes.json").read())
	all_weights = json.loads(open(DATA_DIR/"meta"/"weights.json").read())

	if labels == "txt":
		CLASSES = ['text', 'tabelleninhalt']
		if illustrations:
			CLASSES.append('illustration')
	elif labels == "sep":
		CLASSES = ['tab', 'h', 'v']
	else:
		raise RuntimeError("unknown label type %s" % labels)
		
	background_w_inv = 0
	for c, w in zip(all_classes, all_weights):
		if c not in CLASSES:
			background_w_inv += 1 / w
	WEIGHTS = [all_weights[all_classes.index(s)] for s in CLASSES] + [1 / background_w_inv]

	# reorder for background label.
	#CLASSES = CLASSES[1:]
	#WEIGHTS = WEIGHTS[1:] + [WEIGHTS[0]]

	# define network parameters
	n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
	activation = 'sigmoid' if n_classes == 1 else 'softmax'

	print("creating model...")

	#create model
	models = dict(unet=sm.Unet, pspnet=sm.PSPNet, linknet=sm.Linknet, fpn=sm.FPN)
	kwargs = dict()
	if model == "pspnet":
		kwargs['input_shape'] = (768, 512, 3)
	my_model = (models[model])(BACKBONE, classes=n_classes, activation=activation, **kwargs)

	# define optomizer
	#clr = tensorflow_addons.optimizers.TriangularCyclicalLearningRate(
	#    initial_learning_rate=LR / 1000, maximal_learning_rate=LR,
	#    step_size=2000)

	#optim = keras.optimizers.SGD(learning_rate=LR)
	#optim = NovoGrad(100)
	#optim = Lookahead(Adam(2 * 1e-3))  # was 2 * 1e-3
	#optim = Lookahead(RAdam(2 * 1e-3, min_lr=1e-5))
	#optim = Lookahead(SGD(2 * 1e-3))

	# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
	# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
	#dice_loss = sm.losses.DiceLoss(class_weights=np.array(WEIGHTS))
	#focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
	#total_loss = dice_loss + (args.focal * focal_loss)
	#total_loss = sm.losses.CategoricalCELoss(class_weights=np.array(WEIGHTS))

	if labels == "txt":
		total_loss = sm.losses.CategoricalCELoss(class_weights=np.array(WEIGHTS))
	elif labels == "sep":
		total_loss = sm.losses.DiceLoss(class_weights=np.array(WEIGHTS))
	else:
		raise RuntimeError("unknown label type %s" % args.labels)

	# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
	# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

	normed_weights = np.array(WEIGHTS)
	normed_weights /= np.sum(normed_weights)

	metrics = [
		sm.metrics.IOUScore(threshold=0.5, name="iou_score"),
		sm.metrics.Precision(threshold=0.5, name="precision"),
		sm.metrics.Recall(threshold=0.5, name="recall")]

	# Dataset for train images
	train_dataset = Dataset(
		x_train_dir, 
		y_train_dir, 
		classes=CLASSES, 
		augmentation=get_training_augmentation(),
		preprocessing=get_preprocessing(preprocess_input),
		partial=partial
	)

	# Dataset for validation images
	valid_dataset = Dataset(
		x_valid_dir, 
		y_valid_dir, 
		classes=CLASSES, 
		augmentation=get_validation_augmentation(),
		preprocessing=get_preprocessing(preprocess_input),
	)

	train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

	optim = Lookahead(AdamW(
		2 * LR,  # was: 5 * LR
		#beta_1=0.8,
		#eta_min=0.1 / 2,
		use_cosine_annealing=True,
		batch_size=batchsize,
		total_iterations=len(train_dataset) // batchsize))  #, slow_step=0.25)

	if ngpus > 1:
		#from keras.utils import multi_gpu_model
		#my_model = multi_gpu_model(my_model, gpus=ngpus)
		my_model = keras.utils.multi_gpu_model(my_model, gpus=ngpus)

	print("compiling model...")

	# compile keras model with defined optimozer, loss and metrics
	my_model.compile(optim, total_loss, metrics)

	if False:
		import matplotlib.pyplot as plt
		lr_finder = LRFinder(my_model)
		lr_finder.find(train_dataloader, num_iter=len(train_dataloader) * 1, start_lr=1e-3, end_lr=1e-1)
		lr_finder.plot_loss()
		with open("temp.csv", "w") as f:
			for a, b in zip(lr_finder.lrs, lr_finder.losses):
				f.write("%f;%f\n" % (a, b))
		plt.savefig('lr.png')
		sys.exit(0)

	# check shapes for errors
	#assert train_dataloader[0][0].shape == (BATCH_SIZE, 768, 512, 3)
	#assert train_dataloader[0][1].shape == (BATCH_SIZE, 768, 512, n_classes)

	#keras.callbacks.ReduceLROnPlateau(mode='max', monitor='val_f0.5', patience=4, factor=0.5),
	#keras.callbacks.EarlyStopping(mode='max', monitor='val_f0.5', patience=25, min_delta=0.01),
	#OneCycleScheduler(3e-3, pct_start=0.9, start_div=3e-3 / 2e-3, end_div=1),

	save_model_path = None
	if save_model:
		save_model_path = get_tmp_model_path()

	# define callbacks for learning rate scheduling and best checkpoints saving
	callbacks = [
		LogMetricsToSacred(),
		EvaluateMoreMetrics(valid_dataset, n_classes, save_model_path),
		keras.callbacks.TerminateOnNaN()
		#keras.callbacks.CSVLogger(str(model_path / "training.csv"), separator=',', append=False),
	]

	if False:
		callbacks.append(keras.callbacks.ModelCheckpoint(
			str(model_path),
			save_weights_only=True,
			save_best_only=True,
			mode='max',
			monitor='val_matthews'))

	print("starting fit...")

	# train model
	history = my_model.fit_generator(
		train_dataloader, 
		steps_per_epoch=len(train_dataloader), 
		epochs=EPOCHS, 
		callbacks=callbacks, 
		validation_data=valid_dataloader, 
		validation_steps=len(valid_dataloader)
	)

	if save_model_path:
		ex.add_artifact(str(save_model_path))

	# evaluation.

	if False:
		lsd_name = ('pixel_accuracy', 'mean_accuracy', 'mean_IU', 'frequency_weighted_IU')
		lsd_value = defaultdict(list)

		for i in range(len(valid_dataset)):
			
			image, gt_mask = valid_dataset[i]
			image = np.expand_dims(image, axis=0)
			pr_mask = my_model.predict(image)
					
			metrics = lsd_metrics(
				np.argmax(pr_mask.squeeze(), axis=-1).astype(np.uint8),
				np.argmax(gt_mask.squeeze(), axis=-1).astype(np.uint8),
				n_classes)

			for k in lsd_name:
				lsd_value[k].append(float(getattr(metrics, k)))


		for k, v in lsd_value.items():
			ex.log_scalar("val_" + k, np.mean(np.array(v)))
