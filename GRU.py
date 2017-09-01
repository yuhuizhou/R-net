from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf
from params import Params

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class GRUCell(RNNCell):
	"""Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078) with layer normalization."""
	def __init__(self,
			   num_units,
			   reuse=None,
			   kernel_initializer=None,
			   bias_initializer=None,
			   is_training = True):

		super(GRUCell, self).__init__(_reuse=reuse)
		self._num_units = num_units
		self._activation = math_ops.tanh
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer
		self._is_training = is_training

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope = None):
		"""Gated recurrent unit (GRU) with nunits cells."""
		with vs.variable_scope("gates"):
			inputs_ = _linear(inputs, 3 * self._num_units, False, scope="inputs_")
			state_ = _linear(state, 3 * self._num_units, False, scope="state_")
			bias = tf.get_variable('bias', [3 * self._num_units], dtype = tf.float32)
			bn_inputs_ = batch_norm(inputs_, "inputs_bn", is_training = self._is_training)
			bn_state_ = batch_norm(state_, "states_bn", is_training = self._is_training)
			hidden = bn_inputs_ + bn_state_ + bias
			r, u, c = array_ops.split(hidden, 3, 1)
			r, u = tf.sigmoid(r), tf.sigmoid(u)
		with vs.variable_scope("candidate"):
			c = self._activation(c)
		new_h = u * state + (1 - u) * c
		return new_h, new_h

class gated_attention_GRUCell(RNNCell):
	def __init__(self,
			   num_units,
			   memory,
			   params,
			   output_argmax = None,
			   self_matching = False,
			   gated_attention = None,
			   reuse=None,
			   kernel_initializer=None,
			   bias_initializer=None,
			   is_training = True):

		super(gated_attention_GRUCell, self).__init__(_reuse=reuse)
		self._num_units = num_units
		self._activation = math_ops.tanh
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer
		self._attention = memory
		self._output_argmax = output_argmax
		self._params = params
		self._self_matching = self_matching
		self._gated_attention = gated_attention
		self._is_training = is_training

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def call(self, inputs, state):
		"""Gated recurrent unit (GRU) with gated attention modules."""
		with vs.variable_scope("attention_pool"):
			inputs = self._gated_attention(self._attention,
									inputs,
									state,
									self._num_units,
									params = self._params,
									self_matching = self._self_matching,
									output_argmax = self._output_argmax)
		with vs.variable_scope("gates"):
			inputs_ = _linear(inputs, 3 * self._num_units, False, scope="inputs_")
			state_ = _linear(state, 3 * self._num_units, False, scope="state_")
			bias = tf.get_variable('bias', [3 * self._num_units], dtype = tf.float32)
			bn_inputs_ = batch_norm(inputs_, "inputs_bn", is_training = self._is_training)
			bn_state_ = batch_norm(state_, "states_bn", is_training = self._is_training)
			hidden = bn_inputs_ + bn_state_ + bias
			r, u, c = array_ops.split(hidden, 3, 1)
			r, u = tf.sigmoid(r), tf.sigmoid(u)
		with vs.variable_scope("candidate"):
			c = self._activation(c)
		new_h = u * state + (1 - u) * c
		return new_h, new_h

def batch_norm(x, name_scope, is_training, epsilon=1e-3, decay=0.999):
	'''Assume 2d [batch, values] tensor'''

	with tf.variable_scope(name_scope):
		size = x.get_shape().as_list()[1]

		scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
		offset = tf.get_variable('offset', [size])

		pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
		pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
		batch_mean, batch_var = tf.nn.moments(x, [0])

		train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
		train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

		if is_training:
			with tf.control_dependencies([train_mean_op, train_var_op]):
				return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
		else:
			return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

def layer_norm(input, dim, epsilon = 1e-5, max = 1000, scope = "layer_norm"):
	with vs.variable_scope(scope):
		""" Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
		s = tf.get_variable("s", shape = dim, initializer=tf.ones_initializer(), dtype=tf.float32)
		b = tf.get_variable("b", shape = dim, initializer=tf.zeros_initializer(), dtype=tf.float32)
		m, v = tf.nn.moments(input, [1], keep_dims=True)
		normalised_input = (input - m) / tf.sqrt(v + epsilon)
		return normalised_input * s + b

def _linear(args,
			output_size,
			bias,
			bias_initializer=None,
			kernel_initializer=None,
			scope = "linear"):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
	args: a 2D Tensor or a list of 2D, batch x n, Tensors.
	output_size: int, second dimension of W[i].
	bias: boolean, whether to add a bias term or not.
	bias_initializer: starting value to initialize the bias
	  (default is all zeros).
	kernel_initializer: starting value to initialize the weight.
  Returns:
	A 2D Tensor with shape [batch x output_size] equal to
	sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
	ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
	raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
	args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
	if shape.ndims != 2:
	  raise ValueError("linear is expecting 2D arguments: %s" % shapes)
	if shape[1].value is None:
	  raise ValueError("linear expects shape[1] to be provided for shape %s, "
					   "but saw %s" % (shape, shape[1]))
	else:
	  total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope) as outer_scope:
	weights = vs.get_variable(
		_WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
		dtype=dtype,
		initializer=kernel_initializer)
	if len(args) == 1:
	  res = math_ops.matmul(args[0], weights)
	else:
	  res = math_ops.matmul(array_ops.concat(args, 1), weights)
	if not bias:
	  return res
	with vs.variable_scope(outer_scope) as inner_scope:
	  inner_scope.set_partitioner(None)
	  if bias_initializer is None:
		bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
	  biases = vs.get_variable(
		  _BIAS_VARIABLE_NAME, [output_size],
		  dtype=dtype,
		  initializer=bias_initializer)
	return nn_ops.bias_add(res, biases)
