# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell
from params import Params
from zoneout import ZoneoutWrapper
'''
attention weights from https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
W_u^Q.shape:    (2 * attn_size, attn_size)
W_u^P.shape:    (2 * attn_size, attn_size)
W_v^P.shape:    (attn_size, attn_size)
W_g.shape:      (4 * attn_size, 4 * attn_size)
W_h^P.shape:    (2 * attn_size, attn_size)
W_v^Phat.shape: (2 * attn_size, attn_size)
W_h^a.shape:    (2 * attn_size, attn_size)
W_v^Q.shape:    (attn_size, attn_size)
'''

def encoding(word, char, word_embeddings, char_embeddings, scope = "embedding"):
	with tf.variable_scope(scope):
		word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
		char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
		return word_encoding, char_encoding

def apply_dropout(inputs, size = None, is_training = True):
	'''
	Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
	'''
	if Params.dropout is None and Params.zoneout is None:
		return inputs
	if Params.zoneout is not None:
		return ZoneoutWrapper(inputs, state_zoneout_prob= Params.zoneout, is_training = is_training)
	elif is_training:
		return tf.contrib.rnn.DropoutWrapper(inputs,
											output_keep_prob = 1 - Params.dropout,
											# variational_recurrent = True,
											# input_size = size,
											dtype = tf.float32)
	else:
		return inputs

def dot_product_attention(q,
						  k,
						  v,
						  bias,
						  seq_len = None,
						  axis = -1,
						  is_training = True,
						  scope=None,
						  reuse = None):
	"""dot-product attention.
	Args:
	q: a Tensor with shape [batch, heads, length_q, depth_k]
	k: a Tensor with shape [batch, heads, length_kv, depth_k]
	v: a Tensor with shape [batch, heads, length_kv, depth_v]
	bias: bias Tensor (see attention_bias())
	is_training: a bool of training
	scope: an optional string
	Returns:
	A Tensor.
	"""
	with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
		# [batch, num_heads, query_length, memory_length]
		logits = tf.matmul(q, k, transpose_b=True)
		if bias:
			b = tf.get_variable("bias", logits.shape[-1], initializer = tf.zeros_initializer())
			logits += b
		if seq_len is not None:
			logits = mask_attn_score(logits, seq_len, axis)
		weights = tf.nn.softmax(logits, name="attention_weights")
		# dropping out the attention links for each of the heads
		if is_training and Params.dropout is not None:
			weights = tf.nn.dropout(weights, 1.0 - Params.dropout)
		return tf.matmul(weights, v)

def additive_attention(inputs, units, memory_len, scope = "additive_attention", reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		if type(inputs) == list:
			inputs = tf.concat(inputs, -1)
		out = tf.nn.tanh(tf.layers.dense(inputs, units, activation = None, use_bias=False, name="W"))
		out = tf.squeeze(tf.layers.dense(out, 1, use_bias = False, name = "v"))
		if Params.bias:
			out += tf.get_variable("bias", out.shape[-1], initializer = tf.zeros_initializer())
		return mask_attn_score(out, memory_len)

def split_last_dimension(x, n):
	"""Reshape x so that the last dimension becomes two dimensions.
	The first of these two dimensions is n.
	Args:
	x: a Tensor with shape [..., m]
	n: an integer.
	Returns:
	a Tensor with shape [..., n, m/n]
	"""
	old_shape = x.get_shape().dims
	last = old_shape[-1]
	new_shape = old_shape[:-1] + [n] + [last // n if last else None]
	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
	ret.set_shape(new_shape)
	return tf.transpose(ret,[0,2,1,3])

def combine_last_two_dimensions(x):
	"""Reshape x so that the last two dimension become one.
	Args:
	x: a Tensor with shape [..., a, b]
	Returns:
	a Tensor with shape [..., ab]
	"""
	old_shape = x.get_shape().dims
	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]
	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
	ret.set_shape(new_shape)
	return ret

def compute_qkv(query_antecedent,
				memory_antecedent,
				total_key_depth,
				total_value_depth):
	"""Computes query, key and value.
	Args:
	query_antecedent: a Tensor with shape [batch, length_q, channels]
	memory_antecedent: a Tensor with shape [batch, length_m, channels]
	total_key_depth: an integer
	total_value_depth: and integer
	q_filter_width: An integer specifying how wide you want the query to be.
	kv_filter_width: An integer specifying how wide you want the keys and values
	to be.
	q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
	kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
	Returns:
	q, k, v : [batch, length, depth] tensors
	"""
	if memory_antecedent is None:
		memory_antecedent = query_antecedent
	def _compute(inp, depth, name):
		return tf.layers.dense(inp, depth, activation = None, use_bias=False, name=name)
	q = _compute(query_antecedent, total_key_depth, "q")
	k = _compute(memory_antecedent, total_key_depth, "k")
	v = _compute(memory_antecedent, total_value_depth, "v")
	return q, k, v

def cudnn_GRU(inputs, inputs_len, layers, num_units, is_training, output_order = 0, concat = True, scope = "cudnn_GRU", batch_size = Params.batch_size ):
	with tf.variable_scope(scope):
		inputs = tf.transpose(inputs, [1,0,2])
		res = []
		for i in range(layers):
			shapes = get_shape(inputs)
			gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(layers, num_units, shapes[-1])
			gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(layers, num_units, shapes[-1])
			h_fw = tf.get_variable("h_fw_%d"%i, dtype = tf.float32, shape = (1, batch_size, num_units), initializer = tf.zeros_initializer())
			h_bw = tf.get_variable("h_bw_%d"%i, dtype = tf.float32, shape = (1, batch_size, num_units), initializer = tf.zeros_initializer())
			# w_fw = tf.get_variable("w_fw_%d"%i, dtype = tf.float32, shape = (shapes[-1], num_units), initializer = tf.contrib.layers.xavier_initializer())
			# w_bw = tf.get_variable("w_bw_%d"%i, dtype = tf.float32, shape = (shapes[-1], num_units), initializer = tf.contrib.layers.xavier_initializer())
			w_fw = tf.Variable(tf.random_uniform([gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
			w_bw = tf.Variable(tf.random_uniform([gru_bw.params_size()], -0.1, 0.1), validate_shape=False)
			with tf.variable_scope("fw_%d"%i):
				outputs_fw, state_fw = gru_fw(inputs, h_fw, w_fw)
			with tf.variable_scope("bw_%d"%i):
				inputs_bw = tf.reverse_sequence(inputs, seq_lengths = inputs_len, seq_dim = 0, batch_dim = 1)
				outputs_bw, state_bw = gru_bw(inputs_bw, h_bw, w_bw)
				outputs_bw = tf.reverse_sequence(outputs_bw, seq_lengths = inputs_len, seq_dim = 0, batch_dim = 1)
			inputs = tf.concat([outputs_fw, outputs_bw], axis = -1)
			res.append(inputs)
		if output_order == 0:
			if concat:
				outputs = tf.concat(res, axis = -1)
			else:
				outputs = sum(res)
			return tf.transpose(outputs, [1,0,2])
		else:
			return tf.squeeze(tf.concat([state_fw, state_bw], axis = -1))

def bidirectional_GRU(inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units = Params.attn_size, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
	'''
	Bidirectional recurrent neural network with GRU cells.

	Args:
		inputs:     rnn input of shape (batch_size, timestep, dim)
		inputs_len: rnn input_len of shape (batch_size, )
		cell:       rnn cell of type RNN_Cell.
		output:     if 0, output returns rnn output for every timestep,
					if 1, output returns concatenated state of backward and
					forward rnn.
	'''
	with tf.variable_scope(scope, reuse = reuse):
		if cell is not None:
			(cell_fw, cell_bw) = cell
		else:
			shapes = get_shape(inputs)
			if len(shapes) > 3:
				inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
				inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

			# if no cells are provided, use standard GRU cell implementation
			if layers > 1:
				cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
				cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
			else:
				cell_fw, cell_bw = [apply_dropout(cell_fn(units), size = inputs.shape[-1], is_training = is_training) for _ in range(2)]

		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
														sequence_length = inputs_len,
														dtype=tf.float32)
		if output == 0:
			return tf.concat(outputs, 2)
		elif output == 1:
			return tf.reshape(tf.concat(states,1),(Params.batch_size, shapes[1], 2*units))

def pointer_net(passage, passage_len, question, question_len, cell, scope = "pointer_network"):
	'''
	Answer pointer network as proposed in https://arxiv.org/pdf/1506.03134.pdf.

	Args:
		passage:        RNN passage output from the bidirectional readout layer (batch_size, timestep, dim)
		passage_len:    variable lengths for passage length
		question:       RNN question output of shape (batch_size, timestep, dim) for question pooling
		question_len:   Variable lengths for question length
		cell:           rnn cell of type RNN_Cell.
		params:         Appropriate weight matrices for attention pooling computation

	Returns:
		Unscaled logits for the answer pointer of the beginning and the end of the answer span
	'''
	with tf.variable_scope(scope):
		shapes = get_shape(passage)
		initial_state = question_pooling(question, units = Params.attn_size, memory_len = question_len, scope = "question_pooling")
		initial_state_ = tf.expand_dims(initial_state, 1)
		p1_logits = attention(passage, initial_state_, Params.attn_size, memory_len = passage_len, scope = "attention", additive = True)
		scores = tf.expand_dims(tf.nn.softmax(p1_logits), -1)
		attention_pool = tf.reduce_sum(scores * passage,1)
		_, state = cell(attention_pool, initial_state)
		state = tf.expand_dims(state, 1)
		p2_logits = attention(passage, state, Params.attn_size, memory_len = passage_len, scope = "attention", reuse = True, additive = True)
		return p1_logits, p2_logits

def attention_rnn(inputs, inputs_len, units, attn_cell, bidirection = True, scope = "gated_attention_rnn", is_training = True):
	with tf.variable_scope(scope):
		if bidirection:
			outputs = bidirectional_GRU(inputs,
										inputs_len,
										cell = attn_cell,
										scope = scope + "_bidirectional",
										output = 0,
										is_training = is_training)
		else:
			outputs, _ = tf.nn.dynamic_rnn(attn_cell, inputs,
											sequence_length = inputs_len,
											dtype=tf.float32)
		return outputs

def question_pooling(memory, units, memory_len = None, scope = "question_pooling"):
	with tf.variable_scope(scope):
		shapes = get_shape(memory)
		# V_r = tf.get_variable("question_param", shape = (1, 1, shapes[-1]), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
		# V_r = tf.tile(V_r, [Params.batch_size, 1, 1])
		attn = attention(memory, None, shapes[-1], memory_len = memory_len, scope = "question_attention_pooling", additive = True)
		score = tf.expand_dims(tf.nn.softmax(attn),-1)
		return tf.reduce_sum(score * memory, 1)

def gated_attention(memory, inputs, units, memory_len = None, is_training = None, scope="gated_attention"):
	with tf.variable_scope(scope):
		outputs = attention(inputs, memory, units, memory_len = memory_len)
		shape = get_shape(outputs)
		W_g = tf.get_variable("W_g",dtype = tf.float32, shape = (shape[-1], shape[-1]), initializer = tf.contrib.layers.xavier_initializer())
		g_t = tf.sigmoid(matmul_3by2(outputs,W_g))
		return g_t * outputs

def matmul_3by2(inputs, weights):
	i_s = get_shape(inputs)
	w_s = get_shape(weights)
	inputs = tf.reshape(inputs, [-1, i_s[-1]])
	outputs = tf.matmul(inputs, weights)
	return tf.reshape(outputs, i_s[:-1] + [w_s[-1]])

def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e30):
	return score
	# shape = get_shape(score)
	# score_mask = tf.sequence_mask(memory_sequence_length, maxlen=shape[-1], dtype = tf.float32)
	# mask_shape = [s if i in [0,len(shape)-1] else 1 for i,s in enumerate(shape)]
	# score_mask = tf.reshape(score_mask, mask_shape)
	# return score + score_mask_value * (1.0 - score_mask)

def get_shape(inputs):
	return inputs.shape.as_list()

def attention(inputs, memory, units, scope = "attention", memory_len = None, reuse = None, additive = False, num_heads = 5):
	with tf.variable_scope(scope, reuse = reuse):
		if additive:
			if memory is None:
				return additive_attention(inputs, units, memory_len, reuse = reuse)
			memory = tf.tile(memory, [1, Params.max_p_len, 1])
			return additive_attention([inputs, memory], units, memory_len, reuse = reuse)
		qkv = compute_qkv(inputs, memory, units, units)
		q, k, v = [split_last_dimension(x, num_heads) for x in qkv]
		key_depth_per_head = units // num_heads
		q *= key_depth_per_head**-0.5
		attn = dot_product_attention(q, k, v, Params.bias, seq_len = memory_len, is_training = True, scope = "multiplicative_attention", reuse = reuse)
		attn = combine_last_two_dimensions(tf.transpose(attn,[0, 2, 1, 3]))
		return attn
		# print(attn)
		# exit()
		# outputs_ = []
		# i_s = get_shape(inputs)
		# m_s = get_shape(memory)
		# inputs = tf.reshape(inputs, (-1, i_s[-1]))
		# memory = tf.reshape(memory, (-1, m_s[-1]))
		# for i, inp in enumerate([inputs, memory]):
		# 	shapes = inp.shape.as_list()
		# 	# inp = tf.reshape(inp, (-1, shapes[-1]))
		# 	w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
		# 	outputs = tf.matmul(inp, w)
		# 	outputs_.append(outputs)
		# outputs, memory = outputs_
		# outputs = tf.reshape(outputs, [Params.batch_size, i_s[1], 1, Params.attn_size])
		# memory = tf.reshape(memory, [Params.batch_size, 1, m_s[1], Params.attn_size])
		# outputs += memory
		# if Params.bias:
		# 	b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		# 	outputs += b
		# v = tf.get_variable("v", shape = Params.attn_size, dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		# scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
		# if memory_len is not None:
		# 	scores = mask_attn_score(scores, memory_len)
		# return tf.nn.softmax(scores) # all attention output is softmaxed now

def attention3(inputs, memory, units, scope = "attention", memory_len = None, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		outputs_ = []
		i_s = get_shape(inputs)
		m_s = get_shape(memory)
		inputs = tf.reshape(inputs, (-1, i_s[-1]))
		memory = tf.reshape(memory, (-1, m_s[-1]))
		for i, inp in enumerate([inputs, memory]):
			shapes = get_shape(inp)
			# inp = tf.reshape(inp, (-1, shapes[-1]))
			w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
			outputs = tf.matmul(inp, w)
			outputs_.append(outputs)
		outputs, memory = outputs_
		outputs = tf.reshape(outputs, [Params.batch_size, i_s[1], 1, Params.attn_size])
		memory = tf.reshape(memory, [Params.batch_size, 1, m_s[1], Params.attn_size])
		outputs += memory
		if Params.bias:
			b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			outputs += b
		v = tf.get_variable("v", shape = Params.attn_size, dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
		if memory_len is not None:
			scores = mask_attn_score(scores, memory_len)
		return tf.nn.softmax(scores) # all attention output is softmaxed now

def attention2(inputs, memory, units, scope = "attention", memory_len = None, reuse = None):
	with tf.variable_scope(scope, reuse = reuse):
		outputs_ = []
		print(inputs)
		print(memory)
		exit()
		for i, inp in enumerate(inputs):
			shapes = inp.shape.as_list()
			inp = tf.reshape(inp, (-1, shapes[-1]))
			w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],Params.attn_size], initializer = tf.contrib.layers.xavier_initializer())
			outputs = tf.matmul(inp, w)
			# Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
			if len(shapes) > 2:
				outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
			elif len(shapes) == 2 and shapes[0] is Params.batch_size:
				outputs = tf.reshape(outputs, (shapes[0],1,-1))
			else:
				outputs = tf.reshape(outputs, (1, shapes[0],-1))
			outputs_.append(outputs)
		print(outputs_)
		exit()
		outputs = sum(outputs_)
		if Params.bias:
			b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
			outputs += b
		v = tf.get_variable("v", shape = Params.attn_size, dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
		if memory_len is not None:
			scores = mask_attn_score(scores, memory_len)
		return tf.nn.softmax(scores) # all attention output is softmaxed now

def cross_entropy(output, target):
	cross_entropy = target * tf.log(output + 1e-8)
	cross_entropy = -tf.reduce_sum(cross_entropy, 2) # sum across passage timestep
	cross_entropy = tf.reduce_mean(cross_entropy, 1) # average across pointer networks output
	return tf.reduce_mean(cross_entropy) # average across batch size

def total_params():
	total_parameters = 0
	for variable in tf.trainable_variables():
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	print("Total number of trainable parameters: {}".format(total_parameters))
