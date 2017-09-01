# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch
from params import Params
from layers import *
from GRU import gated_attention_GRUCell, GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
import numpy as np

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer(learning_rate = Params.learning_rate, epsilon = 1e-06),
					"adam":tf.train.AdamOptimizer(learning_rate = Params.learning_rate),
					"gradientdescent":tf.train.GradientDescentOptimizer(learning_rate = Params.learning_rate),
					"adagrad":tf.train.AdagradOptimizer(learning_rate = Params.learning_rate)}

class Model(object):
	def __init__(self,is_training = True):
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			data, self.num_batch = get_batch()
			(self.passage_w,
			self.question_w,
			self.passage_c,
			self.question_c,
			self.passage_w_len,
			self.question_w_len,
			self.passage_c_len,
			self.question_c_len,
			self.indices) = data

			self.encode_ids()
			self.params = get_attn_params(Params.attn_size)
			self.attention_match_rnn()
			self.bidirectional_readout()
			self.pointer_network()
			self.loss_function()
			self.summary()
			self.init_op = tf.global_variables_initializer()
			total_params()

	def encode_ids(self):
		with tf.variable_scope("encode"):
			# Character encoding
			self.passage_c = bidirectional_GRU(self.passage_c,
												self.passage_c_len,
												scope = "passage_char_encoding",
												output = 1,
												is_training = self.is_training)
			self.question_c = bidirectional_GRU(self.question_c,
												self.question_c_len,
												scope = "question_char_encoding",
												output = 1,
												is_training = self.is_training)

			# Concat with last dim (75 * 2 = 150)
			self.passage_encoding = tf.concat((self.passage_w, self.passage_c),axis = 2)
			self.question_encoding = tf.concat((self.question_w, self.question_c),axis = 2)

			# Passage and question encoding
			self.passage_encoding = bidirectional_GRU(self.passage_encoding,
														self.passage_w_len,
														layers = Params.num_layers,
														scope = "passage_encoding",
														output = 0,
														is_training = self.is_training)
			self.question_encoding = bidirectional_GRU(self.question_encoding,
														self.question_w_len,
														layers = Params.num_layers,
														scope = "question_encoding",
														output = 0,
														is_training = self.is_training)

	def attention_match_rnn(self):
		with tf.variable_scope("attention_match_rnn"):
			memory = self.question_encoding
			inputs = self.passage_encoding
			scopes = ["question_passage_matching", "self_matching"]
			params = [((tf.concat((self.params["W_u_Q"],
									self.params["W_u_P"],
									self.params["W_v_P"]),axis = 0),
									self.params["v"]),self.params["W_g"]),
						((tf.concat((self.params["W_v_P"],
									self.params["W_v_P"],
									self.params["W_v_Phat"]),axis = 0),
									self.params["v"]),self.params["W_g"])]
			for i in range(2):
				# cell_fw = MultiRNNCell([apply_dropout(gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i]),is_training = self.is_training) for _ in range(Params.num_layers)])
				# cell_bw = MultiRNNCell([apply_dropout(gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i]),is_training = self.is_training) for _ in range(Params.num_layers)])
				cell_fw = gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i], self_matching = True if i == 1 else False, gated_attention = gated_attention, is_training = self.is_training)
				cell_bw = gated_attention_GRUCell(Params.attn_size, memory = memory, params = params[i], self_matching = True if i == 1 else False, gated_attention = gated_attention, is_training = self.is_training)
				inputs = attention_rnn(inputs,
										self.passage_w_len,
										Params.attn_size,
										(cell_fw,cell_bw),
										scope = scopes[i])
				memory = inputs # self_matching
				inputs = apply_dropout(inputs, is_training = self.is_training)
			self.self_matching_output = inputs

	def bidirectional_readout(self):
		self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output,
			self.passage_w_len,
			# layers = Params.num_layers, # 1 or 3 layers?
			scope = "bidirectional_readout",
			output = 0,
			is_training = self.is_training)

	def pointer_network(self):
		params = ((tf.concat((self.params["W_u_Q"],self.params["W_v_Q"]),axis = 0),self.params["v"]),
					(tf.concat((self.params["W_h_P"],self.params["W_h_a"]),axis = 0),self.params["v"]))
		cell = apply_dropout(GRUCell(Params.attn_size*2, is_training = self.is_training), is_training = self.is_training)
		self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding, cell, params, scope = "pointer_network")

	def loss_function(self):
		with tf.variable_scope("loss"):
			shapes = self.passage_w.shape
			self.mask = tf.to_float(tf.sequence_mask(self.passage_w_len, shapes[1]))
			self.points_logits *= tf.expand_dims(self.mask,1)

			# Causes NaN error
			# self.mean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.indices, logits = self.points_logits))

			# Use non-sparse softmax
			self.indices_prob = tf.one_hot(self.indices, shapes[1])
			self.mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.indices_prob, logits = self.points_logits))
			self.optimizer = optimizer_factory[Params.optimizer]

			if Params.clip:
				# gradient clipping by norm
				gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
				gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
				self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step = self.global_step)
			else:
				self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

	def summary(self):
		tf.summary.scalar('mean_loss', self.mean_loss)
		tf.summary.scalar('passage_word_encoded',tf.reduce_mean(self.passage_w))
		tf.summary.scalar('passage_char_encoded',tf.reduce_mean(self.passage_c))
		tf.summary.scalar('question_word_encoded',tf.reduce_mean(self.question_w))
		tf.summary.scalar('question_char_encoded',tf.reduce_mean(self.question_c))
		tf.summary.scalar('question_encoding',tf.reduce_mean(self.question_encoding))
		tf.summary.scalar('passage_encoding',tf.reduce_mean(self.passage_encoding))
		tf.summary.scalar('self_matching',tf.reduce_mean(self.self_matching_output))
		tf.summary.scalar('pointer',tf.reduce_mean(self.points_logits))
		tf.summary.scalar('learning_rate', Params.learning_rate)
		self.merged = tf.summary.merge_all()

def debug():
	model = Model(is_training = True)
	print("Built model")

def main():
	model = Model(is_training = True); print("Built model")
	with model.graph.as_default():
		sv = tf.train.Supervisor(logdir=Params.logdir,
								save_model_secs=0,
								global_step = model.global_step,
								init_op = model.init_op)
		with sv.managed_session() as sess:
			for epoch in range(1, Params.num_epochs+1):
				if sv.should_stop(): break
				for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
					sess.run(model.train_op)
					if step % Params.save_steps == 0:
						sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(epoch,step))

if __name__ == '__main__':
	if Params.debug == True:
		print("debugging...")
		debug()
	else:
		print("Running...")
		main()
