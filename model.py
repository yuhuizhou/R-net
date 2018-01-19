# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, get_dev
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
from evaluate import *
import numpy as np
import cPickle as pickle
from process import *

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
			"adam":tf.train.AdamOptimizer,
			"gradientdescent":tf.train.GradientDescentOptimizer,
			"adagrad":tf.train.AdagradOptimizer}

class Model(object):
	def __init__(self,is_training = True):
		# Build the computational graph when initializing
		self.is_training = is_training
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.data, self.num_batch = get_batch(is_training = is_training)
			(self.passage_w,
			self.question_w,
			self.passage_c,
			self.question_c,
			self.passage_w_len_,
			self.question_w_len_,
			self.passage_c_len,
			self.question_c_len,
			self.indices) = self.data

			self.passage_w_len = tf.squeeze(self.passage_w_len_)
			self.question_w_len = tf.squeeze(self.question_w_len_)

			self.encode_ids()
			self.attention_match_rnn()
			self.bidirectional_readout()
			self.pointer_network()

			if is_training:
				self.loss_function()
				self.summary()
				self.init_op = tf.global_variables_initializer()
			else:
				self.outputs()

	def encode_ids(self):
		with tf.variable_scope("encoder"):
			with tf.device('/cpu:0'):
				self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.char_vocab_size, Params.char_emb_size]),trainable=True, name="char_embeddings")
				self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),trainable=False, name="word_embeddings")
				self.word_embeddings_placeholder = tf.placeholder(tf.float32,[Params.vocab_size, Params.emb_size],"word_embeddings_placeholder")
				self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

			# Embed the question and passage information for word and character tokens
			self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
											self.passage_c,
											word_embeddings = self.word_embeddings,
											char_embeddings = self.char_embeddings,
											scope = "passage_embeddings")
			self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
											self.question_c,
											word_embeddings = self.word_embeddings,
											char_embeddings = self.char_embeddings,
											scope = "question_embeddings")
			shape = [-1, Params.max_char_len, Params.char_emb_size]
			self.passage_char_encoded = tf.reshape(self.passage_char_encoded, shape)
			self.passage_c_len = tf.reshape(self.passage_c_len, (-1,))
			self.question_char_encoded = tf.reshape(self.question_char_encoded, shape)
			self.question_c_len = tf.reshape(self.question_c_len, (-1,))
			self.passage_char_encoded = cudnn_GRU(self.passage_char_encoded, self.passage_c_len, 1,
													Params.attn_size, self.is_training, output_order = 1,
													scope = "char_passage_encoding", batch_size = Params.batch_size * Params.max_p_len)
			self.question_char_encoded = cudnn_GRU(self.question_char_encoded, self.question_c_len, 1,
													Params.attn_size, self.is_training, output_order = 1,
													scope = "char_question_encoding", batch_size = Params.batch_size * Params.max_q_len)
			self.passage_char_encoded = tf.reshape(self.passage_char_encoded, (Params.batch_size, Params.max_p_len,-1))
			self.question_char_encoded = tf.reshape(self.question_char_encoded, (Params.batch_size, Params.max_q_len,-1))
			self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded),axis = 2)
			self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded),axis = 2)

			self.passage_encoding = cudnn_GRU(self.passage_encoding, self.passage_w_len, Params.num_layers,
												Params.attn_size, self.is_training, concat = False, scope = "passage_encoding")
			self.question_encoding = cudnn_GRU(self.question_encoding, self.question_w_len, Params.num_layers,
												Params.attn_size, self.is_training, concat = False, scope = "question_encoding")

	def attention_match_rnn(self):
		# Apply gated attention recurrent network for both query-passage matching and self matching networks
		with tf.variable_scope("attention_match_rnn"):
			memory = self.question_encoding
			inputs = self.passage_encoding
			scopes = ["question_passage_matching", "self_matching"]
			for i in range(2):
				args = {"inputs": inputs,
						"units": Params.attn_size,
						"memory": memory,
						"memory_len": self.question_w_len if i == 0 else self.passage_w_len,
						"is_training": self.is_training,
						"scope": scopes[i]}
				attn_outputs = gated_attention(**args)
				inputs = cudnn_GRU(attn_outputs, self.passage_w_len, 1,
									Params.attn_size, self.is_training, scope = scopes[i])
				memory = inputs # self matching (attention over itself)
				if i == 0:
					self.question_matching = inputs
			self.self_matching_output = inputs

	def bidirectional_readout(self):
		self.final_bidirectional_outputs = cudnn_GRU(self.self_matching_output, self.passage_w_len, 1,
							Params.attn_size, self.is_training, scope = "bidirectional_readout")

	def pointer_network(self):
		cell = SRUCell(Params.attn_size * 2)
		self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding, self.question_w_len, cell, is_training = self.is_training, scope = "pointer_network")
		self.points_logits_stacked = tf.stack(self.points_logits, 1)

	def outputs(self):
		self.output_index = tf.argmax(self.points_logits, axis = 2)

	def loss_function(self):
		with tf.variable_scope("loss"):
			shapes = self.passage_w.shape
			self.indices = tf.one_hot(self.indices, shapes[1])
			# self.mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.points_logits_stacked, labels = self.indices))
			# self.indices_prob = [tf.squeeze(x) for x in tf.split(tf.one_hot(self.indices, shapes[1]),2,axis = 1)]
			# self.mean_loss1 = tf.nn.softmax_cross_entropy_with_logits(labels = self.indices_prob[0], logits = self.points_logits[0])
			# self.mean_loss2 = tf.nn.softmax_cross_entropy_with_logits(labels = self.indices_prob[1], logits = self.points_logits[1])
			# self.mean_loss = tf.reduce_mean(self.mean_loss1 + self.mean_loss2)
			self.mean_loss = cross_entropy(self.points_logits_stacked, self.indices)
			# self.mean_loss = cross_entropy(self.points_logits, self.indices_prob)
			self.optimizer = optimizer_factory[Params.optimizer](**Params.opt_arg[Params.optimizer])
			if Params.clip:
				# gradient clipping by norm
				gradients, variables = zip(*self.optimizer.compute_gradients(self.mean_loss))
				gradients, _ = tf.clip_by_global_norm(gradients, Params.norm)
				self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step = self.global_step)
			else:
				self.train_op = self.optimizer.minimize(self.mean_loss, global_step = self.global_step)

	def summary(self):
		self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="F1")
		self.F1_placeholder = tf.placeholder(tf.float32, shape = (), name = "F1_placeholder")
		self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="EM")
		self.EM_placeholder = tf.placeholder(tf.float32, shape = (), name = "EM_placeholder")
		self.dev_loss = tf.Variable(tf.constant(10.0, shape=(), dtype = tf.float32),trainable=False, name="dev_loss")
		self.dev_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "dev_loss")
		self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.dev_loss, self.dev_loss_placeholder))
		tf.summary.scalar('loss_training', self.mean_loss)
		tf.summary.scalar('loss_dev', self.dev_loss)
		tf.summary.scalar("F1_Score",self.F1)
		tf.summary.scalar("Exact_Match",self.EM)
		tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
		self.merged = tf.summary.merge_all()

def debug():
	model = Model(is_training = True)
	print("Built model")

def test():
	model = Model(is_training = False); print("Built model")
	dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
	with model.graph.as_default():
		sv = tf.train.Supervisor()
		with sv.managed_session() as sess:
			sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
			EM, F1 = 0.0, 0.0
			for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
				index, ground_truth, passage = sess.run([model.output_index, model.indices, model.passage_w])
				for batch in range(Params.batch_size):
					f1, em = f1_and_EM(index[batch], ground_truth[batch], passage[batch], dict_)
					F1 += f1
					EM += em
			F1 /= float(model.num_batch * Params.batch_size)
			EM /= float(model.num_batch * Params.batch_size)
			print("Exact_match: {}\nF1_score: {}".format(EM,F1))

def main():
	model = Model(is_training = True); print("Built model")
	dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))
	init = False
	devdata, dev_ind = get_dev()
	if not os.path.isfile(os.path.join(Params.logdir,"checkpoint")):
		init = True
		glove = np.memmap(Params.data_dir + "glove.np", dtype = np.float32, mode = "r")
		glove = np.reshape(glove,(Params.vocab_size,Params.emb_size))
	with model.graph.as_default():
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sv = tf.train.Supervisor(logdir=Params.logdir,
						save_model_secs=0,
						global_step = model.global_step,
						init_op = model.init_op)
		with sv.managed_session(config = config) as sess:
			if init: sess.run(model.emb_assign, {model.word_embeddings_placeholder:glove})
			for epoch in range(1, Params.num_epochs+1):
				if sv.should_stop(): break
				mean_loss = []
				for step in tqdm(range(model.num_batch), total = model.num_batch, ncols=70, leave=False, unit='b'):
					_, loss = sess.run([model.train_op, model.mean_loss])
					mean_loss.append(loss)
					if step % Params.save_steps == 0:
						gs = sess.run(model.global_step)
						sv.saver.save(sess, Params.logdir + '/model_epoch_%d_step_%d'%(gs//model.num_batch, gs%model.num_batch))
						sample = np.random.choice(dev_ind, Params.batch_size)
						feed_dict = {data: devdata[i][sample] for i,data in enumerate(model.data)}
						logits, dev_loss = sess.run([model.points_logits_stacked,
													# model.passage_char_encoded,
													# model.passage_encoding,
													# model.question_matching,
													# model.self_matching_output,
													# model.final_bidirectional_outputs,
													# model.points_logits,
													model.mean_loss], feed_dict = feed_dict)
						# print("")
						# print("loss: {}".format(np.sum(np.isnan(dev_loss))))
						# print("logits: {}".format(np.sum(np.isnan(logits))))
						# print("passage_char_encoded: {}".format(np.sum(np.isnan(a))))
						# print("passage_encoding: {}".format(np.sum(np.isnan(b))))
						# print("question_matching: {}".format(np.sum(np.isnan(c))))
						# print("self_matching_output: {}".format(np.sum(np.isnan(d))))
						# print("final_bidirectional_outputs: {}".format(np.sum(np.isnan(e))))
						# print("points_logits: {}".format(np.sum(np.isnan(f))))
						# exit()
						index = np.argmax(logits, axis = 2)
						F1, EM = 0.0, 0.0
						for batch in range(Params.batch_size):
							f1, em = f1_and_EM(index[batch], devdata[8][sample][batch], devdata[0][sample][batch], dict_)
							F1 += f1
							EM += em
						F1 /= float(Params.batch_size)
						EM /= float(Params.batch_size)
						sess.run(model.metric_assign,{model.F1_placeholder: F1, model.EM_placeholder: EM, model.dev_loss_placeholder: dev_loss})
						print("\nTrain_loss: {}\nDev_loss: {}\nDev_Exact_match: {}\nDev_F1_score: {}".format(np.mean(mean_loss), dev_loss,EM,F1))
						mean_loss = []

if __name__ == '__main__':
	if Params.mode.lower() == "debug":
		print("Debugging...")
		main()
	elif Params.mode.lower() == "test":
		print("Testing on dev set...")
		test()
	elif Params.mode.lower() == "train":
		print("Training...")
		main()
	else:
		print("Invalid mode.")
