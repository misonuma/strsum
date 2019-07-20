#coding: utf-8

import os
import sys
import pdb

import numpy as np
import cPickle

import tensorflow as tf

from utils import load_data
from model import StrSumModel
from run import run

PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences

def run(config):
    num_examples,  train_batches, dev_batches, test_batches, embedding_matrix, vocab, word_to_id = load_data(config)
    
    config.n_embed, config.d_embed = embedding_matrix.shape
    config.maximum_iterations = max([max([d._max_sent_len(None) for d in batch]) for ct, batch in dev_batches])
    config.PAD_IDX, config.UNK_IDX, config.BOS_IDX, config.EOS_IDX = word_to_id[config.PAD], word_to_id[config.UNK], word_to_id[config.BOS], word_to_id[config.EOS]
    config.vocab = vocab
    
#     model = StrSumModel(config)
#     model.build()
    
#     sess = tf.Session()
#     gvi = tf.global_variables_initializer()
#     sess.run(gvi)
#     sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32)))
    
#     sample_batch = test_batches[0][1]
#     loss_log = []
#     rouge_log = []
#     run(sess, model, train_batches, dev_batches, test_batches, sample_batch, loss_log, rouge_log, stop_ct=90000)