#coding: utf-8

import os
import sys
import pdb

import numpy as np
import cPickle

import tensorflow as tf

from model import StrSumModel

PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences

def train(config, train_batches, dev_batches, embedding_matrix, vocab):
    
    model = StrSumModel(config)
    model.build()
    
    sess = tf.Session()
    gvi = tf.global_variables_initializer()
    sess.run(gvi)
    sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32)))