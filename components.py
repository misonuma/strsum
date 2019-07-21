#coding:utf-8

import tensorflow as tf
import numpy as np

def dynamic_bi_rnn(inputs, seqlen, n_hidden, keep_prob, cell_name='', reuse=False):
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = keep_prob)
        fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name + 'bw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = keep_prob)
        bw_initial_state = bw_cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope(cell_name, reuse=reuse):
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 sequence_length=seqlen)
    return outputs, output_states    

def get_matrix_tree(r, A):
    L = tf.reduce_sum(A, 1)
    L = tf.matrix_diag(L)
    L = L - A

    r_diag = tf.matrix_diag(r)
    LL = L + r_diag

    LL_inv = tf.matrix_inverse(LL)  #batch_l, doc_l, doc_l
    LL_inv_diag_ = tf.matrix_diag_part(LL_inv)

    d0 = tf.multiply(r, LL_inv_diag_)

    LL_inv_diag = tf.expand_dims(LL_inv_diag_, 2)

    tmp1 = tf.multiply(A, tf.matrix_transpose(LL_inv_diag))
    tmp2 = tf.multiply(A, tf.matrix_transpose(LL_inv))

    d = tmp1 - tmp2
    d = tf.concat([tf.expand_dims(d0,[1]), d], 1)
    return d

def discourse_rank(str_scores, damp):
    batch_l = tf.shape(str_scores)[0]
    max_doc_l = tf.shape(str_scores)[2]
    
    str_scores_root = str_scores[:, 0, :]
    str_scores_words = str_scores[:, 1:, :]

    str_root = tf.expand_dims(tf.one_hot(indices=0, depth=(max_doc_l+1), 
                on_value=0.0, off_value=1.0/tf.cast(max_doc_l, tf.float32), dtype=tf.float32), 0)
    str_roots = tf.expand_dims(tf.tile(str_root, [batch_l, 1]), -1)

    adj_scores = tf.concat([str_roots, str_scores], 2)

    eye = tf.expand_dims(tf.diag(tf.ones(max_doc_l+1)), 0)
    eye_words = tf.tile(eye, [batch_l, 1, 1])

    eig_matrix_inv = tf.matrix_inverse(eye_words - damp*adj_scores)
    damp_vec = tf.ones([batch_l, (max_doc_l+1), 1]) / tf.cast((max_doc_l+1), tf.float32)

    eig_scores_ = tf.multiply((1-damp), tf.matmul(eig_matrix_inv, damp_vec))
    eig_scores_ = tf.squeeze(eig_scores_, 2)[:, 1:]
    eig_scores = tf.nn.softmax(eig_scores_, 1)

    eig_str_scores_root = tf.expand_dims(tf.multiply(eig_scores, str_scores_root), 1)
    eig_str_scores = tf.concat([eig_str_scores_root, str_scores_words], 1)

    return eig_str_scores
