#coding: utf-8

import os
import sys
import subprocess
import logging

import numpy as np
import tensorflow as tf

from model import StrSumModel
from rouge import rouge_n, rouge_l_sentence_level

PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences

def get_txt_from_idx(idxs, model, vocab):
    tokens = []
    for idx in idxs:
        if idx == model.config.EOS_IDX: break
        tokens.append(vocab[idx])
    sent = [' '.join(tokens)]
    return sent

def get_txt_from_tokens(tokens):
    return [' '.join([token for token in l]) for l in tokens]

def get_rouges(sess, model, batch, vocab, modes=[1, 2, 'l']):
    feed_dict = model.get_feed_dict(batch, mode='test')
    batch_root_token_idxs = sess.run(model.root_token_idxs, feed_dict = feed_dict)
    rouges = []
    for instance, root_token_idxs in zip(batch, batch_root_token_idxs):
        out_tokens = get_txt_from_idx(root_token_idxs, model, vocab)
        ref_tokens = get_txt_from_tokens([instance.summary_tokens])
        
        rouge_1_f1 = rouge_n(out_tokens, ref_tokens, 1)[0]
        rouge_2_f1 = rouge_n(out_tokens, ref_tokens, 2)[0]
        rouge_l_f1 = rouge_l_sentence_level(out_tokens, ref_tokens)[0]
        
        rouge_batch = [rouge_1_f1, rouge_2_f1, rouge_l_f1]
        rouges.append(rouge_batch)
    return rouges

def evaluate(sess, batches, model, vocab):
    losses, rouges = [], []
    for ct, batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        loss_batch = sess.run(model.loss, feed_dict = feed_dict)
        rouge_batch = get_rouges(sess, model, batch, vocab)
        losses += [loss_batch]
        rouges += rouge_batch
        
    loss_mean = np.mean(losses)
    rouge_mean = tuple(np.mean(rouges, 0))
    return loss_mean, rouge_mean

def train(config, train_batches, dev_batches, test_batches, embedding_matrix, vocab):
    # clean and create model directory
    cmd_rm = 'rm -r %s' % config.modeldir
    res = subprocess.call(cmd_rm.split())

    cmd_mk = 'mkdir %s' % config.modeldir
    res = subprocess.call(cmd_mk.split())
    
    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logpath = os.path.join(config.modeldir, config.modelname+'.log')
    ah = logging.FileHandler(logpath)
    ah.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ah.setFormatter(formatter)
    logger.addHandler(ah)
    
    logger.critical(str(config.flag_values_dict()))
    
    # build model
    model = StrSumModel(config)
    model.build()
    
    # initialize model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32)))
    saver = tf.train.Saver(max_to_keep=5)
    
    # train model
    print('training starts...')
    losses_train = []
    rouge_l_max = 0
    for ct, batch in train_batches:
        feed_dict = model.get_feed_dict(batch)
        _, loss_batch = sess.run([model.opt, model.loss], feed_dict = feed_dict)
        losses_train += [loss_batch]
        if ct%config.log_period==0:
            loss_train = np.mean(losses_train)
            loss_dev, rouge_dev = evaluate(sess, dev_batches, model, vocab)
            rouge_l = rouge_dev[-1]

            if rouge_l >= rouge_l_max:
                rouge_l_max = rouge_l
                loss_test, rouge_test = evaluate(sess, test_batches, model, vocab)
                modelpath = os.path.join(config.modeldir, config.modelname)
                saver.save(sess, modelpath, global_step=ct)
 
            print('Step: %i | LOSS TRAIN: %.3f, DEV: %.3f, TEST: %.3f | DEV ROUGE-1: %.3f, -2: %.3f, -L: %.3f | TEST ROUGE: -1: %.3f, -2: %.3f, -L: %.3f' %  ((ct, loss_train, loss_dev, loss_test) + rouge_dev + rouge_test))
            
            logger.debug('Step: %i | LOSS TRAIN: %.3f, DEV: %.3f, TEST: %.3f | DEV ROUGE-1: %.3f, -2: %.3f, -L: %.3f | TEST ROUGE: -1: %.3f, -2: %.3f, -L: %.3f' %  ((ct, loss_train, loss_dev, loss_test) + rouge_dev + rouge_test))
            logger.handlers[0].flush()