#coding:utf-8
import math
import random
import itertools
from six.moves import zip_longest
import cPickle
from rouge import rouge_n, rouge_l_sentence_level

def get_txt_from_idx(idxs, model, vocab):
    return [' '.join([vocab[idx] for idx in idxs if (idx != model.config.EOS_IDX and idx != model.config.PAD_IDX)])]

def get_txt_from_tokens(tokens):
    return [' '.join([token for token in l]) for l in tokens]

def get_rouge(o_tokens, r_tokens, mode):
    if o_tokens == '': return (0.0, 0.0, 0.0)
    rouge = rouge_l_sentence_level(o_tokens, r_tokens) if mode == 'l' else rouge_n(o_tokens, r_tokens, mode)
    return rouge[0]

def get_rouges(sess, model, batch, vocab, modes=[1, 2, 'l']):
    feed_dict = model.get_feed_dict(batch, mode='test')
    _output_token_idxs = sess.run(model.beam_output_token_idxs, feed_dict = feed_dict)
    rouges = []
    for r_d, o_d in zip(batch, _output_token_idxs):
        o_idxs = o_d[0] if len(o_d.shape) == 2 else o_d
        o_tokens = get_txt_from_idx(o_idxs, model, vocab)
        r_tokens = get_txt_from_tokens([r_d.summary_tokens])
        rouge_batch = tuple([get_rouge(o_tokens, r_tokens, mode) for mode in modes])
        rouges.append(rouge_batch)
    return rouges