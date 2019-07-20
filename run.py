#coding: utf-8

import tensorflow as tf
import numpy as np
import subprocess
import pdb
from collections import defaultdict
import itertools


from IPython.display import display, HTML, clear_output

from rouge import rouge_n, rouge_l_sentence_level

def get_txt_from_idx(idxs, model=None, config=None):
    if model is not None:
        return [' '.join([model.config.vocab[idx] for idx in idxs if (idx != model.config.EOS_IDX and idx != model.config.PAD_IDX)])]
    elif config is not None:
        return [' '.join([config.vocab[idx] for idx in idxs if (idx != config.EOS_IDX and idx != config.PAD_IDX)])]
    else:
        raise

def get_txt_from_tokens(tokens):
    return [' '.join([token for token in l]) for l in tokens]

def get_nevative_batches(batches, config):
    doc_l_instances = defaultdict(list)
    for idx, batch in batches:
        for instance in batch:
            if instance.goldLabel < 3:
                doc_l = instance.doc_l
                doc_l_instances[doc_l].append(instance)
        
    negative_batches = []
    negative_batch = []
    for instances in doc_l_instances.values():
        for instance in instances:
            negative_batch.append(instance)
            if len(negative_batch) >= config.test_batch_size:
                negative_batches.append(negative_batch)
                negative_batch = []
        if len(negative_batch) > 0:
            negative_batches.append(negative_batch)
            negative_batch = []
                
    return negative_batches

def get_neutral_nevative_batches(batches, config):
    doc_l_instances = defaultdict(list)
    for idx, batch in batches:
        for instance in batch:
            if instance.goldLabel <= 3:
                doc_l = instance.doc_l
                doc_l_instances[doc_l].append(instance)
        
    negative_batches = []
    negative_batch = []
    for instances in doc_l_instances.values():
        for instance in instances:
            negative_batch.append(instance)
            if len(negative_batch) >= config.test_batch_size:
                negative_batches.append(negative_batch)
                negative_batch = []
        if len(negative_batch) > 0:
            negative_batches.append(negative_batch)
            negative_batch = []
                
    return negative_batches


def print_sample(sess, batch, model, doc_idx=None, modes=[1, 2, 'l'], printflg=True, printbody=True):
    tokens_list = []
    def print_doc(r_d, o_d, printbody):
        r_tokens = get_txt_from_tokens([r_d.summary_tokens])
        
        if len(o_d.shape) == 2:
            root_s = o_d[0]
            main_d = o_d[1:]
            o_tokens = get_txt_from_idx(root_s, model)

            if printflg:
                rouge_batch = tuple([get_rouge(o_tokens, r_tokens, mode) for mode in modes])
                print 'ROUGE -1: %.3f, -2: %.3f, -L: %.3f' % rouge_batch
                print 'ROOT'
                print 'ref: ', r_tokens[0]
                print 'out: ', o_tokens[0]
                
                for sent_idx, (r_s, o_s) in enumerate(zip(r_d.token_idxs, main_d)):
                    if np.sum(r_s) == 0: break # all tokens are <pad>
                    print sent_idx + 1, get_txt_from_idx(r_s, model)[0]
                    if printbody: print 'out:' , get_txt_from_idx(o_s, model)[0]
                    
        elif len(o_d.shape) == 1:
            o_tokens = get_txt_from_idx(o_d, model)
            if printflg:
                rouge_batch = tuple([get_rouge(o_tokens, r_tokens, mode) for mode in modes])
                print 'ROUGE -1: %.3f, -2: %.3f, -L: %.3f' % rouge_batch
                print 'ref: ', r_tokens
                print 'out: ', o_tokens
        return r_tokens, o_tokens
    
    feed_dict = model.get_feed_dict(batch, mode='test')
    _output_token_idxs = sess.run(model.beam_output_token_idxs, feed_dict = feed_dict)
    
    for batch_idx, (r_d, o_d) in enumerate(zip(batch, _output_token_idxs)):
        if doc_idx is not None:
            if doc_idx != batch_idx: continue
        if printflg: 
            if 'goldLabel' in r_d.__dict__.keys():
                print 'Doc: {}, ID: {}, GoldLabel: {}, ASIN: {}'.format(batch_idx, r_d.idx, r_d.goldLabel, r_d.asin)
            else:
                print 'Doc: {}, ID: {}'.format(batch_idx, r_d.idx)
        r_tokens, o_tokens = print_doc(r_d, o_d, printbody)
        tokens_list.append((r_tokens, o_tokens))
    return tokens_list

def get_rouge(o_tokens, r_tokens, mode):
    if o_tokens == '': return (0.0, 0.0, 0.0)
    rouge = rouge_l_sentence_level(o_tokens, r_tokens) if mode == 'l' else rouge_n(o_tokens, r_tokens, mode)
    return rouge[0]

def get_rouges(sess, model, batch, modes=[1, 2, 'l']):
    feed_dict = model.get_feed_dict(batch, mode='test')
    _output_token_idxs = sess.run(model.beam_output_token_idxs, feed_dict = feed_dict)
    rouges = []
    for r_d, o_d in zip(batch, _output_token_idxs):
        o_idxs = o_d[0] if len(o_d.shape) == 2 else o_d
        o_tokens = get_txt_from_idx(o_idxs, model)
        r_tokens = get_txt_from_tokens([r_d.summary_tokens])
        rouge_batch = tuple([get_rouge(o_tokens, r_tokens, mode) for mode in modes])
        rouges.append(rouge_batch)
    return rouges

def get_rouges_tokens(sess, model, batches, modes=[1, 2, 'l']):
    rouge_list, o_token_list = [], []
    for i, batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        _output_token_idxs = sess.run(model.beam_output_token_idxs, feed_dict = feed_dict)

        for r_d, o_d in zip(batch, _output_token_idxs):
            o_idxs = o_d[0] if len(o_d.shape) == 2 else o_d
            o_tokens = get_txt_from_idx(o_idxs, model)
            r_tokens = get_txt_from_tokens([r_d.summary_tokens])
            rouges = tuple([get_rouge(o_tokens, r_tokens, mode) for mode in modes])
            rouge_list.append(rouges)
            o_token_list.append(o_tokens)
    return rouge_list, o_token_list

def evaluate_all(sess, batches, model):
    losses, rouges = [], []
    for ct, batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        loss_batch = sess.run(model.loss, feed_dict = feed_dict)
        rouge_batch = get_rouges(sess, model, batch)
        losses += [loss_batch]
        rouges += rouge_batch
    return losses, rouges

def evaluate(sess, batches, model):
    losses, rouges = evaluate_all(sess, batches, model)
    loss_mean = np.mean(losses)
    rouge_mean = tuple(np.mean(rouges, 0))
    return loss_mean, rouge_mean

def evaluate_doc_l(sess, batches, model, redu=False, r_hist=5, epochs=1):
    doc_l_rouges = defaultdict(list)
    for epoch in range(epochs):
        for ct, batch in batches:
            feed_dict = model.get_feed_dict(batch, mode='test')
            rouge_batch = get_rouges(sess, model, batch)
            if redu:
                doc_l_batch = [instance.ex_doc_l for instance in batch]
            else:
                doc_l_batch = [instance.doc_l for instance in batch]
            for doc_l, rouge in zip(doc_l_batch, rouge_batch):
                doc_l_rouges[doc_l].append(rouge)
    return doc_l_rouges

def get_rouges_base(config, embs, instances, base_func, modes=[1, 2, 'l']):
    rouges_list = []
    o_idxs_instances = base_func(config, embs, instances)
    for instance, o_idxs in zip(instances, o_idxs_instances):
        o_tokens = get_txt_from_idx(o_idxs, config=config)
        r_tokens = get_txt_from_tokens([instance.summary_tokens])
        rouges = tuple([get_rouge(o_tokens, r_tokens, mode) for mode in modes])
        rouges_list.append(rouges)
    return rouges_list

def evaluate_doc_l_base(config, embs, batches, base_func):
    doc_l_rouges = defaultdict(list)
    instances = list(itertools.chain.from_iterable([batch for _, batch in batches]))
    rouge_instances = get_rouges_base(config, embs, instances, base_func)
    doc_l_instances = [instance.doc_l for instance in instances]
    for doc_l, rouges in zip(doc_l_instances, rouge_instances):
        doc_l_rouges[doc_l].append(rouges)
    return doc_l_rouges

def evaluate_label(sess, batches, model, epochs=1):
    rouges = []
    label_rouges = defaultdict(list)
    for epoch in range(epochs):
        for ct, batch in batches:
            feed_dict = model.get_feed_dict(batch, mode='test')
            rouge_batch = get_rouges(sess, model, batch)
            label_batch = [instance.goldLabel for instance in batch]
            for label, rouge in zip(label_batch, rouge_batch):
                label_rouges[label].append(rouge)
    label_rouge = {label: np.mean(rouges, 0) for label, rouges in label_rouges.items()}
    
    return label_rouge

def run(sess, model, train_batches, dev_batches, test_batches, sample_batch, loss_log, rouge_log, stop_ct=None):
    config = model.config
    losses_train = []
    saver = tf.train.Saver(max_to_keep=20)
    save_ind = config.save_ct/config.log_period
    if len(loss_log) == 0:
        cmd_rm = 'rm -r %s' % config.writedir
        res = subprocess.call(cmd_rm.split())

        cmd_mk = 'mkdir %s' % config.writedir
        res = subprocess.call(cmd_mk.split())

    for ct, batch in train_batches:
        feed_dict = model.get_feed_dict(batch)
        _, loss_train = sess.run([model.opt, model.loss], feed_dict = feed_dict)
        losses_train += [loss_train]
        if(ct%config.log_period==0):
            loss_train = np.mean(losses_train)
            loss_dev, rouge_dev = evaluate(sess, dev_batches, model)
            
            if config.dev_metric == 'loss':
                loss_min = min(zip(*loss_log)[2]) if len(loss_log) > 0 else np.inf
                do_test = (loss_min >= loss_dev)
            elif config.dev_metric == 'rouge':
                if len(rouge_log) == 0:
                    do_test = True
                elif ct < config.save_ct:
                    do_test = False
                elif len(rouge_log) <= save_ind:
                    do_test = True
                else:
                    norm = np.mean(np.array(zip(*rouge_log))[:3], 1)
                    if 0.0 in norm: norm = np.array([1, 1, 1], dtype=np.float32)
                    rouge_judge = np.sum(np.array(rouge_dev)/norm)
                    rouge_max = np.max(np.sum(np.array(zip(*rouge_log))[:3]/norm[:, np.newaxis], 0)[save_ind:])
                    do_test = (rouge_max <= rouge_judge)
            else:
                raise
                
            if do_test:
                loss_test, rouge_test = evaluate(sess, test_batches, model)    
                saver.save(sess, config.modelpath, global_step=ct)
            else:
                loss_test = zip(*loss_log)[3][-1]
                rouge_test = tuple(np.array(zip(*rouge_log))[3:, -1])

            loss_log += [(ct, loss_train, loss_dev, loss_test)]
            rouge_log += [rouge_dev + rouge_test]
            losses_train = []

            clear_output()
            for i in range(len(loss_log)): 
                print 'Step: %i | LOSS TRAIN: %.3f, DEV: %.3f, TEST: %.3f ' %  loss_log[i], 
                print '| DEV ROUGE-1: %.3f, -2: %.3f, -L: %.3f | TEST ROUGE: -1: %.3f, -2: %.3f, -L: %.3f' % rouge_log[i]
            print_sample(sess, sample_batch, model)

        if stop_ct is not None:
            if ct == stop_ct: break 
            
def print_log(sess, model, sample_batch, loss_log, rouge_log):
    for i in range(len(loss_log)): 
        print 'Step: %i | LOSS TRAIN: %.3f, DEV: %.3f, TEST: %.3f ' %  loss_log[i], 
        print '| DEV ROUGE-1: %.3f, -2: %.3f, -L: %.3f | TEST ROUGE: -1: %.3f, -2: %.3f, -L: %.3f' % rouge_log[i]
    print_sample(sess, sample_batch, model)