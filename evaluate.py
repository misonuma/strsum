import os
import codecs
import itertools
import logging

import tensorflow as tf
import pyrouge

from model import StrSumModel

def get_txt_from_idx(idxs, model, vocab):
    tokens = []
    for idx in idxs:
        if idx == model.config.EOS_IDX: break
        tokens.append(vocab[idx])
    sent = [' '.join(tokens)]
    return sent

def write_files(write_dir, sents_dict):
    for idx, sents in sents_dict.items():
        file_path = os.path.join(write_dir, "%04d.txt" % idx)

        f = codecs.open(file_path, mode="w", encoding="utf-8")
        for i, sent in enumerate(sents):
            f.write(sent) if i==len(sents)-1 else f.write(sent+"\n")

        f.close()
        
def write_ref(batches, config):
    instances = list(itertools.chain.from_iterable([batch for _, batch in batches]))
    ref_sents_dict = {}
    for ct, batch in batches:
        for instance in batch:
            ref_sents = [' '.join(instance.summary_tokens)]
            ref_sents_dict[instance.idx] = ref_sents

    write_files(config.refdir, ref_sents_dict)
    
    
def write_out(batches, config, vocab):
    model = StrSumModel(config)
    model.build()

    ckpt = tf.train.get_checkpoint_state(config.modeldir)
    model_path = ckpt.all_model_checkpoint_paths[-1]
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        instances = list(itertools.chain.from_iterable([batch for _, batch in batches]))
        out_sents_dict = {}
        for ct, batch in batches:
            feed_dict = model.get_feed_dict(batch, mode='test')
            batch_root_token_idxs = sess.run(model.root_token_idxs, feed_dict = feed_dict)
            for root_token_idxs, instance in zip(batch_root_token_idxs, batch):
                idx = instance.idx
                out_sents = get_txt_from_idx(root_token_idxs, model, vocab)
                out_sents_dict[idx] = out_sents

        write_files(config.outdir, out_sents_dict)
        
def print_pyrouge(config):
    logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging

    r = pyrouge.Rouge155()
    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '#ID#.txt'
    
    r.system_dir = config.outdir
    r.model_dir = config.refdir

    rouge_results = r.convert_and_evaluate()
    rouge_dict = r.output_to_dict(rouge_results)
    
    print(rouge_results)
    
def evaluate(config, test_batches, vocab):
    print('writing reference summaries...')
    write_ref(test_batches, config)
    print('writing system summaries...')
    write_out(test_batches, config, vocab)
    
    print_pyrouge(config)