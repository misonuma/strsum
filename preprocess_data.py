import os
import numpy as np
import pandas as pd

import _pickle as cPickle
from collections import OrderedDict, defaultdict

import tensorflow as tf

from data_structure import Instance

flags = tf.app.flags

flags.DEFINE_string('input_path', 'data/sports_df.pkl', 'path of output data')
flags.DEFINE_string('output_path', 'data/sports.pkl', 'path of input data')
flags.DEFINE_string('word_vec_path', 'data/crawl-300d-2M.vec', 'path of pretrained word vec')

flags.DEFINE_integer('n_vocab', 50000, 'size of vocab')

# special tokens
PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences

def get_word_list(tokens_list):
    # create vocab of words
    word_dict = defaultdict(int)
    word_dict[BOS] = np.inf
    word_dict[EOS] = np.inf
    word_dict['.'] = np.inf
    for tokens in tokens_list:
        for word in tokens:
            word_dict[word] += 1
    word_dict = sorted(word_dict.items(), key=lambda x: x[1])[::-1]
    return [w for w, cnt in word_dict]

def get_fasttext(word_vec_path):
    # create pretrained word_vec 
    word_vec = {}
    header = True
    with open(word_vec_path) as f:
        for line in f:
            if header:
                header = False
                continue
            word, vec = line.split(' ', 1)
            word_vec[word] = np.array(list(vec.split())).astype(np.float32)
    return word_vec

def get_word_vec(word_list, fasttext_vec):
    word_vec = []
    for word in word_list:
        try:
            vec = fasttext_vec[word]
            word_vec.append((word, vec))
        except:
            continue
    return OrderedDict(word_vec)

def get_vocab_emb(word_vec, word_emb_dim, n_vocab=50000):
    # build vocab and embedding matrix
    word_vec_list = list(word_vec.items())
    word_vec_list.insert(0, (UNK, np.zeros([word_emb_dim], dtype=np.float32)))
    word_vec_list.insert(0, (PAD, np.zeros([word_emb_dim], dtype=np.float32)))
    
    word_vec_list = word_vec_list[:n_vocab]
    vocab = {word: i for i, (word, vec) in enumerate(word_vec_list)}
    embeddings = np.array([vec for word, vec in word_vec_list]).astype(np.float32)
    
    assert len(vocab) == len(embeddings)
    
    return vocab, embeddings

def prepare_instancelst(data_df, vocab):
    def to_line_idxs(token_idxs, vocab):
        tokens_bos_eos = [token + [vocab['.']] for token in token_idxs]
        line_idxs = [token for tokens_line in tokens_bos_eos for token in tokens_line]
        return line_idxs
    
    instancelst = []
    for i_doc, doc in data_df.iterrows():
        instance = Instance()
        instance.idx = i_doc
        instance.asin = doc.asin
        doc_token_idxs = []
        for i, sent_tokens in enumerate(doc.tokens):
            sent_token_idxs = []
            for token in sent_tokens:
                if(token in vocab):
                    sent_token_idxs.append(vocab[token])
                else:
                    sent_token_idxs.append(vocab[UNK])
            doc_token_idxs.append(sent_token_idxs)
        instance.token_idxs = doc_token_idxs
        instance.line_idxs = to_line_idxs(doc_token_idxs, vocab)
        instance.goldLabel = doc.overall
        instance.summary = doc.summary
        instance.summary_tokens = doc.summary_tokens
        instance.summary_idxs = [vocab[token] if token in vocab else vocab[UNK] for token in instance.summary_tokens]
        instance.doc_l = doc.doc_l
        instance.max_sent_l = doc.max_sent_l
        instancelst.append(instance)
    return instancelst

def main():
    config = flags.FLAGS
    print(str(config.flag_values_dict()))

    print('loading input data...')
    train_df, dev_df, test_df = cPickle.load(open(config.input_path, 'rb'))
    
    tokens = []
    for doc in train_df.tokens:
        tokens.extend(doc)
    for doc in dev_df.tokens:
        tokens.extend(doc)
    
    print('loading pretrained word vectors...')
    word_list = get_word_list(tokens)
    fasttext_vec = get_fasttext(config.word_vec_path)
    word_vec = get_word_vec(word_list, fasttext_vec)
    
    word_emb_dim = list(fasttext_vec.values())[0].shape[0]
    vocab, embeddings = get_vocab_emb(word_vec, word_emb_dim, n_vocab=config.n_vocab)    
    
    print('building instances...')
    instances_train = prepare_instancelst(train_df, vocab)
    instances_dev = prepare_instancelst(dev_df, vocab)
    instances_test = prepare_instancelst(test_df, vocab)
    
    print('saving preprocessed data...')
    cPickle.dump((instances_train, instances_dev, instances_test, embeddings, vocab),open(config.output_path,'wb'))
        
if __name__ == "__main__":
    main()
    
    
    