#coding:utf-8
import os
import sys
import math
import random
import itertools
from six.moves import zip_longest
#import cPickle
import _pickle as cPickle

class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

    def _doc_len(self, idx):
        k = len(self.token_idxs)
        return k

    def _max_sent_len(self, idxs):
        k = max([len(sent) for sent in self.token_idxs])
        return k

class DataSet:
    def __init__(self, data, train):
        self.data_dict = sort_data(data, train)
        self.num_examples = sum([len(data) for data in self.data_dict.values()])
        
    def sort(self):
        random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x._max_sent_len)
        self.data = sorted(self.data, key=lambda x: x._doc_len)

    def get_by_idxs(self, idxs):
        return [self.data[idx] for idx in idxs]

    def get_batches(self, batch_size, num_epochs=None, rand = True):
        _batches = []
        for doc_l, data in self.data_dict.items():
            batches_doc_l = list(grouper(data, batch_size))
            _batches += batches_doc_l

        _batches = [tuple([instance for instance in batch if instance is not None]) for batch in _batches]            
        num_batches = len(_batches)
        if(rand):
            batches = random.sample(_batches, num_batches)
        else:
            batches = _batches

        batches_epochs = itertools.chain.from_iterable(batches for _ in range(num_epochs))
        num_steps = num_epochs*num_batches
        
        for i in range(num_steps):
            batch = next(batches_epochs)
            yield i, batch

def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
    args = [iter(iterable)] * n
    out = zip_longest(*args, fillvalue=fillvalue)
    out = list(out)
    if num_groups is not None:
        default = (fillvalue,) * n
        assert isinstance(num_groups, int)
        out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
    if shorten:
        assert fillvalue is None
        out = (tuple(e for e in each if e is not None) for each in out)
    return out

def sort_data(data, train):
    docl_dict = {}
    for d in data:
        doc_l = d.doc_l
        docl_dict.setdefault(doc_l, [])
        docl_dict[doc_l] += [d]

    if train:
        doc_l_dict = {doc_l: random.shuffle(data) for doc_l, data in docl_dict.items()}
    return docl_dict

def load_data(config):
    datapath = os.path.join(config.datadir, config.dataname)
    train, dev, test, embeddings, word_to_id = cPickle.load(open(datapath, 'rb'), encoding='latin1')
    trainset, devset, testset = DataSet(train, train=True), DataSet(dev, train=False), DataSet(test, train=False)
    print('Number of train examples: %i' % trainset.num_examples)
    vocab = dict([(v, k) for k,v in word_to_id.items()])
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    return train_batches, dev_batches, test_batches, embeddings, vocab, word_to_id
