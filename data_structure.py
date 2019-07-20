import gensim
import numpy as np
import re
import random
import math
import unicodedata
import itertools
from utils import grouper
import pdb 

def strip_accents(s):
     return ''.join(c for c in unicodedata.normalize('NFD', unicode(s,'utf-8'))
                  if unicodedata.category(c) != 'Mn')

class RawData:
    def __init__(self):
        self.userStr = ''
        self.productStr = ''
        self.reviewText = ''
        self.goldRating = -1
        self.predictedRating = -1
        self.userStr = ''

class DataSet:
    def __init__(self, data):
        self.data = data
        self.num_examples = len(self.data)
        self.idx_ind = self.get_idx_ind()
        self.embeddings_root = None
        
    def get_idx_ind(self):
        idx_ind = {}
        for ind, instance in enumerate(self.data):
            idx_ind[instance.idx] = ind
        return idx_ind

    def sort(self):
        random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x._max_sent_len)
        self.data = sorted(self.data, key=lambda x: x._doc_len)

    def get_by_idxs(self, idxs):
        return [self.data[idx] for idx in idxs]

    def get_batches(self, batch_size, num_epochs=None, rand = True):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        idxs = list(range(self.num_examples))
        _grouped = lambda: list(grouper(idxs, batch_size))

        if(rand):
            grouped = lambda: random.sample(_grouped(), num_batches_per_epoch)
        else:
            grouped = _grouped
        num_steps = num_epochs*num_batches_per_epoch
        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for i in range(num_steps):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            yield i, batch_data

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

class Corpus:
    def __init__(self):
        self.doclst = {}

    def load(self, in_path, name):
        self.doclst[name] = []
        for line in open(in_path):
            items = line.split('<split1>')
            doc = RawData()
            doc.goldRating = int(items[0])
            doc.reviewText = items[1]
            self.doclst[name].append(doc)
            
    def preprocess(self):
        random.shuffle(self.doclst['train'])
        for dataset in self.doclst:
            for doc in self.doclst[dataset]:
                doc.sent_lst = doc.reviewText.split('<split2>')
                doc.sent_lst = [re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sent) for sent in doc.sent_lst]
                doc.sent_token_lst = [sent.split() for sent in doc.sent_lst]
                doc.sent_token_lst = [sent_tokens for sent_tokens in doc.sent_token_lst if(len(sent_tokens)!=0)]
            self.doclst[dataset] = [doc for doc in self.doclst[dataset] if len(doc.sent_token_lst)!=0]

    def w2v(self, options):
        sentences = []
        for doc in self.doclst['train']:
            sentences.extend(doc.sent_token_lst)
        if('dev' in self.doclst):
            for doc in self.doclst['dev']:
                sentences.extend(doc.sent_token_lst)
        print(sentences[0])
        if(options['skip_gram']):
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4, sg=1)
        else:
            self.w2v_model = gensim.models.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=5, workers=4)
        self.w2v_model.scan_vocab(sentences)  # initial survey
        rtn = self.w2v_model.scale_vocab(dry_run = True)  # trim by min_count & precalculate downsampling
        print(rtn)
        self.w2v_model.finalize_vocab()  # build tables & arrays
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        self.vocab = self.w2v_model.wv.vocab
        print('Vocab size: {}'.format(len(self.vocab)))

        # model.save('../data/w2v.data')