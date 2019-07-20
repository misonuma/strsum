#coding:utf-8

import os
import numpy as np
import pdb

class Config:
    def __init__(self, flags):
        self.PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        self.UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
        self.BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
        self.EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences

        