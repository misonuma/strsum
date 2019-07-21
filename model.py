#coding:utf-8

import tensorflow as tf
import numpy as np

from components import dynamic_bi_rnn, get_matrix_tree, discourse_rank

class StrSumModel():
    def __init__(self, config):
        tf.reset_default_graph()
        
        t_variables = {}
        t_variables['keep_prob'] = tf.placeholder(tf.float32)
        t_variables['keep_prob_input'] = tf.placeholder(tf.float32)
        t_variables['batch_l'] = tf.placeholder(tf.int32, [])
        t_variables['token_idxs'] = tf.placeholder(tf.int32, [None, None, None])
        t_variables['dec_input_idxs'] = tf.placeholder(tf.int32, [None, None, None])
        t_variables['dec_target_idxs'] = tf.placeholder(tf.int32, [None, None, None])
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None, None])
        t_variables['dec_sent_l'] = tf.placeholder(tf.int32, [None, None])
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None])
        t_variables['max_sent_l'] = tf.placeholder(tf.int32, [])
        t_variables['max_doc_l'] = tf.placeholder(tf.int32, [])
        
        self.t_variables = t_variables
        self.config = config
        
    def build(self):
        dim_hidden = self.config.dim_hidden
        dim_bi_hidden = dim_hidden*2
        dim_str = self.config.dim_str
        dim_sent = self.config.dim_sent
        
        # define variables        
        with tf.variable_scope("self.embeddings", reuse=tf.AUTO_REUSE):
            self.embeddings = tf.get_variable("emb", [self.config.n_embed, self.config.d_embed], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            
        with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
            w_comb = tf.get_variable("w_comb", [dim_sent, dim_sent], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_comb = tf.get_variable("bias_comb", [dim_sent], dtype=tf.float32, initializer=tf.constant_initializer())

        with tf.variable_scope("Structure", reuse=tf.AUTO_REUSE):
            w_parser_p = tf.get_variable("w_parser_p", [dim_str, dim_str], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_parser_p = tf.get_variable("bias_parser_p", [dim_str], dtype=tf.float32, initializer=tf.constant_initializer())

            w_parser_c = tf.get_variable("w_parser_c", [dim_str, dim_str], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_parser_c = tf.get_variable("bias_parser_c", [dim_str], dtype=tf.float32, initializer=tf.constant_initializer())

            w_parser_s = tf.get_variable("w_parser_s", [dim_str, dim_str], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            w_parser_root = tf.get_variable("w_parser_root", [dim_str, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        # input
        batch_l = self.t_variables['batch_l']
        max_doc_l = self.t_variables['max_doc_l']
        max_sent_l = self.t_variables['max_sent_l']
        token_idxs = self.t_variables['token_idxs'][:, :max_doc_l, :max_sent_l]

        # get word embedding
        tokens_input_ = tf.nn.embedding_lookup(self.embeddings, token_idxs)
        tokens_input = tf.nn.dropout(tokens_input_, self.t_variables['keep_prob'])
        tokens_input_do = tf.reshape(tokens_input, [batch_l * max_doc_l, max_sent_l, self.config.d_embed])
                
        # get sentence embedding
        sent_l = self.t_variables['sent_l']
        sent_l_do = tf.reshape(sent_l, [batch_l * max_doc_l])
        
        tokens_outputs, _ = dynamic_bi_rnn(tokens_input_do, sent_l_do, dim_hidden, self.t_variables['keep_prob'], cell_name='Model/sent')
        tokens_output_do_ = tf.concat(tokens_outputs, 2)
        mask_tokens_do = tf.sequence_mask(sent_l_do, maxlen=max_sent_l, dtype=tf.float32)
        tokens_output_do = tokens_output_do_ + tf.expand_dims((mask_tokens_do-1)*999,2)

        doc_l = self.t_variables['doc_l']
        mask_sents = tf.sequence_mask(doc_l, maxlen=max_doc_l, dtype=tf.float32)
        mask_sents_do = tf.reshape(mask_sents, [batch_l * max_doc_l, 1])

        sents_input_con_do = tf.reduce_max(tokens_output_do, 1) * mask_sents_do
        sents_input_con = tf.reshape(sents_input_con_do, [batch_l, max_doc_l, dim_bi_hidden])
        
        # devide sents_input_con to sents_input_str and sents_input
        sents_input_str_ = sents_input_con[:, :, :dim_str]
        sents_input_str = tf.nn.dropout(sents_input_str_, self.t_variables['keep_prob'])
        
        sents_input_ = sents_input_con[:, :, dim_str:]
        sents_input = tf.nn.dropout(sents_input_, self.t_variables['keep_prob'])
    
        # get document structure
        parent = tf.tanh(tf.tensordot(sents_input_str, w_parser_p, [[2], [0]]) + b_parser_p)
        child = tf.tanh(tf.tensordot(sents_input_str, w_parser_c, [[2], [0]]) + b_parser_c)

        raw_scores_words_tmp = tf.tanh(tf.matmul(tf.tensordot(parent, w_parser_s, [[-1],[0]]),tf.matrix_transpose(child)))
        raw_scores_words_ = tf.exp(raw_scores_words_tmp)
        diag_zero = tf.zeros_like(raw_scores_words_[:,:,0])
        raw_scores_words = tf.matrix_set_diag(raw_scores_words_, diag_zero)

        raw_scores_root_ = tf.squeeze(tf.tensordot(sents_input_str, w_parser_root, [[2], [0]]) , [2])
        raw_scores_root = tf.exp(tf.tanh(raw_scores_root_))
        
        str_scores_ = get_matrix_tree(raw_scores_root, raw_scores_words)
        str_scores = tf.multiply(str_scores_, tf.expand_dims(mask_sents, 1))
        self.str_scores = str_scores
        
        # update str_scores with discourse_rank
        if self.config.discourserank:
            str_scores = discourse_rank(str_scores, self.config.damp)
        
        str_scores_sum = tf.expand_dims(tf.reduce_sum(str_scores, 2), 2)
        str_scores_norm = str_scores/str_scores_sum

        # get structured sentence embedding
        str_output_root = tf.matmul(str_scores_norm, sents_input)
        sents_output_root = tf.tanh(tf.tensordot(str_output_root, w_comb, [[2], [0]]) + b_comb)
        
        sents_output_ = sents_output_root[:, 1:, :]
        sents_output = tf.nn.dropout(tf.multiply(sents_output_,tf.expand_dims(mask_sents, 2)), self.t_variables['keep_prob'])
        sents_output_do = tf.reshape(sents_output, [batch_l*max_doc_l, dim_sent])
        
        # prepare for decoding
        dec_input_idxs = self.t_variables['dec_input_idxs']
        dec_input_idxs_do = tf.reshape(dec_input_idxs, [batch_l * max_doc_l, max_sent_l+1])
        dec_input_do = tf.nn.embedding_lookup(self.embeddings, dec_input_idxs_do)
        
        # decode for training
        dec_sent_l = self.t_variables['dec_sent_l']
        dec_sent_l_do = tf.reshape(dec_sent_l, [batch_l * max_doc_l])

        with tf.variable_scope('Model/sent/decoder', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=tf.AUTO_REUSE):
            dec_cell = tf.contrib.rnn.GRUCell(dim_sent)
            dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob = self.t_variables['keep_prob'])
            
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_input_do, sequence_length=dec_sent_l_do)
            
            decoder_initial_state = sents_output_do
            decoder_train = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=helper,
                initial_state=decoder_initial_state)

            decoder_outputs, _, output_sent_l = tf.contrib.seq2seq.dynamic_decode(decoder_train)
            
            self.output_sent_l = output_sent_l
            output_layer = tf.layers.Dense(self.config.n_embed, use_bias=False, name="output_projection")

            logits = output_layer(decoder_outputs.rnn_output)
            self.logits = logits

        # target and mask
        dec_target_idxs = self.t_variables['dec_target_idxs']
        dec_target_idxs_do = tf.reshape(dec_target_idxs, [batch_l * max_doc_l, max_sent_l+1])                
        dec_mask_tokens_do = tf.sequence_mask(dec_sent_l_do, maxlen=max_sent_l+1, dtype=tf.float32)
        
        # define loss
        loss = tf.contrib.seq2seq.sequence_loss(logits, dec_target_idxs_do, dec_mask_tokens_do)
            
        model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Model')
        str_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Structure')
        norm = 0.0
        for p in model_params + str_params:
            if ('bias' not in p.name): norm += self.config.norm * tf.nn.l2_loss(p)
        loss += norm
        
        # define optimizer
        if (self.config.opt == 'Adam'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif (self.config.opt == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(self.config.lr)
        opt = optimizer.minimize(loss)
        
        self.loss = loss
        self.opt = opt
        
        #  infer root token idxs only
        root_output = sents_output_root[:, 0, :]
        beam_root_output = tf.contrib.seq2seq.tile_batch(root_output, multiplier=self.config.beam_width)

        start_tokens = tf.fill([batch_l], self.config.BOS_IDX)
        end_token = self.config.EOS_IDX

        beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=dec_cell,
            embedding=self.embeddings,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=beam_root_output,
            beam_width=self.config.beam_width, 
            output_layer=output_layer,
            length_penalty_weight=self.config.length_penalty_weight)

        beam_decoder_outputs, _, root_beam_sent_l = tf.contrib.seq2seq.dynamic_decode(
            beam_decoder,
            maximum_iterations = self.config.maximum_iterations)

        root_token_idxs = beam_decoder_outputs.predicted_ids[:, :, 0]
        self.root_token_idxs = root_token_idxs
    
    def get_feed_dict(self, batch, mode='train'):
        batch_size = len(batch)
        doc_l_matrix = np.array([instance.doc_l for instance in batch]).astype(np.int32)

        max_doc_l = np.max(doc_l_matrix)
        max_sent_l = max([instance.max_sent_l for instance in batch])

        token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l], np.int32)
        dec_input_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l+1], np.int32)
        dec_target_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l+1], np.int32)
        sent_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
        dec_sent_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)

        for i, instance in enumerate(batch):
            for j, sent in enumerate(instance.token_idxs):
                token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
                dec_input_idxs_matrix[i, j, :len(sent)+1] = np.asarray([self.config.BOS_IDX] + sent)
                dec_target_idxs_matrix[i, j, :len(sent)+1] = np.asarray(sent + [self.config.EOS_IDX])
                sent_l_matrix[i, j] = len(sent)
                dec_sent_l_matrix[i, j] = len(sent)+1

        keep_prob = self.config.keep_prob if mode == 'train' else 1.0
        
        feed_dict = {
                    self.t_variables['token_idxs']: token_idxs_matrix, 
                    self.t_variables['dec_input_idxs']: dec_input_idxs_matrix, self.t_variables['dec_target_idxs']: dec_target_idxs_matrix, 
                    self.t_variables['batch_l']: batch_size, self.t_variables['doc_l']: doc_l_matrix, self.t_variables['sent_l']: sent_l_matrix, self.t_variables['dec_sent_l']: dec_sent_l_matrix,
                    self.t_variables['max_doc_l']: max_doc_l, self.t_variables['max_sent_l']: max_sent_l, 
                    self.t_variables['keep_prob']: keep_prob}
        return  feed_dict
