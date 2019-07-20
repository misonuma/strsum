#coding:utf-8

import tensorflow as tf
import numpy as np
import pdb

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
            self.embeddings = tf.get_variable("emb", [self.config.n_embed, self.config.d_embed], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            
        with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
            w_comb = tf.get_variable("w_comb", [dim_sent, dim_sent], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb = tf.get_variable("bias_comb", [dim_sent], dtype=tf.float32, initializer=tf.constant_initializer())

        with tf.variable_scope("Structure", reuse=tf.AUTO_REUSE):
            w_parser_p = tf.get_variable("w_parser_p", [dim_str, dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_parser_p = tf.get_variable("bias_parser_p", [dim_str], dtype=tf.float32,
                            initializer=tf.constant_initializer())

            w_parser_c = tf.get_variable("w_parser_c", [dim_str, dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_parser_c = tf.get_variable("bias_parser_c", [dim_str], dtype=tf.float32,
                            initializer=tf.constant_initializer())

            w_parser_s = tf.get_variable("w_parser_s", [dim_str, dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

            w_parser_root = tf.get_variable("w_parser_root", [dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        # get word embedding
        batch_l = self.t_variables['batch_l']
        max_doc_l = self.t_variables['max_doc_l']
        max_sent_l = self.t_variables['max_sent_l']

        token_idxs = self.t_variables['token_idxs'][:, :max_doc_l, :max_sent_l]

        tokens_input_ = tf.nn.embedding_lookup(self.embeddings, token_idxs)
        tokens_input = tf.nn.dropout(tokens_input_, self.t_variables['keep_prob'])

        tokens_input_do = tf.reshape(tokens_input, [batch_l * max_doc_l, max_sent_l, self.config.d_embed])
        
        def dynamicBiRNN(input, seqlen, n_hidden, cell_name='', reuse=False):
            batch_size = tf.shape(input)[0]
            with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
                fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = self.t_variables['keep_prob'])
                fw_initial_state = fw_cell.zero_state(batch_size, tf.float32)
            with tf.variable_scope(cell_name + 'bw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
                bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
                bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = self.t_variables['keep_prob'])
                bw_initial_state = bw_cell.zero_state(batch_size, tf.float32)
            with tf.variable_scope(cell_name, reuse=reuse):
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                                         initial_state_fw=fw_initial_state,
                                                                         initial_state_bw=bw_initial_state,
                                                                         sequence_length=seqlen)
            return outputs, output_states    
        
        # get sentence embedding
        sent_l = self.t_variables['sent_l']
        sent_l_do = tf.reshape(sent_l, [batch_l * max_doc_l])
        
        tokens_outputs, tokens_output_states = dynamicBiRNN(tokens_input_do, sent_l_do, n_hidden=dim_hidden, cell_name='Model/sent')
        
        mask_tokens_do = tf.sequence_mask(sent_l_do, maxlen=max_sent_l, dtype=tf.float32)

        tokens_output_do_ = tf.concat(tokens_outputs, 2)
        tokens_output_do = tokens_output_do_ + tf.expand_dims((mask_tokens_do-1)*999,2)

        doc_l = self.t_variables['doc_l']
        mask_sents = tf.sequence_mask(doc_l, maxlen=max_doc_l, dtype=tf.float32)
        mask_sents_do = tf.reshape(mask_sents, [batch_l * max_doc_l, 1])

        sents_input_do = tf.reduce_max(tokens_output_do, 1) * mask_sents_do
        sents_input_ = tf.reshape(sents_input_do, [batch_l, max_doc_l, dim_bi_hidden])
        
        sents_input_str_ = sents_input_[:, :, :dim_str]
        sents_input_ = sents_input_[:, :, dim_str:]
    
        sents_input_str = tf.nn.dropout(sents_input_str_, self.t_variables['keep_prob'])
        sents_input = tf.nn.dropout(sents_input_, self.t_variables['keep_prob'])
    
        # get document structure
        parent = tf.tanh(tf.tensordot(sents_input_str, w_parser_p, [[2], [0]]) + b_parser_p)
        child = tf.tanh(tf.tensordot(sents_input_str, w_parser_c, [[2], [0]]) + b_parser_c)

        raw_scores_words_tmp = tf.tanh(tf.matmul(tf.tensordot(parent, w_parser_s, [[-1],[0]]),tf.matrix_transpose(child)))
        raw_scores_words_ = tf.exp(raw_scores_words_tmp)
        diag_zero = tf.zeros_like(raw_scores_words_[:,:,0])
        raw_scores_words = tf.matrix_set_diag(raw_scores_words_, diag_zero)
        self.raw_scores_words = raw_scores_words

        raw_scores_root_ = tf.squeeze(tf.tensordot(sents_input_str, w_parser_root, [[2], [0]]) , [2])
        raw_scores_root = tf.exp(tf.tanh(raw_scores_root_))
        self.raw_scores_root = raw_scores_root

        def _getMatrixTree(r, A):
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
            return d, LL
        
        str_scores_, LL = _getMatrixTree(raw_scores_root, raw_scores_words)
        str_scores = tf.multiply(str_scores_, tf.expand_dims(mask_sents, 1))
        self.str_scores = str_scores
        
        def get_eig_str_scores(str_scores):
            str_scores_root = str_scores[:, 0, :]
            str_scores_words = str_scores[:, 1:, :]

            str_root = tf.expand_dims(tf.one_hot(indices=0, depth=(max_doc_l+1), 
                        on_value=0.0, off_value=1.0/tf.cast(max_doc_l, tf.float32), dtype=tf.float32), 0)
            str_roots = tf.expand_dims(tf.tile(str_root, [batch_l, 1]), -1)
#             str_roots = tf.zeros([batch_l, (max_doc_l+1), 1], dtype=tf.float32)
            adj_scores = tf.concat([str_roots, str_scores], 2)

            eye = tf.expand_dims(tf.diag(tf.ones(max_doc_l+1)), 0)
            eye_words = tf.tile(eye, [batch_l, 1, 1])

            eig_matrix_inv = tf.matrix_inverse(eye_words - self.config.damp*adj_scores)
            damp_vec = tf.ones([batch_l, (max_doc_l+1), 1]) / tf.cast((max_doc_l+1), tf.float32)

            eig_scores_ = tf.multiply((1-self.config.damp), tf.matmul(eig_matrix_inv, damp_vec))
            eig_scores_ = tf.squeeze(eig_scores_, 2)[:, 1:]
            eig_scores = tf.nn.softmax(eig_scores_, 1)

            eig_str_scores_root = tf.expand_dims(tf.multiply(eig_scores, str_scores_root), 1)
            eig_str_scores = tf.concat([eig_str_scores_root, str_scores_words], 1)
            
            return eig_str_scores, eig_scores_

        if self.config.discourserank:
            eig_str_scores, eig_scores_ = get_eig_str_scores(str_scores)
            self.eig_str_scores = eig_str_scores
            self.eig_scores = eig_scores_
            str_scores = eig_str_scores
        
        str_scores_sum = tf.expand_dims(tf.reduce_sum(str_scores, 2), 2)
        str_scores_norm = str_scores/str_scores_sum

        # get structured sentence embedding
        sents_output_root_raw = tf.matmul(str_scores_norm, sents_input)
        sents_output_root_ = tf.nn.dropout(tf.tanh(tf.tensordot(sents_output_root_raw, w_comb, [[2], [0]]) + b_comb), self.t_variables['keep_prob'])
        sents_output_ = sents_output_root_[:, 1:, :]

        mask_sents_root = tf.concat([tf.ones([batch_l, 1]), mask_sents], 1)

        sents_output = tf.multiply(sents_output_,tf.expand_dims(mask_sents, 2))
        sents_output_do = tf.reshape(sents_output, [batch_l*max_doc_l, dim_sent])

        sents_output_root = tf.multiply(sents_output_root_, tf.expand_dims(mask_sents_root,2))
        sents_output_root_do = tf.reshape(sents_output_root, [batch_l*(max_doc_l+1), dim_sent])
        
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
            
        # decode for inference of only summary tokens
        root_start_tokens = tf.fill([batch_l], self.config.BOS_IDX)
        end_token = self.config.EOS_IDX

        root_sent_input = sents_output_root_[:, 0, :]
        self.root_sent_input = root_sent_input
        beam_root_sent_input = tf.contrib.seq2seq.tile_batch(root_sent_input, multiplier=self.config.beam_width)

        root_beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=dec_cell,
            embedding=self.embeddings,
            start_tokens=root_start_tokens,
            end_token=end_token,
            initial_state=beam_root_sent_input,
            beam_width=self.config.beam_width, 
            output_layer=output_layer,
            length_penalty_weight=self.config.length_penalty_weight)

        root_beam_decoder_outputs, _, root_beam_sent_l = tf.contrib.seq2seq.dynamic_decode(
            root_beam_decoder,
            maximum_iterations = self.config.maximum_iterations)

        root_output_token_idxs_ = root_beam_decoder_outputs.predicted_ids[:, :, 0]
        self.summary_output_token_idxs = root_output_token_idxs_

        # decode for inferrence
        start_tokens = tf.fill([batch_l * (max_doc_l+1)], self.config.BOS_IDX)
        end_token = self.config.EOS_IDX            
        with tf.variable_scope('Model/sent/decoder', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=tf.AUTO_REUSE):
            tiled_sents_output_root_do = tf.contrib.seq2seq.tile_batch(
                    sents_output_root_do, multiplier=self.config.beam_width)
            
            tiled_decoder_initial_state = tiled_sents_output_root_do
            beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=self.embeddings,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=tiled_decoder_initial_state,
                beam_width=self.config.beam_width, 
                output_layer=output_layer,
                length_penalty_weight=self.config.length_penalty_weight)

            beam_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                            beam_decoder,
                            maximum_iterations = self.config.maximum_iterations)

            beam_output_token_idxs_do = beam_decoder_outputs.predicted_ids[:, :, 0]
            beam_output_token_idxs = tf.reshape(beam_output_token_idxs_do, [batch_l, max_doc_l+1, tf.shape(beam_output_token_idxs_do)[1]])
            self.beam_output_token_idxs = beam_output_token_idxs

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
