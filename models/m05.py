from operator import mul
from functools import reduce

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import rnn
import my_rnn_cell
from pprint import pprint

import nn
from models.bm05 import BaseTower, BaseRunner


class Sentence(object):
    def __init__(self, shape, name='sentence'):
        self.name = name
        self.shape = shape
        self.x = tf.placeholder('int32', shape, name="%s" % name)
        self.x_mask = tf.placeholder('float', shape, name="%s_mask" % name)
        self.x_len = tf.placeholder('int16', shape[:-1], name="%s_len" % name)
        self.x_mask_aug = tf.expand_dims(self.x_mask, -1, name='%s_mask_aug' % name)

    def add(self, feed_dict, *batch):
        x, x_mask, x_len = batch
        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.x_len] = x_len


class Memory(Sentence):
    def __init__(self, params, name='memory'):
        N, M, K = params.batch_size, params.max_num_facts, params.max_fact_size
        shape = [N, M, K]
        super().__init__(shape, name=name)
        self.m_mask = tf.placeholder('float', [N, M], name='m_mask')

    def add(self, feed_dict, *batch):
        x, x_mask, x_len, m_mask = batch
        super().add(feed_dict, x, x_mask, x_len)
        feed_dict[self.m_mask] = m_mask


class PESentenceEncoder(object):
    def __init__(self, params, emb_mat):
        self.params = params
        V, d, L, e = params.vocab_size, params.hidden_size, params.rnn_num_layers, params.word_size
        # self.init_emb_mat = tf.get_variable("init_emb_mat", [self.V, self.d])
        emb_hidden_sizes = [d for _ in range(params.emb_num_layers)]
        prev_size = e
        for layer_idx in range(params.emb_num_layers):
            with tf.variable_scope("Ax_%d" % layer_idx):
                cur_size = emb_hidden_sizes[layer_idx]
                mat = tf.get_variable("mat_%d" % layer_idx, shape=[prev_size, cur_size])
                bias = tf.get_variable("bias_%d" % layer_idx, shape=[cur_size])
                emb_mat = tf.tanh(tf.matmul(emb_mat, mat) + bias)
        self.emb_mat = emb_mat  # [V, d]


    def __call__(self, sentence, name='u'):
        assert isinstance(sentence, Sentence)
        params = self.params
        d, e = params.hidden_size, params.word_size
        J = sentence.shape[-1]
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)

        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]

        _l = [g(j) for j in range(J)]
        self.l = tf.constant(_l, shape=[J, d], name='l')
        assert isinstance(sentence, Sentence)
        Ax = tf.nn.embedding_lookup(self.emb_mat, sentence.x, name='Ax')
        # TODO : dimension transformation
        lAx = self.l * Ax
        lAx_masked = lAx * tf.expand_dims(sentence.x_mask, -1)
        m = tf.reduce_sum(lAx_masked, len(sentence.shape) - 1, name=name)
        return m


class LSTMSentenceEncoder(object):
    def __init__(self, params, emb_mat):
        self.params = params
        V, d, L, e = params.vocab_size, params.hidden_size, params.rnn_num_layers, params.word_size
        prev_size = e
        hidden_sizes = [d for _ in range(params.emb_num_layers)]
        for layer_idx in range(params.emb_num_layers):
            with tf.variable_scope("emb_%d" % layer_idx):
                cur_hidden_size = hidden_sizes[layer_idx]
                emb_mat = tf.tanh(nn.linear([V, prev_size], cur_hidden_size, emb_mat))
                prev_size = cur_hidden_size
        self.emb_mat = emb_mat

        self.emb_hidden_sizes = [d for _ in range(params.emb_num_layers)]
        self.input_size = self.emb_hidden_sizes[-1] if self.emb_hidden_sizes else e

        if params.lstm == 'basic':
            self.first_cell = my_rnn_cell.BasicLSTMCell(d, input_size=self.input_size, forget_bias=params.forget_bias)
            self.second_cell = my_rnn_cell.BasicLSTMCell(d, forget_bias=params.forget_bias)
        elif params.lstm == 'regular':
            self.first_cell = rnn_cell.LSTMCell(d, self.input_size, cell_clip=params.cell_clip)
            self.second_cell = rnn_cell.LSTMCell(d, d, cell_clip=params.cell_clip)
        elif params.lstm == 'gru':
            self.first_cell = rnn_cell.GRUCell(d, input_size=self.input_size)
            self.second_cell = rnn_cell.GRUCell(d)
        else:
            raise Exception()

        if params.train and params.keep_prob < 1.0:
            self.first_cell = tf.nn.rnn_cell.DropoutWrapper(self.first_cell, input_keep_prob=params.keep_prob, output_keep_prob=params.keep_prob)
        self.cell = rnn_cell.MultiRNNCell([self.first_cell] + [self.second_cell] * (L-1))
        self.scope = tf.get_variable_scope()
        self.used = False

    def __call__(self, sentence, init_hidden_state=None, name='s'):
        params = self.params
        L, d = params.rnn_num_layers, params.hidden_size
        h_flat = self.get_last_hidden_state(sentence, init_hidden_state=init_hidden_state)
        if params.lstm in ['basic', 'regular']:
            h_last = tf.reshape(h_flat, sentence.shape[:-1] + [2*L*d])
            s = tf.identity(tf.split(2, 2*L, h_last)[2*L-1], name=name)
        elif params.lstm == 'gru':
            h_last = tf.reshape(h_flat, sentence.shape[:-1] + [L*d])
            s = tf.identity(tf.split(2, L, h_last)[L-1], name=name)
        else:
            raise Exception()
        return s

    def get_last_hidden_state(self, sentence, init_hidden_state=None):
        assert isinstance(sentence, Sentence)
        params = self.params
        with tf.variable_scope(self.scope, reuse=self.used):
            J = sentence.shape[-1]
            Ax = tf.nn.embedding_lookup(self.emb_mat, sentence.x)  # [N, C, J, e]

            F = reduce(mul, sentence.shape[:-1], 1)
            init_hidden_state = init_hidden_state or self.cell.zero_state(F, tf.float32)
            Ax_flat = tf.reshape(Ax, [F, J, self.input_size])
            x_len_flat = tf.reshape(sentence.x_len, [F])

            # Ax_flat_split = [tf.squeeze(x_flat_each, [1]) for x_flat_each in tf.split(1, J, Ax_flat)]
            o_flat, h_flat = rnn.dynamic_rnn(self.cell, Ax_flat, x_len_flat, initial_state=init_hidden_state)
            self.used = True
            return h_flat


class Sim(object):
    def __init__(self, params, memory, encoder, u):
        N, C, R, d = params.batch_size, params.num_choices, params.max_num_facts, params.hidden_size
        f = encoder(memory, name='f')
        f_aug = tf.expand_dims(f, 1)  # [N, 1, R, d]
        u_aug = tf.expand_dims(u, 2)  # [N, C, 1, d]
        u_tiled = tf.tile(u_aug, [1, 1, R, 1])
        if params.sim_func == 'man_dist':
            uf = nn.man_sim([N, C, R, d], f_aug, u_tiled, name='uf')  # [N, C, R]
        elif params.sim_func == 'dot':
            uf = tf.reduce_sum(u_tiled * f_aug, 3)
        else:
            raise Exception()
        max_logit = tf.reduce_max(uf, 2)  # [N, C]
        uf_flat = tf.reshape(uf, [N*C, R])
        uf_sm_flat = tf.nn.softmax(uf_flat)
        uf_sm = tf.reshape(uf_sm_flat, [N, C, R])
        var_logit = tf.reduce_max(uf_sm, 2)
        if params.max_func == 'max':
            logit = max_logit
        elif params.max_func == 'var':
            logit = var_logit
        elif params.max_func == 'combined':
            logit = var_logit * max_logit

        f_mask_aug = tf.expand_dims(memory.m_mask, 1)
        p = nn.softmax_with_mask([N, C, R], uf, f_mask_aug, name='p')
        # p = tf.reshape(p_flat, [N, C, R], name='p')

        self.logit = logit
        self.p = p


class Layer(object):
    def __init__(self, params, memory, prev_layer=None, sent_encoder=None, u=None):
        assert isinstance(memory, Memory)
        self.params = params
        N, C, R, d = params.batch_size, params.num_choices, params.max_num_facts, params.hidden_size

        with tf.variable_scope("input"):
            if sent_encoder:
                # input_encoder = RelationEncoder(params, sent_encoder=sent_encoder)
                input_encoder = sent_encoder
            else:
                # input_encoder = RelationEncoder(params, rel_encoder=prev_layer.output_encoder)
                input_encoder = LSTMSentenceEncoder(params)
        with tf.variable_scope("output"):
            output_encoder = input_encoder  # RelationEncoder(params)

        f = input_encoder(memory, name='f')  # [N, R, d]
        c = f  # output_encoder(memory)  # [N, R, d]
        if u is None:
            u = tf.identity(prev_layer.u + prev_layer.o, name="u")  # [N, C, d]

        with tf.name_scope('p'):
            f_aug = tf.expand_dims(f, 1)  # [N, 1, R, d]
            c_aug = tf.expand_dims(c, 1)  # [N, 1, R, d]
            u_aug = tf.expand_dims(u, 2)  # [N, C, 1, d]
            u_tiled = tf.tile(u_aug, [1, 1, R, 1])  # [N, C, R, d]
            uf = tf.reduce_sum(u_tiled * f_aug, 3, name='uf')  # [N, C, R]
            f_mask_aug = tf.expand_dims(memory.m_mask, 1)  # [N, 1, R]
            # f_mask_tiled = tf.tile(f_mask_aug, [1, C, 1])  # [N, C, R]
            # uf_flat = tf.reshape(uf, [N, C*R])
            # f_mask_tiled_flat = tf.reshape(f_mask_tiled, [N, C*R])
            p_flat = nn.softmax_with_mask([N, C, R], uf, f_mask_aug, name='p_flat')  # [N, C, R]
            p = tf.reshape(p_flat, [N, C, R], name='p')
            # p_debug = tf.reduce_sum(p, 2)

        with tf.name_scope('o'):
            c_tiled = tf.tile(c_aug, [1, C, 1, 1])  # [N, C, R, d]
            o = tf.reduce_sum(c_tiled * tf.expand_dims(p, -1), 2)  # [N, C, d]

        self.f = f
        self.c = c
        self.p = p  # [N, C, R]
        self.u = u
        self.o = o
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder


class AttentionTower(BaseTower):
    def initialize(self):
        params = self.params
        tensors = self.tensors
        placeholders = self.placeholders

        V, d, G = params.vocab_size, params.hidden_size, params.image_size
        N, C, J = params.batch_size, params.num_choices, params.max_sent_size
        e = params.word_size

        # initialize self
        # placeholders
        with tf.name_scope('ph'):
            s = Sentence([N, C, J], 's')
            f = Memory(params, 'f')
            image = tf.placeholder('float', [N, G], name='i')
            if params.use_null:
                y = tf.placeholder('float', [N, C+1], name='y')
            else:
                y = tf.placeholder('int8', [N, C], name='y')
            init_emb_mat = tf.placeholder('float', shape=[V, e], name='init_emb_mat')
            placeholders['s'] = s
            placeholders['f'] = f
            placeholders['image'] = image
            placeholders['y'] = y
            placeholders['init_emb_mat'] = init_emb_mat

        with tf.variable_scope('first_u'):
            u_encoder = LSTMSentenceEncoder(params, init_emb_mat)
            # u_encoder = PESentenceEncoder(params, init_emb_mat)
            first_u = u_encoder(s, name='first_u')
        with tf.variable_scope('first_v'):
            v_encoder = LSTMSentenceEncoder(params, init_emb_mat)
            # v_encoder = PESentenceEncoder(params, init_emb_mat)
            first_v = v_encoder(s, name='first_v')


        layers = []
        prev_layer = None
        for layer_index in range(params.num_layers):
            with tf.variable_scope('layer_%d' % layer_index):
                if prev_layer:
                    cur_layer = Layer(params, f, prev_layer=prev_layer)
                else:
                    cur_layer = Layer(params, f, u=first_u, sent_encoder=u_encoder)
                layers.append(cur_layer)
                prev_layer = cur_layer
        last_layer = layers[-1]

        sim = Sim(params, f, u_encoder, first_u)

        if params.model == 'sim':
            with tf.variable_scope("sim"):
                logit = sim.logit
        elif params.model == 'att':
            with tf.variable_scope("att"):
                logit = tf.reduce_sum(first_u * last_layer.o, 2)
        else:
            raise Exception()

        with tf.variable_scope("merge"):
            # raw_gate_aug = nn.linear([N, C], 1, logit) # [N, 1]
            raw_gate = tf.reduce_sum(logit, 1)
            raw_gate_aug = tf.expand_dims(raw_gate, -1, name='raw_gate_aug')
            gate = tf.nn.sigmoid(raw_gate, name='gate')
            gate_aug = tf.expand_dims(gate, -1, name='gate_aug')
            gate_avg = tf.reduce_mean(gate, 0, name='gate_avg')
            sent_logit_aug = nn.linear([N, C, d], 1, first_v, 'sent_logit')
            sent_logit = tf.squeeze(sent_logit_aug, [2])
            mem_logit = logit
            if params.mode == 'l':
                logit = sent_logit
            elif params.mode == 'a':
                logit = mem_logit
            elif params.mode == 'la':
                logit = (1 - gate_aug) * mem_logit + gate_aug * sent_logit
                # self.logit = sig * memory_logit + (1 - sig) * sent_logit

            if params.use_null:
                logit = tf.concat(1, [logit, raw_gate_aug], name='logit_concat')
            tensors['logit'] = logit

        with tf.variable_scope('yp'):
            yp = tf.nn.softmax(logit, name='yp')  # [N, C]
            if params.use_null:
                yp = tf.concat(1, [yp, gate_aug], 'yp_concat')
            tensors['yp'] = yp

        with tf.name_scope('loss') as loss_scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit, tf.cast(y, 'float'), name='cross_entropy')
            avg_cross_entropy = tf.reduce_mean(cross_entropy, 0, name='avg_cross_entropy')
            tf.add_to_collection('losses', avg_cross_entropy)
            loss = tf.add_n(tf.get_collection('losses'), name='loss')
            tensors['loss'] = cross_entropy
            tensors['avg_cross_entropy'] = avg_cross_entropy

        with tf.name_scope('acc'):
            correct_vec = tf.equal(tf.argmax(yp, 1), tf.argmax(y, 1))
            num_corrects = tf.reduce_sum(tf.cast(correct_vec, 'float'), name='num_corrects')
            acc = tf.reduce_mean(tf.cast(correct_vec, 'float'), name='acc')
            tensors['correct'] = correct_vec
            tensors['num_corrects'] = num_corrects
            tensors['acc'] = acc

    def get_feed_dict(self, batch, mode, **kwargs):
        placeholders = self.placeholders
        if batch is None:
            assert mode != 'train', "Cannot pass empty batch during training, for now."
            sents_batch, facts_batch, images_batch, label_batch = None, None, None, None
        else:
            sents_batch, facts_batch, images_batch = batch[:-1]
            if len(batch) > 3:
                label_batch = batch[-1]
            else:
                label_batch = np.zeros([len(sents_batch)])
        s = self._prepro_sents_batch(sents_batch)  # [N, C, J], [N, C]
        f = self._prepro_facts_batch(facts_batch)
        g = self._prepro_images_batch(images_batch)
        feed_dict = {placeholders['image']: g, placeholders['init_emb_mat']: self.params.init_emb_mat}
        if mode == 'train':
            null_weight = kwargs['null_weight']
            y_batch = self._prepro_label_batch(label_batch, null_weight=null_weight)
        elif mode == 'eval':
            y_batch = self._prepro_label_batch(label_batch)
        else:
            raise Exception()
        feed_dict[placeholders['y']] = y_batch
        placeholders['s'].add(feed_dict, *s)
        placeholders['f'].add(feed_dict, *f)
        return feed_dict

    def _prepro_images_batch(self, images_batch):
        params = self.params
        N, G = params.batch_size, params.image_size
        g = np.zeros([N, G])
        if images_batch is None:
            return g
        g[:len(images_batch)] = images_batch
        return g

    def _prepro_sents_batch(self, sents_batch):
        p = self.params
        N, C, J = p.batch_size, p.num_choices, p.max_sent_size
        s_batch = np.zeros([N, C, J], dtype='int32')
        s_mask_batch = np.zeros([N, C, J], dtype='float')
        s_len_batch = np.zeros([N, C], dtype='int16')
        out = s_batch, s_mask_batch, s_len_batch
        if sents_batch is None:
            return out
        for n, sents in enumerate(sents_batch):
            for c, sent in enumerate(sents):
                for j, idx in enumerate(sent):
                    s_batch[n, c, j] = idx
                    s_mask_batch[n, c, j] = 1.0
                s_len_batch[n, c] = len(sent)

        return out

    def _prepro_facts_batch(self, facts_batch):
        p = self.params
        N, M, K = p.batch_size, p.max_num_facts, p.max_fact_size
        s_batch = np.zeros([N, M, K], dtype='int32')
        s_mask_batch = np.zeros([N, M, K], dtype='float')
        s_len_batch = np.zeros([N, M], dtype='int16')
        m_mask_batch = np.zeros([N, M], dtype='float')
        out = s_batch, s_mask_batch, s_len_batch, m_mask_batch
        if facts_batch is None:
            return out
        for n, sents in enumerate(facts_batch):
            for m, sent in enumerate(sents):
                for k, idx in enumerate(sent):
                    s_batch[n, m, k] = idx
                    s_mask_batch[n, m, k] = 1.0
                s_len_batch[n, m] = len(sent)
                m_mask_batch[n, m] = 1.0
        return out

    def _prepro_label_batch(self, label_batch, null_weight=0.0):
        p = self.params
        N, C = p.batch_size, p.num_choices
        if p.use_null:
            y = np.zeros([N, C+1], dtype='float')
        else:
            y = np.zeros([N, C], dtype='float')
        if label_batch is None:
            return y
        for i, label in enumerate(label_batch):
            y[i, label] = np.random.rand() * self.params.rand_y
            rand_other = (1.0 - self.params.rand_y)/(C-1)
            for cur in range(C):
                if cur != label:
                    y[i, cur] = np.random.rand() * rand_other
            y[i] = y[i] / sum(y[i])
            if p.use_null:
                y[i, C] = null_weight

        return y


class AttentionRunner(BaseRunner):
    def _get_train_args(self, epoch_idx):
        params = self.params
        learning_rate = params.init_lr
        null_weight = params.init_nw

        anneal_period = params.anneal_period
        anneal_ratio = params.anneal_ratio
        num_periods = int(epoch_idx / anneal_period)
        factor = anneal_ratio ** num_periods

        if params.use_null:
            nw_period = params.nw_period
            nw_ratio = params.nw_ratio
            nw_num_periods = int(epoch_idx / nw_period)
            nw_factor = nw_ratio ** nw_num_periods
            null_weight *= nw_factor
        if params.opt == 'basic':
            learning_rate *= factor

        train_args = {'null_weight': null_weight, 'learning_rate': learning_rate}
        return train_args
