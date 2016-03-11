from operator import mul
from functools import reduce

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import rnn
from pprint import pprint

import nn
from models.base_model_04 import BaseModel


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


class SentenceEncoder(object):
    def __init__(self, V, J, d, sent_encoder=None):

        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)

        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]

        _l = [g(j) for j in range(J)]
        self.A = tf.identity(sent_encoder.A, 'A') if sent_encoder else tf.get_variable('A', shape=[V, d])
        self.l = tf.constant(_l, shape=[J, d], name='l')

    def __call__(self, sentence, name='u'):
        assert isinstance(sentence, Sentence)
        Ax = tf.nn.embedding_lookup(self.A, sentence.x)
        lAx = self.l * Ax
        lAx_masked = lAx * tf.expand_dims(sentence.x_mask, -1)
        m = tf.reduce_sum(lAx_masked, len(sentence.shape) - 1, name=name)
        return m


class LSTMSentenceEncoder(object):
    def __init__(self, params):
        self.params = params
        self.V, self.d, self.L, self.e = params.vocab_size, params.hidden_size, params.rnn_num_layers, params.word_size
        # self.init_emb_mat = tf.get_variable("init_emb_mat", [self.V, self.d])
        self.init_emb_mat = tf.placeholder('float', shape=[self.V, self.e], name='init_emb_mat')
        emb_mat = self.init_emb_mat
        prev_size = self.e
        hidden_sizes = [self.d for _ in range(params.emb_num_layers)]
        for layer_idx in range(params.emb_num_layers):
            with tf.variable_scope("emb_%d" % layer_idx):
                cur_hidden_size = hidden_sizes[layer_idx]
                next_emb_mat = tf.tanh(nn.linear([self.V, prev_size], cur_hidden_size, emb_mat))
                emb_mat = next_emb_mat
                prev_size = cur_hidden_size
        self.emb_mat = emb_mat
        self.single_cell = rnn_cell.BasicLSTMCell(self.d, forget_bias=0.0)
        self.cell = rnn_cell.MultiRNNCell([self.single_cell] * self.L)

    def __call__(self, sentence, init_hidden_state=None, name='s'):
        h_flat = self.get_last_hidden_state(sentence, init_hidden_state=init_hidden_state)
        h_last = tf.reshape(h_flat, sentence.shape[:-1] + [2*self.L*self.d])
        s = tf.identity(tf.split(2, 2*self.L, h_last)[2*self.L-1], name=name)
        return s

    def get_last_hidden_state(self, sentence, init_hidden_state=None):
        assert isinstance(sentence, Sentence)
        d, L, e =  self.d, self.L, self.e
        J = sentence.shape[-1]
        Ax = tf.nn.embedding_lookup(self.emb_mat, sentence.x, "Ax")  # [N, C, J, d]
        F = reduce(mul, sentence.shape[:-1], 1)
        init_hidden_state = init_hidden_state or self.cell.zero_state(F, tf.float32)
        Ax_flat = tf.reshape(Ax, [F, J, d])
        x_len_flat = tf.reshape(sentence.x_len, [F])

        Ax_flat_split = [tf.squeeze(x_flat_each, [1])
                         for x_flat_each in tf.split(1, J, Ax_flat)]
        o_flat, h_flat = rnn.rnn(self.cell, Ax_flat_split, init_hidden_state, sequence_length=x_len_flat)
        # tf.get_variable_scope().reuse_variables()
        return h_flat


class Layer(object):
    def __init__(self, params, memory, prev_layer=None, sent_encoder=None, u=None):
        assert isinstance(memory, Memory)
        self.params = params
        N, C, R, d = params.batch_size, params.num_choices, params.max_num_facts, params.hidden_size
        linear_start = params.linear_start

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
        u = tf.identity(u or prev_layer.u + prev_layer.o, name="u")  # [N, C, d]

        with tf.name_scope('p'):
            f_aug = tf.expand_dims(f, 1)  # [N, 1, R, d]
            c_aug = tf.expand_dims(c, 1)  # [N, 1, R, d]
            u_aug = tf.expand_dims(u, 2)  # [N, C, 1, d]
            u_tiled = tf.tile(u_aug, [1, 1, R, 1])  # [N, C, R, d]
            uf = tf.reduce_sum(u_tiled * f_aug, 3, name='uf')  # [N, C, R]
            f_mask_aug = tf.expand_dims(memory.m_mask, 1)  # [N, 1, R]
            if linear_start:
                p = tf.reduce_sum(tf.mul(uf, f_mask_aug, name='p'), 3)  # [N, C, R]
            else:
                p = nn.softmax_with_mask([N, C, R], uf, f_mask_aug, name='p')  # [N, C, R]
                p_debug = tf.reduce_sum(p, 2)  # must be 1!

            if prev_layer is None:
                base = tf.get_variable('base', shape=[], dtype='float')
            else:
                base = prev_layer.base
            sig, _ = nn.softmax_with_base([N, C, R], base, uf, f_mask_aug)  # [N, C]

        with tf.name_scope('o'):
            c_tiled = tf.tile(c_aug, [1, C, 1, 1])  # [N, C, R, d]
            o = tf.reduce_sum(c_tiled * tf.expand_dims(p, -1), 2)  # [N, C, d]

        self.f = f
        self.c = c
        self.p = p
        self.p_debug = p_debug
        self.u = u
        self.o = o
        self.sig = sig
        self.base = base
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder


class AttentionModel(BaseModel):
    def _build_tower(self):
        params = self.params
        V, d, G = params.vocab_size, params.hidden_size, params.image_size
        N, C, J = params.batch_size, params.num_choices, params.max_sent_size

        summaries = []

        # initialize self
        # placeholders
        with tf.name_scope('ph'):
            self.s = Sentence([N, C, J], 's')
            self.f = Memory(params, 'f')
            self.image = tf.placeholder('float', [N, G], name='i')
            self.y = tf.placeholder('int8', [N, C], name='y')

        with tf.variable_scope('first_u'):
            sent_encoder = LSTMSentenceEncoder(params)
            self.init_emb_mat = sent_encoder.init_emb_mat
            first_u = sent_encoder(self.s, name='first_u')

        layers = []
        prev_layer = None
        for layer_index in range(params.num_layers):
            with tf.variable_scope('layer_%d' % layer_index):
                if prev_layer:
                    cur_layer = Layer(params, self.f, prev_layer=prev_layer)
                else:
                    cur_layer = Layer(params, self.f, u=first_u, sent_encoder=sent_encoder)
                layers.append(cur_layer)
                prev_layer = cur_layer
        last_layer = layers[-1]
        o_sum = sum(layer.o for layer in layers)

        with tf.variable_scope('f'):
            image = self.image  # [N, G]
            g = tf.tanh(nn.linear([N, G], d, image))  # [N, d]
            aug_g = tf.expand_dims(g, 2, name='aug_g')  # [N, d, 1]

        with tf.variable_scope('yp'):
            # self.logit = tf.squeeze(tf.batch_matmul(last_layer.u + last_layer.o, aug_g), [2])  # [N, C]
            image_logit = tf.squeeze(tf.batch_matmul(first_u, aug_g), [2])  # [N, C]
            memory_logit = tf.reduce_sum(first_u * o_sum, 2)# nn.prod_sum_sim([N, C, d], first_u, o_sum)
            sent_logit =  tf.reduce_sum(first_u, 2)
            if params.mode == 'l':
                self.logit = sent_logit
            elif params.mode == 'la':
                sig = last_layer.sig
                self.logit = sig * memory_logit + (1 - sig) * sent_logit
            self.yp = tf.nn.softmax(self.logit, name='yp')  # [N, C]

        with tf.name_scope('loss') as loss_scope:
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logit, tf.cast(self.y, 'float'), name='cross_entropy')
            self.avg_cross_entropy = tf.reduce_mean(self.cross_entropy, 0, name='avg_cross_entropy')
            tf.add_to_collection('losses', self.avg_cross_entropy)
            self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            self.losses = tf.get_collection('losses', loss_scope)

        with tf.name_scope('acc'):
            self.correct_vec = tf.equal(tf.argmax(self.yp, 1), tf.argmax(self.y, 1))
            self.num_corrects = tf.reduce_sum(tf.cast(self.correct_vec, 'float'), name='num_corrects')
            self.acc = tf.reduce_mean(tf.cast(self.correct_vec, 'float'), name='acc')

        with tf.name_scope('opt'):
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            # FIXME : This must muse cross_entropy for some reason!
            grads_and_vars = opt.compute_gradients(self.cross_entropy)
            # clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            self.opt_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)

        # summaries
        summaries.append(tf.histogram_summary(first_u.op.name, first_u))
        summaries.append(tf.histogram_summary(last_layer.f.op.name, last_layer.f))
        summaries.append(tf.histogram_summary(last_layer.u.op.name, last_layer.u))
        summaries.append(tf.histogram_summary(last_layer.o.op.name, last_layer.o))
        summaries.append(tf.histogram_summary(last_layer.p.op.name, last_layer.p))
        summaries.append(tf.histogram_summary(last_layer.sig.op.name, last_layer.sig))
        summaries.append(tf.scalar_summary("%s (raw)" % self.total_loss.op.name, self.total_loss))
        summaries.append(tf.scalar_summary("%s" % last_layer.base.op.name, last_layer.base))
        self.merged_summary = tf.merge_summary(summaries)
        self.last_layer = last_layer

    def _get_feed_dict(self, batch):
        sents_batch, facts_batch, images_batch = batch[:-1]
        if len(batch) > 3:
            label_batch = batch[-1]
        else:
            label_batch = np.zeros([len(sents_batch)])
        s = self._prepro_sents_batch(sents_batch)  # [N, C, J], [N, C]
        f = self._prepro_facts_batch(facts_batch)
        g = self._prepro_images_batch(images_batch)
        y_batch = self._prepro_label_batch(label_batch)
        feed_dict = {self.y: y_batch, self.image: g, self.init_emb_mat: self.params.init_emb_mat}
        self.s.add(feed_dict, *s)
        self.f.add(feed_dict, *f)
        return feed_dict

    def _prepro_images_batch(self, images_batch):
        params = self.params
        N, G = params.batch_size, params.image_size
        g = np.zeros([N, G])
        g[:len(images_batch)] = images_batch
        return g

    def _prepro_sents_batch(self, sents_batch):
        p = self.params
        N, C, J = p.batch_size, p.num_choices, p.max_sent_size
        s_batch = np.zeros([N, C, J], dtype='int32')
        s_mask_batch = np.zeros([N, C, J], dtype='float')
        s_len_batch = np.zeros([N, C], dtype='int16')
        for n, sents in enumerate(sents_batch):
            for c, sent in enumerate(sents):
                for j, idx in enumerate(sent):
                    s_batch[n, c, j] = idx
                    s_mask_batch[n, c, j] = 1.0
                s_len_batch[n, c] = len(sent)

        return s_batch, s_mask_batch, s_len_batch

    def _prepro_facts_batch(self, facts_batch):
        p = self.params
        N, M, K = p.batch_size, p.max_num_facts, p.max_fact_size
        s_batch = np.zeros([N, M, K], dtype='int32')
        s_mask_batch = np.zeros([N, M, K], dtype='float')
        s_len_batch = np.zeros([N, M], dtype='int16')
        m_mask_batch = np.zeros([N, M], dtype='float')
        for n, sents in enumerate(facts_batch):
            for m, sent in enumerate(sents):
                for k, idx in enumerate(sent):
                    s_batch[n, m, k] = idx
                    s_mask_batch[n, m, k] = 1.0
                s_len_batch[n, m] = len(sent)
                m_mask_batch[n, m] = 1.0

        return s_batch, s_mask_batch, s_len_batch, m_mask_batch

    def _prepro_label_batch(self, label_batch):
        p = self.params
        N, C = p.batch_size, p.num_choices
        y = np.zeros([N, C], dtype='int8')
        for i, label in enumerate(label_batch):
            y[i, label] = 1
        return y
