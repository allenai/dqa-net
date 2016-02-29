import tensorflow as tf
import numpy as np

import nn
from models.base_model import BaseModel


class Sentence(object):
    def __init__(self, shape, name='sentence'):
        self.name = name
        self.shape = shape
        self.x = tf.placeholder('int32', shape, name="%s" % name)
        self.x_len = tf.placeholder('int16', shape[:-1], name="%s_len" % name)
        self.x_mask = tf.placeholder('float', shape, name="%s_mask" % name)
        self.x_mask_aug = tf.expand_dims(self.x_mask, -1, name='%s_mask_aug' % name)

    def add(self, feed_dict, x, x_len, x_mask):
        feed_dict[self.x] = x
        feed_dict[self.x_len] = x_len
        feed_dict[self.x_mask] = x_mask


class Memory(object):
    def __init__(self, params):
        N, R, K, P = params.batch_size, params.max_num_rels, params.max_label_size, params.pred_size
        self.pred = tf.placeholder('float', [N, R, P], name='pred')
        self.num_rels = tf.placeholder('int16', [N], name='num_rels')
        self.rel_mask = tf.placeholder('float', [N, R], name="rel_mask")
        self.a1 = Sentence([N, R, K], name='a1')
        self.a2 = Sentence([N, R, K], name='a2')

    def add(self, feed_dict, rel_mask, num_rels, pred, a1, a2):
        feed_dict[self.rel_mask] = rel_mask
        feed_dict[self.num_rels] = num_rels
        feed_dict[self.pred] = pred
        self.a1.add(feed_dict, *a1)
        self.a2.add(feed_dict, *a2)


class SentenceEncoder(object):
    def __init__(self, V, J, d, encoder=None):

        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)

        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]

        _l = [g(j) for j in range(J)]
        self.A = tf.identity(encoder.A, 'A') if encoder else tf.get_variable('A', shape=[V, d])
        self.l = tf.constant(_l, shape=[J, d], name='l')

    def __call__(self, sentence, name='u'):
        assert isinstance(sentence, Sentence)
        Ax = tf.nn.embedding_lookup(self.A, sentence.x)
        lAx = self.l * Ax
        lAx_masked = lAx * tf.expand_dims(sentence.x_mask, -1)
        m = tf.reduce_sum(lAx_masked, len(sentence.shape) - 2, name=name)
        return m


class RelationEncoder(object):
    def __init__(self, params, rel_encoder=None, sent_encoder=None):
        self.params = params
        V, K, P, d = params.vocab_size, params.max_label_size, params.pred_size, params.hidden_size
        self.G = tf.identity(rel_encoder.G, name='G') if rel_encoder else tf.get_variable('G', dtype='float', shape=[P, d])
        self.sent_encoder = rel_encoder.sent_encoder if rel_encoder else sent_encoder or SentenceEncoder(V, K, d)

        J = 3

        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)

        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]

        _l = [g(j) for j in range(J)]
        self.l = tf.constant(_l, shape=[J, d], name='l')

    def _ground_pred(self, pred, name="p"):
        params = self.params
        N, R, P, d = params.batch_size, params.max_num_rels, params.pred_size, params.hidden_size
        pred_flat = tf.reshape(pred, [N*R, P])
        p_flat = tf.matmul(pred_flat, self.G)
        p = tf.reshape(p_flat, [N, R, d], name=name)
        return p

    def _ground_rel(self, rel, name='r'):
        lrel = self.l * rel
        r = tf.reduce_sum(lrel, 2, name=name)
        return r

    def __call__(self, memory):
        assert isinstance(memory, Memory)
        p = self._ground_pred(memory.pred, name='p')  # [N, R, d]
        v1 = self.sent_encoder(memory.a1, name='v1')  # [N, R, d]
        v2 = self.sent_encoder(memory.a2, name='v2')  # [N, R, d]
        p_aug = tf.expand_dims(p, 2)
        v1_aug = tf.expand_dims(v1, 2)
        v2_aug = tf.expand_dims(v2, 2)
        rel = tf.concat(2, [p_aug, v1_aug, v2_aug], name='rel')  # [N, R, 3, d]
        r = self._ground_rel(rel, name='r')  # [N, R, d]
        return r


class Layer(object):
    def __init__(self, params, memory, prev_layer=None, sent_encoder=None, u=None):
        assert isinstance(memory, Memory)
        self.params = params
        N, C, R, d = params.batch_size, params.num_choices, params.max_num_rels, params.hidden_size
        linear_start = params.linear_start

        if sent_encoder:
            input_encoder = RelationEncoder(params, sent_encoder=sent_encoder)
        else:
            input_encoder = RelationEncoder(params, rel_encoder=prev_layer.output_encoder)
        output_encoder = RelationEncoder(params)

        r = input_encoder(memory)  # [N, R, d]
        c = output_encoder(memory)  # [N, R, d]
        u = tf.identity(u or prev_layer.u + prev_layer.o, "u")  # [N, C, d]

        with tf.name_scope('p'):
            r_aug = tf.expand_dims(r, 1)  # [N, 1, R, d]
            c_aug = tf.expand_dims(c, 1)  # [N, 1, R, d]
            u_aug = tf.expand_dims(u, 2)  # [N, C, 1, d]
            ur = u_aug * r_aug  # [N, C, R, d]
            rel_mask_aug = tf.expand_dims(tf.expand_dims(memory.rel_mask, 1), -1)  # [N, 1, R, 1]
            if linear_start:
                p = tf.reduce_sum(tf.mul(ur, rel_mask_aug, name='p'), 3)  # [N, C, R]
            else:
                p = nn.softmax_with_mask([N, C, R, d], ur, rel_mask_aug)  # [N, C, R]

        with tf.name_scope('o'):
            o = tf.reduce_sum(c_aug * tf.expand_dims(p, -1), 2)  # [N, C, d]

        self.u = u
        self.o = o


class AttentionModel(BaseModel):
    def _get_l(self, J, name="l"):
        d = self.params.hidden_size
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)
        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]
        l = [g(j) for j in range(J)]
        l_tensor = tf.constant(l, shape=[J, d], name=name)
        return l_tensor

    def _build_tower(self):
        params = self.params
        V, d = params.vocab_size, params.hidden_size
        N, C, J = params.batch_size, params.num_choices, params.max_sent_size
        R, K, P = params.max_num_rels, params.max_label_size, params.pred_size

        summaries = []

        # initialize self
        # placeholders
        with tf.name_scope('ph'):
            self.s = Sentence([N, C, J], 's')
            self.m = Memory(params)
            self.y = tf.placeholder('int8', [N, C], name='y')

        with tf.name_scope('first_u'):
            B = tf.get_variable('B', dtype='float', shape=[V, d])
            sent_encoder = SentenceEncoder(B, J, d)
            first_u = sent_encoder(self.s, 'first_u')

        layers = []
        prev_layer = None
        for layer_index in xrange(params.num_layers):
            with tf.variable_scope('layer_%d' % layer_index):
                if prev_layer:
                    cur_layer = Layer(params, self.m, prev_layer=prev_layer)
                else:
                    cur_layer = Layer(params, self.m, u=first_u, sent_encoder=sent_encoder)
                layers.append(cur_layer)
                prev_layer = cur_layer
        last_layer = layers[-1]

        with tf.variable_scope('last_u'):
            last_u = tf.add(last_layer.u, last_layer.o, name='last_u')  # [N, C, d]

        with tf.variable_scope('yp'):
            self.W = tf.get_variable('W', dtype='float', shape=[d, 1])
            last_u_flat = tf.reshape(last_u, [N*C, d])
            logit_flat = tf.matmul(last_u_flat, self.W)  # [N*C, 1]
            self.logit = tf.reshape(logit_flat, [N, C])
            self.yp = tf.nn.softmax(self.logit, name='yp')

        with tf.name_scope('loss') as loss_scope:
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logit, self.y, name='cross_entropy')
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
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            self.opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

        # summaries
        summaries.append(tf.scalar_summary("%s (raw)" % self.total_loss.op.name, self.total_loss))
        self.merged_summary = tf.merge_summary(summaries)

    def _get_feed_dict(self, batch):
        sents_batch, relations_batch = batch[:-1]
        if len(batch) > 2:
            label_batch = batch[-1]
        else:
            label_batch = np.zeros([len(sents_batch)])
        s = self._prepro_sents_batch(sents_batch)  # [N, C, J], [N, C]
        m = self._prepro_relations_batch(relations_batch)
        y_batch = self._prepro_label_batch(label_batch)
        feed_dict = {self.y: y_batch}
        self.s.add(feed_dict, *s)
        self.m.add(feed_dict, *m)
        return feed_dict

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

    def _prepro_relations_batch(self, relations_batch):
        p = self.params
        N, R, K, P = p.batch_size, p.max_num_relations, p.max_label_size, p.pred_size
        rel_mask_batch = np.zeros([N, R], dtype='float')
        num_rels_batch = np.zeros([N], dtype='int16')
        pred_batch = np.zeros([N, R, P], dtype='float')
        a1_sent_batch = np.zeros([N, R, K], dtype='int32')
        a1_mask_batch = np.zeros([N, R, K], dtype='float')
        a1_len_batch = np.zeros([N, R], dtype='int16')
        a2_sent_batch = np.zeros([N, R, K], dtype='int32')
        a2_mask_batch = np.zeros([N, R, K], dtype='float')
        a2_len_batch = np.zeros([N, R], dtype='int16')

        for n, relations in enumerate(relations_batch):
            num_rels_batch[n] = len(relations)
            for r, relation in enumerate(relations):
                rel_mask_batch[n, r] = 1.0
                for k, idx in enumerate(relation['a1']):
                    a1_sent_batch[n, r, k] = idx
                    a1_mask_batch[n, r, k] = 1.0
                a1_len_batch[n, r] = len(relation['a1'])

                pred_batch[n, r] = np.array(relation['pred'])

                for k, idx in enumerate(relation['a2']):
                    a2_sent_batch[n, r, k] = idx
                    a2_mask_batch[n, r, k] = 1.0
                a2_len_batch[n, r] = len(relation['a2'])

        a1 = (a1_sent_batch, a1_mask_batch, a1_len_batch)
        a2 = (a2_sent_batch, a2_mask_batch, a2_len_batch)

        return rel_mask_batch, num_rels_batch, pred_batch, a1, a2

    def _prepro_label_batch(self, label_batch):
        p = self.params
        N, C = p.batch_size, p.num_choices
        y = np.zeros([N, C], dtype='int8')
        for label in label_batch:
            y[label] = 1
        return y
