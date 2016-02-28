import tensorflow as tf
import numpy as np

from models.base_model import BaseModel


class Container(object):
    pass


class MemoryLayer(object):
    def __init__(self, params, prev_layer, phs, consts, tensors):
        self.params = params
        N, M, J, V, d = params.batch_size, params.memory_size, params.max_sent_size, params.vocab_size, params.hidden_size
        linear_start = params.linear_start

        x_batch, x_mask_aug_batch, m_mask_batch = phs.x_batch, phs.x_mask_aug_batch, phs.m_mask_batch
        l_aug_aug = consts.l_aug_aug

        B, first_u_batch = tensors.B, tensors.first_u_batch

        if not prev_layer:
            A = tf.identity(B, name='A') if params.tying == 'adj' else tf.get_variable('A', dtype='float', shape=[V, d])
            TA = tf.get_variable('TA', dtype='float', shape=[M, d])
            C = tf.get_variable('C', dtype='float', shape=[V, d])
            TC = tf.get_variable('TC', dtype='float', shape=[M, d])
        else:
            if params.tying == 'adj':
                A = tf.identity(prev_layer.C, name='A')
                TA = tf.identity(prev_layer.TC, name='TA')
                C = tf.get_variable('C', dtype='float', shape=[V, d])
                TC = tf.get_variable('TC', dtype='float', shape=[M, d])
            elif params.tying == 'rnn':
                A = tf.identity(prev_layer.A, name='A')
                TA = tf.identity(prev_layer.TA, name='TA')
                C = tf.identity(prev_layer.C, name='C')
                TC = tf.identity(prev_layer.TC, name='TC')
            else:
                raise Exception('Unknown tying method: %s' % params.tying)

        if not prev_layer:
            u_batch = tf.identity(tensors.first_u_batch, name='u')
        else:
            u_batch = tf.add(prev_layer.u_batch, prev_layer.o_batch, name='u')

        with tf.name_scope('m'):
            Ax_batch = tf.nn.embedding_lookup(A, x_batch)  # [N, M, J, d]
            if params.position_encoding:
                Ax_batch *= l_aug_aug  # position encoding
            Ax_batch *= x_mask_aug_batch  # masking
            m_batch = tf.reduce_sum(Ax_batch, 2)  # [N, M, d]
            m_batch = tf.add(tf.expand_dims(TA, 0), m_batch, name='m')  # temporal encoding

        with tf.name_scope('c'):
            Cx_batch = tf.nn.embedding_lookup(C, x_batch)  # [N, M, J, d]
            if params.position_encoding:
                Cx_batch *= l_aug_aug  # position encoding
            Cx_batch *= x_mask_aug_batch
            c_batch = tf.reduce_sum(Cx_batch, 2)
            c_batch = tf.add(tf.expand_dims(TC, 0), c_batch, name='c')  # temporal encoding

        with tf.name_scope('p'):
            u_batch_aug = tf.expand_dims(u_batch, -1)  # [N, d, 1]
            um_batch = tf.squeeze(tf.batch_matmul(m_batch, u_batch_aug), [2])  # [N, M]
            if linear_start:
                p_batch = tf.mul(um_batch, m_mask_batch, name='p')
            else:
                p_batch = self._softmax_with_mask(um_batch, m_mask_batch)

        with tf.name_scope('o'):
            o_batch = tf.reduce_sum(c_batch * tf.expand_dims(p_batch, -1), 1)  # [N, d]

        self.A, self.TA, self.C, self.TC = A, TA, C, TC
        self.u_batch, self.o_batch = u_batch, o_batch

    def _softmax_with_mask(self, um_batch, m_mask_batch):
        exp_um_batch = tf.exp(um_batch)  # [N, M]
        masked_batch = exp_um_batch * m_mask_batch  # [N, M]
        sum_2d_batch = tf.expand_dims(tf.reduce_sum(masked_batch, 1), -1)  # [N, 1]
        p_batch = tf.div(masked_batch, sum_2d_batch, name='p')  # [N, M]
        return p_batch


class MemNNModel(BaseModel):
    def _get_l(self):
        J, d = self.params.max_sent_size, self.params.hidden_size
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)
        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]
        l = [g(j) for j in range(J)]
        l_tensor = tf.constant(l, shape=[J, d], name='l')
        return l_tensor

    def _softmax_with_mask(self, um_batch, m_mask_batch):
        exp_um_batch = tf.exp(um_batch)  # [N, M]
        masked_batch = exp_um_batch * m_mask_batch  # [N, M]
        sum_2d_batch = tf.expand_dims(tf.reduce_sum(masked_batch, 1), -1)  # [N, 1]
        p_batch = tf.div(masked_batch, sum_2d_batch, name='p')  # [N, M]
        return p_batch

    def _build_tower(self):
        params = self.params
        linear_start = params.linear_start
        N, M, J, V, d = params.batch_size, params.memory_size, params.max_sent_size, params.vocab_size, params.hidden_size

        summaries = []

        # initialize self
        # placeholders
        with tf.name_scope('ph'):
            with tf.name_scope('x'):
                x_batch = tf.placeholder('int32', shape=[N, M, J], name='x')
                x_mask_batch = tf.placeholder('float', shape=[N, M, J], name='x_mask')
                x_mask_aug_batch = tf.expand_dims(x_mask_batch, -1, 'x_mask_aug')
                m_mask_batch = tf.placeholder('float', shape=[N, M], name='m_mask')

            with tf.name_scope('q'):
                q_batch = tf.placeholder('int32', shape=[N, J], name='q')
                q_mask_batch = tf.placeholder('float', shape=[N, J], name='q_mask')
                q_mask_aug_batch = tf.expand_dims(q_mask_batch, -1, 'q_mask_aug')

            y_batch = tf.placeholder('int32', shape=[N], name='y')

            learning_rate = tf.placeholder('float', name='lr')

        with tf.name_scope('const'):
            l = self._get_l()  # [J, d]
            l_aug = tf.expand_dims(l, 0, name='l_aug')
            l_aug_aug = tf.expand_dims(l_aug, 0, name='l_aug_aug')  # [1, 1, J, d]


        with tf.name_scope('a'):
            a_batch = tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[V])), y_batch, name='a')  # [N, d]


        with tf.name_scope('first_u'):
            B = tf.get_variable('B', dtype='float', shape=[V, d])
            Bq_batch = tf.nn.embedding_lookup(B, q_batch)  # [N, J, d]
            if params.position_encoding:
                Bq_batch *= l_aug
            Bq_batch *= q_mask_aug_batch
            first_u_batch = tf.reduce_sum(Bq_batch, 1, name='first_u')  # [N, d]

        phs, consts, tensors = Container(), Container(), Container()
        phs.x_batch, phs.x_mask_batch, phs.x_mask_aug_batch, phs.m_mask_batch = x_batch, x_mask_batch, x_mask_aug_batch, m_mask_batch
        consts.l_aug_aug = l_aug_aug
        tensors.B, tensors.first_u_batch = B, first_u_batch

        memory_layers = []
        cur_layer = None
        for layer_index in xrange(params.num_layers):
            with tf.variable_scope('layer_%d' % layer_index):
                memory_layer = MemoryLayer(params, cur_layer, phs, consts, tensors)
                memory_layers.append(memory_layer)
                cur_layer = memory_layer

        with tf.variable_scope('last_u'):
            if params.tying == 'rnn':
                H = tf.get_variable('H', dtype='float', shape=[d, d])
                last_u_batch = tf.add(tf.matmul(cur_layer.u_batch, H), cur_layer.o_batch, name='last_u')
            else:
                last_u_batch = tf.add(cur_layer.u_batch, cur_layer.o_batch, name='last_u')

        with tf.variable_scope('ap'):
            if params.tying == 'adj':
                W = tf.transpose(cur_layer.C, name='W')
            elif params.tying == 'rnn':
                W = tf.get_variable('W', dtype='float', shape=[d, V])
            else:
                raise Exception('Unknown tying method: %s' % params.tying)
            logit_batch = tf.matmul(last_u_batch, W, name='logit')  # [N d] X [d V] = [N V]
            ap_batch = tf.nn.softmax(logit_batch, name='ap')

        with tf.name_scope('loss') as loss_scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_batch, a_batch, name='cross_entropy')
            avg_cross_entropy = tf.reduce_mean(cross_entropy, 0, name='avg_cross_entropy')
            tf.add_to_collection('losses', avg_cross_entropy)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            losses = tf.get_collection('losses', loss_scope)

        with tf.name_scope('acc'):
            correct_vec = tf.equal(tf.argmax(ap_batch, 1), tf.argmax(a_batch, 1))
            num_corrects = tf.reduce_sum(tf.cast(correct_vec, 'float'), name='num_corrects')
            acc = tf.reduce_mean(tf.cast(correct_vec, 'float'), name='acc')

        with tf.name_scope('opt'):
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            # FIXME : This must muse cross_entropy for some reason!
            grads_and_vars = opt.compute_gradients(cross_entropy)
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

        # placeholders
        self.x = x_batch
        self.x_mask = x_mask_batch
        self.m_mask = m_mask_batch
        self.q = q_batch
        self.q_mask = q_mask_batch
        self.y = y_batch
        self.learning_rate = learning_rate

        # tensors
        self.total_loss = total_loss
        self.correct_vec = correct_vec
        self.num_corrects = num_corrects
        self.acc = acc
        self.opt_op = opt_op

        # summaries
        summaries.append(tf.scalar_summary("%s (raw)" % total_loss.op.name, total_loss))
        self.merged_summary = tf.merge_summary(summaries)

    def _get_feed_dict(self, batch):
        sents_batch, relations_batch = batch[:-1]
        if len(batch) > 2:
            label_batch = batch[-1]
        else:
            label_batch = np.zeros([len(sents_batch)])
        s_batch, s_len_batch = self._prepro_sents_batch(sents_batch)  # [N, C, J], [N, C]
        a1_batch, a1_len_batch, pred_batch, num_rel_batch, a2_batch, a2_len_batch = \
            self._prepro_relations_batch(relations_batch)  # [N, R, K], [N, R], [N, R, P], [N], [N, R, K], [N, R]
        y_batch = self._prepro_label_batch(label_batch)
        feed_dict = {self.s_batch: s_batch, self.s_len_batch: s_len_batch,
                     self.a1_batch: a1_batch, self.a1_len_batch: a1_len_batch,
                     self.pred_batch: pred_batch, self.num_rel_batch: num_rel_batch,
                     self.a2_batch: a2_batch, self.a2_len_batch: a2_len_batch,
                     self.y_batch: y_batch}
        return feed_dict

    def _prepro_sents_batch(self, sents_batch):
        p = self.params
        N, C, J = p.batch_size, p.num_choices, p.max_sent_size
        s_batch = np.zeros([N, C, J], dtype='int32')
        s_len_batch = np.zeros([N, C], dtype='int16')
        for n, sents in enumerate(sents_batch):
            for c, sent in enumerate(sents):
                for j, idx in enumerate(sent):
                    s_batch[n, c, j] = idx
                s_len_batch[n, c] = len(sent)
        return s_batch, s_len_batch

    def _prepro_relations_batch(self, relations_batch):
        p = self.params
        N, R, K, P = p.batch_size, p.max_num_relations, p.max_label_size, p.pred_size
        a1_batch = np.zeros([N, R, K], dtype='int32')
        a1_len_batch = np.zeros([N, R], dtype='int16')
        pred_batch = np.zeros([N, R, P])
        num_rel_batch = np.zeros([N], dtype='int16')
        a2_batch = np.zeros([N, R, K], dtype='int32')
        a2_len_batch = np.zeros([N, R], dtype='int16')

        for n, relations in enumerate(relations_batch):
            num_rel_batch[n] = len(relations)
            for r, relation in enumerate(relations):
                for k, idx in enumerate(relation['a1']):
                    a1_batch[n, r, k] = idx
                a1_len_batch[n, r] = len(relation['a1'])

                pred_batch[n, r] = np.array(relation['pred'])

                for k, idx in enumerate(relation['a2']):
                    a2_batch[n, r, k] = idx
                a2_len_batch[n, r] = len(relation['a2'])

        return a1_batch, a1_len_batch, pred_batch, num_rel_batch, a2_batch, a2_len_batch

    def _prepro_label_batch(self, label_batch):
        p = self.params
        N, C = p.batch_size, p.num_choices
        y = np.zeros([N, C], dtype='int8')
        for label in label_batch:
            y[label] = 1
        return y