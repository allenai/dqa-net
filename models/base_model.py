import os

import numpy as np
import tensorflow as tf
import progressbar as pb

from read_data import DataSet


class BaseModel(object):
    def __init__(self, graph, params, name=None):
        self.graph = graph
        self.params = params
        self.save_dir = params.save_dir
        self.name = name or self.__class__.__name__
        self.initializer = tf.random_normal_initializer(params.init_mean, params.init_std)
        with graph.as_default(), tf.variable_scope(self.name, initializer=self.initializer):
            print("building %s tower ..." % self.name)
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.learning_rate = tf.placeholder('float32', name='lr')
            self.opt_op = None
            self.correct_vec = None
            self.total_loss = None
            self.merged_summary = None
            self._build_tower()
            self.saver = tf.train.Saver()

    def _build_tower(self):
        # TODO : Implement this! Either here or by creating a child class.
        raise Exception("Implement this!")

    def _get_feed_dict(self, batch):
        # TODO : Implement this!
        raise Exception("Implement this!")

    def train_batch(self, sess, learning_rate, batch):
        feed_dict = self._get_feed_dict(batch)
        feed_dict[self.learning_rate] = learning_rate
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        actual_batch_size = len(batch[0])
        feed_dict = self._get_feed_dict(batch)
        correct_vec, total_loss, summary_str, global_step = \
            sess.run([self.correct_vec, self.total_loss, self.merged_summary, self.global_step], feed_dict=feed_dict)
        num_corrects = np.sum(correct_vec[:actual_batch_size])

        return num_corrects, total_loss, summary_str, global_step

    def train(self, sess, writer, train_data_set, val_data_set):
        assert isinstance(train_data_set, DataSet)
        assert isinstance(val_data_set, DataSet)
        params = self.params
        learning_rate = params.init_lr
        num_epochs = params.num_epochs
        num_batches = params.train_num_batches
        anneal_period = params.anneal_period
        anneal_ratio = params.anneal_ratio

        print("training %d epochs ..." % num_epochs)
        for epoch_idx in xrange(num_epochs):
            if epoch_idx > 0 and epoch_idx % anneal_period == 0:
                learning_rate *= anneal_ratio
            pbar = pb.ProgressBar(widgets=["epoch %d|" % (train_data_set.num_epochs_completed + 1),
                                           pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
            pbar.start()
            for num_batches_completed in xrange(num_batches):
                batch = train_data_set.get_next_labeled_batch()
                _, summary_str, global_step = self.train_batch(sess, learning_rate, batch)
                writer.add_summary(summary_str, global_step)
                pbar.update(num_batches_completed)
            pbar.finish()
            train_data_set.complete_epoch()

            if val_data_set and (epoch_idx + 1) % params.val_period == 0:
                self.eval(sess, train_data_set, is_val=True)
                self.eval(sess, val_data_set, is_val=True)

            if (epoch_idx + 1) % params.save_period == 0:
                self.save(sess)
        print("training done.")

    def eval(self, sess, eval_data_set, is_val=False):
        params = self.params
        if is_val:
            eval_data_set.reset()
            num_batches = params.val_num_batches
        else:
            num_batches = params.test_num_batches
        num_corrects, total = 0, 0
        string = "%s:N=%d|" % (eval_data_set.name, eval_data_set.batch_size * num_batches)
        pbar = pb.ProgressBar(widgets=[string, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
        losses = []
        pbar.start()
        for num_batches_completed in xrange(num_batches):
            batch = eval_data_set.get_next_labeled_batch()
            cur_num_corrects, cur_loss, _, global_step = self.test_batch(sess, batch)
            num_corrects += cur_num_corrects
            total += len(batch[0])
            losses.append(cur_loss)
            pbar.update(num_batches_completed)
        pbar.finish()
        loss = np.mean(losses)

        print("at %d: acc = %.2f%% = %d / %d, loss = %.4f" %
              (global_step, 100 * float(num_corrects)/total, num_corrects, total, loss))

    def save(self, sess):
        print("saving model ...")
        save_path = os.path.join(self.save_dir, self.name)
        self.saver.save(sess, save_path, self.global_step)

    def load(self, sess):
        print("loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

