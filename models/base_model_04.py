import json
import os

import itertools
import numpy as np
import tensorflow as tf

from read_data.r04 import DataSet
from utils import get_pbar


class BaseModel(object):
    def __init__(self, graph, params, name=None):
        self.graph = graph
        self.params = params
        self.save_dir = params.save_dir
        self.name = name or self.__class__.__name__
        self.initializer = tf.random_normal_initializer(params.init_mean, params.init_std)
        with graph.as_default(), tf.variable_scope(self.name, initializer=self.initializer):
            self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.num_epochs_completed = tf.get_variable('epoch', shape=[], dtype='int32',
                                                        initializer=tf.constant_initializer(0), trainable=False)
            self.learning_rate = tf.placeholder('float32', name='learning_rate')
            self.is_train = tf.placeholder('bool', name='is_train')  # build different network for train and test
            self.opt_op = None
            self.correct_vec = None
            self.total_loss = None
            self.merged_summary = None
            print("building %s tower ..." % self.name)
            self._build_tower()
            print("building done.")
            self.saver = tf.train.Saver()

    def _build_tower(self):
        # TODO : Implement this! Either here or by creating a child class.
        raise Exception("Implement this!")

    def _get_feed_dict(self, batch, train, **kwargs):
        # TODO : Implement this!
        raise Exception("Implement this!")

    def _get_train_args(self, epoch_idx):
        # TODO : Implement this!
        raise Exception("Implement this!")

    def train_batch(self, sess, batch, **kwargs):
        feed_dict = self._get_feed_dict(batch, 'train', **kwargs)
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def eval_batch(self, sess, batch, eval_tensors=None):
        actual_batch_size = len(batch[0])
        feed_dict = self._get_feed_dict(batch, 'eval')
        correct_vec, total_loss, summary_str, global_step = \
            sess.run([self.correct_vec, self.total_loss, self.merged_summary, self.global_step], feed_dict=feed_dict)
        num_corrects = np.sum(correct_vec[:actual_batch_size])
        values = sess.run(eval_tensors, feed_dict=feed_dict) if eval_tensors else []

        return (num_corrects, total_loss, summary_str, global_step), values

    def train(self, sess, writer, train_data_set, val_data_set, eval_tensors=None):
        assert isinstance(train_data_set, DataSet), train_data_set.__class__.__name__
        assert isinstance(val_data_set, DataSet), train_data_set.__class__.__name__

        params = self.params
        num_epochs = params.num_epochs
        num_batches = params.train_num_batches if params.train_num_batches >= 0 else train_data_set.num_batches
        num_digits = int(np.log10(num_batches))

        epoch = sess.run(self.num_epochs_completed)
        print("training %d epochs ... " % num_epochs)
        print("starting from epoch %d." % (epoch+1))
        while epoch < num_epochs:
            train_args = self._get_train_args(epoch)
            pbar = get_pbar(num_batches, "epoch %s|" % str(epoch+1).zfill(num_digits)).start()
            for num_batches_completed in range(num_batches):
                batch = train_data_set.get_next_labeled_batch()
                _, summary_str, global_step = self.train_batch(sess, batch, **train_args)
                writer.add_summary(summary_str, global_step)
                pbar.update(num_batches_completed)
            pbar.finish()
            train_data_set.complete_epoch()

            assign_op = self.num_epochs_completed.assign_add(1)
            _, epoch = sess.run([assign_op, self.num_epochs_completed])

            if val_data_set and epoch % params.val_period == 0:
                eval_tensors = eval_tensors if eval_tensors else []
                self.eval(sess, train_data_set, is_val=True, eval_tensors=eval_tensors)
                self.eval(sess, val_data_set, is_val=True, eval_tensors=eval_tensors)

            if epoch % params.save_period == 0:
                self.save(sess)

        print("training done.")

    def eval(self, sess, eval_data_set, is_val=False, eval_tensors=None):
        eval_names = [os.path.basename(tensor.name) for tensor in eval_tensors]
        params = self.params
        # We are using num_batches form params instead of eval_data_set's,
        # because we might want to evaluate only a few batches, not all batches.
        if is_val:
            num_batches = params.val_num_batches if params.val_num_batches >= 0 else eval_data_set.num_batches
        else:
            num_batches = params.test_num_batches if params.test_num_batches >= 0 else eval_data_set.num_batches
        num_corrects, total = 0, 0
        eval_values = []
        idxs= []
        losses = []
        string = "eval on %s, N=%d|" % (eval_data_set.name, eval_data_set.batch_size * num_batches)
        pbar = get_pbar(num_batches, prefix=string).start()
        for num_batches_completed in range(num_batches):
            idxs.extend(eval_data_set.get_batch_idxs())
            batch = eval_data_set.get_next_labeled_batch()
            (cur_num_corrects, cur_loss, _, global_step), eval_value_batch = self.eval_batch(sess, batch, eval_tensors=eval_tensors)
            num_corrects += cur_num_corrects
            total += len(batch[0])
            eval_values.append([x.tolist() for x in eval_value_batch])  # numpy.array.toList
            losses.append(cur_loss)
            pbar.update(num_batches_completed)
        pbar.finish()
        loss = np.mean(losses)
        eval_data_set.reset()

        ids = [eval_data_set.idx2id[idx] for idx in idxs]
        zipped_eval_values = [list(itertools.chain(*each)) for each in zip(*eval_values)]
        values = {name: values for name, values in zip(eval_names, zipped_eval_values)}
        out = {'ids': ids, 'values': values}
        epoch = sess.run(self.num_epochs_completed)
        eval_path = os.path.join(params.eval_dir, "%s_%s.json" % (eval_data_set.name, str(epoch).zfill(4)))
        json.dump(out, open(eval_path, 'w'))

        print("at epoch %d: acc = %.2f%% = %d / %d, loss = %.4f" %
              (epoch, 100 * float(num_corrects)/total, num_corrects, total, loss))

    def save(self, sess):
        print("saving model ...")
        save_path = os.path.join(self.save_dir, self.name)
        self.saver.save(sess, save_path, self.global_step)
        print("saving done.")

    def load(self, sess):
        print("loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        assert checkpoint is not None, "Cannot load checkpoint at %s" % self.save_dir
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
        print("loading done.")
