import json
import os

import itertools
import numpy as np
import progressbar as pb
import tensorflow as tf

from read_data.r04 import DataSet


class BaseModel(object):
    def __init__(self, graph, params, name=None):
        self.graph = graph
        self.params = params
        self.save_dir = params.save_dir
        self.name = name or self.__class__.__name__
        self.initializer = tf.random_normal_initializer(params.init_mean, params.init_std)
        self.num_epochs_completed = 0
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
        self.num_epochs_completed += 1
        return sess.run([self.opt_op, self.merged_summary, self.global_step], feed_dict=feed_dict)

    def eval_batch(self, sess, batch, eval_tensors=None):
        actual_batch_size = len(batch[0])
        feed_dict = self._get_feed_dict(batch)
        correct_vec, total_loss, summary_str, global_step = \
            sess.run([self.correct_vec, self.total_loss, self.merged_summary, self.global_step], feed_dict=feed_dict)
        num_corrects = np.sum(correct_vec[:actual_batch_size])
        values = sess.run(eval_tensors, feed_dict=feed_dict) if eval_tensors else []

        return (num_corrects, total_loss, summary_str, global_step), values

    def train(self, sess, writer, train_data_set, val_data_set, eval_tensors=None):
        assert isinstance(train_data_set, DataSet), train_data_set.__class__.__name__
        assert isinstance(val_data_set, DataSet), train_data_set.__class__.__name__
        params = self.params
        learning_rate = params.init_lr
        num_epochs = params.num_epochs
        num_batches = params.train_num_batches
        anneal_period = params.anneal_period
        anneal_ratio = params.anneal_ratio

        print("training %d epochs ..." % num_epochs)
        for epoch_idx in range(num_epochs):
            if epoch_idx > 0 and epoch_idx % anneal_period == 0:
                learning_rate *= anneal_ratio
            pbar = pb.ProgressBar(widgets=["epoch %d|" % (train_data_set.num_epochs_completed + 1),
                                           pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
            pbar.start()
            for num_batches_completed in range(num_batches):
                batch = train_data_set.get_next_labeled_batch()
                _, summary_str, global_step = self.train_batch(sess, learning_rate, batch)
                writer.add_summary(summary_str, global_step)
                pbar.update(num_batches_completed)
            pbar.finish()
            train_data_set.complete_epoch()

            if val_data_set and (epoch_idx + 1) % params.val_period == 0:
                eval_tensors = eval_tensors if eval_tensors else []
                self.eval(sess, train_data_set, is_val=True, eval_tensors=eval_tensors)
                self.eval(sess, val_data_set, is_val=True, eval_tensors=eval_tensors)

            if (epoch_idx + 1) % params.save_period == 0:
                self.save(sess)
        print("training done.")

    def eval(self, sess, eval_data_set, is_val=False, eval_tensors=None):
        eval_names = [os.path.basename(tensor.name) for tensor in eval_tensors]
        params = self.params
        """
        if is_val:
            num_batches = params.val_num_batches
        else:
            num_batches = params.test_num_batches
        """
        num_batches = eval_data_set.num_batches
        num_corrects, total = 0, 0
        eval_values = []
        idxs= []
        string = "%s:N=%d|" % (eval_data_set.name, eval_data_set.batch_size * num_batches)
        pbar = pb.ProgressBar(widgets=[string, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_batches)
        losses = []
        pbar.start()
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
        eval_path = os.path.join(params.eval_dir, "%s_%s.json" % (eval_data_set.name, str(self.num_epochs_completed).zfill(4)))
        json.dump(out, open(eval_path, 'w'))

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

