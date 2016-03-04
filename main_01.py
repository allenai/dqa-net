import json
from copy import deepcopy
from pprint import pprint
import os

import tensorflow as tf

from models.attention_model_01 import AttentionModel
from read_data import read_data

flags = tf.app.flags

# File directories
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("save_dir", "save", "Save directory [save]")
flags.DEFINE_string("train_data_dir", 'data/1500r-train', "Train data directory [data/1500-train]")
flags.DEFINE_string("val_data_dir", 'data/1500r-test', "Val data directory [data/1500-test]")
flags.DEFINE_string("test_data_dir", 'data/1500r-test', "Test data directory [data/1500-test]")

# Training parameters
flags.DEFINE_integer("batch_size", 100, "Batch size for the network [100]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [50]")
flags.DEFINE_integer("num_layers", 1, "Number of layers [1]")
flags.DEFINE_integer("rnn_num_layers", 1, "Number of rnn layers [1]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 0.1, "Initial weight std [0.1]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_integer("anneal_period", 10, "Anneal period [10]")
flags.DEFINE_float("anneal_ratio", 0.5, "Anneal ratio [0.5")
flags.DEFINE_integer("num_epochs", 100, "Total number of epochs for training [100]")
flags.DEFINE_boolean("linear_start", False, "Start training with linear model? [False]")
flags.DEFINE_float("max_grad_norm", 40, "Max grad norm; above this number is clipped [40")

# Training and testing options
flags.DEFINE_boolean("train", False, "Train? Test if False [False]")
flags.DEFINE_integer("val_num_batches", 5, "Val num batches [5]")
flags.DEFINE_boolean("load", False, "Load from saved model? [False]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_boolean("gpu", False, 'Enable GPU? (Linux only) [False]')
flags.DEFINE_integer("val_period", 5, "Val period (for display purpose only) [5]")
flags.DEFINE_integer("save_period", 10, "Save period [10]")

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick build) [False]")

# App-specific training parameters
# TODO : Any other parameters

# App-specific options
# TODO : Any other options

FLAGS = flags.FLAGS


def combine_meta_data(train_meta_data, val_meta_data):
    new_meta_data = deepcopy(train_meta_data)
    new_meta_data['max_sent_size'] = max(train_meta_data['max_sent_size'], val_meta_data['max_sent_size'])
    new_meta_data['max_label_size'] = max(train_meta_data['max_label_size'], val_meta_data['max_label_size'])
    new_meta_data['max_num_rels'] = max(train_meta_data['max_num_rels'], val_meta_data['max_num_rels'])
    return new_meta_data

def main(_):
    if FLAGS.train:
        train_ds = read_data('train', FLAGS, FLAGS.train_data_dir)
        val_ds = read_data('val', FLAGS, FLAGS.val_data_dir)
        FLAGS.train_num_batches = train_ds.num_batches
        FLAGS.val_num_batches = FLAGS.val_num_batches
        train_meta_data_path = os.path.join(FLAGS.train_data_dir, "meta_data.json")
        train_meta_data = json.load(open(train_meta_data_path, "rb"))
        val_meta_data_path = os.path.join(FLAGS.val_data_dir, "meta_data.json")
        val_meta_data = json.load(open(val_meta_data_path, "rb"))
        meta_data = combine_meta_data(train_meta_data, val_meta_data)
        vocab_path = os.path.join(FLAGS.train_data_dir, "vocab.json")
        if not os.path.exists(FLAGS.save_dir):
            os.mkdir(FLAGS.save_dir)
    else:
        test_ds = read_data('test', FLAGS, FLAGS.test_data_dir)
        FLAGS.test_num_batches = test_ds.num_batches
        meta_data_path = os.path.join(FLAGS.test_data_dir, "meta_data.json")
        meta_data = json.load(open(meta_data_path, "rb"))
        vocab_path = os.path.join(FLAGS.test_data_dir, "vocab.json")

    # Other parameters
    FLAGS.max_sent_size = meta_data['max_sent_size']
    FLAGS.max_label_size = meta_data['max_label_size']
    FLAGS.max_num_rels = meta_data['max_num_rels']
    FLAGS.pred_size = meta_data['pred_size']
    FLAGS.num_choices = meta_data['num_choices']
    vocab = json.load(open(vocab_path, "rb"))
    FLAGS.vocab_size = len(vocab)

    # For quick draft build (deubgging).
    if FLAGS.draft:
        FLAGS.train_num_batches = 1
        FLAGS.val_num_batches = 1
        FLAGS.test_num_batches = 1
        FLAGS.num_epochs = 1
        FLAGS.eval_period = 1
        FLAGS.save_period = 1
        # TODO : Add any other parameter that induces a lot of computations
        FLAGS.num_layers = 1

    pprint(FLAGS.__flags)

    graph = tf.Graph()
    model = AttentionModel(graph, FLAGS)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.train:
            writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph_def)
            if FLAGS.load:
                model.load(sess)
            model.train(sess, writer, train_ds, val_ds)
        else:
            model.load(sess)
            model.eval(sess, test_ds)

if __name__ == "__main__":
    tf.app.run()
