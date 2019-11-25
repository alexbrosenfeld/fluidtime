import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr

from evaluation.BaseEndTask import BaseEndTask
from iterators.DataIterator import DataIterator
from utils.batch_runner import batch_runner

class SynchronicTask(BaseEndTask):
    def __init__(self, args):
        super().__init__(args)
        self.test_year = 1995
        self.MEN_triples = []
        self.MEN_triples_reduced = []
        self.MEN_triples_indices = []
        self._load_data()

    def _load_data(self):
        with open("datasets/synchronic_task/EN-MEN-TR-3k.txt", "r") as fold_data:
            for line in fold_data:
                w1, w2, value = line.rstrip().split()
                self.MEN_triples.append((w1, w2, float(value)))

    def modify_data(self, word2id):
        for w1, w2, score in self.MEN_triples:
            if w1 in word2id and w2 in word2id:
                #TODO: add warnings of missing words
                self.MEN_triples_reduced.append((w1, w2, score))
                self.MEN_triples_indices.append((word2id[w1], word2id[w2], score))

    def evaluate(self, sess, model):


        targets_placeholder = tf.placeholder(tf.int32, shape=(self.args.eval_batch_size,))
        synonym_placeholder = tf.placeholder(tf.int32, shape=(self.args.eval_batch_size,))
        times_placeholder = tf.placeholder(tf.float32, shape=(self.args.eval_batch_size,))

        target_vector = model.get_target_vector(targets_placeholder, times_placeholder)
        target_vector = tf.nn.l2_normalize(target_vector, 1)
        syns_vector = model.get_target_vector(synonym_placeholder, times_placeholder)
        syns_vector = tf.nn.l2_normalize(syns_vector, 1)
        cosine_tensor = tf.reduce_sum(tf.multiply(target_vector, syns_vector), axis=1)

        data_dict = {}
        data_dict[targets_placeholder] = [x[0] for x in self.MEN_triples_indices]
        data_dict[synonym_placeholder] = [x[1] for x in self.MEN_triples_indices]
        data_dict[times_placeholder] = len(data_dict[targets_placeholder])*[(self.test_year - 1900.0)/(2009.0-1900.0)]

        pred_scores = batch_runner(sess, model, self.args.eval_batch_size, cosine_tensor, data_dict, self.args)
        gold_scores = [x[2] for x in self.MEN_triples_indices]

        rho, pvalue = spearmanr(pred_scores, gold_scores)

        print("Synchronic Task Results")
        print()
        print("Correlation: {0:.3f}".format(rho))
        print("p-value: {0:.3f}".format(pvalue))
        print()

