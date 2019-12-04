import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr
import logging
logger = logging.getLogger(__name__)
import os


from evaluation.BaseEndTask import BaseEndTask
from iterators.DataIterator import DataIterator
from utils.batch_runner import batch_runner
from utils.time_tool import year2dec

class SynchronicTask(BaseEndTask):
    def __init__(self, args):
        super().__init__(args)
        self.MEN_location = args.MEN_location
        self.test_year = 1995
        self.start_year = args.start_year
        self.end_year = args.end_year
        self.test_dec = year2dec(self.test_year, self.start_year, self.end_year)
        self.MEN_triples = []
        self.MEN_triples_reduced = []
        self.MEN_triples_indices = []
        self._load_data()
        self.eval_batch_size = args.eval_batch_size

    def _load_data(self):
        with open(os.path.join(self.MEN_location, "EN-MEN-TR-3k.txt"), "r") as fold_data:
            for line in fold_data:
                w1, w2, value = line.rstrip().split()
                self.MEN_triples.append((w1, w2, float(value)))

    def modify_data(self, word2id, word_counts):
        missing_triples_flag = False
        for w1, w2, score in self.MEN_triples:
            if w1 not in word2id or w2 not in word2id:
                if not missing_triples_flag:
                    logger.warning("MEN missing triples from data vocab:")
                    logger.warning("")
                    missing_triples_flag = True
                w1_symbol = "O" if w1 in word2id else "X"
                w2_symbol = "O" if w2 in word2id else "X"
                logger.warning("Word 1: {0} ({1}) Word 2: {2} ({3}) Gold score: {4}".format(w1, w1_symbol, w2, w2_symbol, score))
            else:
                self.MEN_triples_reduced.append((w1, w2, score))
                self.MEN_triples_indices.append((word2id[w1], word2id[w2], score))
        if missing_triples_flag:
            logger.warning("")


    def evaluate(self, sess, model):


        targets_placeholder = tf.placeholder(tf.int32, shape=(self.eval_batch_size,))
        synonym_placeholder = tf.placeholder(tf.int32, shape=(self.eval_batch_size,))
        times_placeholder = tf.placeholder(tf.float32, shape=(self.eval_batch_size,))

        target_vector = model.get_target_vector(targets_placeholder, times_placeholder)
        target_vector = tf.nn.l2_normalize(target_vector, 1)
        syns_vector = model.get_target_vector(synonym_placeholder, times_placeholder)
        syns_vector = tf.nn.l2_normalize(syns_vector, 1)
        cosine_tensor = tf.reduce_sum(tf.multiply(target_vector, syns_vector), axis=1)

        # for x, y in zip(self.MEN_triples_reduced, self.MEN_triples_indices):
        #     print(x, y)
        # exit()

        data_dict = {}
        data_dict[targets_placeholder] = [x[0] for x in self.MEN_triples_indices]
        data_dict[synonym_placeholder] = [x[1] for x in self.MEN_triples_indices]
        data_dict[times_placeholder] = len(data_dict[targets_placeholder])*[self.test_dec]

        pred_scores = batch_runner(sess, model, self.eval_batch_size, cosine_tensor, data_dict)
        gold_scores = [x[2] for x in self.MEN_triples_indices]

        for (w1, w2, gold_score), pred_score in zip(self.MEN_triples_reduced, pred_scores):
            print(w1, w2, gold_score, pred_score)
        print()

        rho, pvalue = spearmanr(pred_scores, gold_scores)

        print("Synchronic Task Results")
        print()
        print("Correlation: {0:.3f}".format(rho))
        print("p-value: {0:.3f}".format(pvalue))
        print()

