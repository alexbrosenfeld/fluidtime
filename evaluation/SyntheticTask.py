import numpy as np


class Datum:
    def __init__(self, w1, w2, m, s):
        self.w1 = w1
        self.w2 = w2
        self.m = m
        self.s = s
        self.w1_syns = None
        self.w2_syns = None
        self.w1_index = None
        self.w2_index = None
        self.w1_syns_indices = {}
        self.w2_syns_indices = {}
        self.form_sigmoids()

    def form_sigmoids(self):
        self.w1_probs = {}
        self.w2_probs = {}

        years = np.arange(1900, 2010)

        self.w1_prob_array = np.reciprocal(1 + np.exp(-self.s * (years - float(self.m))))
        self.w2_prob_array = 1.0 - self.w1_prob_array

        for yind, year in enumerate(range(1900, 2010)):
            self.w1_probs[year] = self.w1_prob_array[yind]
            self.w2_probs[year] = self.w2_prob_array[yind]


class SyntheticTask:
    def __init__(self, fold):
        self.fold = fold
        self.synth_task_data = []
        self.synth_task_w1_map = {}
        self.synth_task_w2_map = {}
        self.synth_task_words = set()
        self.bless2word = {}
        self.word2bless = {}

        self._load_data()

    def _load_data(self):
        with open("datasets/synthetic_task/synthetic_words_set_{0}.csv".format(self.fold), "r") as fold_data:
            fold_data.readline()
            for line in fold_data:
                w1, w2, m, s = line.rstrip().split()
                self.add_datum(w1, w2, int(m), float(s))

        with open("datasets/synthetic_task/BLESS_classes.txt".format(self.fold), "r") as bless_file:
            for line in bless_file:
                cat, syns = line.rstrip().split("\t")
                self.bless2word[cat] = set(syns.split())
                for word in self.bless2word[cat]:
                    self.word2bless[word] = cat

        self.iso_words = {}
        for cat in self.bless2word:
            self.iso_words[cat] = self.bless2word[cat] - self.synth_task_words

        for datum in self.synth_task_data:  # type: Datum
            datum.w1_syns = self.iso_words[self.word2bless[datum.w1]]
            datum.w2_syns = self.iso_words[self.word2bless[datum.w2]]

    def add_datum(self, w1, w2, m, s):
        new_index = len(self.synth_task_data)
        datum = Datum(w1, w2, m, s)
        self.synth_task_data.append(datum)
        self.synth_task_w1_map[w1] = new_index
        self.synth_task_w2_map[w2] = new_index
        self.synth_task_words.add(w1)
        self.synth_task_words.add(w2)

    @staticmethod
    def standardize(array):
        return (array - np.mean(array)) / np.std(array)

    def evaluate(self, sess, model):
        import tensorflow as tf

        years = np.arange(1900, 2010)
        years_dec = (years - 1900.0) / (2009.0 - 1900.0)
        num_years = len(years)
        targets_placeholder = tf.placeholder(tf.int32, shape=(num_years,))
        synonym_placeholder = tf.placeholder(tf.int32, shape=(num_years,))
        times_placeholder = tf.placeholder(tf.float32, shape=(num_years,))

        target_vector = model.get_target_vector(targets_placeholder, times_placeholder)
        target_vector = tf.nn.l2_normalize(target_vector, 1)
        syns_vector = model.get_target_vector(synonym_placeholder, times_placeholder)
        syns_vector = tf.nn.l2_normalize(syns_vector, 1)
        cos = tf.reduce_sum(tf.multiply(target_vector, syns_vector), axis=1)

        for datum in self.synth_task_data:
            pos_vector = self.standardize(datum.w1_prob_array)
            if datum.w1_index is None:
                continue
            if datum.w2_index is None:
                continue
            if datum.w1_syns_indices == set():
                continue
            if datum.w2_syns_indices == set():
                continue
            pos_avg = np.zeros((num_years,), dtype=np.float)
            for syn_index in datum.w1_syns_indices:
                feed_dict = {
                    targets_placeholder: num_years * [datum.w1_index],
                    synonym_placeholder: num_years * [syn_index],
                    times_placeholder: years_dec,
                }
                cosines = sess.run(cos, feed_dict=feed_dict)
                pos_avg += cosines
            pos_avg /= len(datum.w1_syns_indices)

            neg_avg = np.zeros((num_years,), dtype=np.float)
            for syn_index in datum.w2_syns_indices:
                feed_dict = {
                    targets_placeholder: num_years * [datum.w1_index],
                    synonym_placeholder: num_years * [syn_index],
                    times_placeholder: years_dec,
                }
                cosines = sess.run(cos, feed_dict=feed_dict)
                neg_avg += cosines
            neg_avg /= len(datum.w2_syns_indices)

            predicted_vector = pos_avg - neg_avg
            predicted_vector = self.standardize(predicted_vector)

            score = np.sum(np.square(pos_vector - predicted_vector))

            print(datum.w1, datum.w2, score)
