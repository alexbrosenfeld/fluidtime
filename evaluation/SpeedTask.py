import tensorflow as tf
import numpy as np

from evaluation.BaseEndTask import BaseEndTask

from utils.batch_runner import batch_runner
from utils.time_tool import get_year_ranges

import matplotlib.pyplot as plt

class SpeedTask(BaseEndTask):
    def __init__(self, args, words_of_interest, seed_words=None):
        super().__init__(args)

        # self.seed_words = ["cat", "dog"]
        self.seed_words = seed_words
        # self.words_of_interest = ["cat", "dog", "fish"]
        self.words_of_interest = words_of_interest

    def modify_data(self, word2id, word_counts):


        temp_words_of_interest = []
        self.words_of_interest_indices = []
        for w in self.words_of_interest:
            if w in word2id:
                self.words_of_interest_indices.append(word2id[w])
                temp_words_of_interest.append(w)
            else:
                print("Word of interest {0} missing.".format(w))
        self.words_of_interest = temp_words_of_interest

        if self.seed_words is None:
            wcounts = list(word_counts.items())
            wcounts.sort(key=lambda x: x[1], reverse=True)
            self.seed_words = [x[0] for x in wcounts[:self.args.seed_vocab_size]]

        self.seed_indices = []
        for w in self.seed_words:
            if w not in word2id:
                print("Seed word '{0}' is missing from data vocab.".format(w))
                continue
            self.seed_indices.append(word2id[w])


    def evaluate(self, sess, model):
        epsilon = 1e-7

        years, years_dec, num_years = get_year_ranges(self.args)

        targets_placeholder = tf.placeholder(tf.int32, shape=(num_years,))
        times_before_placeholder = tf.placeholder(tf.float32, shape=(num_years,))
        times_after_placeholder = tf.placeholder(tf.float32, shape=(num_years,))

        targets_before_vector = model.get_target_vector(targets_placeholder, times_before_placeholder)
        targets_before_vector = tf.nn.l2_normalize(targets_before_vector, 1)
        targets_after_vector = model.get_target_vector(targets_placeholder, times_after_placeholder)
        targets_after_vector = tf.nn.l2_normalize(targets_after_vector, 1)
        velocity = tf.divide(tf.subtract(targets_after_vector, targets_before_vector), tf.constant(2*epsilon))
        speed = tf.norm(velocity, axis=1)

        mean_seed_speeds = None
        squared_seed_speeds = None

        for seed in self.seed_indices:
            feed_dict = {}
            feed_dict[targets_placeholder] = num_years*[seed]
            feed_dict[times_before_placeholder] = years_dec - epsilon
            feed_dict[times_after_placeholder] = years_dec + epsilon

            results = sess.run(speed, feed_dict=feed_dict)

            if mean_seed_speeds is None:
                mean_seed_speeds = results
                squared_seed_speeds = np.square(results)
            else:
                mean_seed_speeds += results
                squared_seed_speeds += np.square(results)

        mean_seed_speeds = mean_seed_speeds/len(self.seed_indices)
        squared_seed_speeds = squared_seed_speeds/len(self.seed_indices)
        sd_seed_speeds = np.sqrt(squared_seed_speeds - np.square(mean_seed_speeds))

        print("Producing Speed Graphs")
        print()
        import os
        for word, word_index in zip(self.words_of_interest, self.words_of_interest_indices):
            feed_dict = {}
            feed_dict[targets_placeholder] = num_years*[word_index]
            feed_dict[times_before_placeholder] = years_dec - epsilon
            feed_dict[times_after_placeholder] = years_dec + epsilon

            results = sess.run(speed, feed_dict=feed_dict)
            speeds = (results-mean_seed_speeds)/sd_seed_speeds

            plt.plot(years, speeds)
            plt.title("Speed over time for '{0}'".format(word))
            plt.xlabel("Year")
            plt.ylabel("Relative speed")
            plt.savefig(os.path.join(self.args.speed_graph_output_dir, "{0}.png".format(word)))
            plt.close()
            print("Saving speed graph for {0}.".format(word))
        print()

