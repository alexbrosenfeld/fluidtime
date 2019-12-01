import tensorflow as tf
import numpy as np
import logging
logger = logging.getLogger(__name__)
import os

from evaluation.BaseEndTask import BaseEndTask

from utils.time_tool import get_year_ranges

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SpeedTask(BaseEndTask):
    def __init__(self, args, words_of_interest, seed_words=None):
        """Class to generate nearest neighbors over time.

        This class graphs the speed of a word over time.

        We define speed as the norm of the derivative of the l2-normalized word vector at time t. To unpack,
        we use word(time)/||word(time)|| as our position instead of just word(time) as cosine similarity is the
        most common measure of word vector similarity. Speed as a norm of a derivative just comes from physics.

        Although hypothetically there are ways to calculate the actual derivative in tensorflow, in practice,
        due to the way tensorflow is set up, getting the actual derivative is slow and infeasible. Thus, we use a
        first order finite difference approximation:

        f'(x) \sim f(x+eps) - f(x-eps) / 2*eps

        Absolute speed is not very informative due to the uninterpretability of word vector dimensions. Thus, we
        use the relative speed where we standardize the speed relative to the absolute speeds of a list of seed_words.

        Arguments:
            args: argparse object, config options
            words_of_interest: list of str, words to get speed graphs for
            seed_words (optional): list of str, list of seed wprds. If None, take the args.seed_vocab_size
                most frequent words from the training data. Seed words are used to standardize speeds.
        """
        super().__init__(args)

        self.seed_words = seed_words
        self.words_of_interest = words_of_interest

    def modify_data(self, word2id, word_counts):


        temp_words_of_interest = []
        self.words_of_interest_indices = []
        for w in self.words_of_interest:
            if w in word2id:
                self.words_of_interest_indices.append(word2id[w])
                temp_words_of_interest.append(w)
            else:
                logger.warning("Word of interest {0} missing.".format(w))
        self.words_of_interest = temp_words_of_interest

        if self.seed_words is None:
            wcounts = list(word_counts.items())
            wcounts.sort(key=lambda x: x[1], reverse=True)
            self.seed_words = [x[0] for x in wcounts[:self.args.seed_vocab_size]]

        self.seed_indices = []
        for w in self.seed_words:
            if w not in word2id:
                logger.warning("Seed word '{0}' is missing from data vocab.".format(w))
                continue
            self.seed_indices.append(word2id[w])


    def evaluate(self, sess, model):

        # epislon is the size of the finite difference interval
        epsilon = 1e-7

        years, years_dec, num_years = get_year_ranges(self.args)

        targets_placeholder = tf.placeholder(tf.int32, shape=(num_years,))
        times_before_placeholder = tf.placeholder(tf.float32, shape=(num_years,))
        times_after_placeholder = tf.placeholder(tf.float32, shape=(num_years,))

        # tensor to calculate speed, which is equal to norm of derivative
        #TODO: implement higher order finite difference method
        targets_before_vector = model.get_target_vector(targets_placeholder, times_before_placeholder)
        targets_before_vector = tf.nn.l2_normalize(targets_before_vector, 1)
        targets_after_vector = model.get_target_vector(targets_placeholder, times_after_placeholder)
        targets_after_vector = tf.nn.l2_normalize(targets_after_vector, 1)
        velocity = tf.divide(tf.subtract(targets_after_vector, targets_before_vector), tf.constant(2*epsilon))
        speed = tf.norm(velocity, axis=1)

        # mean_seed_speeds = E[X]
        # squared_seed_speeds = E[X^2]
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
        # sd = sqrt(E[X^2] - E[X]^2)
        sd_seed_speeds = np.sqrt(squared_seed_speeds - np.square(mean_seed_speeds))

        print("Producing Speed Graphs")
        print()

        if not os.path.exists(self.args.speed_graph_output_dir):
            logger.info("Creating output directory.")
            logger.info("")
            os.makedirs(self.args.speed_graph_output_dir)

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
            logger.info("Saving speed graph for {0}.".format(word))
        logger.info("")

