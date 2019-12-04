import os
import sys
from random import choices, random

import numpy as np

from evaluation.BaseEndTask import BaseEndTask
from evaluation.SyntheticTask import SyntheticTask
from iterators.DataIterator import DataIterator

from iterators.alias import alias_draw, alias_setup


class AggGBIterator(DataIterator):
    """DataIterator for Google Books ngram corpus aggregations.

    The Google Books ngram corpora (http://storage.googleapis.com/books/ngrams/books/datasetsv2.html) are an excellent
    resource for diachronic distributional models. However, the data is infeasible to train on as is. To counter that,
    I processed an entire corpus into a sequence of (target, context, year, expected frequency) such that the
    expected frequency is the expected amount of times (target, context) is extracted using the skipgram
    algorithm on ngrams from that year.

    Ngram: w1 w2 w3 w4 w5 year freq

    (w1, w2, year) += freq * 4 * subsamp(w1) * subsamp(w2)
    (w1, w3, year) += freq * 3 * subsamp(w1) * subsamp(w3)
    (w1, w4, year) += freq * 2 * subsamp(w1) * subsamp(w4)
    (w1, w5, year) += freq * 1 * subsamp(w1) * subsamp(w5)

    (w5, w4, year) += freq * 4 * subsamp(w5) * subsamp(w4)
    (w5, w3, year) += freq * 3 * subsamp(w5) * subsamp(w3)
    (w5, w2, year) += freq * 2 * subsamp(w5) * subsamp(w2)
    (w5, w1, year) += freq * 1 * subsamp(w5) * subsamp(w1)

    This information is put into the following numpy arrays:

    Let n be the number of (target, context, year, expected frequency) tuples.

    train_targets, (n,) numpy array, index of targets
    train_contexts, (n,) numpy array, index of contexts
    train_times, (n,) numpy array, years converted to be between 0 and 1
    train_rawtimes, (n,) numpy array, years
    train_nums, (n,) numpy array, expected frequencies

    train_neg_sample_probs, (vocab_size,), unigram probability for negative sampling. Frequency to 0.75 and turned
        into probability

    alias_J, alias_q, output of alias_setup applied to train_nums. Weighted sampling in both numpy and random
        is O(n), so we use alias sampling
        (https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/)
        to get O(log(n)) sampling. Applying alias_setup to train_nums takes several hours, so we precompute.



    Arguments:
        args: argparse, config options
        synth_task (optional): SyntheticTask object, if present, this modifies the training data appropriate for the
            synthetic task.
        tasks (optional): list of BaseEndTask, after loading data, provides the tasks with the appropriate indexing
            info.

    Properties:
        training_data_location: str, location of training data numpy arrays
        vocab_file_name: str, file name of file containing vocab
        num_neg_samples: number of negative samples
        alias_flag: boolean, True use alias sampling, False use random.choices

        vocab: list of str, vocabulary
        id2vocab: map from id to vocab
    """
    def __init__(self, args, synth_task: SyntheticTask = None, tasks=None):
        super().__init__(args)

        self.training_data_location = args.GB_data_dir
        self.vocab_file_name = args.vocab_file
        self.num_neg_samples = args.num_negative_samples
        self.alias_flag = args.alias_flag
        self.vocab_size = args.vocab_size
        self.synth_task = synth_task

        self.vocab = []
        # TODO: Check if redundant.
        self.id2vocab = {}

        self.tasks = tasks

        self._load_constant_data()
        self._load_training_data()





    def _load_constant_data(self):
        """Load vocab list from file.
        """

        with open(self.vocab_file_name, "r") as vocab_file:
            word_counter = 0
            for line in vocab_file:
                word = line.rstrip()
                self.vocab.append(word)
                self.id2vocab[word] = word_counter
                word_counter += 1

        assert len(self.vocab) == self.vocab_size


    def _load_training_data(self):
        """Load preprocessed training data into memory.
        """
        self.target_arr = np.load(os.path.join(self.training_data_location, "train_targets.npy"))
        self.context_arr = np.load(self.training_data_location + "train_contexts.npy")
        self.time_arr = np.load(self.training_data_location + "train_times.npy")
        self.rawtime_arr = np.load(self.training_data_location + "train_rawtimes.npy")
        self.num_arr = np.load(self.training_data_location + "train_nums.npy")

        self.neg_sample_arr = np.load(self.training_data_location + "train_neg_sample_probs.npy")

        if self.alias_flag:
            self.alias_J = np.load(self.training_data_location + "alias_J.npy")
            self.alias_q = np.load(self.training_data_location + "alias_q.npy")
            self.neg_sampler = alias_setup(self.neg_sample_arr)
        else:
            self.train_data_length = self.target_arr.shape[0]
            self.neg_data_length = self.neg_sample_arr.shape[0]

        #TODO: Check vocab matches vocab size

        word2id = dict((w, k) for k,w in enumerate(self.vocab))
        word2freq = dict((w, self.neg_sample_arr[k]) for k,w in enumerate(self.vocab))
        if self.synth_task is not None:
            self.synth_task.modify_data(word2id, word2freq)

        for task in self.tasks:  # type: BaseEndTask
            task.modify_data(word2id, word2freq)


    def draw_pos_sample(self):
        """Draw a positive sample from the training data.
        """
        if self.alias_flag:
            curr_index = alias_draw(self.alias_J, self.alias_q)
        else:
            curr_index = choices(range(0, self.train_data_length), weights=self.num_arr)
        self.true_target = self.target_arr[curr_index]
        self.true_context = self.context_arr[curr_index]
        self.true_time = self.time_arr[curr_index]
        self.true_raw_time = self.rawtime_arr[curr_index]
        # [times, targets, contexts, vals]
        if self.synth_task is not None:
            if self.true_target in self.synth_task.synth_task_w1_index_map:
                datum = self.synth_task.synth_task_w1_index_map[self.true_target]
                if datum.w1_probs[self.true_raw_time] > random():
                    "do nothing"
                else:
                    return self.draw_pos_sample()
            if self.true_target in self.synth_task.synth_task_w1_index_map:
                datum = self.synth_task.synth_task_w1_index_map[self.true_target]
                if datum.w1_probs[self.true_raw_time] > random():
                    self.true_target = datum.w1_index
                else:
                    return self.draw_pos_sample()


            # words = line.rstrip().split()
            # adjusted_line = []
            # for word in words:
            #     if word in self.synth_task.synth_task_words:
            #         if word in self.synth_task.synth_task_w1_map:
            #             datum = self.synth_task.synth_task_data[self.synth_task.synth_task_w1_map[word]]
            #             if datum.w1_probs[year] > random():
            #                 adjusted_line.append(word)
            #         else:
            #             datum = self.synth_task.synth_task_data[self.synth_task.synth_task_w2_map[word]]
            #             if datum.w2_probs[year] > random():
            #                 adjusted_line.append(datum.w1)
            #
            #     else:
            #         adjusted_line.append(word)
            # line = " ".join(adjusted_line)
        return self.true_target, self.true_context, self.true_time, 1

    def draw_neg_sample(self):
        """Draw a negative sample from the training data. Negative samples are built from a positive sample.
        """
        if self.alias_flag:
            alias_J, alias_q = self.neg_sampler
            neg_sample = alias_draw(alias_J, alias_q)
        else:
            neg_sample = choices(range(0, self.neg_data_length), weights=self.neg_sample_arr)
        return self.true_target, neg_sample, self.true_time, 0

    def add_datum_to_batch(self, datum):
        """Adds datum to batch.
        """
        target, context, time, label = datum
        self.targets.append(target)
        self.contexts.append(context)
        self.times.append(time)
        self.labels.append(label)
        # This checks if the batch is complete and ready to return
        return len(self.targets) >= self.batch_size


    def get_batch(self):
        """Returns a single batch.
        """
        self.targets = []
        self.contexts = []
        self.times = []
        self.labels = []
        while True:
            # Add positive sample.
            if self.add_datum_to_batch(self.draw_pos_sample()):
                return self.targets, self.contexts, self.times, self.labels
            # Add negative samples.
            for _ in range(0, self.num_neg_samples):
                if self.add_datum_to_batch(self.draw_neg_sample()):
                    return self.targets, self.contexts, self.times, self.labels

