import os
import sys
from random import choices

import numpy as np

from evaluation.BaseEndTask import BaseEndTask
from evaluation.SyntheticTask import SyntheticTask
from iterators.DataIterator import DataIterator

from iterators.alias import alias_draw, alias_setup

# if os.name == "nt":
#     sys.path.append(r"C:\Users\Alex Rosenfeld\PycharmProjects\aftertime_tf\\")
# else:
#     sys.path.append("TACC path")
#
# from main_network.data_generator.GoogleBooksGenerator import GoogleBooksGenerator
# from main_network.constants.EngFicGooBooks import EngFicGooBooks


class AggGBIterator(DataIterator):
    """Ignore this class. I'm using it as a way to test some of the functions with real data.
    """
    def __init__(self, args, synth_task: SyntheticTask = None, tasks=None):
        super().__init__(args)
        lbl = "overall_redo"

        self.training_data_location = args.GB_data_dir
        self.vocab_file_name = args.vocab_file
        self.num_neg_samples = args.num_negative_samples
        self.alias_flag = args.alias_flag

        self.vocab = []
        self.id2vocab = {}

        self.tasks = tasks

        self._load_constant_data()
        self._load_training_data()




        # constants = EngFicGooBooks()
        # self.dataiter = GoogleBooksGenerator(lbl, constants, 5, 10000, start_year=1900, end_year=2009)

    def _load_constant_data(self):

        with open(self.vocab_file_name, "r") as vocab_file:
            word_counter = 0
            for line in vocab_file:
                word = line.rstrip()
                self.vocab.append(word)
                self.id2vocab[word] = word_counter
                word_counter += 1


    def _load_training_data(self):
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

        for task in self.tasks:  # type: BaseEndTask
            task.modify_data(dict((w, k) for k,w in enumerate(self.vocab)), dict((w, 1) for k,w in enumerate(self.vocab)))


    def draw_pos_sample(self):
        if self.alias_flag:
            curr_index = alias_draw(self.alias_J, self.alias_q)
        else:
            curr_index = choices(range(0, self.train_data_length), weights=self.num_arr)
        self.true_target = self.target_arr[curr_index]
        self.true_context = self.context_arr[curr_index]
        self.true_time = self.time_arr[curr_index]
        self.true_raw_time = self.rawtime_arr[curr_index]
        # [times, targets, contexts, vals]
        return self.true_target, self.true_context, self.true_time, 1

    def draw_neg_sample(self):
        if self.alias_flag:
            alias_J, alias_q = self.neg_sampler
            neg_sample = alias_draw(alias_J, alias_q)
        else:
            neg_sample = choices(range(0, self.neg_data_length), weights=self.neg_sample_arr)
        return self.true_target, neg_sample, self.true_time, 0

    def add_datum_to_batch(self, datum):
        target, context, time, label = datum
        self.targets.append(target)
        self.contexts.append(context)
        self.times.append(time)
        self.labels.append(label)
        return len(self.targets) >= self.batch_size


    def get_batch(self):
        self.targets = []
        self.contexts = []
        self.times = []
        self.labels = []
        while True:
            if self.add_datum_to_batch(self.draw_pos_sample()):
                return self.targets, self.contexts, self.times, self.labels
            for _ in range(0, self.num_neg_samples):
                if self.add_datum_to_batch(self.draw_neg_sample()):
                    return self.targets, self.contexts, self.times, self.labels

