from random import randint, random

from keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.preprocessing.text import Tokenizer

from evaluation.BaseEndTask import BaseEndTask
from evaluation.SyntheticTask import SyntheticTask
from iterators.DataIterator import DataIterator


class COHASampleIterator(DataIterator):
    def __init__(self, args, synth_task: SyntheticTask = None, tasks = None):
        super().__init__(args)
        self.vocab_size = args.vocab_size
        self.genre = args.coha_genre
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.relevant_file_names = []
        self.synth_task = synth_task
        self.tasks = tasks
        self.load_file_data()
        self.curr_targets = []
        self.curr_contexts = []
        self.curr_times = []
        self.curr_labels = []
        self.word2id = self.tokenizer.word_index


    def load_file_data(self):
        import os

        for dirName, subdirList, fileList in os.walk(self.args.coha_data_dir):
            for fname in fileList:
                genre, year, code = fname.split("_")
                if self.genre is not None and self.genre != genre:
                    continue
                year = int(year)
                if year > self.end_year or year < self.start_year:
                    continue
                full_dir = os.path.join(dirName, fname)
                self.relevant_file_names.append((year, full_dir))
                with open(full_dir, "r") as coha_file:
                    # First line in a COHA file is the file number and thus should be skipped.
                    coha_file.readline()
                    for line in coha_file:
                        if line.strip() == "":
                            continue
                        self.tokenizer.fit_on_texts([line])

        for w in self.tokenizer.word_index.keys():
            if self.tokenizer.word_index[w] >= self.vocab_size:
                del self.tokenizer.word_index[w]

        if self.synth_task is not None:
            self.synth_task.modify_data(self.tokenizer.word_index, self.tokenizer.word_counts)

        for task in self.tasks: #type: BaseEndTask
            task.modify_data(self.tokenizer.word_index, self.tokenizer.word_counts)

    def add_to_data(self):
        file_num = randint(0, len(self.relevant_file_names) - 1)
        year, file_name = self.relevant_file_names[file_num]
        dec = self.year2dec(year)
        with open(file_name, "r") as coha_file:
            coha_file.readline()
            for line in coha_file:
                if line.strip() == "":
                    continue

                if self.synth_task is not None:
                    words = line.rstrip().split()
                    adjusted_line = []
                    for word in words:
                        if word in self.synth_task.synth_task_words:
                            if word in self.synth_task.synth_task_w1_map:
                                datum = self.synth_task.synth_task_data[self.synth_task.synth_task_w1_map[word]]
                                if datum.w1_probs[year] > random():
                                    adjusted_line.append(word)
                            else:
                                datum = self.synth_task.synth_task_data[self.synth_task.synth_task_w2_map[word]]
                                if datum.w2_probs[year] > random():
                                    adjusted_line.append(datum.w1)

                        else:
                            adjusted_line.append(word)
                    line = " ".join(adjusted_line)

                wids = self.tokenizer.texts_to_sequences([line])[0]
                # Note that make_sampling_table estimates the sample probabilities using Zipf's law and does not
                # use the word counts in determining probabilities.
                sampling_table = make_sampling_table(self.vocab_size)
                #TODO: Add negative sample option.
                #Note: skipgrams does not weigh sampling probabilities by unigram probability.
                #TODO: fix negative sampling
                pairs, labels = skipgrams(wids, self.vocab_size, window_size=self.args.window_size, negative_samples=5, sampling_table=sampling_table)
                self.curr_targets += [pair[0] for pair in pairs]
                self.curr_contexts += [pair[1] for pair in pairs]
                self.curr_labels += labels
                self.curr_times += len(pairs) * [dec]

    def get_batch(self):
        while len(self.curr_targets) < self.batch_size:
            self.add_to_data()
        target_indices = self.curr_targets[:self.batch_size]
        context_indices = self.curr_contexts[:self.batch_size]
        times = self.curr_times[:self.batch_size]
        labels = self.curr_labels[:self.batch_size]

        self.curr_targets = self.curr_targets[self.batch_size:]
        self.curr_contexts = self.curr_contexts[self.batch_size:]
        self.curr_times = self.curr_times[self.batch_size:]
        self.curr_labels = self.curr_labels[self.batch_size:]
        return target_indices, context_indices, times, labels
