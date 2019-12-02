from random import randint, random

from keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.preprocessing.text import Tokenizer

from evaluation.BaseEndTask import BaseEndTask
from evaluation.SyntheticTask import SyntheticTask
from iterators.DataIterator import DataIterator


class COHASampleIterator(DataIterator):
    """DataIterator for COHA samples.

    This randomly pulls COHA files with the appropriate characteristics (year, genre) and produces batches.
    This class is for demonstration purposes only as 1. this approach would not sufficiently produce batches
    with diverse timestamps and 2. I used keras's sequence methods, which only approximate skipgram.

    Arguments:
        args: argparse, config options
        synth_task (optional): SyntheticTask object, if present, this modifies the training data appropriate for the
            synthetic task.
        tasks (optional): list of BaseEndTask, after loading data, provides the tasks with the appropriate indexing
            info.

    Properties:
        vocab_size: int, size_of_vocab
        genre: COHA genre, if None: use all data, if not None: only use data from files of that genre.
        tokenizer: keras Tokenizer, word indexing/frequency information
        relevant_file_names: list of str, COHA file names of right year and genre
    """

    def __init__(self, args, synth_task: SyntheticTask = None, tasks=None):
        super().__init__(args)
        self.vocab_size = args.vocab_size
        self.genre = args.coha_genre
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.relevant_file_names = []
        self.synth_task = synth_task
        self.tasks = tasks

        self.coha_data_dir = args.coha_data_dir

        self.window_size = args.window_size
        self.num_negative_samples = args.num_negative_samples

        self.load_file_data()

        self.word2id = self.tokenizer.word_index

        # Batch storage
        self._curr_targets = []
        self._curr_contexts = []
        self._curr_times = []
        self._curr_labels = []

    def load_file_data(self):
        import os

        for dirName, subdirList, fileList in os.walk(self.coha_data_dir):
            for fname in fileList:
                genre, year, code = fname.split("_")
                if self.genre is not None and self.genre != genre:
                    continue
                year = int(year)
                # ignore files from years out of scope.
                if year > self.end_year or year < self.start_year:
                    continue
                full_dir = os.path.join(dirName, fname)
                # keep track of relevant files.
                self.relevant_file_names.append((year, full_dir))
                with open(full_dir, "r") as coha_file:
                    # First line in a COHA file is the file number and thus should be skipped.
                    coha_file.readline()
                    for line in coha_file:
                        if line.strip() == "":
                            continue
                        self.tokenizer.fit_on_texts([line])

        # remove words from word_index of insufficient frequency
        for w in list(self.tokenizer.word_index.keys()):
            if self.tokenizer.word_index[w] >= self.vocab_size:
                del self.tokenizer.word_index[w]

        # add word index information to tasks
        if self.synth_task is not None:
            self.synth_task.modify_data(self.tokenizer.word_index, self.tokenizer.word_counts)

        for task in self.tasks:  # type: BaseEndTask
            task.modify_data(self.tokenizer.word_index, self.tokenizer.word_counts)

    def add_to_data(self):
        """Adds a randomly chosen file to the batch data
        """
        file_num = randint(0, len(self.relevant_file_names) - 1)
        year, file_name = self.relevant_file_names[file_num]
        # convert year to number between 0 and 1
        dec = self.year2dec(year)
        with open(file_name, "r") as coha_file:
            # first line is file number
            coha_file.readline()
            for line in coha_file:
                if line.strip() == "":
                    continue

                # converts words in training data to pseudowords for synthetic task
                # See paper for details
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
                # Note: skipgrams does not weigh sampling probabilities by unigram probability.
                pairs, labels = skipgrams(wids, self.vocab_size, window_size=self.window_size,
                                          negative_samples=self.num_negative_samples,
                                          sampling_table=sampling_table)
                # Add pair data to batch data
                self._curr_targets += [pair[0] for pair in pairs]
                self._curr_contexts += [pair[1] for pair in pairs]
                self._curr_labels += labels
                self._curr_times += len(pairs) * [dec]

    def get_batch(self):
        """Returns a single training batch."""
        while len(self._curr_targets) < self.batch_size:
            self.add_to_data()
        target_indices = self._curr_targets[:self.batch_size]
        context_indices = self._curr_contexts[:self.batch_size]
        times = self._curr_times[:self.batch_size]
        labels = self._curr_labels[:self.batch_size]

        self._curr_targets = self._curr_targets[self.batch_size:]
        self._curr_contexts = self._curr_contexts[self.batch_size:]
        self._curr_times = self._curr_times[self.batch_size:]
        self._curr_labels = self._curr_labels[self.batch_size:]
        return target_indices, context_indices, times, labels
