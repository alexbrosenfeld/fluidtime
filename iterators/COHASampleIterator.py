from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import skipgrams

from iterators.DataIterator import DataIterator

from random import randint

class COHASampleIterator(DataIterator):
    """This class is to demonstrate the format that your DataIterator should produce.
    """
    def __init__(self, batch_size, start_year, end_year, vocab_size, genre=None):
        super().__init__(batch_size, start_year, end_year)
        self.vocab_size = vocab_size
        self.genre = genre
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.relevant_file_names = []
        self.load_file_data()
        self.curr_targets = []
        self.curr_contexts = []
        self.curr_times = []
        self.curr_labels = []
        self.word2id = self.tokenizer.word_index

    def load_file_data(self):
        import os


        for dirName, subdirList, fileList in os.walk("datasets/coha_sample/"):
            for fname in fileList:
                genre, year, code = fname.split("_")
                if self.genre is not None and self.genre != genre:
                    continue
                year = int(year)
                if year > self.end_year or year < self.start_year:
                    continue
                full_dir = os.path.join(dirName, fname)
                self.relevant_file_names.append((year, full_dir))
                # print(full_dir)
                with open(full_dir, "r") as coha_file:
                    coha_file.readline()
                    for line in coha_file:
                        if line.strip() == "":
                            continue
                        self.tokenizer.fit_on_texts([line])
                        # print(genre, year, line)
            # print(dirName, subdirList, fileList)
        # print(self.tokenizer.word_counts)
        # print(len(self.relevant_file_names))
        # exit()

    def add_to_data(self):
        file_num = randint(0, len(self.relevant_file_names) - 1)
        year, file_name = self.relevant_file_names[file_num]
        dec = self.year2dec(year)
        # print(year, dec)
        # exit()
        with open(file_name, "r") as coha_file:
            coha_file.readline()
            for line in coha_file:
                if line.strip() == "":
                    continue

                # wids = [self.word2id[w] for w in text_to_word_sequence(line)]
                wids = self.tokenizer.texts_to_sequences([line])[0]
                # wids = [self.word2id[w] for w in out for out in self.tokenizer.texts_to_sequences((line,))]
                # print(wids)
                # exit()
                # print(len(self.word2id))
                # exit()
                pairs, labels = skipgrams(wids, self.vocab_size, window_size=5, negative_samples=5)
                self.curr_targets += [pair[0] for pair in pairs]
                self.curr_contexts += [pair[1] for pair in pairs]
                self.curr_labels += labels
                self.curr_times += len(pairs)*[dec]

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

    # def get_batch(self):
    #     target_indices = [random.randint(0, self.vocab_size - 1) for _ in range(self.batch_size)]
    #     context_indices = [random.randint(0, self.vocab_size - 1) for _ in range(self.batch_size)]
    #     times = [random.random() for _ in range(self.batch_size)]
    #     labels = [random.randint(0, 1) for _ in range(self.batch_size)]
    #     return target_indices, context_indices, times, labels