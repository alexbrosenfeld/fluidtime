import tensorflow as tf
import numpy as np

from evaluation.BaseEndTask import BaseEndTask

from utils.batch_runner import batch_runner, AGGREGATION_METHOD
from utils.time_tool import get_year_ranges

from collections import defaultdict


class NearestNeighborsTask(BaseEndTask):
    def __init__(self, args):
        super().__init__(args)

        #TODO: input neighbor_vocab and words_of_interest
        # self.neighbor_vocab = ["cat", "dog", "fish"]
        self.neighbor_vocab = ['the', 'of', 'to', 'and', 'in', 'a', 'is', 'that', 'it', 'be', 'as', 'for', 'are', 'by', 'this', 'with', 'which', 'not', 'or', 'on', 'from', 'at', 'an', 'we', 'can', 'but', 'have', 'one', 'was', 'will', 'if', 'its', 'i', 'has', 'they', 'other', 'their', 'there', 'more', "'s", 'than', 'all', 'when', 'would', 'these', 'may', 'so', 'such', 'into', 'only', 'our', 'no', 'two', 'been', 'some', 'were', 'very', 'about', 'he', 'any', 'must', 'between', 'also', 'out', 'what', 'energy', 'current', 'same', 'each', 'time', 'most', 'then', 'first', 'through', 'object', 'house', 'up', 'his', 'should', 'large', 'used', 'do', 'way', 'them', 'had', 'even', 'matter', 'being', 'however', 'new', 'high', 'sortal', 'field', 'many', 'case', 'us', 'made', 'see', 'because']

        self.words_of_interest = ["fish", "cat"]

    def modify_data(self, word2id):
        #TODO: have cutoff words if not exist.
        self.neighbor_indices = []
        self.neighbor_index_word = {}

        for w in self.neighbor_vocab:
            if w not in word2id:
                continue
            self.neighbor_indices.append(word2id[w])
            self.neighbor_index_word[word2id[w]] = w

        self.words_of_interest_indices = [word2id[w] for w in self.words_of_interest if w in word2id]

    def evaluate(self, sess, model):
        num_words_of_interest = len(self.words_of_interest)

        years, years_dec, num_years = get_year_ranges(self.args)

        targets_placeholder = tf.placeholder(tf.int32, shape=(num_words_of_interest,))
        targets_times_placeholder = tf.placeholder(tf.float32, shape=(num_words_of_interest,))
        synonym_placeholder = tf.placeholder(tf.int32, shape=(self.args.eval_batch_size,))
        synonyms_times_placeholder = tf.placeholder(tf.float32, shape=(self.args.eval_batch_size,))


        target_vector = model.get_target_vector(targets_placeholder, targets_times_placeholder)
        target_vector = tf.nn.l2_normalize(target_vector, 1)
        syns_vector = model.get_target_vector(synonym_placeholder, synonyms_times_placeholder)
        syns_vector = tf.nn.l2_normalize(syns_vector, 1)
        cosine_tensor = tf.tensordot(target_vector, syns_vector, axes=[[1], [1]])

        fixed_data = {}
        fixed_data[targets_placeholder] = self.words_of_interest_indices

        nearest_neghbor_results = defaultdict(list)

        for year, year_dec in zip(years, years_dec):
            fixed_data[targets_times_placeholder] = num_words_of_interest*[year_dec]

            data_dict = {}
            data_dict[synonym_placeholder] = [x for x in self.neighbor_indices]
            data_dict[synonyms_times_placeholder] = len(data_dict[synonym_placeholder]) * [year_dec]

            values, indices = batch_runner(sess, model, self.args.eval_batch_size, cosine_tensor, data_dict, self.args, fixed_data=fixed_data,
                                   indexed_key=synonym_placeholder, aggregation_method=AGGREGATION_METHOD.top_k)

            for word_index, word in enumerate(self.words_of_interest):
                line_data = [self.neighbor_index_word[i] for i in indices[word_index] if word != self.neighbor_index_word[i]][:self.args.num_nearest_neighbors]
                nearest_neghbor_results[word].append([year] + line_data)

        print("Nearest Neighbors Tables")
        print()

        import os
        for word in self.words_of_interest:
            with open(os.path.join(self.args.nearest_neighbor_output_dir, "{0}.tsv".format(word)), "w") as out_file:
                print("Target", "Year", *["NN{0}".format(i) for i in range(1, self.args.num_nearest_neighbors + 1)], sep="\t", file=out_file)
                for year, *line_data in nearest_neghbor_results[word]:
                    print(word, year, *line_data, sep="\t", file=out_file)
                print("Saved {0} nearest neighbor data.".format(word))
        print()

