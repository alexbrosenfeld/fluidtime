from iterators.DataIterator import DataIterator


class BaseEndTask:
    def __init__(self, args):
        self.args = args

    def modify_data(self, word2id, word_counts):
        raise NotImplementedError

    def evaluate(self, sess, model):
        raise NotImplementedError