import random

from iterators.DataIterator import DataTterator

class DumbIterator(DataTterator):
    """This class is to demonstrate the format that your DataIterator should produce.
    """
    def __init__(self, batch_size, vocab_size):
        super().__init__(batch_size)
        self.vocab_size = vocab_size

    def get_batch(self):
        target_indices = [random.randint(0, self.vocab_size - 1) for _ in range(self.batch_size)]
        context_indices = [random.randint(0, self.vocab_size - 1) for _ in range(self.batch_size)]
        times = [random.random() for _ in range(self.batch_size)]
        labels = [random.randint(0, 1) for _ in range(self.batch_size)]
        return target_indices, context_indices, times, labels
