from DataIterator import DataTterator
import random

class DumbIterator(DataTterator):
    def __init__(self, batch_size, vocab_size):
        super().__init__(batch_size)
        self.vocab_size = vocab_size

    def get_batch(self):
        target_indices = [random.randint(self.vocab_size) for _ in range(self.batch_size)]
        context_indices = [random.randint(self.vocab_size) for _ in range(self.batch_size)]
        times = [random.random() for _ in range(self.batch_size)]
        labels = [random.randint(1) for _ in range(self.batch_size)]
        return target_indices, context_indices, times, labels

