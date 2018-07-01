import sys
import os

from DataIterator import DataTterator

if os.name == "nt":
    sys.path.append(r"C:\Users\Alex Rosenfeld\PycharmProjects\aftertime_tf\\")
else:
    sys.path.append("TACC path")

from main_network.data_generator.GoogleBooksGenerator import GoogleBooksGenerator
from main_network.constants.EngFicGooBooks import EngFicGooBooks


class TACCDataIterator(DataTterator):
    def __init__(self):
        super().__init__()
        lbl = "overall_redo"
        constants = EngFicGooBooks()
        self.dataiter = GoogleBooksGenerator(lbl, constants, 5, 10000, start_year=1900, end_year=2009)

    def get_batch(self):
        #returns 4 1-d numpy arrays of shape (batch_size,)
        #target_ids, context_ids, times (scaled between 0 and 1), labels (1 for positive sample, 0 for negative sample)
        return self.dataiter.get_batch()