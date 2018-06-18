import sys
import os

from DataIterator import DataTterator

if os.name == "nt":
    sys.path.append(r"C:\Users\Alex Rosenfeld\PycharmProjects\aftertime_tf\\")
else:
    sys.path.append("TACC path")

from main_network.data_generator.GoogleBooksGenerator import GoogleBooksGenerator


class TACCDataIterator(DataTterator):
    def __init__(self):
        super().__init__()
        self.dataiter = GoogleBooksGenerator()

    def get_batch(self):
        return self.dataiter.get_batch()