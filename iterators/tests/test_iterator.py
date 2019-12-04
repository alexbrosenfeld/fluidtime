import unittest
import numpy as np

from evaluation.SpeedTask import SpeedTask
from iterators.COHASampleIterator import COHASampleIterator as IterTestBase

from utils.configparser import parser


class IterTestWrapper(IterTestBase):
    task_args = ["--words_of_interest", "cat"]
    def __init__(self, arg_list, synth_task, tasks):
        iter_spec_args = ["--coha_data_dir", "../../datasets/coha_sample/", "--coha_genre", "nf"]

        args = parser.parse_args(arg_list + iter_spec_args)
        super().__init__(args, synth_task, tasks)


class NoTasksTests(unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        arg_list = ["--batch_size", str(self.batch_size)]
        self.test_iter = IterTestWrapper(arg_list, None, [])

    def test_01_superclass(self):
        self.assertTrue(any(map(lambda cls: cls.__name__ == "DataIterator", self.test_iter.__class__.__mro__)))

    def test_02_get_batch_implemented(self):
        try:
            self.test_iter.get_batch()
        except NotImplementedError:
            self.fail("get_batch not implemented.")

        self.assertTrue(type(self.test_iter.get_batch()) is tuple, "get_batch should return a tuple")
        self.assertEqual(len(self.test_iter.get_batch()), 4, "Wrong number of outputs, should be 4")

        targets, contexts, times, labels = self.test_iter.get_batch()
        self.assertTrue(type(targets) is list, "get_batch should return 4 lists.")
        self.assertTrue(type(contexts) is list, "get_batch should return 4 lists.")
        self.assertTrue(type(times) is list, "get_batch should return 4 lists.")
        self.assertTrue(type(labels) is list, "get_batch should return 4 lists.")

        self.assertEqual(len(targets), self.batch_size)
        self.assertEqual(len(contexts), self.batch_size)
        self.assertEqual(len(times), self.batch_size)
        self.assertEqual(len(labels), self.batch_size)

        self.assertTrue(all(map(lambda x: type(x) is int, targets)))
        self.assertTrue(all(map(lambda x: type(x) is int, contexts)))
        self.assertTrue(all(map(lambda x: type(x) is float, times)))

        #TODO: check vocab size
        self.assertTrue(all(map(lambda x: x >= 0, targets)), "target indices should be non-negative.")
        self.assertTrue(all(map(lambda x: x >= 0, contexts)), "context indices should be non-negative.")
        self.assertTrue(all(map(lambda x: 1.0 >= x and x >= 0.0, times)), "times should be floats between 0 and 1.")
        self.assertTrue(all(map(lambda x: x == 0 or x == 1, labels)), "labels should be 0 or 1")

class TaskModTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_01_modify(self):
        arg_list = IterTestWrapper.task_args
        args = parser.parse_args(arg_list)
        self.speed_task = SpeedTask(args, args.words_of_interest)
        self.assertFalse(hasattr(self.speed_task, "words_of_interest_indices"))
        self.test_iter = IterTestWrapper(arg_list, None, [self.speed_task])
        self.assertTrue(hasattr(self.speed_task, "words_of_interest_indices"), "Iterator does not call task.modify_data")



if __name__ == '__main__':
    unittest.main()