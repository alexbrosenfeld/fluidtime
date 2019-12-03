import unittest

from iterators.COHASampleIterator import COHASampleIterator as IterTest

class ModelTests(unittest.TestCase):
    def setUp(self):
        from utils.configparser import parser
        self.batch_size = 100
        args = parser.parse_args(
            ["--batch_size", str(self.batch_size), "--coha_data_dir", "../datasets/coha_sample/", "--coha_genre", "nf"])
        self.test_iter = IterTest(args, None, [])

    def test_iter_01_get_batch_implemented(self):
        try:
            self.test_iter.get_batch()
        except NotImplementedError:
            self.fail("get_batch not implemented.")

    def test_iter_02_get_batch_format(self):
        self.assertTrue(type(self.test_iter.get_batch()) is tuple)

    def test_iter_03_get_batch_num_outputs(self):
        self.assertEqual(len(self.test_iter.get_batch()), 4, "Wrong number of outputs, should be 4")




if __name__ == '__main__':
    unittest.main()