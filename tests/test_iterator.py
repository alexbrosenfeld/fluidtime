import unittest

from iterators.COHASampleIterator import COHASampleIterator as IterTest

class ModelTests(unittest.TestCase):


    def test_iter_1(self):
        from utils.configparser import parser
        args = parser.parse_args(["--batch_size", "100", "--coha_data_dir", "../datasets/coha_sample/", "--coha_genre", "nf"])
        test_iter = IterTest(args, None, [])
        print(test_iter.get_batch())
        self.assertTrue(True)




if __name__ == '__main__':
    unittest.main()