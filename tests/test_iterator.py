import unittest

from iterators.COHASampleIterator import COHASampleIterator as IterTest

class ModelTests(unittest.TestCase):


    def test_iter_1(self):
        from utils.configparser import parser
        print(parser)
        args = parser.parse_args(["--batch_size", "100"])
        test_iter = IterTest(args, None, None)
        print(test_iter.get_batch())
        self.assertTrue(True)




if __name__ == '__main__':
    unittest.main()