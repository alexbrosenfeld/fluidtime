from utils.time_tool import year2dec, dec2year


class DataIterator(object):
    """Base class for training data iterators.

        Arguments:
            args: argparse, config options

        Properties:
            batch_size: int, size of batch
            start_year: int, start year
            end_year: int, end year
        """

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.start_year = args.start_year
        self.end_year = args.end_year
        pass

    # These functions convert years to number between 0 and 1 (and vice versa)
    def year2dec(self, year):
        return year2dec(year, self.args)

    def dec2year(self, dec):
        return dec2year(dec, self.args)

    # Each call returns a sngle training batch.
    def get_batch(self):
        raise NotImplementedError
