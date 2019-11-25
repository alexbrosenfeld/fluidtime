from utils.time_tool import year2dec, dec2year


class DataIterator(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.start_year = args.start_year
        self.end_year = args.end_year
        pass

    def year2dec(self, year):
        return year2dec(year, self.args)

    def dec2year(self, dec):
        return dec2year(dec, self.args)

    def get_batch(self):
        raise NotImplementedError
