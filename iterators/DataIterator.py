class DataIterator(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.start_year = args.start_year
        self.end_year = args.end_year
        pass

    def year2dec(self, year):
        return (year - self.start_year)/(self.end_year-self.start_year)

    def dec2year(self, dec):
        return (self.end_year-self.start_year)*dec + self.start_year

    def get_batch(self):
        raise NotImplementedError
