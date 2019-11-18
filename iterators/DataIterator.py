class DataIterator(object):
    def __init__(self, batch_size, start_year, end_year):
        self.batch_size = batch_size
        self.start_year = start_year
        self.end_year = end_year
        pass

    def year2dec(self, year):
        return (year - self.start_year)/(self.end_year-self.start_year)

    def dec2year(self, dec):
        return (self.end_year-self.start_year)*dec + self.start_year

    def get_batch(self):
        raise NotImplementedError
