class DataTterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        pass

    def get_batch(self):
        raise NotImplementedError
