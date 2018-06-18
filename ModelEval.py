from Model import Model

class ModelEval(object):
    def __init__(self, sess, model:Model):
        self.sess = sess
        self.model = model
        pass

