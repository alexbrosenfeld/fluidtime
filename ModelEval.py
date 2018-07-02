import tensorflow as tf

from models.Model import Model


class ModelEval(object):
    def __init__(self, sess:tf.Session, model:Model):
        self.sess = sess
        self.model = model
        pass

