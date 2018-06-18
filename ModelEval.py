from Model import Model
import tensorflow as tf

class ModelEval(object):
    def __init__(self, sess:tf.Session, model:Model):
        self.sess = sess
        self.model = model
        pass

