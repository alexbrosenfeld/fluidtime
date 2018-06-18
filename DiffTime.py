import numpy as np
import tensorflow as tf
from Model import Model

class DiffTime(Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = 300
        self.h1_dim = 100
        self.h2_dim = 100
        self.prod_dim = 100
        self.embedding_dim = 300
        RANDOFFSET = np.random.rand(self.h1_dim, 1)
        RANDA = (1.0 / np.sqrt(2.0)) * np.random.randn(self.h1_dim, 1)
        RANDB = (-1 * RANDOFFSET * RANDA).reshape((self.h1_dim,))

        with tf.variable_scope("targemb"):
            tf.get_variable("targetemb", [self.vocab_size, self.embedding_dim])

        with tf.variable_scope("contemb"):
            tf.get_variable("contextemb", [self.vocab_size, self.embedding_dim])

        tf.get_variable("h1/kernel", [1, self.h1_dim],
                        initializer=tf.constant_initializer(RANDA))
        tf.get_variable("h1/bias", [self.h1_dim],
                        initializer=tf.constant_initializer(RANDB))
        tf.get_variable("h2/kernel", [self.h1_dim, self.h2_dim])
        tf.get_variable("h2/bias", [self.h2_dim])

        tf.get_variable("evoke/kernel", [self.embedding_dim, self.h2_dim * self.prod_dim])
        tf.get_variable("evoke/bias", [self.h2_dim * self.prod_dim])

        tf.get_variable("last_out/kernel", [self.prod_dim, self.output_dim])
        tf.get_variable("last_out/bias", [self.output_dim])



    def get_time_vector(self, time):
        time_reshape = tf.reshape(time, (-1, 1))

        h1 = tf.layers.dense(time_reshape, self.h1_dim, name="h1", activation=tf.tanh, reuse=True)
        h2 = tf.layers.dense(h1, self.h2_dim, name="h1", activation=tf.tanh, reuse=True)
        return h2

    def embedding2matrix(self, embedding):
        out1 = tf.layers.dense(embedding, self.h2_dim * self.prod_dim, name="evoke", reuse=True)
        out2 = tf.reshape(out1, (-1, self.prod_dim, self.h2_dim))
        return out2

    def get_target_vector(self, target, time):
        with tf.variable_scope("targemb", reuse=True):
            targetemb = tf.get_variable("targetemb")
        target_embeddings = tf.nn.embedding_lookup(targetemb, target)
        tv = self.get_time_vector(time)

        mat = self.embedding2matrix(target_embeddings)

        timevect_expand = tf.expand_dims(tv, -1)
        matmul = tf.matmul(mat, timevect_expand)
        matmul = tf.reshape(matmul, (-1, self.prod_dim))
        out = tf.layers.dense(matmul, self.output_dim, name="last_out", reuse=True)
        return out

    def get_context_vector(self, context, time):
        with tf.variable_scope("contemb", reuse=True):
            contemb = tf.get_variable("contextemb")
        context_embeddings = tf.nn.embedding_lookup(contemb, context)
        tv = self.get_time_vector(time)

        mat = self.embedding2matrix(context_embeddings)

        timevect_expand = tf.expand_dims(tv, -1)
        matmul = tf.matmul(mat, timevect_expand)
        matmul = tf.reshape(matmul, (-1, self.prod_dim))
        out = tf.layers.dense(matmul, self.output_dim, name="last_out", reuse=True)
        return out

    def loss_function(self, features, labels):
        # | ||
        #|| |_
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=features)
        return tf.reduce_mean(loss)

    def get_loss(self, targets, contexts, times, labels):
        targ_vect = self.get_target_vector(targets, times)
        cont_vect = self.get_context_vector(contexts, times)
        mult_vect = tf.multiply(targ_vect, cont_vect)
        logits = tf.reduce_sum(mult_vect, 1)
        loss = self.loss_function(logits, labels)
        return loss
