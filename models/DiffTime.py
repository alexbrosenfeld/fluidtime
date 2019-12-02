import numpy as np
import tensorflow as tf

from models.BaseModel import BaseModel


class DiffTime(BaseModel):
    """This is the model from our NAACL 2018 paper.
    """
    def __init__(self, args):
        super().__init__(args)
        self.vocab_size = args.vocab_size

        #TODO: Place model dimensions as args
        self.output_dim = 300
        self.h1_dim = 100
        self.h2_dim = 100
        self.prod_dim = 100
        self.embedding_dim = 300
        self.oneoverembdim = 1.0 / float(self.embedding_dim)


        # variables for target/context embeddings
        with tf.variable_scope("targemb"):
            tf.get_variable("targetemb", [self.vocab_size, self.embedding_dim],
                                initializer=tf.random_uniform_initializer(-self.oneoverembdim, self.oneoverembdim))
        with tf.variable_scope("contemb"):
            tf.get_variable("contextemb", [self.vocab_size, self.embedding_dim],
                                initializer=tf.random_uniform_initializer(-self.oneoverembdim, self.oneoverembdim))

        # variables for time embedding method
        # As lowest layer deals directly with time as a value between 0 and 1, we want these layers to be initialized
        # such that the change point of each tanh is within the interval.
        RANDOFFSET = np.random.rand(self.h1_dim, 1)
        RANDA = (1.0 / np.sqrt(2.0)) * np.random.randn(self.h1_dim, 1)
        RANDB = (-1 * RANDOFFSET * RANDA).reshape((self.h1_dim,))
        tf.get_variable("h1/kernel", [1, self.h1_dim],
                        initializer=tf.constant_initializer(RANDA))
        tf.get_variable("h1/bias", [self.h1_dim],
                        initializer=tf.constant_initializer(RANDB))
        tf.get_variable("h2/kernel", [self.h1_dim, self.h2_dim])
        tf.get_variable("h2/bias", [self.h2_dim])

        # variables to convert word embedding to matrix
        tf.get_variable("evoke/kernel", [self.embedding_dim, self.h2_dim * self.prod_dim])
        tf.get_variable("evoke/bias", [self.h2_dim * self.prod_dim])

        # variables for last linear layer
        tf.get_variable("last_out/kernel", [self.prod_dim, self.output_dim])
        tf.get_variable("last_out/bias", [self.output_dim])



    def get_time_vector(self, time):
        """Takes time and produces an embedding for that time"""
        time_reshape = tf.reshape(time, (-1, 1))

        h1 = tf.layers.dense(time_reshape, self.h1_dim, name="h1", activation=tf.tanh, reuse=True)
        h2 = tf.layers.dense(h1, self.h2_dim, name="h2", activation=tf.tanh, reuse=True)
        return h2

    def embedding2matrix(self, embedding):
        """Converts a word embedding to a matrix"""
        out1 = tf.layers.dense(embedding, self.h2_dim * self.prod_dim, name="evoke", reuse=True)
        out2 = tf.reshape(out1, (-1, self.prod_dim, self.h2_dim))
        return out2

    def get_target_vector(self, target, time):
        # Get word matrix
        with tf.variable_scope("targemb", reuse=True):
            targetemb = tf.get_variable("targetemb")
        target_embeddings = tf.nn.embedding_lookup(targetemb, target)
        mat = self.embedding2matrix(target_embeddings)

        # Get time embedding
        tv = self.get_time_vector(time)
        timevect_expand = tf.expand_dims(tv, -1)

        # apply word matrix to time embedding
        matmul = tf.matmul(mat, timevect_expand)
        matmul = tf.reshape(matmul, (-1, self.prod_dim))

        #linear layer on top
        out = tf.layers.dense(matmul, self.output_dim, name="last_out", reuse=True)
        return out

    def get_context_vector(self, context, time):
        # Get word matrix
        with tf.variable_scope("contemb", reuse=True):
            contemb = tf.get_variable("contextemb")
        context_embeddings = tf.nn.embedding_lookup(contemb, context)
        mat = self.embedding2matrix(context_embeddings)

        # Get time embedding
        tv = self.get_time_vector(time)
        timevect_expand = tf.expand_dims(tv, -1)

        # apply word matrix to time embedding
        matmul = tf.matmul(mat, timevect_expand)
        matmul = tf.reshape(matmul, (-1, self.prod_dim))

        # linear layer on top
        out = tf.layers.dense(matmul, self.output_dim, name="last_out", reuse=True)
        return out

    @staticmethod
    def _loss_function(features, labels):
        """binary cross entropy loss function"""
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=features)
        # loss = tf.Print(loss, [loss])
        return tf.reduce_mean(loss)

    def get_loss(self, targets, contexts, times, labels):
        """SGNS loss

        It is useful to separate _loss_function from get_loss as each can be extended in a wide variety of ways,
            e.g. regularization, alternate loss functions, other systems on top, etc.
        """
        targ_vect = self.get_target_vector(targets, times)
        cont_vect = self.get_context_vector(contexts, times)
        mult_vect = tf.multiply(targ_vect, cont_vect)
        logits = tf.reduce_sum(mult_vect, 1)
        loss = self._loss_function(logits, labels)
        return loss
