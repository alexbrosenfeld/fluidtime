import tensorflow as tf
import time
import logging
logger = logging.getLogger(__name__)


from iterators.DataIterator import DataIterator

class BaseModel(object):
    def __init__(self, args):
        self.args = args
        pass

    def get_loss(self, targets, contexts, times, labels):
        raise NotImplementedError

    def get_target_vector(self, target, time):
        raise NotImplementedError

    def train(self, sess, data_iterator:DataIterator, batch_size:int, num_iterations:int, reporting_freq:int):

        targets_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        contexts_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        times_placeholder = tf.placeholder(tf.float32, shape=(batch_size,))
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,))

        optimizer = tf.train.AdamOptimizer(1e-4)
        loss = self.get_loss(targets_placeholder, contexts_placeholder, times_placeholder, labels_placeholder)
        tf.summary.scalar('loss', loss)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)

        print("Begin training.")
        print("")

        start_time = time.time()
        for step in range(0, num_iterations):
            targets, contexts, times, labels = data_iterator.get_batch()
            feed_dict = {
                targets_placeholder: targets,
                contexts_placeholder: contexts,
                times_placeholder: times,
                labels_placeholder: labels,
            }
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % reporting_freq == 0:
                logger.info('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))

        logger.info("")
        print("Training complete.")
        print("")

    def save(self, sess):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, self.args.model_location)

    def load(self, sess):
        saver = tf.train.import_meta_graph(self.args.model_location + ".meta")
        saver.restore(sess, self.args.model_location)