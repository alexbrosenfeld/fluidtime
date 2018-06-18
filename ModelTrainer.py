from DataIterator import DataTterator
from Model import Model
import tensorflow as tf
import time

class ModelTrainer(object):
    def __init__(self, sess, dataiter:DataTterator, model:Model):
        self.dataiter = dataiter
        self.model = model
        self.sess = sess
        pass

    def train_model(self, batch_size:int, num_iterations:int, num_iter_per_epoch:int):
        targets_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        contexts_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        times_placeholder = tf.placeholder(tf.float32, shape=(batch_size,))
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,))

        optimizer = tf.train.AdamOptimizer(1e-4)
        loss = self.model.get_loss(targets_placeholder, contexts_placeholder, times_placeholder, labels_placeholder)
        tf.summary.scalar('loss', loss)
        train_op = optimizer.minimize(loss)

        start_time = time.time()
        for step in range(0, num_iterations):
            targets, contexts, times, labels = self.dataiter.get_batch()
            # print(model.get_loss(targets, contexts, times, labels))
            # sess.run(model.get_loss, feed_dict={"targets":targets, "contexts":contexts, "times":times, "labels":labels})

            feed_dict = {
                targets_placeholder: targets,
                contexts_placeholder: contexts,
                times_placeholder: times,
                labels_placeholder: labels,
            }
            _, loss_value = self.sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % num_iter_per_epoch == 0:
                print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))