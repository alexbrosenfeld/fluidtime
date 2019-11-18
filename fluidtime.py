
import tensorflow as tf

from iterators.COHASampleIterator import COHASampleIterator
from models.DiffTime import DiffTime
from evaluation.SyntheticTask import SyntheticTask

# vocab_size = 100000
vocab_size = 100000
# batch_size = 10000
batch_size = 10
# num_iterations = 99000
num_iterations = 1000
num_iter_per_epoch = 10





def main():

    synthetic_task = 1
    synth_task = None
    if synthetic_task in {1, 2, 3}:
        synth_task = SyntheticTask(synthetic_task)

    dataiter = COHASampleIterator(batch_size, 1900, 2009, vocab_size, genre=None, synth_task=synth_task)
    # print(dataiter.get_batch())
    #
    # exit()

    with tf.Session() as sess:
        model = DiffTime(vocab_size)

        model.train(sess, dataiter, batch_size, num_iterations, num_iter_per_epoch)

        synth_task.evaluate(sess, model)
        # model_eval = ModelEval(sess, model)

if __name__ == '__main__':
    main()