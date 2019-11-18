
import tensorflow as tf

from ModelEval import ModelEval
from ModelTrainer import ModelTrainer
from iterators.DumbIterator import DumbIterator
from iterators.COHASampleIterator import COHASampleIterator
from models.DiffTime import DiffTime

# vocab_size = 100000
vocab_size = 1000
# batch_size = 10000
batch_size = 10
# num_iterations = 99000
num_iterations = 100
num_iter_per_epoch = 2





def main():
    # dataiter = DumbIterator(batch_size, vocab_size)
    dataiter = COHASampleIterator(batch_size, 1900, 2009, vocab_size, genre="nf")
    # print(dataiter.get_batch())
    #
    # exit()

    with tf.Session() as sess:
        model = DiffTime(vocab_size)

        model.train(sess, dataiter, batch_size, num_iterations, num_iter_per_epoch)

        # trainer = ModelTrainer(sess, dataiter, model)
        # trainer.train_model(batch_size, num_iterations, num_iter_per_epoch)
        modeleval = ModelEval(sess, model)

if __name__ == '__main__':
    main()