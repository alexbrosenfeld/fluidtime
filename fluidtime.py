
import tensorflow as tf
from DiffTime import DiffTime
from ModelTrainer import ModelTrainer
from TACCDataIterator import TACCDataIterator
from ModelEval import ModelEval

vocab_size = 100000
batch_size = 10000
num_iterations = 99000
num_iter_per_epoch = 100


dataiter = TACCDataIterator()



with tf.Session() as sess:
    model = DiffTime(vocab_size)
    trainer = ModelTrainer(sess, dataiter, model)
    trainer.train_model(batch_size, num_iterations, num_iter_per_epoch)
    modeleval = ModelEval(sess, model)

