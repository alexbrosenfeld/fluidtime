
import tensorflow as tf
from DiffTime import DiffTime
from ModelTrainer import ModelTrainer
from TACCDataIterator import TACCDataIterator

vocab_size = 100000

dataiter = TACCDataIterator()
model = DiffTime(vocab_size)

with tf.Session() as sess:
    trainer = ModelTrainer(sess, dataiter, model)

