import sys
import os
import tensorflow as tf
from DiffTime import DiffTime
from ModelTrainer import ModelTrainer

if os.name == "nt":
    sys.path.append(r"C:\Users\Alex Rosenfeld\PycharmProjects\aftertime_tf\\")
else:
    sys.path.append("TACC path")

from main_network.data_generator.GoogleBooksGenerator import GoogleBooksGenerator

vocab_size = 100000

dataiter = GoogleBooksGenerator()
model = DiffTime(vocab_size)

with tf.Session() as sess:
    trainer = ModelTrainer(sess, dataiter, model)

