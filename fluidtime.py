import sys

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
# tf_logger = logging.getLogger("tensorflow")
# tf_logger.disabled = True
# tf_logger.propagate = False
tf.logging.set_verbosity(tf.logging.ERROR)

# tf.get_logger().propagate = False
# tf.logger().propagate = False


from evaluation.SyntheticTask import SyntheticTask
from evaluation.SynchronicTask import SynchronicTask
from evaluation.SpeedTask import SpeedTask
from evaluation.NearestNeighborsTask import NearestNeighborsTask

from iterators.COHASampleIterator import COHASampleIterator
from iterators.AggGBIterator import AggGBIterator
from models.DiffTime import DiffTime

from utils.configparser import args


def main():


    # set up logging
    if args.verbose_level == 0:
        logger.disabled = True
        warning_level = logging.CRITICAL
    elif args.verbose_level == 1:
        warning_level = logging.INFO
    else:
        raise ValueError
    logger.setLevel(warning_level)
    logging.basicConfig(stream=sys.stdout, level=warning_level)

    # instantiate evaluations
    synthetic_task = args.synth_task_fold
    synth_task = None
    synch_task = None
    speed_task = None
    nn_task = None
    main_tasks = []
    # The synthetic task is mutually exclusive to the other tasks as only the former modifies the training data.
    if synthetic_task in {1, 2, 3}:
        synth_task = SyntheticTask(synthetic_task, args)
    else:
        synch_task = SynchronicTask(args)
        speed_task = SpeedTask(args, args.words_of_interest)
        nn_task = NearestNeighborsTask(args, args.words_of_interest)
        main_tasks = [synch_task, speed_task, nn_task]

    # data_iterator loads the training data and produces batches for training.
    # data_iterator = COHASampleIterator(args, synth_task=synth_task, tasks=main_tasks)
    data_iterator = AggGBIterator(args, synth_task=synth_task, tasks=main_tasks)

    with tf.Session() as sess:
        # Instantiate and train model.
        model = DiffTime(args)
        # print(sess.run(tf.get_variable("h1")))
        model.train(sess, data_iterator, args.batch_size, args.num_iterations, args.report_freq)

        # Run each evaluation.
        # As the synthetic task modifies the training data, it is ran mutually exclusively of the other tasks.
        if synthetic_task in {1, 2, 3}:
            synth_task.evaluate(sess, model)
        else:
            synch_task.evaluate(sess, model)
            # words_of_interest must be provided for these tasks to make sense.
            if args.words_of_interest is None or args.words_of_interest == []:
                logger.info("No words of interest are given.")
                logger.info("Thus, nearest neighbors and speed graphs will not be created.")
            else:
                speed_task.evaluate(sess, model)
                nn_task.evaluate(sess, model)


if __name__ == '__main__':
    main()
