import argparse
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




def main():
    parser = argparse.ArgumentParser(description='DiffTime')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 32)')
    parser.add_argument('--vocab_size', type=int, default=50000, metavar='N', help='vocab size (default: 50000)')
    parser.add_argument('--num_iterations', type=int, default=100, help='number of iterations (default: 100)')
    parser.add_argument('--report_freq', type=int, default=10,
                        help='number of iters between loss reporting (default: 10)')
    parser.add_argument('--start_year', type=int, default=1900, help='start year of analysis (default: 1900)')
    parser.add_argument('--end_year', type=int, default=2009, help='end year of analysis (default: 2009)')
    parser.add_argument('--window_size', type=int, default=4, help='window size (default: 4)')
    parser.add_argument('--num_negative_samples', type=int, default=5, help='number of negative samples (default: 5)')
    parser.add_argument('--verbose_level', type=int, default=1,
                        help='verbosity level. 0=only results, 1=include details and warnings (default: 1)', choices={0, 1})
    parser.add_argument('--model_location', type=str, default="output/saved_models/model.ckpt", help='location to save model (default: output/saved_models/model.ckpt)')


    # COHASampleIterator arguments
    parser.add_argument('--coha_genre', type=str, default=None, help='which COHA genre to use (default: None)')
    parser.add_argument('--coha_data_dir', type=str, default="datasets/coha_sample/", help='directory of coha data (default: datasets/coha_sample/)')

    # AggGBIterator arguments
    parser.add_argument('--GB_data_dir', type=str, default="datasets/agg_gb/", help='directory of aggregated Google Books data (default: datasets/agg_gb/)')
    parser.add_argument('--vocab_file', type=str, default="datasets/agg_gb/vocab.txt", help='vocab file name (default: datasets/agg_gb/vocab.txt)')
    parser.add_argument('--alias_flag', default=False, help='use alias sampling (default: False)', action="store_true")


    # General task arguments
    parser.add_argument('--eval_batch_size', type=int, default=32, metavar='N', help='batch size for evaluation calculation (default: 128)')
    parser.add_argument('--seed_vocab_size', type=int, default=1000, help='number of most frequent words used to calculate (default: 1000)')
    parser.add_argument('--words_of_interest', nargs='+', help='words to generate nearest neighbors and speed graphs', default=None, type=str)

    # SyntheticTask arguments
    parser.add_argument('--synth_task_fold', type=int, default=None,
                        help='which synth task fold to use (default: None)', choices={1, 2, 3})

    # SynchronicTask arguments
    parser.add_argument('--MEN_location', type=str, default="datasets/synchronic_task/",
                        help='location of MEN dataset (default: datasets/synchronic_task/)')

    # SpeedTask arguments
    parser.add_argument('--speed_graph_output_dir', type=str, default="output/speed_graphs/", help='directory to save speed results (default: output/speed_graphs/)')

    # NearestNeighborsTask arguments
    parser.add_argument('--num_nearest_neighbors', type=int, default=10,
                        help='number of nearest neighbors to produce (default: 10)')
    parser.add_argument('--nearest_neighbor_output_dir', type=str, default="output/nearest_neighbors/", help='directory to save nearest neighbor results (default: output/nearest_neighbors/)')
    parser.add_argument('--nearest_neighbor_show_cosine', default=False, help='show cosine similarity values in nearest neighbors output (default: False)', action="store_true")

    args = parser.parse_args()

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
