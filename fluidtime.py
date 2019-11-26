import argparse
import sys

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
tf.get_logger().propagate = False


from evaluation.SyntheticTask import SyntheticTask
from evaluation.SynchronicTask import SynchronicTask
from evaluation.SpeedTask import SpeedTask
from evaluation.NearestNeighborsTask import NearestNeighborsTask

from iterators.COHASampleIterator import COHASampleIterator
from models.DiffTime import DiffTime





# vocab_size = 100000
# vocab_size = 100000
# batch_size = 10000
# batch_size = 10
# num_iterations = 99000
# num_iterations = 1000
# num_iter_per_epoch = 10





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
    parser.add_argument('--verbose_level', type=int, default=1,
                        help='verbosity level. 0=only results, 1=include point in process, 2=warnings (default: 1)', choices={0, 1, 2})
    parser.add_argument('--model_location', type=str, default="output/saved_models/model.ckpt", help='location to save model (default: output/saved_models/model.ckpt)')


    # COHASampleIterator arguments
    parser.add_argument('--coha_genre', type=str, default=None, help='which COHA genre to use (default: None)')
    parser.add_argument('--coha_data_dir', type=str, default="datasets/coha_sample/", help='directory of coha data (default: datasets/coha_sample/)')

    #General task arguments
    parser.add_argument('--eval_batch_size', type=int, default=32, metavar='N', help='batch size for evaluation calculation (default: 128)')
    parser.add_argument('--seed_vocab_size', type=int, default=1000, help='number of most frequent words used to calculate (default: 1000)')


    # SyntheticTask arguments
    parser.add_argument('--synth_task_fold', type=int, default=None,
                        help='which synth task fold to use (default: None)', choices={1, 2, 3})

    # SpeedTask arguments
    parser.add_argument('--speed_graph_output_dir', type=str, default="output/speed_graphs/", help='directory to save speed results (default: output/speed_graphs/)')

    # NearestNeighborsTask arguments
    parser.add_argument('--num_nearest_neighbors', type=int, default=10,
                        help='number of nearest neighbors to produce (default: 10)')
    parser.add_argument('--nearest_neighbor_output_dir', type=str, default="output/nearest_neighbors/", help='directory to save nearest neighbor results (default: output/nearest_neighbors/)')
    parser.add_argument('--nearest_neighbor_show_cosine', default=False, help='show cosine similarity values in nearest neighbors output (default: False)', action="store_true")



    args = parser.parse_args()

    if args.verbose_level == 0:
        logger.disabled = True
        warning_level = logging.CRITICAL
    elif args.verbose_level == 1:
        #TODO: only show info on info and not warnings
        warning_level = logging.INFO
    elif args.verbose_level == 2:
        warning_level = logging.WARNING
    else:
        raise ValueError
    logger.setLevel(warning_level)
    logging.basicConfig(stream=sys.stdout, level=warning_level)

    synthetic_task = args.synth_task_fold
    synth_task = None
    if synthetic_task in {1, 2, 3}:
        synth_task = SyntheticTask(synthetic_task, args)
    synch_task = SynchronicTask(args)
    speed_task = SpeedTask(args, ["cat", "fish"])

    # TODO: make words of interest something to enter in
    nn_task = NearestNeighborsTask(args, ["cat", "fish"])

    data_iterator = COHASampleIterator(args, synth_task=synth_task, tasks=[synch_task, speed_task, nn_task])

    with tf.Session() as sess:
        model = DiffTime(args)

        model.train(sess, data_iterator, args.batch_size, args.num_iterations, args.report_freq)

        model.save(sess)

        model.load(sess)
        if synthetic_task in {1, 2, 3}:
            synth_task.evaluate(sess, model)
        synch_task.evaluate(sess, model)
        speed_task.evaluate(sess, model)
        nn_task.evaluate(sess, model)


if __name__ == '__main__':
    main()
