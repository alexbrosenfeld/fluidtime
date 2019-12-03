import argparse

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

# args = parser.parse_args()