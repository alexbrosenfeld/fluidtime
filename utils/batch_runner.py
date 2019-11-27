import numpy as np
from enum import Enum

# Aggregation method: method to aggregate runner_batches
# concatenate: vstack one on top of another
# top_k: k+1 nearest neighbors (the extra plus 1 is in case the data contains a word as a neighbor).
#   The extra plus 1 is removed by the method that calls batch_runner
AGGREGATION_METHOD = Enum("AGGREGATION_METHOD", "concatenate top_k")


def batch_runner(sess, model, runner_batch_size, tensor, data_dict, args, fixed_data=None, indexed_key=None,
                 aggregation_method=AGGREGATION_METHOD.concatenate):
    """Run a tf model with batching the input data.

    In practice, it's often infeasible to load every entry into memory.
    To rectify this, the following is code to batch executions.

    Arguments:
        sess: tf session
        model: models.BaseModel, model where tensor comes from. Model is unnecessary.
        runner_batch_size: size of each batch
        tensor: tf.Tensor, the tensor to be run, i.e. sess.run(tensor)
        data_dict: dict[tf.placeholder, array/list], input data that will be divided into batches
        args: argparse, config options
        fixed_data (optional): dict[tf.placeholder, array/list], input data that remains fixed in each batch
        indexed_key (optional): tf.placeholder, key for data_dict/fixed_data that
        aggregation_method (optional): AGGREGATION_METHOD value, which way to combine batches

    Returns:
        results: np array or list of np arrays, the aggregated results.
    """
    data_dict_keys = list(data_dict.keys())

    # reference_data_length is the length of each input data array
    # assuming all input data is the same length, any input key can be chosen.
    reference_key = data_dict_keys[0]
    reference_data_length = len(data_dict[reference_key])

    assert all(len(data_dict[key]) == reference_data_length for key in
               data_dict_keys), "different number of examples provided."

    # dummy elements for extending batches to fit runner_batch_size
    dummy_data = dict((key, data_dict[key][0]) for key in data_dict_keys)

    results = None

    feed_dict = {}
    if fixed_data is not None:
        feed_dict = fixed_data

    for i in range(0, reference_data_length, runner_batch_size):
        # size_adjustment is how many dummy variables are needed to complete the last batch.
        size_adjustment = 0
        if reference_data_length < i + runner_batch_size:
            size_adjustment = i + runner_batch_size - reference_data_length

        for key in data_dict_keys:
            feed_dict[key] = data_dict[key][i:min(reference_data_length, i + runner_batch_size)] + size_adjustment * [
                dummy_data[key]]

        batch_result = sess.run(tensor, feed_dict=feed_dict)

        if aggregation_method == AGGREGATION_METHOD.top_k:
            if size_adjustment != 0:
                batch_result = batch_result[:, :-size_adjustment]
            raw_indices = np.tile(np.array(feed_dict[indexed_key]), (batch_result.shape[0], 1))
            if results is None:
                arg_part_indices = np.argpartition(batch_result, -args.num_nearest_neighbors, axis=1)[:,
                                   -args.num_nearest_neighbors:]
                values = batch_result[np.arange(batch_result.shape[0])[:, None], arg_part_indices]
                indices = raw_indices[np.arange(raw_indices.shape[0])[:, None], arg_part_indices]
            else:
                raw_values = np.concatenate((results[0], batch_result), axis=1)
                raw_indices = np.concatenate((results[1], raw_indices), axis=1)
                arg_part_indices = np.argpartition(raw_values, -args.num_nearest_neighbors, axis=1)[:,
                                   -args.num_nearest_neighbors:]
                values = raw_values[np.arange(raw_values.shape[0])[:, None], arg_part_indices]
                indices = raw_indices[np.arange(raw_indices.shape[0])[:, None], arg_part_indices]
            results = [values, indices]
        elif aggregation_method == AGGREGATION_METHOD.concatenate:
            if size_adjustment != 0:
                batch_result = batch_result[:-size_adjustment]
            if results is None:
                results = batch_result
            else:
                results = np.concatenate((results, batch_result))
        else:
            raise ValueError

    # for top_k aggregation, sort the nearest neighbors
    if aggregation_method == AGGREGATION_METHOD.top_k:
        values, indices = results
        arg_sort_indices = np.argsort(values, axis=1)
        values = values[np.arange(values.shape[0])[:, None], arg_sort_indices]
        indices = indices[np.arange(indices.shape[0])[:, None], arg_sort_indices]
        results = [values, indices]

    return results
