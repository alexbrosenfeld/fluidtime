import tensorflow as tf
import numpy as np
from enum import Enum

AGGREGATION_METHOD = Enum("AGGREGATION_METHOD", "concatenate top_k")

def batch_runner(sess, model, batch_size, tensor, data_dict, args, fixed_data=None, indexed_key=None, aggregation_method=AGGREGATION_METHOD.concatenate):
    data_dict_keys = list(data_dict.keys())
    reference_key = data_dict_keys[0]
    reference_data_length = len(data_dict[reference_key])

    assert all(len(data_dict[key]) == reference_data_length for key in data_dict_keys), "different number of examples provided."


    dummy_data = dict((key, data_dict[key][0]) for key in data_dict_keys)

    results = None

    feed_dict = {}
    if fixed_data is not None:
        feed_dict = fixed_data

    for i in range(0, reference_data_length, batch_size):
        size_adjustment = 0
        if reference_data_length < i + batch_size:
            size_adjustment = i + batch_size - reference_data_length


        for key in data_dict_keys:
            feed_dict[key] = data_dict[key][i:min(reference_data_length, i + batch_size)] + size_adjustment*[dummy_data[key]]

        result = sess.run(tensor, feed_dict=feed_dict)

        if type(result) is np.ndarray:


            if results is None:
                if aggregation_method == AGGREGATION_METHOD.top_k:
                    if size_adjustment != 0:
                        result = result[:, :-size_adjustment]
                    raw_indices = np.tile(np.array(feed_dict[indexed_key]), (result.shape[0], 1))
                    arg_part_indices = np.argpartition(result, -args.num_nearest_neighbors, axis=1)[:, -args.num_nearest_neighbors:]
                    values = result[np.arange(result.shape[0])[:, None], arg_part_indices]
                    indices = raw_indices[np.arange(raw_indices.shape[0])[:, None], arg_part_indices]
                    results = [values, indices]
                elif aggregation_method == AGGREGATION_METHOD.concatenate:
                    if size_adjustment != 0:
                        result = result[:-size_adjustment]
                    results = result
                else:
                    raise ValueError
            else:
                if aggregation_method == AGGREGATION_METHOD.top_k:
                    if size_adjustment != 0:
                        result = result[:, :-size_adjustment]

                    raw_values = np.concatenate((results[0], result), axis=1)

                    raw_indices = np.tile(np.array(feed_dict[indexed_key]), (result.shape[0], 1))
                    raw_indices = np.concatenate((results[1], raw_indices), axis=1)

                    arg_part_indices = np.argpartition(raw_values, -args.num_nearest_neighbors, axis=1)[:, -args.num_nearest_neighbors:]
                    values = raw_values[np.arange(raw_values.shape[0])[:, None], arg_part_indices]
                    indices = raw_indices[np.arange(raw_indices.shape[0])[:, None], arg_part_indices]

                    results = [values, indices]
                elif aggregation_method == AGGREGATION_METHOD.concatenate:
                    if size_adjustment != 0:
                        result = result[:-size_adjustment]
                    results = np.concatenate((results, result))
                else:
                    raise ValueError

        else:
            raise ValueError

    if aggregation_method == AGGREGATION_METHOD.top_k:
        values, indices = results
        arg_sort_indices = np.argsort(values, axis=1)
        values = values[np.arange(values.shape[0])[:, None], arg_sort_indices]
        indices = indices[np.arange(indices.shape[0])[:, None], arg_sort_indices]
        results = [values, indices]


    return results



