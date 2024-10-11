from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distance_df, normalized_k=0.1):
    num_sensors = 170
    sensor_ids = [x for x in range(num_sensors)]

    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distances_filename', type=str, default='../datasets/original_data/PEMS08/PEMS08.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_dir', type=str, default='../datasets/PEMS08',
                        help='Path of the output file.')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        pass
    else:
        os.makedirs(args.output_dir)

    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'int', 'to': 'int'})
    sensor_ids, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df)
    # Save to pickle file.
    with open(os.path.join(args.output_dir, 'adj_mx.pkl'), 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
