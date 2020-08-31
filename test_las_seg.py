#!/usr/bin/python3
"""Testing On Segmentation Task."""

import logging
import os
import sys
import math
import h5py
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_utils import strfdelta

from logger import setup_logging



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-t', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--max_point_num', '-p', help='Max point number of each sample', type=int, default=12288)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', action='store_true')
    parser.add_argument(
        '--log_path', '-lp', help='Path where log file should be saved.')

    args = parser.parse_args()

    setup_logging(args.log_path)
    logger = logging.getLogger(__name__)

    logger.info(args)

    start_time = datetime.utcnow()

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num
    max_point_num = args.max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    logger.info(f'Parameter number: {parameter_num}.')



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        logger.info(f'Checkpoint loaded from {args.load_ckpt}!')

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        folder = os.path.dirname(args.filelist)
        filenames = [os.path.join(folder, line.strip()) for line in open(args.filelist)]
        for filename in filenames:
            logger.info(f'Reading {filename}...')
            file_start_time = datetime.utcnow()
            data_h5 = h5py.File(filename)

            data = data_h5['data'][...].astype(np.float32)
            data_num = data_h5['data_num'][...].astype(np.int32)
            batch_num = data.shape[0]

            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)

            logger.info(f'{batch_num} testing batches.')
            for batch_idx in range(batch_num):
                if batch_idx % 10 == 0:
                    logger.info(f'Processing {batch_idx} of {batch_num} batches.')


                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]

                tile_num = math.ceil((sample_num * batch_size) / point_num)
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
                indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

                seg_probs = sess.run([seg_probs_op],
                                        feed_dict={
                                            pts_fts: points_batch,
                                            indices: indices_batch,
                                            is_training: False,
                                        })
                probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

                predictions = [(-1, 0.0)] * point_num
                for idx in range(sample_num * batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = probs_2d[idx, :]
                    confidence = np.amax(probs)
                    label = np.argmax(probs)
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                labels_pred[batch_idx, 0:point_num] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num] = np.array([confidence for _, confidence in predictions])



            filename_pred = filename[:-3] + '_pred.h5'
            logger.info(f'Saving {filename_pred}...')
            file = h5py.File(filename_pred, 'w')
            file.create_dataset('data_num', data=data_num)
            file.create_dataset('label_seg', data=labels_pred)
            file.create_dataset('confidence', data=confidences_pred)
            has_indices = 'indices_split_to_full' in data_h5
            if has_indices:
                file.create_dataset('indices_split_to_full', data=data_h5['indices_split_to_full'][...])
            file.close()

            if args.save_ply:
                logger.info('{}-Saving ply of {}...'.format(datetime.now(), filename_pred))
                msg = 'Saving ply of {}...'.format(filename_pred)
                # timer_start(msg)
                filepath_label_ply = os.path.join(filename_pred[:-3] + 'ply_label')
                data_utils.save_ply_property_batch(data[:, :, 0:3], labels_pred[...],
                                                   filepath_label_ply, data_num[...], setting.num_class)
                # timer_stop(msg)

            ######################################################################
            logger.info(f'Done predicting for {filename}. Time spent: {strfdelta(datetime.utcnow() - file_start_time)}')
        logger.info('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
