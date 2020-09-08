#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import plyfile
import numpy as np
import argparse
import h5py

# reduced_length_dict = {"MarketplaceFeldkirch":[10538633,"marketsquarefeldkirch4-reduced"],
#                        "StGallenCathedral":[14608690,"stgallencathedral6-reduced"],
#                        "sg27":[28931322,"sg27_10-reduced"],
#                        "sg28":[24620684,"sg28_2-reduced"]}
#
# full_length_dict = {"stgallencathedral_station1":[31179769,"stgallencathedral1"],
#                     "stgallencathedral_station3":[31643853,"stgallencathedral3"],
#                     "stgallencathedral_station6":[32486227,"stgallencathedral6"],
#                     "marketplacefeldkirch_station1":[26884140,"marketsquarefeldkirch1"],
#                     "marketplacefeldkirch_station4":[23137668,"marketsquarefeldkirch4"],
#                     "marketplacefeldkirch_station7":[23419114,"marketsquarefeldkirch7"],
#                     "birdfountain_station1":[40133912,"birdfountain1"],
#                     "castleblatten_station1":[31806225,"castleblatten1"],
#                     "castleblatten_station5":[49152311,"castleblatten5"],
#                     "sg27_station3":[422445052,"sg27_3"],
#                     "sg27_station6":[226790878,"sg27_6"],
#                     "sg27_station8":[429615314,"sg27_8"],
#                     "sg27_station10":[285579196,"sg27_10"],
#                     "sg28_station2":[170158281,"sg28_2"],
#                     "sg28_station5":[267520082,"sg28_5"]}
from data_utils import read_xyz_label_from_las, save_xyz_label_to_las, read_points_from_las, \
    rewrite_las_with_new_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    # parser.add_argument('--version', '-v', help='full or reduced', type=str, required=True)
    args = parser.parse_args()
    print(args)

    # if args.version == 'full':
    #     length_dict = full_length_dict
    # else:
    #     length_dict = reduced_length_dict

    import time
    start_time_dict = {}
    total_time_dict = {}

    current_time_ms = lambda: int(round(time.time() * 1000))

    def timer_start(msg):
        global start_time_dict
        global total_time_dict
        start_time_dict[msg] = current_time_ms()
        if not msg in total_time_dict:
            total_time_dict[msg] = 0

    def timer_pause(msg):
        global start_time_dict
        global total_time_dict
        total_time_dict[msg] += current_time_ms() - start_time_dict[msg]

    def timer_stop(msg):
        global total_time_dict
        timer_pause(msg)
        print("{} completed in {}ms".format(msg, total_time_dict[msg]))
        total_time_dict[msg] = 0

    def get_pred_prefixes(datafolder):
        fs = os.listdir(datafolder)
        preds = []
        for f in fs:
            if f[-8:] == '_pred.h5':
                preds += [f]
        pred_pfx = []
        for p in preds:
            if '_zero' in p:
                pred_pfx += [p.split('_zero')[0]]
        return np.unique(pred_pfx)

    SAVE_TO_EXT = '.las'
    LOAD_FROM_EXT = '.las'
    categories_list = get_pred_prefixes(args.datafolder)

    # categories_list = [category for category in length_dict]
    print(categories_list)

    for category in categories_list:
        output_path = os.path.join(args.datafolder, "results", category + "_pred" + SAVE_TO_EXT)
        if not os.path.exists(os.path.join(args.datafolder,"results")):
            os.makedirs(os.path.join(args.datafolder,"results"))
        pred_list = [pred for pred in os.listdir(args.datafolder)
                     if category in pred  and pred.split(".")[0].split("_")[-1] == 'pred']

        print('pred_list: {}'.format(pred_list))

        #     label_length = length_dict[category][0]
        #     merged_label = np.zeros((label_length),dtype=int)
        #     merged_confidence = np.zeros((label_length),dtype=float)
        merged_label = None
        merged_confidence = None

        for pred_file in pred_list:
            print(os.path.join(args.datafolder, pred_file))
            data = h5py.File(os.path.join(args.datafolder, pred_file))
            labels_seg = data['label_seg'][...].astype(np.int64)
            indices = data['indices_split_to_full'][...].astype(np.int64)
            confidence = data['confidence'][...].astype(np.float32)
            data_num = data['data_num'][...].astype(np.int64)

            if merged_label is None:
                # calculating how many labels need to be there in the output
                label_length = 0
                for i in range(indices.shape[0]):
                    label_length = np.max([label_length, np.max(indices[i][:data_num[i]])])
                label_length += 1
                merged_label = np.zeros((label_length), dtype=int)
                merged_confidence = np.zeros((label_length), dtype=float)
            else:
                label_length2 = 0
                for i in range(indices.shape[0]):
                    label_length2 = np.max([label_length2, np.max(indices[i][:data_num[i]])])
                label_length2 += 1
                if label_length < label_length2:
                    # expaning labels and confidence arrays, as the new file appears having mode of them
                    extend_array_label = np.zeros(label_length2 - label_length, dtype=int)
                    extend_array_confidence = np.zeros(label_length2 - label_length, dtype=float)
                    merged_label = np.append(merged_label, extend_array_label, 0)
                    merged_confidence = np.append(merged_confidence, extend_array_confidence, 0)
                    # for i in range(label_length2 - label_length):
                    #     merged_label = np.append(merged_label, 0)
                    #     merged_confidence = np.append(merged_confidence, 0.0)
                    label_length = label_length2

            for i in range(labels_seg.shape[0]):
                temp_label = np.zeros((data_num[i]),dtype=int)
                pred_confidence = confidence[i][:data_num[i]]
                temp_confidence = merged_confidence[indices[i][:data_num[i]]]

                temp_label[temp_confidence >= pred_confidence] = merged_label[indices[i][:data_num[i]]][temp_confidence >= pred_confidence]
                temp_label[pred_confidence > temp_confidence] = labels_seg[i][:data_num[i]][pred_confidence > temp_confidence]

                merged_confidence[indices[i][:data_num[i]][pred_confidence > temp_confidence]] = pred_confidence[pred_confidence > temp_confidence]
                merged_label[indices[i][:data_num[i]]] = temp_label

        if len(pred_list) > 0:
            # concatenating source points with the final labels and writing out resulting file
            final_labels = np.ndarray((merged_label.shape[0], 1), np.int64)
            final_labels[:, 0] = merged_label  # + 1
            points_path = os.path.join(args.datafolder, category + LOAD_FROM_EXT)

            rewrite_las_with_new_labels(points_path, output_path, merged_label)
            # if LOAD_FROM_EXT == ".las":
            #     points, h = read_points_from_las(points_path)
            # else:
            #     print('Reading {}'.format(points_path))
            #     points = np.loadtxt(points_path)
            # if SAVE_TO_EXT == '.las':
            #     save_xyz_label_to_las(output_path, points, final_labels, h)
            # else:
            #     final = np.concatenate([points, final_labels], axis=-1)
            #     print('Writing {}'.format(output_path))
            #     np.savetxt(output_path, final, fmt='%1.3f %1.3f %1.3f %i %i')

if __name__ == '__main__':
    main()
