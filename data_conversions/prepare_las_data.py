"""Prepare Data for LAS training task."""
import math
import os
import random

import h5py
import argparse
import logging
import pathlib
import numpy as np
from datetime import datetime
from data_utils import read_xyz_label_from_las_laspy, strfdelta

from logger import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', '-f', help='Base path folder')
    parser.add_argument(
        '--folder_child', '-fc', help='Child folders in the base folder to use, coma separated')
    parser.add_argument(
        '--log_path', '-lp', help='Path where log file should be saved.')
    parser.add_argument(
        '--max_point_num', '-m', help='Max point number of each sample', type=int, default=8192)
    parser.add_argument('--block_size', '-b', help='Block size', type=float, default=5.0)
    parser.add_argument('--grid_size', '-g', help='Grid size', type=float, default=0.1)
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
    parser.add_argument('--use_hag_as_z', '-hag', help='Use height above ground instead of Z', action='store_true')
    # file list preparing
    parser.add_argument('--h5_num', '-d', help='Number of h5 files to be loaded each time', type=int, default=4)
    parser.add_argument('--repeat_num', '-r', help='Number of repeatly using each loaded h5 list', type=int, default=2)


    args = parser.parse_args()
    setup_logging(args.log_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Start preparing... "
                f"Base Folder: {args.folder}, "
                f"Child Folders: {args.folder_child}, "
                f"block size: {args.block_size}, "
                f"grid size: {args.grid_size}, "
                f"max_point_num: {args.max_point_num}"
                )
    start_time = datetime.utcnow()
    # args = argparse.Namespace(
    #     folders=None,
    #     max_point_num=24576,  # 8192,
    #     save_ply=False,  # True,
    #     # meters
    #     block_size=50.0,  # 5.0,
    #     grid_size=1.0,  # 0.1,
    #     # feet
    #     #     block_size = 150.0,#5.0,
    #     #     grid_size = 3.0,#0.1,
    # )

    max_point_num = args.max_point_num

    batch_size = 2048 # how many blocks of block_size * block_size with max_point_num points are written in a single H5
    data = np.zeros((batch_size, max_point_num, 4))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_point_num), dtype=np.int32)

    if args.save_ply:
        data_center = np.zeros((batch_size, max_point_num, 3))
    LOAD_FROM_EXT = '.las'
    base_folder = pathlib.Path(args.folder)

    for folder_child in args.folder_child.split(','):
        folder = base_folder / folder_child
        datasets = [
            path_to_las_file for path_to_las_file in folder.iterdir() if path_to_las_file.suffix.lower() == LOAD_FROM_EXT
        ]
        for las_file_idx, path_to_las_file in enumerate(datasets):
            start_time_file = datetime.utcnow()

            xyz, i, rcrn, labels, xyzirgb_num = read_xyz_label_from_las_laspy(path_to_las_file, use_hag_as_z=args.use_hag_as_z)
            i = i / 2000 + 0.5
            
            # todo what these offsets are for?
            offsets = [('zero', 0.0), ('half', args.block_size / 2)]
            for offset_name, offset in offsets:
                idx_h5 = 0
                idx = 0

                logger.info(f'Computing block id of {xyzirgb_num} points...')
                xyz_min = np.amin(xyz, axis=0, keepdims=True) - offset
                xyz_max = np.amax(xyz, axis=0, keepdims=True)
                logger.info(f"xyz_min: {xyz_min}")
                logger.info(f"xyz_max: {xyz_max}")
                block_size = (args.block_size, args.block_size, 2 * (xyz_max[0, -1] - xyz_min[0, -1]))
                xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int)

                logger.info('Collecting points belong to each block...')
                blocks, point_block_indices, block_point_counts = np.unique(
                    xyz_blocks, return_inverse=True, return_counts=True, axis=0,
                )
                block_point_indices = np.split(
                    np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]),
                )
                logger.info(f'{path_to_las_file} is split into {blocks.shape[0]} blocks.')

                block_to_block_idx_map = dict()
                for block_idx in range(blocks.shape[0]):
                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    block_to_block_idx_map[(block[0], block[1])] = block_idx

                # merge small blocks into one of their big neighbors
                block_point_count_threshold = max_point_num / 10
                nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
                block_merge_count = 0
                for block_idx in range(blocks.shape[0]):
                    if block_point_counts[block_idx] >= block_point_count_threshold:
                        continue

                    block = (blocks[block_idx][0], blocks[block_idx][1])
                    for x, y in nbr_block_offsets:
                        nbr_block = (block[0] + x, block[1] + y)
                        if nbr_block not in block_to_block_idx_map:
                            continue

                        nbr_block_idx = block_to_block_idx_map[nbr_block]
                        if block_point_counts[nbr_block_idx] < block_point_count_threshold:
                            continue

                        block_point_indices[nbr_block_idx] = np.concatenate(
                            [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
                        block_point_indices[block_idx] = np.array([], dtype=np.int)
                        block_merge_count = block_merge_count + 1
                        break
                logger.info(f'{block_merge_count} of {blocks.shape[0]} blocks are merged.')

                idx_last_non_empty_block = 0
                for block_idx in reversed(range(blocks.shape[0])):
                    if block_point_indices[block_idx].shape[0] != 0:
                        idx_last_non_empty_block = block_idx
                        break

                # uniformly sample each block
                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = block_point_indices[block_idx]
                    if point_indices.shape[0] == 0:
                        continue
                    block_points = xyz[point_indices]
                    block_min = np.amin(block_points, axis=0, keepdims=True)
                    xyz_grids = np.floor((block_points - block_min) / args.grid_size).astype(np.int)
                    grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                             return_counts=True, axis=0)
                    grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
                    grid_point_count_avg = int(np.average(grid_point_counts))
                    point_indices_repeated = []
                    for grid_idx in range(grids.shape[0]):
                        point_indices_in_block = grid_point_indices[grid_idx]
                        repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0])
                        if repeat_num > 1:
                            point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)
                            np.random.shuffle(point_indices_in_block)
                            point_indices_in_block = point_indices_in_block[:grid_point_count_avg]
                        point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
                    block_point_indices[block_idx] = np.array(point_indices_repeated)
                    block_point_counts[block_idx] = len(point_indices_repeated)

                for block_idx in range(idx_last_non_empty_block + 1):
                    point_indices = block_point_indices[block_idx]
                    if point_indices.shape[0] == 0:
                        continue

                    block_point_num = point_indices.shape[0]
                    block_split_num = int(math.ceil(block_point_num * 1.0 / max_point_num))
                    point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num))
                    point_nums = [point_num_avg] * block_split_num
                    point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
                    starts = [0] + list(np.cumsum(point_nums))

                    np.random.shuffle(point_indices)
                    block_points = xyz[point_indices]
                    block_min = np.amin(block_points, axis=0, keepdims=True)
                    block_max = np.amax(block_points, axis=0, keepdims=True)
                    block_center = (block_min + block_max) / 2
                    block_center[0][-1] = block_min[0][-1]
                    block_points = block_points - block_center  # align to block bottom center
                    x, y, z = np.split(block_points, (1, 2), axis=-1)
                    block_xzyrgbi = np.concatenate([x, z, y, i[point_indices]], axis=-1)
                    # TODO
                    # block_xzyrgbi = np.concatenate([x, z, y], axis=-1)

                    block_labels = labels[point_indices]

                    for block_split_idx in range(block_split_num):
                        start = starts[block_split_idx]
                        point_num = point_nums[block_split_idx]
                        end = start + point_num
                        idx_in_batch = idx % batch_size
                        data[idx_in_batch, 0:point_num, ...] = block_xzyrgbi[start:end, :]
                        data_num[idx_in_batch] = point_num
                        label[idx_in_batch] = las_file_idx  # won't be used...
                        label_seg[idx_in_batch, 0:point_num] = block_labels[start:end]
                        indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end]
                        if args.save_ply:
                            block_center_xzy = np.array([[block_center[0][0], block_center[0][2], block_center[0][1]]])
                            data_center[idx_in_batch, 0:point_num, ...] = block_center_xzy

                        if ((idx + 1) % batch_size == 0) or \
                                (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1):
                            item_num = idx_in_batch + 1
                            filename_h5 = path_to_las_file.parent / f'{path_to_las_file.stem}_{offset_name}_{idx_h5}.h5'
                            logger.info(f'Saving {filename_h5}...')

                            file = h5py.File(filename_h5, 'w')
                            file.create_dataset('data', data=data[0:item_num, ...])
                            file.create_dataset('data_num', data=data_num[0:item_num, ...])
                            file.create_dataset('label', data=label[0:item_num, ...])
                            file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                            file.create_dataset('indices_split_to_full', data=indices_split_to_full[0:item_num, ...])
                            file.close()

                            # if args.save_ply and offset_name == 'zero':
                            #     logger.info(f'Saving ply of {filename_h5}...')
                            #     filepath_label_ply = os.path.join(folder, 'ply_label',
                            #                                       dataset + '_label_%s_%d' % (offset_name, idx_h5))
                            #
                            #     data_utils.save_ply_property_batch(
                            #         data[0:item_num, :, 0:3] + data_center[0:item_num, ...],
                            #         label_seg[0:item_num, ...],
                            #         filepath_label_ply, data_num[0:item_num, ...], 8)

                            #                             filepath_label_aligned_ply = os.path.join(folder, 'ply_label_aligned',
                            #                                                                       dataset + '_label_%s_%d' % (
                            #                                                                           offset_name, idx_h5))
                            #                             data_utils.save_ply_property_batch(data[0:item_num, :, 0:3],
                            #                                                                label_seg[0:item_num, ...],
                            #                                                                filepath_label_aligned_ply,
                            #                                                                data_num[0:item_num, ...], 8)
                            idx_h5 = idx_h5 + 1
                        idx = idx + 1
            logger.info(f"Done preparing file: {path_to_las_file}")
            logger.info(f"Time spent: {strfdelta(datetime.utcnow() - start_time_file)}")
    logger.info(f"Done preparing folders: {args.folder_child}")
    logger.info(f"Time spent: {strfdelta(datetime.utcnow() - start_time)}")

    ################################################################
    logger.info(f"Start creating the files lists in: {args.folder}")

    root = args.folder

    splits = args.folder_child.split(',')
    split_filelists = dict()
    for split in splits:
        split_filelists[split] = ['./%s/%s\n' % (split, filename) for filename in
                                  os.listdir(os.path.join(root, split))
                                  if filename.endswith('.h5')]

    train_h5 = split_filelists['train']
    random.shuffle(train_h5)
    train_list = os.path.join(root, 'train_data_files.txt')
    logger.info('{}-Saving {}...'.format(datetime.now(), train_list))
    with open(train_list, 'w') as filelist:
        list_num = math.ceil(len(train_h5) / args.h5_num)
        for list_idx in range(list_num):
            train_list_i = os.path.join(root, 'filelists', 'train_files_g_%d.txt' % list_idx)
            with open(train_list_i, 'w') as filelist_i:
                for h5_idx in range(args.h5_num):
                    filename_idx = list_idx * args.h5_num + h5_idx
                    if filename_idx > len(train_h5) - 1:
                        break
                    filename_h5 = train_h5[filename_idx]
                    filelist_i.write('../' + filename_h5)
            for repeat_idx in range(args.repeat_num):
                filelist.write('./filelists/train_files_g_%d.txt\n' % list_idx)

    val_h5 = split_filelists['val']
    val_list = os.path.join(root, 'val_data_files.txt')
    logger.info('{}-Saving {}...'.format(datetime.now(), val_list))
    with open(val_list, 'w') as filelist:
        for filename_h5 in val_h5:
            filelist.write(filename_h5)

    test_h5 = split_filelists['test']
    test_list = os.path.join(root, 'test_files.txt')
    logger.info('{}-Saving {}...'.format(datetime.now(), test_list))
    with open(test_list, 'w') as filelist:
        for filename_h5 in test_h5:
            filelist.write(filename_h5)



if __name__ == '__main__':
    main()
