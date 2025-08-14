import argparse
import copy
import os
import random
import shutil

from .av_dataset import load_video
import cv2
from tqdm import tqdm
import torch

from .distortions import (block_wise, color_contrast, color_saturation,
                         gaussian_blur, gaussian_noise_color, jpeg_compression,
                         video_compression)

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Add a distortion to video.')
    parser.add_argument('--vid_in_tensor',
                        type=str,
                        required=True,
                        help='path to the input video')
    parser.add_argument('--vid_out_path',
                        type=str,
                        default=None,
                        help='path to the output video')
    parser.add_argument(
        '--type',
        type=str,
        default='random',
        help='distortion type: CS | CC | BW | GNC | GB | JPEG | VC | random')
    parser.add_argument('--level',
                        type=str,
                        default='random',
                        help='distortion level: 1 | 2 | 3 | 4 | 5 | random')
    parser.add_argument('--meta_path',
                        type=str,
                        default=None,
                        help='path to the output video meta file')
    parser.add_argument(
        '--via_xvid',
        action='store_true',
        help='if add this argument, write to XVID .avi video first, '
        "then convert it to 'vid_out_path' by ffmpeg.")
    args = parser.parse_args()

    return args


def convert_tensor_to_cv2_format(input):
    '''Input is a tensor, output is a list'''
    frame_list = []
    
    for frame_tensor in input:
        frame = frame_tensor.permute(1,2,0)
        frame_np = frame.cpu().numpy()
        frame_np = frame_np[..., ::-1]
        frame_list.append(frame_np)

    return frame_list


def convert_cv2_to_tensor_format(input):
    '''Input is a list, output is a tensor'''
    tensor_list = []

    for frame in tensor_to_frame_list:
        frame = frame[..., ::-1]
        tensor_frame = torch.from_numpy(frame.copy()).permute(2,0,1).float()
        tensor_list.append(tensor_frame)

    return tensor_list


def get_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0, 1]  # smaller, worse
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35, 1]  # smaller, worse
    param_dict['BW'] = [16, 32, 48, 64, 80, 1]  # larger, worse
    param_dict['GNC'] = [0.001, 0.002, 0.005, 0.01, 0.05, 1]  # larger, worse
    param_dict['GB'] = [7, 9, 13, 17, 21, 1]  # larger, worse
    param_dict['JPEG'] = [2, 3, 4, 5, 6, 1]  # larger, worse

    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]


def get_distortion_function(type):
    func_dict = dict()  # a dict of function
    func_dict['CS'] = color_saturation
    func_dict['CC'] = color_contrast
    func_dict['BW'] = block_wise
    func_dict['GNC'] = gaussian_noise_color
    func_dict['GB'] = gaussian_blur
    func_dict['JPEG'] = jpeg_compression

    return func_dict[type]


def apply_distortion_log(type, level):
    if type == 'CS':
        print(f'Apply level-{level} color saturation change distortion...')
    elif type == 'CC':
        print(f'Apply level-{level} color contrast change distortion...')
    elif type == 'BW':
        print(f'Apply level-{level} local block-wise distortion...')
    elif type == 'GNC':
        print(f'Apply level-{level} white Gaussian noise in color components '
              'distortion...')
    elif type == 'GB':
        print(f'Apply level-{level} Gaussian blur distortion...')
    elif type == 'JPEG':
        print(f'Apply level-{level} JPEG compression distortion...')


def distortion_vid(vid_in_tensor,
                   vid_in_path=None,
                   vid_out_path=None,
                   dist_type='random',
                   dist_level='random'):

    # get distortion type
    if dist_type == 'random':
        dist_types = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG']
        type_id = random.randint(0, 5)
        dist_type = dist_types[type_id]
    else:
        dist_type = dist_type

    # get distortion level
    if dist_level == 'random':
        dist_level = random.randint(1, 5)
    else:
        dist_level = int(dist_level)
   
    # do not apply distortion if not requested
    if dist_level==0 or dist_type==None:
        if vid_out_path is not None:
                shutil.copy(vid_in_path, vid_out_path)
        return vid_in_tensor

    # get distortion parameter
    dist_param = get_distortion_parameter(dist_type, dist_level)

    # get distortion function
    dist_function = get_distortion_function(dist_type)

    # convert input from (T x C x H x W) to (H, W, 3)
    input_list = convert_tensor_to_cv2_format(vid_in_tensor)

    # optionally save distortion output as mp4 file
    if vid_out_path is not None:
        
        # create output file path
        root = os.path.split(vid_out_path)[0]
        root = '.' if root == '' else root
        os.makedirs(root, exist_ok=True)
        
        # create writer
        vid = cv2.VideoCapture(vid_in_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(vid_out_path, fourcc, fps, (w, h))

    # apply distortion
    output_list = []
    for frame in input_list:
        new_frame = dist_function(frame, dist_param)

        # write to output mp4
        if vid_out_path is not None:
                writer.write(new_frame)
        new_frame = new_frame[..., ::-1]
        output_frame = torch.from_numpy(new_frame.copy()).permute(2,0,1).float()
        output_list.append(output_frame)

    if vid_out_path is not None:
            writer.release()

    output_tensor = torch.stack(output_list)

    # confirm tensor size
    assert vid_in_tensor.size() == output_tensor.size()

    return output_tensor

def write_to_meta_file(meta_path, vid_in_tensor, vid_out_path, dist_type, dist_level):
    # create meta root
    root = os.path.split(meta_path)[0]
    root = '.' if root == '' else root
    os.makedirs(root, exist_ok=True)

    meta_dict = dict()  # a dict of list
    # if exist, get original meta
    if os.path.exists(meta_path):
        f = open(meta_path, 'r')
        lines = f.read().splitlines()
        f.close()
        for l in lines:
            vid_path, dist_meta = l.split()[0], l.split()[1:]
            meta_dict[vid_path] = dist_meta

    # update meta
    meta_list = copy.deepcopy(meta_dict[vid_in_tensor]) if vid_in_tensor in meta_dict else []
    meta_list.append(f'{dist_type}:{dist_level}')
    meta_dict[vid_out_path] = meta_list

    # write meta
    f = open(meta_path, 'w')
    for k, v in meta_dict.items():
        f.write(' '.join([k] + v) + '\n')
    f.close()


def main():
    args = parse_args()
    vid_in_tensor = args.vid_in_tensor
    vid_out_path = args.vid_out_path
    type = args.type
    level = args.level
    meta_path = args.meta_path
    via_xvid = args.via_xvid

    # check input args
    assert os.path.exists(vid_in_tensor), 'Input video does not exist.'
    assert vid_in_tensor != vid_out_path, ('Paths to the input and output videos '
                               'should NOT be the same.')
    type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG', 'VC', 'random']
    if type not in type_list:
        raise ValueError(
            f"Expect distortion type in {type_list}, but got '{type}'.")
    level_list = ['1', '2', '3', '4', '5', 'random']
    if level not in level_list:
        raise ValueError(
            f"Expect distortion level in {level_list}, but got '{level}'.")

    # add distortion to the input video and write to 'vid_out_path'
    dist_tensor = distortion_vid(vid_in_tensor, vid_out_path, type, level,
                                           via_xvid)

    # if meta_path is not None, write meta
    if meta_path is not None:
        # write to meta file
        write_to_meta_file(meta_path, vid_in_tensor, vid_out_path, dist_type, dist_level)


if __name__ == '__main__':
    main()
