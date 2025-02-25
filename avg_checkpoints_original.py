#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:32:05 2024

@author: umbertocappellazzo
"""

import os

import torch


def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def ensemble_original(args, num_average_epochs=10):
    last = [
        os.path.join(args.exp_dir, args.exp_name, f"epoch={n}.ckpt")
        for n in range(
            args.max_epochs - num_average_epochs,
            args.max_epochs,
        )
    ]
    model_path = os.path.join(args.exp_dir, args.exp_name, f"model_avg_{num_average_epochs}.pth")
    torch.save(average_checkpoints(last), model_path)
    return model_path