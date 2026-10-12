# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from models.meanflow import MeanFlow

from models.unet import SongUNet

MODEL_ARCHS = {
    "unet": SongUNet,
    "unet_ffhq256": SongUNet,
}

MODEL_CONFIGS = {
    "unet": {
        "img_resolution": 32,
        "in_channels": 6,
        "out_channels": 3,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
    },
    "unet_ffhq256": {
        "img_resolution": 256,
        "in_channels": 6,
        "out_channels": 3,
        "model_channels": 128,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [1, 2, 2, 2],
        "attn_resolutions": [16],
        "encoder_type": "residual",
        "decoder_type": "standard",
        "num_blocks": 4,
    },
}


def instantiate_model(args) -> nn.Module:
    architechture = args.arch
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    configs = MODEL_CONFIGS[architechture]
    configs['dropout'] = args.dropout
    arch = MODEL_ARCHS[architechture]
    if args.use_edm_aug:
        configs['augment_dim'] = 6
    model = MeanFlow(arch=arch, net_configs=configs, args=args)

    return model
