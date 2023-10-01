# Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything

from typing import Dict, List
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torchvision

from segment_anything import sam_model_registry, SamPredictor
from YSDT.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
import numpy as np
import torch

from YSDT.inference.object_info import ObjectInfo

def get_sam_model(config: Dict, device: str) -> (SamPredictor):
    # Building SAM Model and SAM Predictor
    variant = config['sam_variant'].lower()
    if variant == 'mobile':
        MOBILE_SAM_CHECKPOINT_PATH = config['MOBILE_SAM_CHECKPOINT_PATH']

        # Building Mobile SAM model
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        sam = SamPredictor(mobile_sam)
    elif variant == 'original':
        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']

        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
            device=device)
        sam = SamPredictor(sam)

    return sam
