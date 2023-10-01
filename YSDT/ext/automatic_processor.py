from os import path
from typing import Dict, List, Optional

import cv2
import torch
import numpy as np

from YSDT.inference.object_info import ObjectInfo
from YSDT.inference.inference_core import DEVAInferenceCore
from YSDT.inference.frame_utils import FrameInfo
from YSDT.inference.result_utils import ResultSaver
from YSDT.inference.demo_utils import get_input_frame_for_YSDT
from YSDT.ext.automatic_sam import auto_segment
from YSDT.utils.tensor_utils import pad_divide_by, unpad

from segment_anything import SamAutomaticMaskGenerator


def make_segmentation(cfg: Dict, image_np: np.ndarray, forward_mask: Optional[torch.Tensor],
                      sam_model: SamAutomaticMaskGenerator, min_side: int,
                      suppress_small_mask: bool) -> (torch.Tensor, List[ObjectInfo]):
    mask, segments_info = auto_segment(cfg, sam_model, image_np, forward_mask, min_side,
                                       suppress_small_mask)
    return mask, segments_info


@torch.inference_mode()
def process_frame_automatic(YSDT: DEVAInferenceCore,
                            sam_model: SamAutomaticMaskGenerator,
                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = YSDT.config

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    suppress_small_mask = cfg['suppress_small_objects']
    need_resize = new_min_side > 0
    image = get_input_frame_for_YSDT(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > YSDT.next_voting_frame:
            # getting a forward mask
            if YSDT.memory.engaged:
                forward_mask = estimate_forward_mask(YSDT, image)
            else:
                forward_mask = None

            mask, segments_info = make_segmentation(cfg, image_np, forward_mask, sam_model,
                                                    new_min_side, suppress_small_mask)
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            YSDT.add_to_temporary_buffer(frame_info)

            if ti == YSDT.next_voting_frame:
                # process this clip
                this_image = YSDT.frame_buffer[0].image
                this_frame_name = YSDT.frame_buffer[0].name
                this_image_np = YSDT.frame_buffer[0].image_np

                _, mask, new_segments_info = YSDT.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = YSDT.incorporate_detection(this_image, mask, new_segments_info)
                YSDT.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np)

                for frame_info in YSDT.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = YSDT.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np)

                YSDT.clear_buffer()
        else:
            # standard propagation
            prob = YSDT.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np)

    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            if YSDT.memory.engaged:
                forward_mask = estimate_forward_mask(YSDT, image)
            else:
                forward_mask = None

            mask, segments_info = make_segmentation(cfg, image_np, forward_mask, sam_model,
                                                    new_min_side, suppress_small_mask)
            frame_info.segments_info = segments_info
            prob = YSDT.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = YSDT.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np)


def estimate_forward_mask(YSDT: DEVAInferenceCore, image: torch.Tensor):
    image, pad = pad_divide_by(image, 16)
    image = image.unsqueeze(0)  # add the batch dimension

    ms_features = YSDT.image_feature_store.get_ms_features(YSDT.curr_ti + 1, image)
    key, _, selection = YSDT.image_feature_store.get_key(YSDT.curr_ti + 1, image)
    prob = YSDT._segment(key, selection, ms_features)
    forward_mask = torch.argmax(prob, dim=0)
    forward_mask = unpad(forward_mask, pad)
    return forward_mask