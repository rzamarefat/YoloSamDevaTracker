from os import path
from typing import Dict, List

import cv2
import torch
import numpy as np

from YSDT.inference.object_info import ObjectInfo
from YSDT.inference.inference_core import DEVAInferenceCore
from YSDT.inference.frame_utils import FrameInfo
from YSDT.inference.result_utils import ResultSaver
from YSDT.inference.demo_utils import get_input_frame_for_YSDT
import torchvision
import torch.nn.functional as F
from YSDT import Config
from YSDT.YOLODet import YOLODet
ypd = YOLODet()


try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamPredictor


def make_segmentation_with_YOLO(config: Dict,image: np.ndarray, sam: SamPredictor,
                        min_side: int) -> (torch.Tensor, List[ObjectInfo]):
    """
    config: the global configuration dictionary
    image: the image to segment; should be a numpy array; H*W*3; unnormalized (0~255)
    prompts: list of class names

    Returns: a torch index mask of the same size as image; H*W
             a list of segment info, see object_utils.py for definition
    """

    BOX_THRESHOLD = TEXT_THRESHOLD = config['DINO_THRESHOLD']
    NMS_THRESHOLD = config['DINO_NMS_THRESHOLD']

    sam.set_image(image, image_format='RGB')
    detections = ypd.detect(image)
    nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy),
                                  torch.from_numpy(detections.confidence),
                                  NMS_THRESHOLD).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    result_masks = []
    for box in detections.xyxy:
        masks, scores, _ = sam.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])

    detections.mask = np.array(result_masks)

    h, w = image.shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=Config.DEVICE)
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    for i in np.flip(np.argsort(detections.area)):
        mask = detections.mask[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
            curr_id += 1

    return output_mask, segments_info




@torch.inference_mode()
def process_frame_with_text(YSDT: DEVAInferenceCore,
                            sam_model: SamPredictor,
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
    need_resize = new_min_side > 0
    image = get_input_frame_for_YSDT(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    if ti + cfg['num_voting_frames'] > YSDT.next_voting_frame:
        mask, segments_info = make_segmentation_with_YOLO(cfg, image_np, sam_model, new_min_side)
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