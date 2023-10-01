from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import torchvision
import os
from os import path
from PIL import Image, ImagePalette,ImageDraw, ImageFont
import pycocotools.mask as mask_util
from threading import Thread
from queue import Queue
from dataclasses import dataclass
import copy

import numpy as np
import supervision as sv

from YSDT.utils.pano_utils import ID2RGBConverter
from YSDT.inference.object_manager import ObjectManager
from YSDT.inference.object_info import ObjectInfo
import traceback
from YSDT import Config

class ResultSaver:
    def __init__(self,
                 output_root: str,
                 video_name: str,
                 *,
                 dataset: str,
                 object_manager: ObjectManager,
                 palette: Optional[ImagePalette.ImagePalette] = None):
        self.output_root = output_root
        self.video_name = video_name
        self.dataset = dataset.lower()
        self.palette = palette
        self.object_manager = object_manager

        self.need_remapping = False
        self.json_style = None
        self.output_postfix = None
        self.visualize = False

        if self.dataset == 'vipseg':
            self.all_annotations = []
            self.video_json = {'video_id': video_name, 'annotations': self.all_annotations}
            self.need_remapping = True
            self.json_style = 'vipseg'
            self.output_postfix = 'pan_pred'
        elif self.dataset == 'burst':
            self.id2rgb_converter = ID2RGBConverter()
            self.need_remapping = True
            self.all_annotations = []
            dataset_name = path.dirname(video_name)
            seq_name = path.basename(video_name)
            self.video_json = {
                'dataset': dataset_name,
                'seq_name': seq_name,
                'segmentations': self.all_annotations
            }
            self.json_style = 'burst'
        elif self.dataset == 'unsup_davis17':
            self.need_remapping = True
        elif self.dataset == 'ref_davis':
            # nothing special is required
            pass
        elif self.dataset == 'demo':
            self.need_remapping = True
            self.all_annotations = []
            self.video_json = {'annotations': self.all_annotations}
            self.json_style = 'vipseg'
            self.visualize = True
            self.visualize_postfix = 'Visualizations'
            self.output_postfix = 'Annotations'
        elif self.dataset == 'gradio':
            # minimal mode, expect a cv2.VideoWriter to be assigned to self.writer asap
            self.writer = None
            self.need_remapping = True
            self.visualize = True
        else:
            raise NotImplementedError

        if self.need_remapping:
            self.id2rgb_converter = ID2RGBConverter()

        self.queue = Queue(maxsize=10)
        self.thread = Thread(target=save_result, args=(self.queue, ))
        self.thread.daemon = True
        self.thread.start()

    def save_mask(self,
                  prob: torch.Tensor,
                  frame_name: str,
                  need_resize: bool = False,
                  shape: Optional[Tuple[int, int]] = None,
                  save_the_mask: bool = True,
                  image_np: np.ndarray = None,
                  path_to_image: str = None):

        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
                                                                                                 0]
        # Probability mask -> index mask
        mask = torch.argmax(prob, dim=0)

        args = ResultArgs(
            saver=self,
            mask=mask.cpu(),
            frame_name=frame_name,
            save_the_mask=save_the_mask,
            image_np=image_np,
            path_to_image=path_to_image,
            tmp_id_to_obj=copy.deepcopy(self.object_manager.tmp_id_to_obj),
            obj_to_tmp_id=copy.deepcopy(self.object_manager.obj_to_tmp_id),
            segments_info=copy.deepcopy(self.object_manager.get_current_segments_info()),
        )

        self.queue.put(args)

    def end(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()


@dataclass
class ResultArgs:
    saver: ResultSaver
    mask: torch.Tensor
    frame_name: str
    save_the_mask: bool
    image_np: np.ndarray
    path_to_image: str
    tmp_id_to_obj: Dict[int, ObjectInfo]
    obj_to_tmp_id: Dict[ObjectInfo, int]
    segments_info: List[Dict]

def detect_spot(img_np, target_color):
    mask = np.all(img_np == target_color, axis=-1)
    coordinates = np.argwhere(mask)
    center = np.round(coordinates.mean(axis=0)).astype(int)
    return center

def save_result(queue: Queue):
    while True:
        args: ResultArgs = queue.get()
        if args is None:
            queue.task_done()
            break

        saver = args.saver
        mask = args.mask
        frame_name = args.frame_name
        save_the_mask = args.save_the_mask
        image_np = args.image_np
        path_to_image = args.path_to_image
        tmp_id_to_obj = args.tmp_id_to_obj
        obj_to_tmp_id = args.obj_to_tmp_id
        segments_info = args.segments_info
        all_obj_ids = [k.id for k in obj_to_tmp_id]
            
        # remap indices
        if saver.need_remapping:
            new_mask = torch.zeros_like(mask)
            for tmp_id, obj in tmp_id_to_obj.items():
                new_mask[mask == tmp_id] = obj.id
            mask = new_mask

        # record output in the json file
        if saver.json_style == 'vipseg':
            for seg in segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = {
                'file_name': frame_name[:-4] + '.jpg',
                'segments_info': segments_info,
            }
            saver.all_annotations.append(this_annotation)
        elif saver.json_style == 'burst':
            for seg in segments_info:
                seg['mask'] = mask == seg['id']
                seg['area'] = int(seg['mask'].sum())
                coco_mask = mask_util.encode(np.asfortranarray(seg['mask'].numpy()))
                coco_mask['counts'] = coco_mask['counts'].decode('utf-8')
                seg['rle_mask'] = coco_mask
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = {
                'file_name':
                frame_name[:-4] + '.jpg',
                'segmentations': [{
                    'id': seg['id'],
                    'score': seg['score'],
                    'rle': seg['rle_mask'],
                } for seg in segments_info],
            }
            saver.all_annotations.append(this_annotation)
        elif saver.visualize:
            # if we are visualizing, we need to preprocess segment info
            for seg in segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]

        # save the mask to disk
        if save_the_mask:
            if saver.object_manager.use_long_id:
                out_mask = mask.numpy().astype(np.uint32)
                rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
                for id in all_obj_ids:
                    colored_mask = saver.id2rgb_converter._id_to_rgb(id)
                    
                    import random
                    # print("◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙")
                    # print(np.max(out_mask))
                    # print("◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙")
                    # temp_rgb = Image.fromarray(colored_mask)
                    # temp_rgb.save(f'/home/rmarefat/projects/tenta/Pig/DEVA/out_img_{random.randint(100,500)}.jpg')
                    
                    obj_mask = (out_mask == id)
                    rgb_mask[obj_mask] = colored_mask
                    # try:
                    #     center = detect_spot(rgb_mask,colored_mask)
                    #     tmp_mask = Image.fromarray(rgb_mask)
                    #     draw = ImageDraw.Draw(tmp_mask)
                    #     font = ImageFont.truetype("/home/rmarefat/projects/tenta/Pig/DEVA/soehne-mono-halbfett.ttf", 16)
                    #     inv = (255-colored_mask[0],255-colored_mask[1],255-colored_mask[2])
                    #     draw.text(center, str(id), font=font, fill=inv)
                    #     rgb_mask = np.array(tmp_mask)
                    # except:
                    #     print(traceback.format_exc())
                    #     # exit()
                    #     print("◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙◙")
                    
                out_img = Image.fromarray(rgb_mask)
            else:
                if saver.palette is not None:
                    out_img.putpalette(saver.palette)
            
            

            if saver.dataset != 'gradio':
                # find a place to save the mask
                if saver.output_postfix is not None:
                    this_out_path = path.join(saver.output_root, saver.output_postfix)
                else:
                    this_out_path = saver.output_root
                if saver.video_name is not None:
                    this_out_path = path.join(this_out_path, saver.video_name)

                os.makedirs(this_out_path, exist_ok=True)
                out_img.save(path.join(this_out_path, frame_name[:-4] + '.png'))

            # if saver.visualize:
            if Config.MODE == 'development':
                if image_np is None:
                    if path_to_image is not None:
                        image_np = np.array(Image.open(path_to_image))
                    else:
                        raise ValueError('Cannot visualize without image_np or path_to_image')
                alpha = (out_mask == 0).astype(np.float32) * 0.5 + 0.5
                alpha = alpha[:, :, None]
                blend = (image_np * alpha + rgb_mask * (1 - alpha)).astype(np.uint8)

                # if prompts is not None:
                if True:
                    # draw bounding boxes for the prompts
                    all_masks = []
                    labels = []
                    all_cat_ids = []
                    all_scores = []
                    try:
                        for seg in segments_info:
                            all_masks.append(mask == seg['id'])
                            # labels.append(f'{prompts[seg["category_id"]]} {seg["score"]:.2f}')
                            labels.append(f'{seg["id"]}')
                            all_cat_ids.append(seg['category_id'])
                            all_scores.append(seg['score'])
                        if len(all_masks) > 0:
                            all_masks = torch.stack(all_masks, dim=0)
                            xyxy = torchvision.ops.masks_to_boxes(all_masks)
                            xyxy = xyxy.numpy()

                            detections = sv.Detections(xyxy,
                                                    confidence=np.array(all_scores),
                                                    class_id=np.array(all_cat_ids))
                            annotator = sv.BoxAnnotator()
                            blend = annotator.annotate(scene=blend,
                                                    detections=detections,
                                                    labels=labels)
                    except Exception as e:
                        print(traceback.format_exc())

                if saver.dataset != 'gradio':
                    # find a place to save the visualization
                    if saver.visualize_postfix is not None:
                        this_out_path = path.join(saver.output_root, saver.visualize_postfix)
                    else:
                        this_out_path = saver.output_root
                    if saver.video_name is not None:
                        this_out_path = path.join(this_out_path, saver.video_name)

                    os.makedirs(this_out_path, exist_ok=True)
                    Image.fromarray(blend).save(path.join(this_out_path, frame_name[:-4] + '.jpg'))
                else:
                    saver.writer.write(blend[:, :, ::-1])

        queue.task_done()