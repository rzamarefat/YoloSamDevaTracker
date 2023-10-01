import inspect
import os
from YSDT.ext.with_text_processor import process_frame_with_text as process_frame_text
import tempfile
import cv2
import os
from os import path
from argparse import ArgumentParser
import torch
import numpy as np
from YSDT.model.network import DEVA
from YSDT.inference.inference_core import DEVAInferenceCore
from YSDT.inference.result_utils import ResultSaver
from YSDT.inference.eval_args import add_common_eval_args, get_model_and_config
from YSDT.inference.demo_utils import flush_buffer
from YSDT.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from YSDT.ext.sam import get_sam_model
from tqdm import tqdm
import json
from YSDT import Config

torch.autograd.set_grad_enabled(False)

# for id2rgb
np.random.seed(42)

# default parameters
parser = ArgumentParser()
add_common_eval_args(parser)
add_ext_eval_args(parser)
add_text_default_args(parser)

# load model and config
args = parser.parse_args([])
cfg = vars(args)
cfg['enable_long_term'] = Config.DEVA.ENABLE_LONG_TERM
cfg['enable_long_term_count_usage'] = Config.DEVA.ENABLE_LONG_TERM_COUNT_USAGE
cfg['max_num_objects'] = Config.DEVA.MAX_NUM_OBJECTS
cfg['size'] = Config.VIDEO_SIZE
cfg['amp'] = Config.DEVA.AMP
cfg['chunk_size'] = Config.DEVA.CHUNK_SIZE
cfg['detection_every'] = Config.DEVA.DETECTION_EVERY
cfg['max_missed_detection_count'] = Config.DEVA.MAX_MISSED_DETECTION_COUNT
cfg['sam_variant'] = 'original'
cfg['pluralize'] = Config.DEVA.PLURALIZE

SOURCE_VIDEO_PATH = Config.SOURCE_VIDEO_PATH
OUTPUT_VIDEO_PATH = Config.OUTPUT_VIDEO_PATH


class YoloSamDevaTracker:
    
    def __init__(self) -> None:
        # Load our checkpoint
        self.YSDT_model = DEVA(cfg).cuda().eval()
        if args.model is not None:
            model_weights = torch.load(args.model)
            self.YSDT_model.load_weights(model_weights)
        else:
            print('No model loaded.')

        self.sam_model = get_sam_model(cfg, Config.DEVICE)
        self.YSDT = DEVAInferenceCore(self.YSDT_model, config=cfg)
        self.YSDT.next_voting_frame = cfg['num_voting_frames'] - 1
        self.YSDT.enabled_long_id()

    
    def track(self):
        # obtain temporary directory
        result_saver = ResultSaver(None, None, dataset='gradio', object_manager=self.YSDT.object_manager)
        writer_initizied = False
        
        cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ti = 0
        
        with torch.cuda.amp.autocast(enabled=cfg['amp']):
            with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        if not writer_initizied:
                            h, w = frame.shape[:2]
                            if Config.MODE == 'development':
                                writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'vp80'), fps, (w, h))
                            else:
                                writer = cv2.VideoWriter()
                            writer_initizied = True
                            result_saver.writer = writer
                        
                        process_frame_text(self.YSDT,
                                            self.sam_model,
                                            'null.png',
                                            result_saver,
                                            ti,
                                            image_np=frame)
                        ti += 1
                        pbar.update(1)
                    else:
                        break
            flush_buffer(self.YSDT, result_saver)
        n_pigs = len(result_saver.object_manager.all_historical_object_ids)
        writer.release()
        cap.release()
        self.YSDT.clear_buffer()
        return n_pigs 
        

if __name__ == '__main__':
    YSDT = YoloSamDevaTracker()
    print(YSDT.track())