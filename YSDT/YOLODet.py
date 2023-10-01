from ultralytics import YOLO
import cv2
import numpy as np
from YSDT.inference.object_info import ObjectInfo
import torch
import traceback
import supervision as sv
from YSDT import Config


class YOLODet:
    def __init__(self):
        self._model = YOLO(Config.YOLO_MODEL)
    
    def _aggregate_masks(self, mask_list, shape):
        
        mask_shape = mask_list[0].shape
        
        mask_agg = np.zeros((720, 1280), dtype='uint8')
        
        for id, mask in enumerate(mask_list):
            mask *= 255.0
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask, (1280, 720))
            mask = mask.astype(np.float16)
            mask /= 255.
            mask = mask * (id + 1)
            mask_agg = mask_agg + mask
        mask_agg = mask_agg.astype(np.uint8)
        
        
        mask_agg = torch.from_numpy(mask_agg).to("cuda")
        
        return mask_agg
        
    def detect(self, image):
        # Load a model
        segment_info = []
        pred = self._model.predict(image, conf=0.4,classes=[2])
        try:
            boxes = pred[0].boxes.xyxy.cpu().numpy()
            conf = pred[0].boxes.conf.cpu().numpy()
            cls = pred[0].boxes.cls.cpu().numpy()
            cls = cls.astype('float64')
            # Detections(xyxy=array([[     118.75,      151.66,      776.69,      1078.2]], dtype=float32), mask=None, confidence=array([    0.76047], dtype=float32), class_id=array([0]), tracker_id=None)
            detections = sv.Detections(xyxy=boxes, confidence=conf)
            detections.class_id=cls
            detections.tracker_id=None
            return detections
            
        except Exception as e:
            # return torch.from_numpy(agg_mask).to("cuda"), []
            detections = sv.Detections(xyxy=np.array([],dtype='float32'), confidence=np.array([],dtype='float32'))
            detections.class_id=np.array([],dtype='float64')
            detections.tracker_id=None
            return detections
            
        
if __name__ == '__main__':
    img = cv2.imread("/home/rmarefat/projects/ultralytics/Screenshot___.png")
    ypd = YOLODet()
    ypd.detect(img)