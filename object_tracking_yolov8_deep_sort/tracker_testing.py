from deep_sort.deep_sort.tracker_testing import Tracker as DeepSortTracker
#from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker

from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching  # 거리 측정 메트릭
from deep_sort.deep_sort.detection_testing import Detection
#from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:

    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        # mars-small128.pb 의 PATH
        #encoder_model_filename = 'model_data/mars-small128.pb'
        encoder_model_filename = '~/object_tracking_yolov8_deep_sort/resources/networks/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # 거리 측정 메트릭 : NearestNeighborDistanceMetric()
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1) # 객체 추적을 위한 신경망 로드

    def update(self, frame, detections):

        #bboxes = np.asarray([d[:-1] for d in detections])
        bboxes = np.asarray([d[:-2] for d in detections])
        try: # 추가 : IndexError 나는 부분 무시함
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        except IndexError:
            pass

        scores = [d[-2] for d in detections]
        class_ids = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)
        
        dets = []
        for bbox_id, bbox in enumerate(bboxes):

            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id], class_ids[bbox_id])) # ⛳️
            #dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()  
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []

        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id = track.class_id # ⛳️

            tracks.append(Track(track_id, bbox, class_id)) # ⛳️
            #tracks.append(Track(track_id, bbox))

        self.tracks = tracks

class Track:

    #class_id = None # 클래스 속성으로 정의

    track_id = None
    bbox = None
    class_id = None # ⛳️

    def __init__(self, id, bbox, class_id): # ⛳️
    #def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id # ⛳️