from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching  # 거리 측정 메트릭
from deep_sort.deep_sort.detection import Detection
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
        encoder_model_filename = '/Users/jeong-geun-o/Library/Mobile Documents/com~apple~CloudDocs/🎓University/4-1/봄학기/04. 융합캡스톤디자인/OT2/object_tracking_yolov8_deep_sort/resources/networks/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # 거리 측정 메트릭 : NearestNeighborDistanceMetric()
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1) # 객체 추적을 위한 신경망 로드

    def update(self, frame, detections):
        
        # x1, y1, x2, y2, score, class_id

        bboxes = np.asarray([d[:-1] for d in detections])
        try: # 추가 : IndexError 나는 부분 무시함
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        except IndexError:
            pass
        scores = [d[-1] for d in detections]
        
        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))


        # class ID
        #class_id = detections[-1]

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()
        

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            
            id = track.track_id
            # print(f"id : {id}\n bbox : {bbox}")
            tracks.append(Track(id, bbox))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    #class_id = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
        #self.class_id = # ??? #
