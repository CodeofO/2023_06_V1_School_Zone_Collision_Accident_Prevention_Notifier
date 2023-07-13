import os
import random
import pandas as pd
import numpy as np

import cv2
from ultralytics import YOLO
from tracker_testing import Tracker


# NOTE : detect_type 별 성능평가
# weight(2.5) : 상, 
    # 움직이는 객체만 인식
    # 깜빡거림 존재
# square(3) : 중 ud*1.5
    # 간혹 다른 객체도 인식
    # 깜빡거림 존재
# exp : 상, e^n
    # 간혹 다른 객체도 인식
    # 깜빡거림 적음
# exp2 : 최상, 2^n
    # 간혹 다른 객체도 인식
    # 깜빡거림 적음



data_path = '~/OT2/object_tracking_yolov8_deep_sort'

video1 = 'blur_person'
video2 = 'blur_car'
detection_threshold = 0.5
detect_type = 'exp2' # detect_type : weight, square, exp, exp2, expm1
#weight = 1.5 # weight : 2.5 / square : 3 / exp, exp2 : 1~2
weight = 4
ud_square = 1
warning_count = 10
alraming_time = 150 # 300 ~ 500

# etc
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
cl_name = pd.read_csv(os.path.join(data_path, 'coco_classes.txt'))
class_names = cl_name['class_name'].tolist()


def load_video(data_path, vidoe_name):
    # video_path
    video_path = os.path.join(data_path, 'data', f"{vidoe_name}.mp4")
    
    return cv2.VideoCapture(video_path)


def yolov8(result, detection_threshold, class_names, target_classes):
            
    detections = []

    for r in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        class_id = int(class_id)
        
        class_name = class_names[class_id] # 🍭
        
        #if (score > detection_threshold) & cond_class :
        if (score > detection_threshold) & (class_name in target_classes) : # 🐣
            detections.append([x1, y1, x2, y2, score, class_id]) # ⛳️


    return detections


def first_track(tracker, sequence, class_names, before_track_infos1, before_track_infos2, target_classes): # tracker : tracker1 or tracker2 / sequence : 1 or 2
    for class_name in target_classes: # 🌍        
        globals()[f'ud_list_{class_name}'] = [] # 🌍
    

    for track in tracker.tracks:
        x1, y1, x2, y2 = track.bbox
        track_id = track.track_id
        #track_ids.append(track_id) # 🍭
        class_id = track.class_id # ⛳️
        class_name = class_names[class_id]

        # Center point
        bbox_center = ((x2 - x1),(y2 - y1))
        
        if sequence == 1:
            before_track_infos = before_track_infos1
        elif sequence == 2:
            before_track_infos = before_track_infos2

        # Compare to Before
        if len(before_track_infos) > 0:

            for info in before_track_infos:

                #if track_id == info[1]:
                if (track_id == info[1]) & (class_name in target_classes):
                        bx, by = info[0]
                        x, y = bbox_center
                        
                        # Computing Uclidian Distance Between Before and Now
                        ud = np.sqrt((bx - x)**2 + (by - y)**2)
                        ud = ud ** ud_square
                        globals()[f'ud_list_{class_name}'].append(ud) # 🐣
    
    # define mean of Uclidian Distances
    ud_mean_dict = dict() # 🌍
    try:
        for class_name in target_classes:
            ud_mean_dict[class_name] = np.mean(globals()[f'ud_list_{class_name}']) # 🌍
    except:
        pass
    
    return ud_mean_dict


def second_track(frame, tracker, sequence, detect_type, ud_mean_dict, warning_text, before_track_infos1, before_track_infos2, warning_count_down, weight=weight, ud_square=ud_square):
            
    # make list for recording frame's tracking infos
    tracks_info = []
    track_ids = [] # 🌍

    # for track 2️⃣
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        track_ids.append(track_id) # 🌍
        class_id = track.class_id # ⛳️
        class_name = class_names[class_id]  # 클래스 ID에 해당하는 클래스 이름 가져오기                

        for key, value in ud_mean_dict.items():
            if key == class_name:
                ud_list_mean = value
        
        # center 
        bbox_center = ((x2 - x1),(y2 - y1))
        
        # record tracks_info
        info = [bbox_center, track_id, class_id]
        tracks_info.append(info)

        if sequence == 1:
            before_track_infos = before_track_infos1
        elif sequence == 2:
            before_track_infos = before_track_infos2

        for info in before_track_infos:
            if track_id == info[1]:                        

                # Computing Uclidian Distance
                bx, by = info[0]
                x, y = bbox_center
                ud = np.sqrt((bx - x) ** 2 + (by - y) ** 2)
                ud = ud ** ud_square


                #Key CODE
                    # 하이퍼 파라미터 
                        # 사용
                            # 1) 기울기 w
                        # 사용 하지 않음
                            # 1) 세제곱
                            # 2) tanh
                            # 3) 자연함수 e
                
                # 차에 비해서 사람은 Identifying Moving Object이 잘 안됨. 따라서 가중치 부여
                if class_name == 'person':
                    ud = np.exp2(ud + (weight * 2)) # ⭕️ # 사람과 차가 겹쳤을 때 사람에게 더 강한 가중치를 주기 위함

                if detect_type == 'weight':
                    cond_ud = (ud > ud_list_mean * weight)
                elif detect_type == 'square':
                    cond_ud = (ud ** weight > ud_list_mean ** weight)
                elif detect_type == 'exp':
                    cond_ud = (np.exp(ud) > np.exp(ud_list_mean) * weight)
                elif detect_type == 'exp2':
                    cond_ud = (np.exp2(ud) > np.exp2(ud_list_mean) * weight) # 🐣
                    
                elif detect_type == 'expm1':
                    cond_ud = (np.expm1(ud) > np.expm1(ud_list_mean) * weight)
                
                # Identifying Moving Object
                if cond_ud: # 🍭

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, f"MOVING_{track_id}_{class_name}",                                         
                                (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                #colors[track_id % len(colors)], 
                                (0, 0, 255), 
                                thickness=2)
                    
                    print(f'A moving {class_name} detected!!!')
                    
                    # Moving Object에 WARNING 부여                            
                    # track_id_{track_id}_count 정의
                    try:
                        globals()[f'track_id_{track_id}_count_{sequence}'] += 1 # 🍭
                    except:
                        globals()[f'track_id_{track_id}_count_{sequence}'] = 1 # 🍭
                                                                                            
                    
                    # WARNING을 주는 기준 : warning_count
                    # warning 조건 달성
                    if globals()[f'track_id_{track_id}_count_{sequence}'] >= warning_count: # 🌈 🌍
                        cv2.putText(frame, "WARNING", 
                                    (int(x1), int(y1) - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    #2.5, (0, 0, 255), thickness=3) # 🌈
                                    2.5, (0, 165, 255), thickness=3) # 🌈
                        
                        print(f'Watch out for {class_name}!!!')

                        # 만약 하나의 track_id 에서 WARNING이 발생한다면 나머지는 0으로 초기화
                        for i in track_ids:  # 🍭
                            if track_id == i: # 🌈
                                pass # 🌈
                            else: # 🌈
                                #globals()[f'track_id_{track_id}_count_{sequence}'] = 0  # 🌈
                                globals()[f'track_id_{track_id}_count_{sequence}'] = globals()[f'track_id_{track_id}_count_{sequence}'] // 2 # ⭕️


                        # LED에 신호를 주는 변수 # 🐣    
                        if (class_name == 'car') | (class_name == 'motorbike') | (class_name == 'truck'): # 🐣
                            warning_text = 'car' # 🐣
                        elif (class_name == 'person') | (class_name == 'bicycle'): # 🐣
                            warning_text = 'person' # 🐣
                                                    
                    # warning 조건 미달성
                    else: 
                        pass
                        
                else: 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    cv2.putText(frame, f"Stopping{track_id}_{class_name}",                                         
                                (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                colors[track_id % len(colors)], 
                                thickness=2)
            
    return [tracks_info, warning_text]


def YOLO_OB_OT_moving(data_path, vidoe_name1, vidoe_name2, 
                      detection_threshold = 0.5, 
                      detect_type = 'None', 
                      weight = None, 
                      ud_square = 1, 
                      warning_count = 10, 
                      alraming_time = 500):

    
    cap1 = load_video(data_path, vidoe_name1) # 🌍
    cap2 =  load_video(data_path, vidoe_name2) # 🌍
    
    ret1, frame1 = cap1.read() # 🌍
    ret2, frame2 = cap2.read() # 🌍

    video_out_path = os.path.join(data_path, 'data', 'output', f'{vidoe_name1}_th{detection_threshold}_{detect_type}_hp{weight}_us{ud_square}_wc{warning_count}_at{alraming_time}.mp4') # 🌍
    
    final_frame_shape = (1500, 500) # 🍭
    cap_out = cv2.VideoWriter(video_out_path, 
                              cv2.VideoWriter_fourcc(*'MP4V'), 
                              cap1.get(cv2.CAP_PROP_FPS), # 🌍 cap1의 프레임속도
                              (final_frame_shape[0], final_frame_shape[1])) # 🍭
    
    # define model
    model = YOLO("yolov8n.pt")
    tracker1 = Tracker() # 🌍
    tracker2 = Tracker() # 🌍
    
    # etc
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
    cl_name = pd.read_csv(os.path.join(data_path, 'coco_classes.txt'))
    class_names = cl_name['class_name'].tolist()

    # live monitoring
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL) # GPT 추가 : 실시간 모니터링
    cv2.resizeWindow("Tracking", final_frame_shape[0], final_frame_shape[1]) # GPT 추가 : 실시간 모니터링

    monitor_width = 1500
    monitor_height = 500

    before_track_infos1 = []
    before_track_infos2 = []

    # LED : WANRING 신호를 주는 기준
    warning_text = ''  # 🍭
    warning_text1 = ''
    warning_text2 = '' 
    warning_count_down = 0
    c = 0

    #  
    target_classes = ['car', 'person', 'motorbike', 'bicycle', 'truck'] # 🐣

    while ret1 and ret2:

        for result1, result2 in zip(model(frame1, device="mps"), model(frame2, device="mps")):
            detections1 = []
            detections2 = []

            detections1 = yolov8(result1, detection_threshold, class_names, target_classes)
            detections2 = yolov8(result2, detection_threshold, class_names, target_classes)

            tracker1.update(frame1, detections1)
            first_track1_dict = first_track(tracker1, 1, class_names, before_track_infos1 = before_track_infos1, before_track_infos2 = before_track_infos2, target_classes = target_classes)
            second_track1 = second_track(frame1, tracker1, 1, detect_type, first_track1_dict, warning_text=warning_text, warning_count_down=warning_count_down, before_track_infos1 = before_track_infos1, before_track_infos2 = before_track_infos2)
            
            tracker2.update(frame2, detections2)
            first_track2_dict = first_track(tracker2, 2, class_names, before_track_infos1 = before_track_infos1, before_track_infos2 = before_track_infos2, target_classes = target_classes)
            second_track2 = second_track(frame2, tracker2, 2, detect_type, first_track2_dict, warning_text=warning_text, warning_count_down=warning_count_down, before_track_infos1 = before_track_infos1, before_track_infos2 = before_track_infos2)
                        
            before_track_infos1 = second_track1[0].copy()
            before_track_infos2 = second_track2[0].copy()
            


        # Resize frame # 🌈
        dvided_width = monitor_width // 3
        frame1_resized = cv2.resize(frame1, (dvided_width, monitor_height)) # 🌍
        frame2_resized = cv2.resize(frame2, (dvided_width, monitor_height)) # 🌍

        # Create final frame # 🌈
        final_frame = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8) # 🐣
        
        # 1) left 
        final_frame[:, :dvided_width] = frame1_resized # 🐣
        # 2) middle 
        final_frame[:, dvided_width:dvided_width * 2] = np.zeros((monitor_height, dvided_width, 3), dtype=np.uint8) # 🐣
        
        # 3) right
        final_frame[:, dvided_width * 2:] = frame2_resized # 🌍



        # 양쪽 모두 warning 발생 시 second_track1[2] 업데이트 됨
        if (second_track1[1] != '') & (second_track2[1] != ''): # 두 영상에 모두 WARNING 신호를 줄 때 # ⭕️
            c += 1
            warning_text1 = second_track1[1]
            warning_text2 = second_track2[1]
            

            try:  # ⭕️
                if cond1: # 알람이 울리는 중 : warning_count_down 업데이트 X
                    pass
                else: # 알람이 마치면 : warning_count_down 업데이트 O
                    warning_count_down = alraming_time # 💌
            except:
                warning_count_down = alraming_time # 💌
        
        if c == 1: # ⭕️
            real_warning_text1 = warning_text1
            real_warning_text2 = warning_text2

        # cond : 
        cond1 = ((alraming_time * 2 // 4) < warning_count_down) & (warning_count_down <= alraming_time) # ⭕️
        # 복잡한 상황에서는 너무 같은 신호를 너무 오래 제공 : alraming_time * 2 // 5) < warning_count_down
        
        if cond1 : # ⭕️
            real_warning_text1 = real_warning_text1
            real_warning_text2 = real_warning_text2
            warning_count_down -= 1
        else: # ⭕️
            real_warning_text1 = warning_text1
            real_warning_text2 = warning_text2
    
        n = 6
        cond2_on = warning_count_down % (alraming_time // n) >= ((alraming_time // n) // 2) # ⭕️
        
        if cond2_on: # ⭕️
            start_warning_alram = 1 
        else: 
            start_warning_alram = 0 



        car_path = './car.png'
        person_path = './person.png'
        warn_path = './warn.png'
        crash_path = './crash.png'

        car_img = cv2.imread(car_path)
        car_img = cv2.resize(car_img, (200, 300))

        person_img = cv2.imread(person_path)
        person_img = cv2.resize(person_img, (200, 300))

        crash_img = cv2.imread(crash_path)
        crash_img = cv2.resize(crash_img, (100, 200))

        warn_img = cv2.imread(warn_path)
        warn_img = cv2.resize(warn_img, (500, 200))

        
        if (cond1) & (start_warning_alram  == 1): # 🍭 # 🐣  # ⭕️

            # left
            if real_warning_text1 == 'car':
                final_frame[:300, 500:700] = car_img
            elif real_warning_text1 == 'person':
                final_frame[:300, 500:700] = person_img
                
            # right
            if real_warning_text2 == 'car':                
                final_frame[:300, 800:1000] = car_img
            elif real_warning_text2 == 'person':
                final_frame[:300, 800:1000] = person_img
            
            final_frame[50:250, 700:800] = crash_img
            final_frame[300:500, 500:1000] = warn_img
            

        # live monitoring
        cv2.imshow("Tracking", final_frame) # 🌈
        cap_out.write(final_frame) # 🍭
        if cv2.waitKey(1) == ord('q'): 
            break

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

    cap1.release()
    cap2.release()
    cap_out.release()
    cv2.destroyAllWindows()


#for video in vidoe_name_list:
YOLO_OB_OT_moving(data_path=data_path,
                vidoe_name1=video1, 
                vidoe_name2=video2, 
                detection_threshold = detection_threshold, 
                detect_type=detect_type,
                weight=weight,
                ud_square = ud_square,
                warning_count = warning_count,
                alraming_time = alraming_time)