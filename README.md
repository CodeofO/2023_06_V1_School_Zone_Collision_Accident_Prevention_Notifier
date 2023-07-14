# 🚸 V01. 스쿨존 충돌사고 예방 알리미 
  
<img width="993" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/32f6055b-b644-4b87-8c07-b83ad876675b">



* 프로젝트 : AI를 이용한 스쿨존(골목길) 충돌사고 예방 알고리즘
* 주최 : 울산대학교 캡스톤디자인
* 제안배경 : 매년 높아지는 스쿨존 교통사고를 예방하기 위한 아이디어입니다.
* 핵심 : 

##### **1. 좁은 길의 특성상 주차되어 있는 차량이 많아, 멈춰있는 객체, 움직이는 객체를 분리합니다.**  
<img width="777" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/8b79eb82-e3e4-4a1c-bb22-60391c8f9b6b">  
```  
if class_name == 'person':
                    ud = np.exp2(ud + (weight * 2)) # 사람과 차가 겹쳤을 때 사람에게 더 강한 가중치를 주기 위함

                if detect_type == 'weight':
                    cond_ud = (ud > ud_list_mean * weight)
                elif detect_type == 'square':
                    cond_ud = (ud ** weight > ud_list_mean ** weight)
                elif detect_type == 'exp':
                    cond_ud = (np.exp(ud) > np.exp(ud_list_mean) * weight)
                elif detect_type == 'exp2':
                    cond_ud = (np.exp2(ud) > np.exp2(ud_list_mean) * weight) 
                    
                elif detect_type == 'expm1':
                    cond_ud = (np.expm1(ud) > np.expm1(ud_list_mean) * weight)
                
                # Identifying Moving Object
                if cond_ud: 

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, f"MOVING_{track_id}_{class_name}",                                         
                                (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                #colors[track_id % len(colors)], 
                                (0, 0, 255), 
                                thickness=2)
                    
                    print(f'A moving {class_name} detected!!!')
```



##### **2. 움직이는 객체가 인식이 된 후 일정 조건이 되면 객체가 Warning상태가 됩니다.(조심해야 하는   객체라는 의미)**    
<img width="763" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/a33fc4fb-94f0-49d3-8acd-1ffbb4db9172">    
<img width="762" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/6eeb6c6b-2314-4b89-8553-b911fbebb21e">  
  
```  
    # Moving Object에 WARNING 부여                            
    # track_id_{track_id}_count 정의
    try:
        globals()[f'track_id_{track_id}_count_{sequence}'] += 1 
    except:
        globals()[f'track_id_{track_id}_count_{sequence}'] = 1 
                                                                            
    
    # WARNING을 주는 기준 : warning_count
    # warning 조건 달성
    if globals()[f'track_id_{track_id}_count_{sequence}'] >= warning_count: 
        cv2.putText(frame, "WARNING", 
                    (int(x1), int(y1) - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    #2.5, (0, 0, 255), thickness=3)
                    2.5, (0, 165, 255), thickness=3) 
        
        print(f'Watch out for {class_name}!!!')

        # 만약 하나의 track_id 에서 WARNING이 발생한다면 나머지는 0으로 초기화
        for i in track_ids:  
            if track_id == i:
                pass 
            else: 
                #globals()[f'track_id_{track_id}_count_{sequence}'] = 0  
                globals()[f'track_id_{track_id}_count_{sequence}'] = globals()[f'track_id_{track_id}_count_{sequence}'] // 2 


        # LED에 신호를 주는 변수 
        if (class_name == 'car') | (class_name == 'motorbike') | (class_name == 'truck'): 
            warning_text = 'car' 
        elif (class_name == 'person') | (class_name == 'bicycle'): 
            warning_text = 'person' 
                                    
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
```



##### **3. 양쪽에서 오는 두 객체가 Warning상태가 되면 알리미는 접근중인 객체의 종류에 맞게 Sign을 점등합니다.**  
<img width="488" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/1da67956-b34a-4fc6-81ef-c923b51fc9e8">  
```
        # 양쪽 모두 warning 발생 시 second_track1[2] 업데이트 됨
        if (second_track1[1] != '') & (second_track2[1] != ''): # 두 영상에 모두 WARNING 신호를 줄 때 
            c += 1
            warning_text1 = second_track1[1]
            warning_text2 = second_track2[1]
            

            try: 
                if cond1: # 알람이 울리는 중 : warning_count_down 업데이트 X
                    pass
                else: # 알람이 마치면 : warning_count_down 업데이트 O
                    warning_count_down = alraming_time # 💌
            except:
                warning_count_down = alraming_time # 💌
        
        if c == 1: 
            real_warning_text1 = warning_text1
            real_warning_text2 = warning_text2

        # cond : 
        cond1 = ((alraming_time * 2 // 4) < warning_count_down) & (warning_count_down <= alraming_time) 
        # 복잡한 상황에서는 너무 같은 신호를 너무 오래 제공 : alraming_time * 2 // 5) < warning_count_down
        
        if cond1 : 
            real_warning_text1 = real_warning_text1
            real_warning_text2 = real_warning_text2
            warning_count_down -= 1
        else: 
            real_warning_text1 = warning_text1
            real_warning_text2 = warning_text2
    
        n = 6
        cond2_on = warning_count_down % (alraming_time // n) >= ((alraming_time // n) // 2)
        
        if cond2_on:
            start_warning_alram = 1 
        else: 
            start_warning_alram = 0 
```

        
✅ **조감도**  
<img width="1146" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/224d4d83-f38b-417c-a15f-d039ec954553">
✅ **최종 결과물**    
<img width="1477" alt="image" src="https://github.com/CodeofO/2023_School_Zone_Collision_Accident_Prevention_Notifier/assets/99871109/9f7fa97d-426f-4fc0-af43-7b36dea14d50">
  
**Youtube 업로드**  
👉 https://youtu.be/BCXjsv-tun4
    
**< 참고 자료 >**  
git-hub : Yolov8   
👉 https://github.com/ultralytics/ultralytics  
  
git-hub : DeepSORT   
👉 https://github.com/nwojke/deep_sort  
  
youtube : Yolov8 object detection + deep sort object tracking | Computer vision tutorial   
👉 https://youtu.be/jIRRuGN0j5E  
