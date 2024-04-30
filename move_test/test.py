import cv2
import numpy as np
import mediapipe as mp

def getLength(landmark_15_x, landmark_13_x, landmark_15_y,landmark_13_y):
    distance = np.sqrt((landmark_15_x - landmark_13_x) ** 2 + (landmark_15_y - landmark_13_y) ** 2)
    return distance

def getAngle(landmark_15_x, landmark_13_x, landmark_15_y,landmark_13_y):
    angle = - np.arctan2(landmark_15_y - landmark_13_y, landmark_15_x - landmark_13_x) * 180 / np.pi
    return angle

def getImage(result, A, B):
    tmp_image = cv2.imread('a.png', cv2.IMREAD_UNCHANGED)
    img_tmp = cv2.imread('back.png', cv2.IMREAD_UNCHANGED)
    center_x = tmp_image.shape[1] // 2
    center_y = tmp_image.shape[0] // 2

    landmarks_pose = result.pose_landmarks.landmark

    # Calculate coordinates of landmarks 13 and 15
    landmark_a_x = int(landmarks_pose[A].x * w)
    landmark_a_y = int(landmarks_pose[A].y * h)
    landmark_b_x = int(landmarks_pose[B].x * w)
    landmark_b_y = int(landmarks_pose[B].y * h)

    # Calculate distance between landmarks 13 and 15
    distance = getLength(landmark_b_x, landmark_a_x, landmark_b_y, landmark_a_y)

    # Calculate image scaling factor based on distance
    scale_factor = distance / tmp_image.shape[1]
    resized_image = cv2.resize(tmp_image, (0, 0), fx=scale_factor, fy=scale_factor)

    # Calculate rotation angle based on the two landmarks
    angle = getAngle(landmark_b_x, landmark_a_x, landmark_b_y, landmark_a_y)
    M = cv2.getRotationMatrix2D((center_x * scale_factor, center_y * scale_factor), angle, 1)
    rotated_image = cv2.warpAffine(resized_image, M, (int(resized_image.shape[1]), int(resized_image.shape[0])))

    middle_x = (landmark_a_x + landmark_b_x)//2
    middle_y = (landmark_a_y + landmark_b_y)//2
    half_length = rotated_image.shape[0]//2

    overlay_start_x = middle_x - half_length
    overlay_start_y = middle_y - half_length
    overlay_end_x = middle_x + half_length
    overlay_end_y = middle_y + half_length

    overlay_width = overlay_end_x - overlay_start_x
    overlay_height = overlay_end_y - overlay_start_y

    if overlay_width > 0 and overlay_height > 0:

        # Affine transformation
        mtrx = np.float32([[1, 0, overlay_start_x],
                        [0, 1, overlay_start_y]]) 
        img_tmp = cv2.warpAffine(rotated_image, mtrx, (overlay_end_x, overlay_end_y), None, \
                    cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0,0,0) )
        
        # 오른쪽과 아래쪽에 padding 추가
        pad_right = w - (overlay_end_x)
        pad_bottom = h - (overlay_end_y)
        img_tmp = cv2.copyMakeBorder(img_tmp, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGBA2BGR)
        
        return img_tmp


# Mediapipe 모델 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 비디오 캡처
cap = cv2.VideoCapture("12963-243165477_small.mp4")

# 비디오 프레임 속성
fps = cap.get(cv2.CAP_PROP_FPS)

# 이미지 로드 및 초기화

while cap.isOpened():   # 비디오 파일 처리 루프
    ret, img = cap.read()   # 프레임 불러와서
    
    if not ret:
        break
    
    # 영상 반전 및 RGB 변환
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 포즈 탐지
    result = holistic.process(img_rgb)
    
    # 이미지 크기
    h, w, _ = img.shape
    
    if result.pose_landmarks is not None:  # Detect pose landmarks
        imgt = getImage(result, 13, 15)
        img = cv2.add(img, imgt)
        
        imgt = getImage(result, 11, 13)
        img = cv2.add(img, imgt)

        imgt = getImage(result, 12, 14)
        img = cv2.add(img, imgt)

        imgt = getImage(result, 14, 16)
        img = cv2.add(img, imgt)

        imgt = getImage(result, 24, 26)
        img = cv2.add(img, imgt)

        imgt = getImage(result, 23, 25)
        img = cv2.add(img, imgt)

        imgt = getImage(result, 26, 28)
        img = cv2.add(img, imgt)

        imgt = getImage(result, 25, 27)
        img = cv2.add(img, imgt)

        # landmarks_pose = result.pose_landmarks.landmark
        
        # # Calculate coordinates of landmarks 13 and 15
        # landmark_13_x = int(landmarks_pose[13].x * w)
        # landmark_13_y = int(landmarks_pose[13].y * h)
        # landmark_15_x = int(landmarks_pose[15].x * w)
        # landmark_15_y = int(landmarks_pose[15].y * h)

        # # Calculate distance between landmarks 13 and 15
        # distance = getLength(landmark_15_x, landmark_13_x, landmark_15_y, landmark_13_y)

        # # Calculate image scaling factor based on distance
        # scale_factor = distance / tmp_image.shape[1]
        # resized_image = cv2.resize(tmp_image, (0, 0), fx=scale_factor, fy=scale_factor)

        # # Calculate rotation angle based on the two landmarks
        # angle = getAngle(landmark_15_x, landmark_13_x, landmark_15_y, landmark_13_y)
        # M = cv2.getRotationMatrix2D((center_x * scale_factor, center_y * scale_factor), angle, 1)
        # rotated_image = cv2.warpAffine(resized_image, M, (int(resized_image.shape[1]), int(resized_image.shape[0])))

        # middle_x = (landmark_13_x + landmark_15_x)//2
        # middle_y = (landmark_13_y + landmark_15_y)//2
        # half_length = rotated_image.shape[0]//2

        # overlay_start_x = middle_x - half_length
        # overlay_start_y = middle_y - half_length
        # overlay_end_x = middle_x + half_length
        # overlay_end_y = middle_y + half_length

        # overlay_width = overlay_end_x - overlay_start_x
        # overlay_height = overlay_end_y - overlay_start_y

        # if overlay_width > 0 and overlay_height > 0:

        #     # Affine transformation
        #     mtrx = np.float32([[1, 0, overlay_start_x],
        #                     [0, 1, overlay_start_y]]) 
        #     img2 = cv2.warpAffine(rotated_image, mtrx, (overlay_end_x, overlay_end_y), None, \
        #                 cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0,0,0) )
            
        #     # 오른쪽과 아래쪽에 padding 추가
        #     pad_right = w - (overlay_end_x)
        #     pad_bottom = h - (overlay_end_y)
        #     img2 = cv2.copyMakeBorder(img2, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        #     img2 = cv2.add(img, img2)

    # 이미지 출력
    cv2.imshow('move', img)

    # 키 입력 감지
    if cv2.waitKey(1) == ord('q'): # q 누르면 창 닫힘
        break

# 비디오 캡처 종료
cap.release()
cv2.destroyAllWindows()
