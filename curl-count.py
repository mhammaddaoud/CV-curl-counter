import cv2
import mediapipe as mp
import numpy as np
import math

count = 1  #starting count with 1 hard coded based on used video
elbow = []
shoulder = []
wrist = []
shoulder_elbow = 0
up = False

mpPoseDetection = mp.solutions.pose

mpDraw = mp.solutions.drawing_utils

poseDetection = mpPoseDetection.Pose()

cap = cv2.VideoCapture('pose1.mp4')

# resizing video size
def resized(img,percent):
    himg, wimg, dimg = img.shape
    h, w = int(himg*percent), int(wimg*percent)

    img = cv2.resize(img, (w,h))
    
    return img

while (cap.isOpened()):
    success, frame = cap.read()

    himg, wimg, dimg = frame.shape

    imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = poseDetection.process(imgrgb)

    list = []

    # get pose land marks and append to list
    for lm in result.pose_landmarks.landmark:
        x, y = lm.x, lm.y
        cx, cy = int(x*wimg), int(y*himg)
        list.append([cx,cy])

    cv2.rectangle(frame, (10,10), (100,100), color = (0,0,255), thickness = -1)

    # take needed poses for shoulder, elbow, and wrist
    shoulder = list[11]
    elbow = list[13]
    wrist = list[15]

    ########################## angle calculations of arm joints ##########################)
    shoulder_elbow = math.sqrt(pow(shoulder[1] - elbow[1], 2) + pow(shoulder[0] - elbow[0], 2))
    shoulder_elbow = int(shoulder_elbow)

    elbow_wrist = math.sqrt(pow(elbow[1] - wrist[1], 2) + pow(elbow[0] - wrist[0], 2))
    elbow_wrist = int(elbow_wrist)

    shoulder_wrist = math.sqrt(pow(shoulder[1] - wrist[1], 2) + pow(shoulder[0] - wrist[0], 2))
    shoulder_wrist = int(shoulder_wrist)

    angle = math.acos((pow(shoulder_elbow, 2) + pow(elbow_wrist, 2) - pow(shoulder_wrist, 2)) / (2 * shoulder_elbow * elbow_wrist))
    angle = int(math.degrees(angle))
    ######################################################################################

    # counting reps if the arm went up then down we increase rep count
    if angle >= 140 and up == False:
        up = True
        count += 1

    elif angle < 140:
        count = count
        up = False

    #display count on video
    cv2.putText(frame, f'{count}', org = (25,85), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=3, color=(0,255,0), thickness=5)


    ########################## draw circles on arm joints ##########################
    cv2.circle(frame, shoulder, 6, (0,0,255), -1)
    cv2.circle(frame, elbow, 6, (0,0,255), -1)
    cv2.circle(frame, wrist, 6, (0,0,255), -1)

    cv2.circle(frame, shoulder, 6, (0,0,255), 3)
    cv2.circle(frame, elbow, 6, (0,0,255), 3)
    cv2.circle(frame, wrist, 6, (0,0,255), 3)
    ################################################################################

    # draw triangle around arm joints and change its color based on arm angle
    triangle_cnt = np.array([shoulder, elbow, wrist])
    cv2.drawContours(frame, [triangle_cnt], 0, (160-angle, angle, angle+50), -1)

    #resize input video
    img_resized = resized(frame, 0.6)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("done")
                         
                         
