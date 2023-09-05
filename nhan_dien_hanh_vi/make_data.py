import cv2
import mediapipe as mp
import pandas as pd


cap = cv2.VideoCapture(0)

#khoi tao thu vien mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


lm_list = []
label = "HANDSWING"
no_of_frames = 600

def make_landmark_timestep(results):
    c_lm =[]
    for id,lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    #ve cac duong noi
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    #ve cac diem nut
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id,lm)
        cx , cy = int(lm.x+w), int(lm.y+h)
        cv2.circle(img, (cx,cy), 5, (0,0,255), cv2.FILLED)
        
    return img




while True:
    ret, frame = cap.read()
    if ret:
        
        #nhan dien pose 
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        
        if results.pose_landmarks :
            #ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            
            #ve khung xuong len anh
            frame = draw_landmark_on_image(mpDraw, results, frame)
         
        
        cv2.imshow("video", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

#write vao csv
df = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")    
        
cap.release()
cv2.destroyAllWindows