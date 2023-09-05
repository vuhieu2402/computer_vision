import cv2
import os
import time
import dem_ngon_tay.hand as htm
import google.protobuf
import sys

pTime =0 #thoi gian ban dau
cap = cv2.VideoCapture(0)




FolderPath = "Fingers"
lst = os.listdir(FolderPath)
lst_2 =[]
for i in lst:
    image = cv2.imread(f"{FolderPath}/{i}")
    lst_2.append(image)


detector = htm.handDetector(detectionCon=0.55)
fingerid = [4,8,12,16,20]

while True:
    ret , frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)#phat hien vi tri 
    
    
    
    if len(lmList)!=0:
        fingers = []
        
        #viet cho ngon cai
        if lmList[fingerid[0]][1] < lmList[fingerid[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        
        
        #viet cho cac ngon dai
        for id in range(1,5):
            if lmList[fingerid[id]][2] < lmList[fingerid[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        songontay = fingers.count(1)       
                
    
        
        h ,w , c = lst_2[songontay-1].shape
        frame[0:h,0:w] = lst_2[songontay-1]

        #ve hinh chu nhat
        cv2.rectangle(frame, (0,200),(150,400),(0,255,0),-1)
        cv2.putText(frame, str(songontay), (30,390), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,10,(255,0,0),3)
    
    #viet ra FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (150,70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255,0,0), 3)
    
    
    
    cv2.imshow("video", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()