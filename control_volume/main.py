import cv2
import time
import math
import numpy as np
import dem_ngon_tay.hand as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.7)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMute()
volume.GetMasterVolumeLevel()

volRange=volume.GetVolumeRange()
# print(volRange)
minVol = volRange[0]
maxVol = volRange[1]





pTime = 0
while True:
    ret , frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)  
    
    if len(lmList) != 0 :
        x1,y1 = lmList[4][1],lmList[4][2] #lay toa do dau nhon tay cai  
        x2,y2 = lmList[8][1],lmList[8][2] #lay toa do dau nhon tay tro
        
        
        #ve
        cv2.circle(frame,(x1,y1),15,(255,0,255),-1)
        cv2.circle(frame,(x2,y2),15,(255,0,255),-1) 
        cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),2)
        
        #ve duong tron giua  2 duong thang noi 2 diem
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(frame,(cx, cy), 15, (255,0,255),-1)
        
        length = math.hypot(x2-x1, y2-y1)
        
        vol = np.interp(length,[30,310],[minVol,maxVol])
        volBar = np.interp(length,[30,310],[400,150])
        vol_type = np.interp(length,[30,310],[0,100])
        volume.SetMasterVolumeLevel(vol, None)
        
        
        if length <25 :
            cv2.circle(frame,(cx, cy),15,(0,255,0),-1)
            
        cv2.rectangle(frame,(50,150),(100,400),(0,255,0),3)
        cv2.rectangle(frame,(50,int(volBar)),(100,400),(0,255,0),-1)
        
        #show % volume
        cv2.putText(frame, f"{int(vol_type)}%", (40,120), cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0), 3)
            
             
    
    
    cTime = time.time()  # trả về số giây, tính từ 0:0:00 ngày 1/1/1970 theo giờ  utc , gọi là(thời điểm bắt đầu thời gian)
    fps=1/(cTime-pTime) # tính fps Frames per second - đây là  chỉ số khung hình trên mỗi giây
    pTime=cTime
    # show fps lên màn hình, fps hiện đang là kiểu float , ktra print(type(fps))
    cv2.putText(frame, f"FPS: {int(fps)}",(150,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    
    cv2.imshow("video" , frame)
    
    if cv2.waitKey(1) == ord("q") :
        break
    
cap.release()
cv2.destroyAllWindows()
