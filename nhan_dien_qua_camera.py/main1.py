import datetime
import cv2
import face_recognition
import os
import numpy as np

path = "pic2"
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}") # pic2/Donal Trump.jpg
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    

def thamdu(name):
    with open("1.csv","r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")
            
            
            
            
#step encoding
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

encodeListKnow = Mahoa(images)

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frameS = cv2.resize(frame, (0,0), None, fx=0.5,fy=0.5)
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
    
    #Xac dinh vi tri khuon mat tren cam va encode hinh anh tren cam 
    frameCurFrame = face_recognition.face_locations(frameS) #lay tung khhuon mat va vi tri khuon mat hien tai
    encodeCurFrame = face_recognition.face_encodings(frameS)
    
    #lay tung khuon mat va vi tri hien tai theo cap
    for encodeFace, faceLoc in zip(encodeCurFrame, frameCurFrame):
        matches = face_recognition.face_compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex] <0.50:
            name = classNames[matchIndex].upper()
            thamdu(name)
        else:
            name = "Unknow"
        
        #viet ten len frame
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*2,x2*2,y2*2,x1*2
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,0),2)
        
    
    cv2.imshow("video" , frame)
    
    if cv2.waitKey(1) == ord("q") :
        break
    
cap.release()
cv2.destroyAllWindows()
