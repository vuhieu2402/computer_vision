import cv2
import face_recognition


img_Elon = face_recognition.load_image_file('pic//elon musk.jpg')
img_Elon = cv2.cvtColor(img_Elon,cv2.COLOR_BGR2RGB)

imgCheck = face_recognition.load_image_file('pic//elon check.jpg')
imgCheck = cv2.cvtColor(imgCheck,cv2.COLOR_BGR2RGB)

#xac dinh vi tri khuon mat
faceLoc = face_recognition.face_locations(img_Elon)[0]
print(faceLoc)#(y1,x2,y2,x1)

#ma hoa hinh anh
encodeElon = face_recognition.face_encodings(img_Elon)[0]
cv2.rectangle(img_Elon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (0,255,0),2)


faceCheck = face_recognition.face_locations(imgCheck)[0]
print(faceCheck)#(y1,x2,y2,x1)
#ma hoa hinh anh
encodeCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]), (0,255,0),2)



result = face_recognition.compare_faces([encodeElon],encodeCheck)

#sai so
faceDis = face_recognition.face_distance([encodeElon],encodeCheck)
cv2.putText(imgCheck,f"{result}{round(faceDis[0],2)}", (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,0,0),2)

cv2.imshow("goc", img_Elon)
cv2.imshow("check", imgCheck)

cv2.waitKey()
cv2.destroyAllWindows()