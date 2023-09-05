import cv2
import numpy as np


def blur_face(img, face_box):
  """
  Làm mờ khuôn mặt bằng cách sử dụng bộ lọc Median.

  Args:
    img: Ảnh đầu vào.
    face_box: Tọa độ vị trí của khuôn mặt.

  Returns:
    Ảnh đã được làm mờ khuôn mặt.
  """

  # Lấy vùng ảnh của khuôn mặt.
  face = img[face_box[1]:face_box[3], face_box[0]:face_box[2]]

  # Áp dụng bộ lọc Median cho vùng ảnh khuôn mặt.
  face = cv2.medianBlur(face, 21)

  # Thay thế vùng ảnh khuôn mặt trong ảnh đầu vào bằng vùng ảnh đã được làm mờ.
  img[face_box[1]:face_box[3], face_box[0]:face_box[2]] = face

  return img

# def blur_face(img, face_box, sigma):
#   """
#   Làm mờ khuôn mặt bằng cách sử dụng bộ lọc Gaussian.
#
#   Args:
#     img: Ảnh đầu vào.
#     face_box: Tọa độ vị trí của khuôn mặt.
#     sigma: Độ lệch chuẩn của bộ lọc Gaussian.
#
#   Returns:
#     Ảnh đã được làm mờ khuôn mặt.
#   """
#
#   # Lấy vùng ảnh của khuôn mặt.
#   face = img[face_box[1]:face_box[3], face_box[0]:face_box[2]]
#
#   # Áp dụng bộ lọc Gaussian cho vùng ảnh khuôn mặt.
#   face = cv2.GaussianBlur(face, (5, 5), sigmaX=sigma)
#
#   # Thay thế vùng ảnh khuôn mặt trong ảnh đầu vào bằng vùng ảnh đã được làm mờ.
#   img[face_box[1]:face_box[3], face_box[0]:face_box[2]] = face
#
#   return img


face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    if ret: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30))
        
        for (x,y,w,h) in faces:
            face_box = (x, y, x + w, y + h)

            # Làm mờ khuôn mặt.
            frame = blur_face(frame, face_box)
            
        cv2.imshow("Webcam",frame)

    phim_bam = cv2.waitKey(1)
    if phim_bam == ord('q'):
        break
   

cv2.destroyAllWindows()
cap.release()