import cv2
import numpy as np
from imutils.object_detection import non_max_suppression # Để loại bỏ chồng lấn
 
# Đưa video vào cùng thư mục với file python của đoạn code này
filename = 'pedestrians_on_street_1.mp4'
file_size = (1920, 1080) # Giả sử kích thước video là 1920x1080
scale_ratio = 1 # Tỉ lệ phóng đại kích thước video khi cần thiết
 
# Lưu video đầu ra
output_filename = 'pedestrians_on_street.mp4'
output_frames_per_second = 20.0
 
def main():
 
    # Khởi tạo đối tượng HOGDescriptor
    hog = cv2.HOGDescriptor()
     
    # Khởi chạy hàm phát hiện người đi bộ
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
     
    # Tải lên video
    cap = cv2.VideoCapture(filename)
 
    # Tạo đối tượng VideoWriter để lưu video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename, 
                                        fourcc,
                                        output_frames_per_second,
                                        file_size)
     
    # Quá trình xử lý video
    while cap.isOpened():
         
         # Lấy từng khung hình
         success, frame = cap.read()
         
         # Sau khi lấy được khung hình thì tiếp tục các bước tiếp theo
         if success:
         
             # Thay đổi kích thước khung hình
             width = int(frame.shape[1] * scale_ratio)
             height = int(frame.shape[0] * scale_ratio)
             frame = cv2.resize(frame, (width, height))
             
             # Lưu khung hình ban đầu
             orig_frame = frame.copy()
             
             # Phát hiện người đi bộ
             # image: mỗi khung hình đơn lẻ trong video
             # winStride: kích thước bước của cửa số trượt theo hướng trục x và y
             # padding: số lượng pixel theo hướng trục x và y  
             # scale: Hệ số tăng kích thước cửa sổ phát hiện  
             # bounding_boxes: Vị trí người đi bộ phát hiện được
             # weights: Trọng số của người đi bộ phát hiện được
             (bounding_boxes, weights) = hog.detectMultiScale(frame,
                                                                                       winStride=(16, 16),
                                                                                       padding=(4, 4),
                                                                                       scale=1.05)
 
             # Vẽ hình chữ nhật bao quanh người đi bộ trên khung hình
             for (x, y, w, h) in bounding_boxes:
                  cv2.rectangle(orig_frame,
                                      (x, y), 
                                      (x + w, y + h), 
                                      (0, 0, 255),
                                      2)
                         
             # Loại bỏ những hình chữ nhật chồng lấn nhau
             # Thay đổi chỉ số overlapThresh để được kết quả tốt nhất
             bounding_boxes = np.array([[x, y, x + w, y + h] for (
                                                      x, y, w, h) in bounding_boxes])
             
             selection = non_max_suppression(bounding_boxes,
                                                               probs=None,
                                                               overlapThresh=0.45)
         
             # vẽ hình chữ nhật là kết quả cuối cùng
             for (x1, y1, x2, y2) in selection:
                   cv2.rectangle(frame,
                                       (x1, y1),
                                       (x2, y2),
                                       (0, 255, 0),
                                       4)
         
             # Lưu khung hình vào video đầu ra
             result.write(frame)
             
             # Hiển thị khung hình
             cv2.imshow("Frame", frame)   
 
             # Hiển thị khung hình trong x mili giây và quit nếu nhấn “q”
             if cv2.waitKey(25) & 0xFF == ord('q'):
                   break
         
             # Thoát ra nếu không còn khung hình nào nữa
         else:
             break
             
    # Dừng lại sau khi video kết thúc
    cap.release()
     
    # Giải phóng quá trình ghi video
    result.release()
     
    # Đóng tất cả các cửa sổ
    cv2.destroyAllWindows()
 
main()