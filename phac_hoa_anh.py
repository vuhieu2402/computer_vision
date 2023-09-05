import cv2

img =cv2.imread("hai_dang.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(gray)
cv2.imshow("invert",invert)
blur = cv2.GaussianBlur(invert, (21,21),0)
cv2.imshow("blur",blur)

invert_blur = cv2.bitwise_not(blur)
cv2.imshow("invert_blur", invert_blur)
sketch = cv2.divide(gray,invert_blur, scale=256.0)


cv2.imshow("anh", sketch)
cv2.waitKey(0)