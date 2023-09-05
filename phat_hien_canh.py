import cv2
import numpy


PREVIEW  = 0  # Preview Mode
BLUR     = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY    = 3  # Canny Edge Detector
SOBEl = 4

scale = 1
delta = 0
ddepth = cv2.CV_16S

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

img = cv2.imread("img.png")

image_filter = PREVIEW
alive = True
result =None
while alive:
    if image_filter == PREVIEW:
        result = img
    elif image_filter == CANNY:
        result = cv2.Canny(img, 80, 150)
    elif image_filter == BLUR:
        result = cv2.blur(img, (13, 13))
    elif image_filter == SOBEl:
        img_blur = cv2.GaussianBlur(img, (3,3),0)

        gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        result = grad

    elif image_filter == FEATURES:
        result = img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_gray, **feature_params)
        if corners is not None:
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

    cv2.imshow('anh', result)
    
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("S") or key == ord("s"):
        image_filter = SOBEl
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW


cv2.destroyAllWindows()
