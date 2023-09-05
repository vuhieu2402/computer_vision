import cv2
import numpy as np
import matplotlib.pyplot as plt



img1 = cv2.imread('anh//original.png')
img2 = cv2.imread('anh//scan.png')

img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Phát hiện các tính năng ORB và tính toán các bộ mô tả
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create()
keypoints1 , descriptior1 = orb.detectAndCompute(img1_gray, None)
keypoints2 , descriptior2 = orb.detectAndCompute(img2_gray, None)

img1_display = cv2.drawKeypoints(img1, keypoints1 , outImage= np.array([]), color=(255,0,0) , flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_display = cv2.drawKeypoints(img2, keypoints2 , outImage= np.array([]), color=(255,0,0) , flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# cv2.imshow("original" , img1_display)
# cv2.imshow("scan" , img2_display)

#Match keypoints in the two image
# Match features
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#Converting to list for sorting as tuples are immutable objects
matches = list(matcher.match(descriptior1, descriptior2, None))

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)
# Remove not so good matches
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw top matches
im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)


# new_width = 800
# new_height = 600
# resized_image = cv2.resize(im_matches, (new_width, new_height))
# cv2.imshow('anh', resized_image)

# Hiển thị ảnh đã được đặt kích thước


#cv2.imshow("anh",im_matches)


#Find Homography
# Trích xuất vị trí của các matche good
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)


#Warp image
height, width, channels = img1.shape
img2_reg = cv2.warpPerspective(img2, h, (width, height))

new_width = 800
new_height = 600

resized_image1 = cv2.resize(img1, (new_width, new_height))
cv2.imshow("original",resized_image1)


resized_image2 = cv2.resize(img2_reg, (new_width, new_height))
cv2.imshow("scanned" , resized_image2)


cv2.waitKey(0)
cv2.destroyAllWindows()
