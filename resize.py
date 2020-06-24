# 画像のリサイズをするプログラム
# サイズの大きな画像を特徴検出のプログラムなどに与えるとOutOfMemoryErrorを起こしてしまうため、
# このプログラムで適切な大きさにリサイズする

import cv2

image = cv2.imread('images/sources/python.jpg')

height = image.shape[0]
width = image.shape[1]

image = cv2.resize(image, (int(width * 0.2), int(height * 0.2)))

cv2.imwrite('images/resized/python.jpg', image)
cv2.imshow('resize', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
