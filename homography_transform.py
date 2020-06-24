# ホモグラフィ変換をして画像を加工するプログラム

import cv2
import numpy as np

image = cv2.imread('images/homography/python_right.jpg')
h, w, c = image.shape

pts1 = np.float32([[120, 175], [450, 30], [110, 710], [500, 790]])
pts2 = np.float32([[0, 0], [624, 0], [0, 832], [624, 832]])

M = cv2.getPerspectiveTransform(pts1, pts2)

out = cv2.warpPerspective(image, M, (w, h))

square1 = np.float32([[120, 175], [450, 30], [500, 790], [110, 710]])
image = cv2.polylines(image, [np.int32(square1)], True, (0, 0, 255), 3, cv2.LINE_AA)

square2 = np.float32([[0, 0], [624, 0], [624, 832], [0, 832]])
out = cv2.polylines(out, [np.int32(square2)], True, (0, 0, 255), 5, cv2.LINE_AA)

out = cv2.hconcat([image, out])

cv2.imshow('homography_transform', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
