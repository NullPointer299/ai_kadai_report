# レポート中の特徴検出の説明画像を作成するプログラム

import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

# cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])

# green fill
cv2.rectangle(img, (128, 128), (384, 384), (255, 255, 255), cv2.FILLED)

# blue patch
cv2.rectangle(img, (96, 96), (160, 160), (255, 0, 0), 3)

# green patch
cv2.rectangle(img, (192, 192), (320, 256), (0, 255, 0), 3)

# red patch
cv2.rectangle(img, (224, 352), (352, 416), (0, 0, 255), 3)

cv2.imshow('draw_square', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
