# 読み込んだ画像から特徴点の検出をして表示するプログラム

import cv2

# 画像読み込み
image = cv2.imread('images/resized/books.jpg', cv2.IMREAD_GRAYSCALE)

# 特徴検出アルゴリズムの選択
# detector = cv2.ORB_create()
detector = cv2.AKAZE_create()

# 特徴点（keypoints）の検出
keypoints = detector.detect(image)

# drawKeypoints(image, keypoints, outImage[, color[, flags]])
out = cv2.drawKeypoints(image, keypoints, None,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 画像の表示
cv2.imshow('detect_keypoints', out)

# なにかキーが押されたら終了
cv2.waitKey(0)
cv2.destroyAllWindows()
