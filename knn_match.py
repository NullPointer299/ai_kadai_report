# KNNを用いて総当たりマッチングをして、結果を出力するプログラム

import cv2

query_image = cv2.imread('images/resized/python.jpg', cv2.IMREAD_GRAYSCALE)
train_image = cv2.imread('images/resized/books.jpg', cv2.IMREAD_GRAYSCALE)

detector = cv2.AKAZE_create()

# 特徴点（keypoints）と特徴記述子（descriptor）の取得
query_keypoints, query_descriptor = detector.detectAndCompute(query_image, None)
train_keypoints, train_descriptor = detector.detectAndCompute(train_image, None)

# Brute-Force matcherの作成
matcher = cv2.BFMatcher()

# KNNでのマッチング kは2なので、上位2つの特徴点を抽出する。
matches = matcher.knnMatch(query_descriptor, train_descriptor, k=2)

# D. Lowe氏が提唱した割合試験
good = []
ratio = 0.6
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 結果の書き込み
out = cv2.drawMatchesKnn(
    query_image, query_keypoints, train_image, train_keypoints, good, None, flags=2)

cv2.imshow('knn_match', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
