# 特徴点マッチングとホモグラフィを使用した物体検出をするプログラム

import cv2
import numpy as np

query_image = cv2.imread('images/resized/scaled/python.jpg')
train_image = cv2.imread('images/resized/books.jpg')

detector = cv2.AKAZE_create()

query_keypoints, query_descriptor = detector.detectAndCompute(query_image, None)
train_keypoints, train_descriptor = detector.detectAndCompute(train_image, None)

matcher = cv2.BFMatcher()

matches = matcher.knnMatch(query_descriptor, train_descriptor, k=2)

good = []
ratio = 0.6
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append(m)

# 検出した特徴点の閾値（10以下ならば物体検出はできていないと判断する）
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    # 物体検出できた場合
    # 検出できた特徴点の座標を配列に加工
    source_points = np.float32([query_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination_points = np.float32([train_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # findHomography関数によってホモグラフィ行列を取得する
    # RANSACアルゴリズムを使用してノイズを除去する
    M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # 取得したホモグラフィ行列を使用して宛先の座標を計算する
    h, w, c = query_image.shape
    points = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    destination = cv2.perspectiveTransform(points, M)

    # 検出した物体の囲い線を描画する
    # cv2.polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]])
    train_image = cv2.polylines(train_image, [np.int32(destination)], True, (0, 0, 255), 3, cv2.LINE_AA)
else:
    # 物体検出できなかった場合
    text = "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 検出失敗のテキストを描画する
    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    cv2.putText(train_image, text, (10, 600), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    print(text)
    matchesMask = None

# 対応点を描画する
# matchColor = (BLUE, GREEN, RED)
draw_params = dict(matchColor=(0, 255, 255),  # Yellow
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

out = cv2.drawMatches(query_image, query_keypoints, train_image, train_keypoints, good, None, **draw_params)

cv2.imshow('find_homography', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
