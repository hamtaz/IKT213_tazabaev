import cv2
import numpy as np



def edge_detection(reference_img):
    gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    reference_img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imwrite('solutions/image_corners.png', reference_img)

def sift_detection(image_to_align, reference_image, max_features, good_match_percent):
    gray1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1,des1 = sift.detectAndCompute(gray1, None)
    kp2,des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < good_match_percent*n.distance:
            good.append(m)

    if len(good)>=max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = reference_image.shape[:2]
        aligned = cv2.warpPerspective(image_to_align, M, (width, height))
        cv2.imwrite("solutions/aligned.png", aligned)
        matchesMask = mask.ravel().tolist()

        h, w = gray1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        gray2 = cv2.polylines(gray2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches found")
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(gray1, kp1, gray2, kp2, good, None, **draw_params)
    cv2.imwrite("solutions/sift_image.png", img3)


if __name__ == "__main__":
    image1_path = 'reference_img.png'
    img1 = cv2.imread(image1_path)

    image2_path = "align_this.jpg"
    img2 = cv2.imread(image2_path)
    edge_detection(img1)
    sift_detection(img1,img2, 10, 0.7)