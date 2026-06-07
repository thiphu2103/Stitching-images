import cv2
import numpy as np


def resize_keep_ratio(img, max_size=1200):

    h, w = img.shape[:2]

    scale = max_size / max(h, w)

    if scale >= 1:
        return img

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )


def get_matches(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(8000)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:

        if m.distance < 0.6 * n.distance:
            good.append(m)

    good = sorted(
        good,
        key=lambda x: x.distance
    )

    good = good[:80]

    return kp1, kp2, good


def estimate_transform(img_ref, img_new):

    kp1, kp2, good = get_matches(
        img_ref,
        img_new
    )

    if len(good) < 10:
        raise Exception("Not enough matches")

    pts_ref = np.float32([
        kp1[m.queryIdx].pt
        for m in good
    ])

    pts_new = np.float32([
        kp2[m.trainIdx].pt
        for m in good
    ])

    M, inliers = cv2.estimateAffinePartial2D(
        pts_new,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=3
    )

    if M is None:
        raise Exception("Transform failed")

    return M


def warp_image(img, M):

    h, w = img.shape[:2]

    corners = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)

    warped_corners = cv2.transform(
        corners,
        M
    )

    return warped_corners


def stitch_four_images(
        top_left,
        top_right,
        bottom_left,
        bottom_right):

    ref = top_left

    M_tr = estimate_transform(
        ref,
        top_right
    )

    M_bl = estimate_transform(
        ref,
        bottom_left
    )

    M_br = estimate_transform(
        ref,
        bottom_right
    )

    all_corners = []

    h0, w0 = ref.shape[:2]

    ref_corners = np.float32([
        [0, 0],
        [w0, 0],
        [w0, h0],
        [0, h0]
    ]).reshape(-1, 1, 2)

    all_corners.extend(
        ref_corners.reshape(-1, 2)
    )

    for img, M in [
        (top_right, M_tr),
        (bottom_left, M_bl),
        (bottom_right, M_br)
    ]:

        corners = warp_image(
            img,
            M
        )

        all_corners.extend(
            corners.reshape(-1, 2)
        )

    all_corners = np.array(all_corners)

    xmin, ymin = np.int32(
        all_corners.min(axis=0) - 1
    )

    xmax, ymax = np.int32(
        all_corners.max(axis=0) + 1
    )

    tx = -xmin
    ty = -ymin

    width = xmax - xmin
    height = ymax - ymin

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    canvas[
        ty:ty+h0,
        tx:tx+w0
    ] = ref

    for img, M in [
        (top_right, M_tr),
        (bottom_left, M_bl),
        (bottom_right, M_br)
    ]:

        T = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)

        M2 = T @ np.vstack([
            M,
            [0, 0, 1]
        ])

        warped = cv2.warpAffine(
            img,
            M2[:2],
            (width, height)
        )

        mask = np.any(
            warped > 0,
            axis=2
        )

        canvas[mask] = warped[mask]

    return canvas


# ==========================
# Load ảnh
# ==========================

top_left = resize_keep_ratio(
    cv2.imread("top_left.jpg")
)

top_right = resize_keep_ratio(
    cv2.imread("top_right.jpg")
)

bottom_left = resize_keep_ratio(
    cv2.imread("bottom_left.jpg")
)

bottom_right = resize_keep_ratio(
    cv2.imread("bottom_right.jpg")
)

result = stitch_four_images(
    top_left,
    top_right,
    bottom_left,
    bottom_right
)

cv2.imwrite(
    "panorama.jpg",
    result
)

print("Done")