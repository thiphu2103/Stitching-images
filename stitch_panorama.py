#!/usr/bin/env python3
"""
Ghép 4 ảnh góc thành panorama - không ghosting.
Dùng Dynamic Programming Optimal Seam tại mỗi vùng overlap.

Yêu cầu: pip install opencv-python numpy
Chạy:    python3 stitch_panorama.py
"""
import cv2
import numpy as np
from pathlib import Path

INPUT_DIR   = Path(".")
OUTPUT_PATH = Path("panorama_output.jpg")
QUALITY     = 93
BLEND_WIDTH = 25   # px mỗi bên seam để blend mềm


def load(name):
    path = INPUT_DIR / f"{name}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    return img


def find_translation(src, dst, scale=0.25):
    """Tìm (dx, dy): dst = src dịch chuyển (dx, dy)."""
    s = cv2.resize(src, (int(src.shape[1]*scale), int(src.shape[0]*scale)))
    d = cv2.resize(dst, (int(dst.shape[1]*scale), int(dst.shape[0]*scale)))
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) / scale
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) / scale
    offsets = pts2 - pts1
    def iqr(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return arr[(arr >= q1-1.5*iqr) & (arr <= q3+1.5*iqr)]
    dx = int(np.median(iqr(offsets[:,0])))
    dy = int(np.median(iqr(offsets[:,1])))
    print(f"  Translation: dx={dx}, dy={dy}  ({len(good)} matches)")
    return dx, dy


def vertical_seam(img1, img2):
    """DP: tìm seam dọc tối ưu. Trả về col[row]."""
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32)).mean(axis=2)
    h, w = diff.shape
    cost = diff.copy()
    for r in range(1, h):
        p = cost[r-1]
        cost[r] += np.minimum(p, np.minimum(
            np.concatenate([[1e9], p[:-1]]),
            np.concatenate([p[1:],  [1e9]])
        ))
    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(cost[-1])
    for r in range(h-2, -1, -1):
        c = seam[r+1]; lo, hi = max(0,c-1), min(w-1,c+1)
        seam[r] = lo + np.argmin(cost[r, lo:hi+1])
    return seam


def horizontal_seam(img1, img2):
    """DP: tìm seam ngang tối ưu. Trả về row[col]."""
    return vertical_seam(img1.transpose(1,0,2), img2.transpose(1,0,2))


def apply_vseam(canvas, img_l, img_r, pos_l, pos_r, iH, iW):
    """Áp dụng vertical seam giữa ảnh trái và ảnh phải."""
    x0_l, y0_l = pos_l; x0_r, y0_r = pos_r
    ox0 = max(x0_l, x0_r); ox1 = min(x0_l+iW, x0_r+iW)
    oy0 = max(y0_l, y0_r); oy1 = min(y0_l+iH, y0_r+iH)
    if ox1 <= ox0 or oy1 <= oy0: return
    rl = img_l[oy0-y0_l:oy1-y0_l, ox0-x0_l:ox1-x0_l]
    rr = img_r[oy0-y0_r:oy1-y0_r, ox0-x0_r:ox1-x0_r]
    seam = vertical_seam(rl, rr)
    ov_w = ox1 - ox0
    for row, col in enumerate(seam):
        cr = oy0 + row
        canvas[cr, ox0:ox0+col] = rl[row, :col]
        canvas[cr, ox0+col:ox1] = rr[row, col:]
        lo, hi = max(0, col-BLEND_WIDTH), min(ov_w, col+BLEND_WIDTH)
        if hi > lo:
            alpha = np.linspace(1, 0, hi-lo)
            canvas[cr, ox0+lo:ox0+hi] = (
                rl[row,lo:hi].astype(float) * alpha[:,None] +
                rr[row,lo:hi].astype(float) * (1-alpha)[:,None]
            ).astype(np.uint8)


def apply_hseam(canvas, img_t, img_b, pos_t, pos_b, iH, iW):
    """Áp dụng horizontal seam giữa ảnh trên và ảnh dưới."""
    x0_t, y0_t = pos_t; x0_b, y0_b = pos_b
    ox0 = max(x0_t, x0_b); ox1 = min(x0_t+iW, x0_b+iW)
    oy0 = max(y0_t, y0_b); oy1 = min(y0_t+iH, y0_b+iH)
    if ox1 <= ox0 or oy1 <= oy0: return
    rt = img_t[oy0-y0_t:oy1-y0_t, ox0-x0_t:ox1-x0_t]
    rb = img_b[oy0-y0_b:oy1-y0_b, ox0-x0_b:ox1-x0_b]
    seam = horizontal_seam(rt, rb)
    ov_h = oy1 - oy0
    for col, row in enumerate(seam):
        cc = ox0 + col
        canvas[oy0:oy0+row, cc] = rt[:row, col]
        canvas[oy0+row:oy1, cc] = rb[row:, col]
        lo, hi = max(0, row-BLEND_WIDTH), min(ov_h, row+BLEND_WIDTH)
        if hi > lo:
            alpha = np.linspace(1, 0, hi-lo)
            canvas[oy0+lo:oy0+hi, cc] = (
                rt[lo:hi,col].astype(float) * alpha[:,None] +
                rb[lo:hi,col].astype(float) * (1-alpha)[:,None]
            ).astype(np.uint8)


def largest_inscribed_rect(mask, step=15):
    H, W = mask.shape
    rc1 = np.array([np.where(mask[r])[0][0]  if mask[r].any() else W for r in range(H)])
    rc2 = np.array([np.where(mask[r])[0][-1] if mask[r].any() else 0  for r in range(H)])
    best, rect = 0, (0,H,0,W)
    for r1 in range(0, H, step):
        c1, c2 = rc1[r1], rc2[r1]
        for r2 in range(r1+step, H, step):
            c1 = max(c1, rc1[r2]); c2 = min(c2, rc2[r2])
            if c1 >= c2: break
            if (r2-r1)*(c2-c1) > best:
                best = (r2-r1)*(c2-c1); rect = (r1,r2,c1,c2)
    return rect


def main():
    print("=== Panorama Stitcher - Optimal Seam ===\n")

    imgs = {k: load(k) for k in ["top_left","top_right","bottom_left","bottom_right"]}
    iH, iW = next(iter(imgs.values())).shape[:2]
    print(f"Ảnh: {iW}x{iH}\n")

    ref = imgs["top_left"]
    print("top_right  → top_left:");  dx_tr, dy_tr = find_translation(ref, imgs["top_right"])
    print("bottom_left → top_left:"); dx_bl, dy_bl = find_translation(ref, imgs["bottom_left"])
    print("bottom_right → top_left:");dx_br, dy_br = find_translation(ref, imgs["bottom_right"])

    pos = {
        "top_left":     (0,       0),
        "top_right":    (-dx_tr, -dy_tr),
        "bottom_left":  (-dx_bl, -dy_bl),
        "bottom_right": (-dx_br, -dy_br),
    }

    all_x = [p[0] for p in pos.values()] + [p[0]+iW for p in pos.values()]
    all_y = [p[1] for p in pos.values()] + [p[1]+iH for p in pos.values()]
    tx, ty = -min(all_x), -min(all_y)
    out_w  = max(all_x) - min(all_x)
    out_h  = max(all_y) - min(all_y)
    pos = {k: (v[0]+tx, v[1]+ty) for k,v in pos.items()}
    print(f"\nCanvas: {out_w}x{out_h}")

    # Xếp ảnh lên canvas (bottom → top)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for name in ["bottom_left","bottom_right","top_left","top_right"]:
        px, py = pos[name]
        canvas[py:py+iH, px:px+iW] = imgs[name]

    # Áp dụng seam tại mỗi vùng overlap
    print("\nApplying optimal seams...")
    apply_vseam(canvas, imgs["bottom_left"], imgs["bottom_right"],
                pos["bottom_left"], pos["bottom_right"], iH, iW)
    apply_vseam(canvas, imgs["top_left"],    imgs["top_right"],
                pos["top_left"],    pos["top_right"],    iH, iW)
    apply_hseam(canvas, imgs["top_left"],    imgs["bottom_left"],
                pos["top_left"],    pos["bottom_left"],  iH, iW)
    apply_hseam(canvas, imgs["top_right"],   imgs["bottom_right"],
                pos["top_right"],   pos["bottom_right"], iH, iW)

    # Crop
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    r1,r2,c1,c2 = largest_inscribed_rect(gray > 5)
    final = canvas[r1:r2, c1:c2]
    print(f"Kích thước cuối: {final.shape[1]}x{final.shape[0]}")

    cv2.imwrite(str(OUTPUT_PATH), final, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    print(f"\n✅ Đã lưu: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()