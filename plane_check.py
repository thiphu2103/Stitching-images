"""
plane_check.py — Phát hiện ảnh bị nghiêng lệch plane trong chuỗi panorama
==========================================================================
Thuật toán:
  1. Match keypoints giữa các cặp ảnh liên tiếp (SIFT + FLANN).
  2. Tính Homography H bằng RANSAC.
  3. Decompose H → [rotation | translation | normal].
  4. Nếu góc pitch/roll vượt ngưỡng → ảnh đó bị nghiêng.

Ngoài ra còn kiểm tra:
  - Quá ít inlier (ảnh blur/too dark)
  - Translation dọc (vertical shift) quá lớn

Dùng trong server.py:
    from plane_check import check_sequence, PlaneCheckResult
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Ngưỡng mặc định ───────────────────────────────────────────────────────────
DEFAULT_MAX_PITCH_DEG   = 20.0   # pitch/roll tối đa cho phép (độ)
DEFAULT_MAX_VERT_SHIFT  = 0.2   # dịch dọc tối đa (tính theo % chiều cao ảnh)
DEFAULT_MIN_INLIERS     = 25     # số inlier tối thiểu để coi là match được
DEFAULT_N_FEATURES      = 3000   # số feature SIFT
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PairResult:
    """Kết quả kiểm tra giữa 2 ảnh liên tiếp."""
    idx_a: int
    idx_b: int
    inliers: int
    yaw_deg: float            # xoay ngang (mong muốn)
    pitch_deg: float          # xoay dọc   (nên ~ 0)
    roll_deg: float           # nghiêng ngang (nên ~ 0)
    vert_shift_ratio: float   # dịch dọc / chiều cao
    ok: bool
    reason: Optional[str] = None  # mô tả lỗi nếu ok=False


@dataclass
class PlaneCheckResult:
    """Kết quả tổng thể cho toàn bộ chuỗi ảnh."""
    all_ok: bool
    bad_indices: list[int] = field(default_factory=list)   # chỉ số ảnh lỗi (0-based)
    messages: list[str]    = field(default_factory=list)   # thông báo từng lỗi
    pairs: list[PairResult] = field(default_factory=list)  # chi tiết từng cặp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_gray(path: str, scale: float = 0.5) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _rotation_angles(R: np.ndarray) -> tuple[float, float, float]:
    """
    Phân tích ma trận quay R (3x3) → (yaw, pitch, roll) theo độ.
    Convention: ZYX Euler angles.
      yaw   = xoay quanh trục Z (trái/phải)  ← mong muốn khi quét panorama
      pitch = xoay quanh trục Y (lên/xuống)  ← phải ~ 0
      roll  = xoay quanh trục X (nghiêng)    ← phải ~ 0
    """
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # Gimbal lock check: |R[2,0]| ~ 1
    if abs(R[2, 0]) < 0.9999:
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        roll  =  np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw   =  np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        # Gimbal lock: pitch = ±90°
        yaw   = 0.0
        if R[2, 0] < 0:
            pitch =  np.pi / 2
            roll  =  np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll  = -np.arctan2(R[0, 1], R[0, 2])
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def _check_pair(
    path_a: str,
    path_b: str,
    idx_a: int,
    idx_b: int,
    max_pitch_deg: float,
    max_vert_shift: float,
    min_inliers: int,
    n_features: int,
) -> PairResult:
    """So sánh 2 ảnh liên tiếp và trả về PairResult."""

    gray_a = _load_gray(path_a)
    gray_b = _load_gray(path_b)
    h, w   = gray_a.shape[:2]

    # ── 1. Detect + Match ────────────────────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        return PairResult(
            idx_a=idx_a, idx_b=idx_b,
            inliers=0, yaw_deg=0, pitch_deg=0, roll_deg=0,
            vert_shift_ratio=0, ok=False,
            reason=f"Ảnh {idx_b + 1} quá tối hoặc không có chi tiết để nhận diện."
        )

    # FLANN matcher
    index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < min_inliers:
        return PairResult(
            idx_a=idx_a, idx_b=idx_b,
            inliers=len(good), yaw_deg=0, pitch_deg=0, roll_deg=0,
            vert_shift_ratio=0, ok=False,
            reason=(
                f"Ảnh {idx_b + 1} không khớp với ảnh trước "
                f"(chỉ có {len(good)} điểm chung). "
                "Đảm bảo ảnh liên tiếp có vùng chồng lặp ~30%."
            )
        )

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    # ── 2. Homography + RANSAC ───────────────────────────────────────────────
    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransacReprojThreshold=4.0)
    if H is None:
        return PairResult(
            idx_a=idx_a, idx_b=idx_b,
            inliers=0, yaw_deg=0, pitch_deg=0, roll_deg=0,
            vert_shift_ratio=0, ok=False,
            reason=f"Ảnh {idx_b + 1}: không tính được homography."
        )

    inliers = int(mask.sum()) if mask is not None else len(good)

    # ── 3. Decompose Homography ──────────────────────────────────────────────
    # Ước lượng camera intrinsics đơn giản (focal ~ max(W,H))
    f  = max(w, h)
    K  = np.array([[f, 0, w / 2],
                   [0, f, h / 2],
                   [0, 0, 1    ]], dtype=np.float64)

    num, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    # Chọn solution hợp lệ nhất (normal gần [0,0,1])
    best_R     = rotations[0]
    best_t     = translations[0]
    best_score = -1.0
    for i in range(num):
        n = normals[i].flatten()
        s = abs(n[2])   # z-component gần 1 → plane ~ frontoparallel
        if s > best_score:
            best_score = s
            best_R     = rotations[i]
            best_t     = translations[i]

    yaw_deg, pitch_deg, roll_deg = _rotation_angles(best_R)

    # ── 4. Vertical shift ────────────────────────────────────────────────────
    # Dịch dọc trung bình của các inlier points
    if mask is not None:
        in_mask = mask.ravel() == 1
        dy_mean = float(np.mean(np.abs(pts_a[in_mask, 1] - pts_b[in_mask, 1])))
    else:
        dy_mean = float(np.mean(np.abs(pts_a[:, 1] - pts_b[:, 1])))
    vert_shift_ratio = dy_mean / h

    # ── 5. Đánh giá ─────────────────────────────────────────────────────────
    reasons = []

    if abs(pitch_deg) > max_pitch_deg:
        reasons.append(
            f"Ảnh {idx_b + 1} bị nghiêng lên/xuống {abs(pitch_deg):.1f}° "
            f"(giới hạn {max_pitch_deg}°). Giữ điện thoại thẳng đứng khi quét ngang."
        )
    if abs(roll_deg) > max_pitch_deg:
        reasons.append(
            f"Ảnh {idx_b + 1} bị xoay nghiêng {abs(roll_deg):.1f}° "
            f"(giới hạn {max_pitch_deg}°). Giữ điện thoại thẳng, không nghiêng sang trái/phải."
        )
    if vert_shift_ratio > max_vert_shift:
        reasons.append(
            f"Ảnh {idx_b + 1} lệch dọc quá nhiều ({vert_shift_ratio * 100:.1f}% chiều cao). "
            "Giữ điện thoại cùng chiều cao khi quét."
        )

    ok = len(reasons) == 0
    return PairResult(
        idx_a=idx_a, idx_b=idx_b,
        inliers=inliers,
        yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg,
        vert_shift_ratio=vert_shift_ratio,
        ok=ok,
        reason="\n".join(reasons) if reasons else None,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def check_sequence(
    image_paths: list[str],
    max_pitch_deg: float  = DEFAULT_MAX_PITCH_DEG,
    max_vert_shift: float = DEFAULT_MAX_VERT_SHIFT,
    min_inliers: int      = DEFAULT_MIN_INLIERS,
    n_features: int       = DEFAULT_N_FEATURES,
) -> PlaneCheckResult:
    """
    Kiểm tra toàn bộ chuỗi ảnh trước khi stitch.

    Trả về PlaneCheckResult:
      .all_ok       → True nếu tất cả ảnh đều ổn
      .bad_indices  → danh sách index ảnh lỗi (0-based)
      .messages     → thông báo tiếng Việt cho từng lỗi
      .pairs        → chi tiết từng cặp ảnh

    Ví dụ dùng trong server.py:
        result = check_sequence(paths)
        if not result.all_ok:
            raise HTTPException(422, {"errors": result.messages, "bad_indices": result.bad_indices})
    """
    if len(image_paths) < 2:
        return PlaneCheckResult(all_ok=True, pairs=[])

    pairs: list[PairResult] = []
    bad_set: set[int]       = set()
    messages: list[str]     = []

    for i in range(len(image_paths) - 1):
        pr = _check_pair(
            path_a        = image_paths[i],
            path_b        = image_paths[i + 1],
            idx_a         = i,
            idx_b         = i + 1,
            max_pitch_deg = max_pitch_deg,
            max_vert_shift= max_vert_shift,
            min_inliers   = min_inliers,
            n_features    = n_features,
        )
        pairs.append(pr)
        if not pr.ok and pr.reason:
            bad_set.add(pr.idx_b)
            messages.append(pr.reason)

    return PlaneCheckResult(
        all_ok      = len(bad_set) == 0,
        bad_indices = sorted(bad_set),
        messages    = messages,
        pairs       = pairs,
    )