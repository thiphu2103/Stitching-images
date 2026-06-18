"""
Panorama Stitcher v4 — True Multiprocessing
═══════════════════════════════════════════════════════════════════════════════
So sánh với v3 (cv2.setNumThreads):
  v3 approach: flip 1 số → OpenCV OpenMP/TBB chạy bên trong từng hàm C++
               → speedup thực tế ~1.3–1.8× vì GIL vẫn held khi Python
                 tạo KeyPoint objects, overhead context-switch cao

  v4 approach: ProcessPoolExecutor → mỗi process có GIL + memory riêng
               → speedup ~N× (gần tuyến tính với core count) cho bước
                 thực sự CPU-bound

═══════════════════════════════════════════════════════════════════════════════
Quyết định parallelise từng bước:

  Bước              Cơ chế          Lý do
  ─────────────────────────────────────────────────────────────────────────
  Load ×12        ThreadPool      I/O bound — GIL released khi đọc disk
  SIFT ×12        ProcessPool     CPU-bound nặng nhất; mỗi ảnh độc lập
  Match ×11       ProcessPool     11 pairs hoàn toàn độc lập nhau
  Warp ×12        ProcessPool     img+mask gộp 1 worker, embarrassingly //
  Soft mask ×12   ProcessPool     GaussianBlur per-image, độc lập
  Bundle adj.     sequential      LM solver iterative — không thể split
  Seam find       sequential      C++ DpSeamFinder ~0.1 s; overhead > gain
  Blend           sequential      Pyramid accumulation cần global state
  Write           sequential      1 output file

═══════════════════════════════════════════════════════════════════════════════
Fix so với v3:
  LEVELS  = 6   (v3 để 200 → bug: ảnh ~3 kpx chỉ 11 octave tối đa;
                 200 tầng pyrDown → 0×0 px → NaN/OOM toàn bộ blend)
  FEATHER = 300  (v3 để 500 > IH//2 tại scale=0.7 → mask bị clamp)

Cài đặt:
    pip install opencv-contrib-python-headless scipy

Chạy:
    python stitch_v4_mt.py
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR    = "uploads"
INPUT_FILES  = [f"img_{i:02d}.jpeg" for i in range(1, 13)]
OUTPUT_PATH  = "panorama_v4.jpeg"
SCALE        = 0.7
N_FEATURES   = 8000
FEATHER      = 300      # phải < min(IH, IW) // 2
LEVELS       = 6        # ← FIX: v3 để 200 là bug (pyrDown tới 0px = NaN)
JPEG_QUALITY = 95
SEAM_SIGMA   = 20
N_WORKERS    = None     # None = os.cpu_count()
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Worker functions — BẮT BUỘC ở module level (không phải lambda/inner func)
# ProcessPoolExecutor dùng pickle để truyền work vào subprocess; pickle chỉ
# serialize được function được định nghĩa ở top-level của module.
# ══════════════════════════════════════════════════════════════════════════════

def _worker_load(args):
    """
    Worker: load + resize 1 ảnh.  Chạy trong ThreadPoolExecutor (I/O bound).

    Tại sao ThreadPool thay vì ProcessPool?
    imread() và resize() block ở disk I/O; GIL được release trong C++ khi
    chờ I/O → các thread Python khác chạy được trong thời gian chờ đó.
    ProcessPool sẽ tốn ~80 ms spawn overhead cho mỗi process mà không
    cải thiện throughput thêm được vì bottleneck là disk, không phải CPU.
    """
    path, scale = args
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    h, w = img.shape[:2]
    img  = cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)
    if   len(img.shape) == 2:    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2]   == 4:    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _worker_sift(args):
    """
    Worker: SIFT feature detection cho 1 ảnh.  Chạy trong ProcessPoolExecutor.

    Tại sao ProcessPool (không phải ThreadPool)?
    detectAndCompute() là pure CPU: Gaussian scale-space, DoG, keypoint
    localization, orientation, descriptor.  GIL được held khi OpenCV tạo
    Python cv2.KeyPoint objects → threads tranh nhau GIL → không speedup.
    ProcessPool cho mỗi process 1 GIL riêng → N process chạy song song thật.

    Tricky: cv2.KeyPoint KHÔNG pickle được natively → serialize thành numpy
    array (7 float32 / keypoint) để truyền qua pipe của ProcessPool.
    Overhead pickle ~40–60 ms / ảnh; với SIFT ~1 s / ảnh, hoàn toàn chấp nhận.
    """
    img_bytes, n_features = args
    # Decode từ JPEG bytes (nhỏ hơn raw ndarray ~10× khi pickle)
    img  = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    sift = cv2.SIFT_create(n_features)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # Serialize keypoints thành numpy array để pickle được
    kp_arr = np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle,
          k.response, k.octave, k.class_id] for k in kp],
        dtype=np.float32
    ) if kp else np.empty((0, 7), np.float32)
    return kp_arr, des   # des là np.ndarray → pickle native


def _kp_from_arr(arr: np.ndarray):
    """Rebuild list[cv2.KeyPoint] từ numpy array đã serialize."""
    return [
        cv2.KeyPoint(x=float(r[0]), y=float(r[1]), size=float(r[2]),
                     angle=float(r[3]), response=float(r[4]),
                     octave=int(r[5]), class_id=int(r[6]))
        for r in arr
    ]


def _worker_match(args):
    """
    Worker: match 1 cặp ảnh (i, i+1).  Chạy trong ProcessPoolExecutor.

    Pairs (i,j) hoàn toàn độc lập → embarrassingly parallel.
    Input là kp_arr + des numpy arrays (không gửi lại ảnh, tiết kiệm bandwidth).
    FLANN + USAC_MAGSAC đều là pure CPU → ProcessPool đúng.
    """
    i, j, kp1_arr, des1, kp2_arr, des2 = args
    kp1 = _kp_from_arr(kp1_arr)
    kp2 = _kp_from_arr(kp2_arr)

    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return i, j, None, None, None

    flann   = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5},
                                     {"checks": 100})
    matches = flann.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.72 * n.distance]
    if len(good) < 15:
        return i, j, None, None, None

    src = np.float32([kp2[m.trainIdx].pt for m in good])
    dst = np.float32([kp1[m.queryIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0,
                                  maxIters=5000, confidence=0.9999)
    if H is None:
        return i, j, None, None, None

    inl = mask.ravel() == 1
    return i, j, H, dst[inl], src[inl]


def _worker_warp(args):
    """
    Worker: warpPerspective cho cả ảnh + feather mask trong 1 lần spawn.
    Chạy trong ProcessPoolExecutor.

    Thiết kế: gộp warp img + warp mask vào cùng 1 worker để:
      - Giảm số lần spawn process từ 24 xuống 12
      - Tái dụng img đã decode, tránh decode 2 lần
    Mỗi warp dùng H riêng, không share state → safe.

    Truyền img dạng JPEG bytes (không phải raw float32 ndarray) vì:
      raw float32 (1344×756×3) ~ 11 MB × 12 = 132 MB qua pipe
      JPEG 95q                 ~ 0.8 MB × 12 = 9.6 MB qua pipe
    """
    img_bytes, base_mask_bytes, H, canvas_wh, img_hw = args
    CW, CH = canvas_wh
    IH, IW = img_hw

    img       = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                              cv2.IMREAD_COLOR)
    base_mask = np.frombuffer(base_mask_bytes, np.float32).reshape(IH, IW)

    wi = cv2.warpPerspective(img.astype(np.float32), H, (CW, CH),
                              flags=cv2.INTER_LINEAR)
    wm = cv2.warpPerspective(base_mask, H, (CW, CH),
                              flags=cv2.INTER_LINEAR)
    return wi, wm


def _worker_softmask(args):
    """
    Worker: hard ownership → GaussianBlur soft mask cho 1 ảnh.
    Chạy trong ProcessPoolExecutor.

    ownership và warped_mask được truyền dưới dạng raw bytes (không pickle
    object Python) → nhanh hơn.  Dùng np.frombuffer để reconstruct.
    """
    idx, own_bytes, shape_hw, wmask_bytes, seam_sigma = args
    CH, CW = shape_hw
    ownership   = np.frombuffer(own_bytes,   np.int32  ).reshape(CH, CW)
    warped_mask = np.frombuffer(wmask_bytes, np.float32).reshape(CH, CW)

    hard = (ownership == idx).astype(np.float32)
    soft = cv2.GaussianBlur(hard, (0, 0), seam_sigma)
    soft *= (warped_mask > 0.01).astype(np.float32)
    return idx, soft


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    """Encode BGR uint8 → JPEG bytes để truyền qua ProcessPool pipe nhỏ hơn."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ══════════════════════════════════════════════════════════════════════════════
# Bundle adjustment — sequential (LM solver không parallelise được)
# ══════════════════════════════════════════════════════════════════════════════

def bundle_adjust(pair_data, n_imgs, img_w, img_h):
    """
    Global bundle adjustment: least_squares (Levenberg–Marquardt).
    LM là iterative solver — mỗi iteration phụ thuộc kết quả iteration trước
    → không thể split thành independent tasks → giữ sequential.
    """
    def H_from_p(p): return np.append(p, 1.0).reshape(3, 3)
    def p_from_H(H): return (H / H[2, 2]).flatten()[:8]

    Hc = [np.eye(3)]
    for i in range(n_imgs - 1):
        if (i, i + 1) in pair_data:
            Hc.append(Hc[-1] @ pair_data[(i, i + 1)][0])
        else:
            T = np.eye(3); T[0, 2] = img_w * 0.45
            Hc.append(Hc[-1] @ T)

    x0 = np.concatenate([p_from_H(H) for H in Hc[1:]])

    def residuals(x):
        Hs  = [np.eye(3)] + [H_from_p(x[8*k:8*k+8]) for k in range(n_imgs-1)]
        res = []
        for (i, j), (_, pi, pj) in pair_data.items():
            pih = np.c_[pi, np.ones(len(pi))].T
            pjh = np.c_[pj, np.ones(len(pj))].T
            ci  = Hs[i] @ pih;  ci /= ci[2]
            cj  = Hs[j] @ pjh;  cj /= cj[2]
            res.append((ci[:2] - cj[:2]).T.flatten())
        return np.concatenate(res)

    init_cost = np.sum(residuals(x0) ** 2) / 2
    opt = least_squares(residuals, x0, method="lm",
                         max_nfev=3000, ftol=1e-7, xtol=1e-7)
    print(f"  BA cost: {opt.cost:.1f}  (init: {init_cost:.1f},  "
          f"reduction: {100*(1-opt.cost/init_cost):.0f}%)")
    return [np.eye(3)] + [H_from_p(opt.x[8*k:8*k+8]) for k in range(n_imgs-1)]


# ══════════════════════════════════════════════════════════════════════════════
# Feather mask — sequential (nhanh, chỉ 1 lần)
# ══════════════════════════════════════════════════════════════════════════════

def make_feather_mask(h, w, feather):
    mask = np.ones((h, w), np.float32)
    f    = min(feather, h // 2 - 1, w // 2 - 1)  # clamp tránh OOB
    for r in range(f):
        v = (r + 1) / f
        mask[r, :];      mask[r, :]     = np.minimum(mask[r, :],     v)
        mask[h-1-r, :] = np.minimum(mask[h-1-r, :], v)
        mask[:, r]     = np.minimum(mask[:, r],     v)
        mask[:, w-1-r] = np.minimum(mask[:, w-1-r], v)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Seam finding — sequential (C++ DpSeamFinder ~0.1 s; overhead > gain)
# ══════════════════════════════════════════════════════════════════════════════

def find_seam_ownership(warped_imgs, warped_masks, n):
    """
    cv2.detail_DpSeamFinder (C++) — sequential.

    Tại sao không parallel?
    DpSeamFinder cần thấy TẤT CẢ ảnh cùng lúc để tìm seam tối ưu giữa các
    cặp chồng lấp.  Nếu split thành N subprocess thì mỗi proc chỉ thấy 1 pair
    → không còn là global DP nữa, chất lượng seam giảm.
    Hơn nữa, C++ implementation đã ~0.1 s → không đáng spawn process.
    """
    CH, CW = warped_masks[0].shape
    corners_cv, images_cv, masks_cv = [], [], []

    for idx in range(n):
        bm   = (warped_masks[idx] > 0.05).astype(np.uint8) * 255
        ys, xs = np.where(bm > 0)
        corners_cv.append((int(xs.min()) if len(xs) else 0,
                           int(ys.min()) if len(ys) else 0))
        images_cv.append(np.clip(warped_imgs[idx], 0, 255).astype(np.uint8))
        masks_cv.append(bm)

    cv2.detail_DpSeamFinder("COLOR").find(images_cv, corners_cv, masks_cv)

    ownership = np.full((CH, CW), -1, dtype=np.int32)
    for idx in range(n - 1, -1, -1):
        ownership[masks_cv[idx] > 0] = idx
    return ownership


# ══════════════════════════════════════════════════════════════════════════════
# Multi-band blend — sequential accumulation
# ══════════════════════════════════════════════════════════════════════════════

def multiband_blend(warped_imgs, soft_masks, levels):
    """
    Laplacian pyramid multi-band blend.

    Tại sao giữ sequential?
    Vòng lặp accumulate (pa[lvl] += lpi[lvl] * gpm[lvl]) cần global pa/pw
    được cập nhật sau mỗi ảnh.  Có thể parallel build_lp cho mỗi ảnh, nhưng
    với LEVELS=6 và 12 ảnh, cost build_lp ~0.04 s/ảnh → overhead ProcessPool
    (~80 ms) lớn hơn gain.  Nếu N > 50 ảnh hoặc LEVELS > 10, mới đáng.
    """
    def build_lp(img, lvls):
        gp = [img]
        for _ in range(lvls - 1): gp.append(cv2.pyrDown(gp[-1]))
        lp = []
        for i in range(lvls - 1):
            up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1]); return lp

    def build_gp(m, lvls):
        gp = [m]
        for _ in range(lvls - 1): gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def recon(lp):
        r = lp[-1]
        for l in reversed(lp[:-1]):
            r = cv2.pyrUp(r, dstsize=(l.shape[1], l.shape[0])) + l
        return r

    pa = pw = None
    for img, mask in zip(warped_imgs, soft_masks):
        m3  = np.stack([mask] * 3, axis=2)
        lpi = build_lp(img, levels)
        gpm = build_gp(m3, levels)
        if pa is None:
            pa = [np.zeros_like(l) for l in lpi]
            pw = [np.zeros_like(g) for g in gpm]
        for lvl in range(levels):
            pa[lvl] += lpi[lvl] * gpm[lvl]
            pw[lvl] += gpm[lvl]

    blended = [a / np.maximum(w, 1e-6) for a, w in zip(pa, pw)]
    return np.clip(recon(blended), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_perf(mode: str, n: int, perf: dict) -> None:
    W   = 56
    sep = "─" * W
    print(f"\n┌{sep}┐")
    print(f"│  BENCHMARK [{mode}]  {n} imgs{'':{W-30}}│")
    print(f"├{sep}┤")
    steps = [
        ("Load + resize (ThreadPool)",   "load_s"),
        ("SIFT × N (ProcessPool)",       "sift_s"),
        ("Match × N-1 (ProcessPool)",    "match_s"),
        ("Bundle adjustment",            "ba_s"),
        ("Warp img+mask (ProcessPool)",  "warp_s"),
        ("Seam finding (C++ seq)",       "seam_s"),
        ("Soft masks (ProcessPool)",     "softmask_s"),
        ("Multi-band blend (seq)",       "blend_s"),
        ("Crop + JPEG write",            "write_s"),
    ]
    total = perf.get("total_s", 0)
    for label, key in steps:
        val = perf.get(key, 0)
        pct = val / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 4)
        print(f"│  {label:<32} {val:>6.3f}s  {pct:>4.0f}%  {bar:<8} │")
    print(f"├{sep}┤")
    print(f"│  {'TOTAL':<32} {total:>6.3f}s{'':{W-42}} │")
    print(f"│  Latency / image:             {total/max(n,1):>6.3f}s{'':{W-42}} │")
    print(f"└{sep}┘\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main — stitch() với True Multiprocessing
# ══════════════════════════════════════════════════════════════════════════════

def stitch(input_dir=INPUT_DIR, input_files=INPUT_FILES,
           output_path=OUTPUT_PATH, scale=SCALE, n_features=N_FEATURES,
           feather=FEATHER, levels=LEVELS, jpeg_quality=JPEG_QUALITY,
           seam_sigma=SEAM_SIGMA, n_workers=N_WORKERS,
           _perf_out: dict | None = None):
    """
    True-multiprocessing stitch — 1 shared ProcessPool cho toàn pipeline.

    Key fix v4.1:
    • Dùng fork (Linux default) thay vì spawn → không import lại interpreter
      → pool.start() từ ~3s xuống ~0.05s
    • 1 ProcessPool duy nhất tồn tại suốt pipeline (SIFT → Match → Warp →
      Softmask), không tạo/hủy pool 4 lần riêng biệt
    • nw = min(workers, N) để không spawn process thừa
    """
    n_cpu = os.cpu_count() or 4
    nw    = min(n_workers or n_cpu, len(input_files))
    N     = len(input_files)
    perf  = {}
    print(f"\n[v4] Workers={nw}/{n_cpu}  N={N}  LEVELS={levels}  FEATHER={feather}")

    # ── 1. Load  (ThreadPool — I/O bound) ────────────────────────────────────
    t0    = time.perf_counter()
    paths = [os.path.join(input_dir, f) for f in input_files]
    with ThreadPoolExecutor(max_workers=nw) as tpool:
        futs = {tpool.submit(_worker_load, (p, scale)): i
                for i, p in enumerate(paths)}
        buf  = {}
        for fut in as_completed(futs):
            buf[futs[fut]] = fut.result()
    imgs = [buf[i] for i in range(N)]
    perf["load_s"] = time.perf_counter() - t0
    IH, IW = imgs[0].shape[:2]
    print(f"[v4] Load:     {perf['load_s']:.3f}s  ({IW}×{IH})")

    # Encode ảnh → JPEG bytes một lần; tái dùng cho SIFT + Warp workers
    imgs_bytes = [_encode_jpeg(img) for img in imgs]

    # ── 1 ProcessPool dùng chung cho bước 2, 3, 5, 7 ────────────────────────
    # fork (Linux default) → child process kế thừa memory của parent,
    # không cần import lại OpenCV/numpy → start time ~50ms thay vì ~800ms/spawn
    with ProcessPoolExecutor(max_workers=nw) as pool:

        # ── 2. SIFT  (CPU bottleneck, bypass GIL) ────────────────────────────
        t0   = time.perf_counter()
        futs = {pool.submit(_worker_sift, (b, n_features)): i
                for i, b in enumerate(imgs_bytes)}
        feat = {}
        for fut in as_completed(futs):
            feat[futs[fut]] = fut.result()
        features = [feat[i] for i in range(N)]
        perf["sift_s"] = time.perf_counter() - t0
        print(f"[v4] SIFT:     {perf['sift_s']:.3f}s  ({nw} processes)")

        # ── 3. Match pairs  (pairs độc lập nhau) ─────────────────────────────
        t0 = time.perf_counter()
        match_args = [
            (i, i+1, features[i][0], features[i][1],
                     features[i+1][0], features[i+1][1])
            for i in range(N - 1)
        ]
        pairs = {}
        mfuts = [pool.submit(_worker_match, arg) for arg in match_args]
        for fut in as_completed(mfuts):
            i, j, H, p1, p2 = fut.result()
            if H is not None:
                pairs[(i, j)] = (H, p1, p2)
            else:
                print(f"  WARNING: no match img{i+1}→img{j+1}")
        perf["match_s"] = time.perf_counter() - t0
        print(f"[v4] Match:    {perf['match_s']:.3f}s  ({len(pairs)}/{N-1} matched)")

        # ── 4. Bundle adjustment  (sequential, LM không split được) ──────────
        t0    = time.perf_counter()
        H_opt = bundle_adjust(pairs, N, IW, IH)
        perf["ba_s"] = time.perf_counter() - t0
        print(f"[v4] BA:       {perf['ba_s']:.3f}s")

        # ── Canvas bounds ─────────────────────────────────────────────────────
        corners = np.float32([[0,0],[IW,0],[IW,IH],[0,IH]]).reshape(-1,1,2)
        all_c   = np.concatenate([cv2.perspectiveTransform(corners, H) for H in H_opt])
        xmin, ymin = all_c[:,0,0].min(), all_c[:,0,1].min()
        xmax, ymax = all_c[:,0,0].max(), all_c[:,0,1].max()
        ox, oy = -xmin + 2, -ymin + 2
        Toff   = np.array([[1,0,ox],[0,1,oy],[0,0,1]], np.float64)
        Hf     = [Toff @ H for H in H_opt]
        CW, CH = int(xmax - xmin) + 4, int(ymax - ymin) + 4
        print(f"[v4] Canvas:   {CW}×{CH}")

        # ── 5. Warp img + mask  (gộp 2 warp / worker) ────────────────────────
        t0        = time.perf_counter()
        base_mask = make_feather_mask(IH, IW, feather)
        bm_bytes  = base_mask.astype(np.float32).tobytes()
        warp_args = [
            (imgs_bytes[idx], bm_bytes, Hf[idx], (CW, CH), (IH, IW))
            for idx in range(N)
        ]
        wfuts = {pool.submit(_worker_warp, arg): idx
                 for idx, arg in enumerate(warp_args)}
        wmap  = {}
        for fut in as_completed(wfuts):
            wmap[wfuts[fut]] = fut.result()
        warped_imgs  = [wmap[i][0] for i in range(N)]
        warped_masks = [wmap[i][1] for i in range(N)]
        perf["warp_s"] = time.perf_counter() - t0
        print(f"[v4] Warp:     {perf['warp_s']:.3f}s  ({nw} processes)")

        # ── 6. Seam finding  (C++ DpSeamFinder — sequential, ~0.1s) ──────────
        t0        = time.perf_counter()
        ownership = find_seam_ownership(warped_imgs, warped_masks, N)
        perf["seam_s"] = time.perf_counter() - t0
        print(f"[v4] Seam:     {perf['seam_s']:.3f}s  (C++ sequential)")

        # ── 7. Soft masks  (GaussianBlur per-image, độc lập) ─────────────────
        t0     = time.perf_counter()
        own_b  = ownership.astype(np.int32).tobytes()
        shp_hw = (CH, CW)
        sm_args = [
            (idx, own_b, shp_hw,
             warped_masks[idx].astype(np.float32).tobytes(), seam_sigma)
            for idx in range(N)
        ]
        sfuts  = {pool.submit(_worker_softmask, arg): None for arg in sm_args}
        sm_map = {}
        for fut in as_completed(sfuts):
            idx, soft = fut.result()
            sm_map[idx] = soft
        soft_masks = [sm_map[i] for i in range(N)]
        perf["softmask_s"] = time.perf_counter() - t0
        print(f"[v4] Softmask: {perf['softmask_s']:.3f}s  ({nw} processes)")

    # pool.__exit__ — workers tắt sạch sau khi ra khỏi context

    # ── 8. Multi-band blend  (sequential — pyramid accumulate) ───────────────
    t0     = time.perf_counter()
    result = multiband_blend(warped_imgs, soft_masks, levels)
    perf["blend_s"] = time.perf_counter() - t0
    print(f"[v4] Blend:    {perf['blend_s']:.3f}s  (LEVELS={levels})")

    # ── 9. Crop + write ───────────────────────────────────────────────────────
    t0   = time.perf_counter()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    nz   = np.where(gray > 2)
    result = result[nz[0].min():nz[0].max()+1,
                    nz[1].min():nz[1].max()+1]
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    perf["write_s"] = time.perf_counter() - t0

    perf["total_s"] = sum(perf.values())
    _print_perf(f"v4-MT  workers={nw}", N, perf)
    print(f"Done: {result.shape[1]}×{result.shape[0]} px → {output_path}")

    if _perf_out is not None:
        _perf_out.update(perf)

    return result


stitch_mt = stitch   # alias cho server.py


if __name__ == "__main__":
    # Linux dùng fork (default) → không cần set_start_method
    # Chỉ cần guard để tránh recursive import trên Windows/macOS khi dùng spawn
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)
    stitch()