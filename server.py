"""
Panorama Stitcher — FastAPI Server  (v4 — True Multiprocessing)
  • Plane check (422 nếu ảnh nghiêng)
  • Stitch bằng stitch_v4_mt (ProcessPool trên SIFT/Match/Warp)
  • YOLO sliding-window detection
  • Trả về perf dict để client debug

Thay đổi so với server cũ:
  - Import từ stitch_v4_mt thay vì stitch_panorama
  - Bỏ stitch ST riêng — v4 đã là MT, chạy 1 lần duy nhất
  - _perf_out được điền đầy đủ: load_s, sift_s, match_s, ba_s, warp_s, seam_s, blend_s
  - if __name__ == "__main__" có multiprocessing guard (bắt buộc cho ProcessPool)

Chạy:
    pip install fastapi uvicorn python-multipart opencv-contrib-python-headless scipy ultralytics
    python server.py
"""
import os
import uuid
import shutil
import tempfile
import time
import multiprocessing
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np

# ── Import từ stitch_v4_mt ────────────────────────────────────────────────────
# stitch()    = True-MT (ProcessPool cho SIFT / Match / Warp)
# stitch_mt() = alias của stitch(), giữ để tương thích với code cũ gọi stitch_mt
from stitch_panorama import stitch, stitch_mt

from plane_check import check_sequence

# ── YOLO ─────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

MODEL_PATH   = Path(__file__).parent / "best_soup_190sku.pt"
_yolo_model: YOLO | None = None

def get_yolo() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model

_PALETTE: list[tuple[int,int,int]] = [
    (124,111,255),(74,222,128),(251,191,36),(248,113,113),
    (56,189,248),(251,146,60),(167,139,250),(52,211,153),
    (244,114,182),(94,234,212),(253,224,71),(134,239,172),
]
def _class_color(cls_id: int) -> tuple[int,int,int]:
    return _PALETTE[cls_id % len(_PALETTE)]

SW_PATCH_SIZE = 1280
SW_OVERLAP    = 0.25
SW_NMS_IOU    = 0.45
SW_CONF       = 0.25

def _sliding_window_detect(img: np.ndarray, model: YOLO) -> list[dict]:
    H, W  = img.shape[:2]
    step  = int(SW_PATCH_SIZE * (1 - SW_OVERLAP))
    names = model.names
    raw: list[dict] = []

    y_starts = list(range(0, max(1, H - SW_PATCH_SIZE + 1), step))
    x_starts = list(range(0, max(1, W - SW_PATCH_SIZE + 1), step))
    if not y_starts or y_starts[-1] + SW_PATCH_SIZE < H:
        y_starts.append(max(0, H - SW_PATCH_SIZE))
    if not x_starts or x_starts[-1] + SW_PATCH_SIZE < W:
        x_starts.append(max(0, W - SW_PATCH_SIZE))

    print(f"[YOLO] {W}×{H} → {len(y_starts)*len(x_starts)} patches")
    for oy in y_starts:
        for ox in x_starts:
            ey = min(oy + SW_PATCH_SIZE, H)
            ex = min(ox + SW_PATCH_SIZE, W)
            patch   = img[oy:ey, ox:ex]
            results = model(patch, conf=SW_CONF, verbose=False)[0]
            for box in results.boxes:
                px1,py1,px2,py2 = map(int, box.xyxy[0].tolist())
                raw.append({
                    "cls_id":     int(box.cls[0]),
                    "label":      names.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "confidence": float(box.conf[0]),
                    "x1": ox+px1, "y1": oy+py1,
                    "x2": ox+px2, "y2": oy+py2,
                })

    if not raw:
        return []

    merged: list[dict] = []
    for cls_id in set(d["cls_id"] for d in raw):
        cls_b  = [d for d in raw if d["cls_id"] == cls_id]
        bx     = np.array([[d["x1"],d["y1"],d["x2"],d["y2"]] for d in cls_b], np.float32)
        sc     = np.array([d["confidence"] for d in cls_b], np.float32)
        xywh   = bx.copy(); xywh[:,2] -= xywh[:,0]; xywh[:,3] -= xywh[:,1]
        keep   = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), SW_CONF, SW_NMS_IOU)
        if keep is None: continue
        for i in (keep.flatten() if hasattr(keep,"flatten") else keep):
            merged.append(cls_b[i])
    print(f"[YOLO] {len(raw)} raw → {len(merged)} sau NMS")
    return merged

def _draw_detections(img: np.ndarray, detections: list[dict]) -> list[dict]:
    out = []
    for d in detections:
        x1,y1,x2,y2 = d["x1"],d["y1"],d["x2"],d["y2"]
        color     = _class_color(d["cls_id"])
        color_bgr = (color[2],color[1],color[0])
        r, thickness = 8, 2

        overlay = img.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),color_bgr,-1)
        cv2.addWeighted(overlay,0.08,img,0.92,0,img)
        cv2.line(img,(x1+r,y1),(x2-r,y1),color_bgr,thickness)
        cv2.line(img,(x1+r,y2),(x2-r,y2),color_bgr,thickness)
        cv2.line(img,(x1,y1+r),(x1,y2-r),color_bgr,thickness)
        cv2.line(img,(x2,y1+r),(x2,y2-r),color_bgr,thickness)
        cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color_bgr,thickness)
        cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color_bgr,thickness)
        cv2.ellipse(img,(x1+r,y2-r),(r,r), 90,0,90,color_bgr,thickness)
        cv2.ellipse(img,(x2-r,y2-r),(r,r),  0,0,90,color_bgr,thickness)

        text = f"{d['label']}  {d['confidence']*100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw,th),bl = cv2.getTextSize(text,font,0.52,1)
        pad=6; lx1=x1; ly1=max(0,y1-th-pad*2-bl); lx2=x1+tw+pad*2; ly2=max(0,y1)
        cv2.rectangle(img,(lx1,ly1),(lx2,ly2),color_bgr,-1)
        cv2.putText(img,text,(lx1+pad,ly2-bl-1),font,0.52,(255,255,255),1,cv2.LINE_AA)

        out.append({"label":d["label"],"confidence":round(d["confidence"],3),
                    "bbox":[x1,y1,x2,y2],"color":list(color)})
    return out

def _run_yolo(panorama_path: str, annotated_path: str) -> list[dict]:
    model = get_yolo()
    img   = cv2.imread(panorama_path)
    if img is None:
        raise ValueError("Không đọc được ảnh panorama")
    raw  = _sliding_window_detect(img, model)
    dets = _draw_detections(img, raw)
    cv2.imwrite(annotated_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dets

# ── Benchmark console printer ─────────────────────────────────────────────────
def _print_server_perf(job_id: str, n: int, perf: dict) -> None:
    W = 58
    print(f"\n┌{'─'*W}┐")
    print(f"│  SERVER BENCHMARK  job={job_id[:8]}  ({n} ảnh){'':>{W-44}}│")
    print(f"├{'─'*W}┤")
    rows = [
        ("Upload & save",              "upload_save_s"),
        ("Plane check",                "plane_check_s"),
        ("Stitch v4-MT (total)",       "stitch_mt_s"),
        ("  ├ Load (ThreadPool)",      "st_load_s"),
        ("  ├ SIFT (ProcessPool)",     "st_sift_s"),
        ("  ├ Match (ProcessPool)",    "st_match_s"),
        ("  ├ Bundle adj.",            "st_ba_s"),
        ("  ├ Warp (ProcessPool)",     "st_warp_s"),
        ("  ├ Seam (C++ seq)",         "st_seam_s"),
        ("  └ Blend (seq)",            "st_blend_s"),
        ("YOLO sliding-window",        "yolo_s"),
        ("TOTAL request",              "total_s"),
    ]
    for label, key in rows:
        val = perf.get(key, 0)
        bar = "█" * min(int(val * 5), 20)
        print(f"│  {label:<32} {val:>6.3f}s  {bar:<20} │")
    print(f"├{'─'*W}┤")
    print(f"│  Latency/img: {perf.get('stitch_mt_s',0)/max(n,1):.3f}s{'':>{W-22}}│")
    print(f"└{'─'*W}┘\n")

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Panorama Stitcher API v4")
UPLOAD_DIR = tempfile.mkdtemp()
OUTPUT_DIR = tempfile.mkdtemp()


@app.get("/result/{job_id}")
def get_result(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"panorama_{job_id}.jpeg")
    if not os.path.exists(path):
        raise HTTPException(404, detail="Not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/annotated/{job_id}")
def get_annotated(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"annotated_{job_id}.jpeg")
    if not os.path.exists(path):
        raise HTTPException(404, detail="Not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/health")
def health():
    return {"status": "ok", "stitch_version": "v4-MT"}


@app.post("/stitch")
async def stitch_images(files: list[UploadFile] = File(...)):
    if len(files) < 2:
        raise HTTPException(400, detail="Cần ít nhất 2 ảnh")
    if len(files) > 20:
        raise HTTPException(400, detail="Tối đa 20 ảnh")

    job_id  = uuid.uuid4().hex
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    saved_paths: list[str] = []
    saved_names: list[str] = []
    t_start = time.perf_counter()
    perf:     dict = {}

    try:
        # ── 1. Upload & save ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        for i, upload in enumerate(files):
            ext  = os.path.splitext(upload.filename or "")[1] or ".jpg"
            name = f"img_{i+1:02d}{ext}"
            path = os.path.join(job_dir, name)
            with open(path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(path)
            saved_names.append(name)
        perf["upload_save_s"] = round(time.perf_counter() - t0, 3)

        # ── 2. Plane check ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        check = check_sequence(saved_paths)
        perf["plane_check_s"] = round(time.perf_counter() - t0, 3)

        if not check.all_ok:
            raise HTTPException(422, detail={
                "bad_photo_numbers": [i+1 for i in check.bad_indices],
                "errors":  check.messages,
                "summary": (f"Ảnh {', '.join(str(i+1) for i in check.bad_indices)} "
                            "bị nghiêng hoặc lệch plane. Vui lòng chụp lại."),
            })

        common = dict(
            input_dir=job_dir, input_files=saved_names,
            scale=0.5, n_features=6000, feather=300,
            levels=6, jpeg_quality=92, seam_sigma=15,
        )

        # ── 3. Stitch v4-MT ───────────────────────────────────────────────────
        # Chỉ chạy 1 lần (v4 đã là True-MT, không cần chạy ST riêng để so sánh)
        output_path = os.path.join(OUTPUT_DIR, f"panorama_{job_id}.jpeg")
        perf_inner: dict = {}   # sẽ được điền bởi _perf_out trong stitch()
        t0 = time.perf_counter()
        stitch(output_path=output_path, _perf_out=perf_inner, **common)
        perf["stitch_mt_s"] = round(time.perf_counter() - t0, 3)

        # Gắn timing từng bước vào perf response
        for key in ("load_s","sift_s","match_s","ba_s",
                    "warp_s","seam_s","softmask_s","blend_s"):
            perf[f"st_{key}"] = round(perf_inner.get(key, 0), 3)

        if not os.path.exists(output_path):
            raise HTTPException(500, detail="Stitch thất bại — output file không tồn tại")

        # ── 4. YOLO detection ─────────────────────────────────────────────────
        annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{job_id}.jpeg")
        t0 = time.perf_counter()
        try:
            detections = _run_yolo(output_path, annotated_path)
        except Exception as e:
            detections = []
            print(f"[YOLO] Lỗi: {e}")
        perf["yolo_s"] = round(time.perf_counter() - t0, 3)

        perf["total_s"] = round(time.perf_counter() - t_start, 3)
        _print_server_perf(job_id, len(files), perf)

        summary: dict[str,int] = {}
        for d in detections:
            summary[d["label"]] = summary.get(d["label"], 0) + 1

        return {
            "job_id":        job_id,
            "url":           f"/result/{job_id}",
            "annotated_url": f"/annotated/{job_id}" if os.path.exists(annotated_path) else None,
            "detections":    detections,
            "summary":       summary,
            "perf":          perf,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    # Guard BẮT BUỘC — ProcessPoolExecutor dùng spawn trên Linux/macOS/Windows.
    # Thiếu guard → uvicorn import server.py khi spawn → fork loop vô hạn.
    multiprocessing.set_start_method("spawn", force=True)
    print("Server: http://0.0.0.0:8000")
    print("======= SERVER STARTED (stitch v4-MT) =======")
    uvicorn.run(app, host="0.0.0.0", port=8000)