#!/usr/bin/env python3
"""
yolo_metadata_to_rl.py
------------------------
Tinh semantic_score tung frame tu output raw cua YOLOv5s (yolo_metadata.jsonl),
theo DUNG cong thuc muc 4.1 "Phan tich ngu canh va diem so ngu nghia (Semantic
Scoring)" trong de xuat nghien cuu:

    S = alpha * num_obj + beta * sum(Area_i)
        + gamma * (sum(Motion_i) / N) + delta * sum(Priority_i)

  - num_obj: so doi tuong phat hien trong khung.
  - Area_i: dien tich tung vung ROI (bbox); sum(Area_i) duoc luu la roi_area.
  - motion_vectors: do chuyen dong tung doi tuong TU KHUNG TRUOC den khung
    hien tai. YOLOv5s KHONG co object tracking ID, nen o day dung GHEP GAN
    NHAT (nearest-centroid matching, cung class, trong nguong khoang cach)
    giua frame t-1 va t de xap xi -- day la XAP XI, khong phai optical flow/
    tracking that (can DeepSORT/ByteTrack neu muon chinh xac hon).
  - priority_class: trong so uu tien theo loai doi tuong (nguoi > xe > con
    lai), xem PRIORITY_WEIGHTS ben duoi -- CHINH SUA LAI cho khop bo class
    that ban dung (COCO 80 class cua YOLOv5s).
  - alpha, beta, gamma, delta: he so can bang, truyen qua CLI de de tinh chinh.

QUAN TRONG: script nay KHONG tao ra "bandwidth" -- bandwidth la dieu kien
MANG, khong lien quan gi den YOLO/VCU (xem giai thich lan truoc). Dung
gen_bandwidth_trace.py de tao rieng, roi merge_traces() de ghep lai.

CACH DUNG:
  # Buoc 1: tinh semantic_score tu yolo_metadata.jsonl
  python3 yolo_metadata_to_rl.py \
      --input yolo_metadata.jsonl \
      --out semantic_trace.jsonl \
      --alpha 1.0 --beta 10.0 --gamma 5.0 --delta 2.0

  # Buoc 2 (da co san neu ban chay gen_bandwidth_trace.py roi):
  python3 -c "
from gen_bandwidth_trace import merge_traces
merge_traces('semantic_trace.jsonl', 'bandwidth_trace.jsonl', 'rl_states.jsonl')
"
"""

import argparse
import json
import math
import sys

# Trong so uu tien theo class (nguoi > phuong tien > con lai). CHINH SUA lai
# cho khop muc do "quan trong" ban muon (vd: bien so xe/khuon mat quan trong
# hon xe hoi thuong). Class name khop COCO (dung boi YOLOv5s).
PRIORITY_WEIGHTS = {
    "person": 1.0,
    "bicycle": 0.6,
    "car": 0.8,
    "motorcycle": 0.6,
    "bus": 0.8,
    "truck": 0.8,
    "train": 0.7,
}
DEFAULT_PRIORITY = 0.3  # class khong nam trong bang tren (vd vat the phu)

# Nguong khoang cach (ty le theo duong cheo khung hinh) de coi 2 detection o
# 2 frame lien tiep la "cung 1 doi tuong" khi ghep gan nhat -- xap xi thay
# the cho tracking ID that.
MAX_MATCH_DIST_RATIO = 0.08


def load_yolo_metadata(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["frame_idx"])
    return rows


def match_and_compute_motion(prev_dets, cur_dets, diag):
    """Ghep gan nhat (nearest-centroid, cung class) giua 2 frame lien tiep de
    xap xi motion_vectors. Tra ve list khoang cach DA CHUAN HOA (0..~1 theo
    duong cheo khung hinh) cho tung doi tuong o frame hien tai co ghep duoc."""
    if not prev_dets or not cur_dets:
        return []

    prev_by_class = {}
    for d in prev_dets:
        prev_by_class.setdefault(d["class_id"], []).append(d)

    motions = []
    used_prev = set()
    for d in cur_dets:
        candidates = prev_by_class.get(d["class_id"], [])
        best_dist, best_idx = None, None
        for i, pd in enumerate(candidates):
            if id(pd) in used_prev:
                continue
            dx = d["centroid"][0] - pd["centroid"][0]
            dy = d["centroid"][1] - pd["centroid"][1]
            dist = math.hypot(dx, dy)
            if best_dist is None or dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx is not None and best_dist is not None:
            dist_norm = best_dist / diag
            if dist_norm <= MAX_MATCH_DIST_RATIO * 5:  # loai bo ghep sai qua xa
                # (nguong rong hon MAX_MATCH_DIST_RATIO vi doi tuong co the
                # di chuyen nhanh giua 2 frame; chi loai truong hop qua vo ly)
                motions.append(dist_norm)
                used_prev.add(id(candidates[best_idx]))
    return motions


def compute_semantic_scores(rows, alpha, beta, gamma, delta):
    out = []
    prev_dets = None
    for row in rows:
        width, height = row["width"], row["height"]
        diag = math.hypot(width, height)
        dets = row.get("detections", [])
        num_obj = len(dets)

        roi_area = sum(d.get("area_ratio", 0.0) for d in dets)  # dien tich ROI (ty le, 0..1..N neu chong lan)
        priority_sum = sum(
            PRIORITY_WEIGHTS.get(d.get("class_name", ""), DEFAULT_PRIORITY) for d in dets
        )

        motions = match_and_compute_motion(prev_dets, dets, diag) if prev_dets is not None else []
        motion_avg = (sum(motions) / num_obj) if num_obj > 0 else 0.0
        motion_sum = sum(motions)  # luu lai de debug/quan sat rieng

        semantic_score = (
            alpha * num_obj
            + beta * roi_area
            + gamma * motion_avg
            + delta * priority_sum
        )

        out.append({
            "frame_idx": row["frame_idx"],
            "semantic_score": round(float(semantic_score), 4),
            "num_objects": num_obj,
            "roi_area": round(float(roi_area), 4),
            "motion": round(float(motion_avg), 4),
            "motion_sum": round(float(motion_sum), 4),
            "priority": round(float(priority_sum), 4),
        })
        prev_dets = dets
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="yolo_metadata.jsonl (raw detections)")
    ap.add_argument("--out", required=True, help="File jsonl output (chi co semantic, "
                     "CHUA co bandwidth -- merge sau bang merge_traces())")
    ap.add_argument("--alpha", type=float, default=1.0, help="Trong so so luong doi tuong num_obj")
    ap.add_argument("--beta", type=float, default=10.0, help="Trong so tong dien tich ROI sum(Area_i)")
    ap.add_argument("--gamma", type=float, default=5.0, help="Trong so chuyen dong trung binh sum(Motion_i)/N")
    ap.add_argument("--delta", type=float, default=2.0, help="Trong so tong uu tien class sum(Priority_i)")
    args = ap.parse_args()

    rows = load_yolo_metadata(args.input)
    results = compute_semantic_scores(rows, args.alpha, args.beta, args.gamma, args.delta)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scores = [r["semantic_score"] for r in results]
    print(f"[done] Da tinh semantic_score cho {len(results)} frame -> {args.out}", file=sys.stderr)
    print(f"[info] semantic_score: min={min(scores):.2f} max={max(scores):.2f} "
          f"trung binh={sum(scores)/len(scores):.2f}", file=sys.stderr)
    nonzero = sum(1 for s in scores if s > 0)
    print(f"[info] so frame co object (semantic_score > 0): {nonzero}/{len(scores)}", file=sys.stderr)


if __name__ == "__main__":
    main()

