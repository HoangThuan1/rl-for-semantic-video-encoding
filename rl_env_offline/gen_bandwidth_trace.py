#!/usr/bin/env python3
"""
gen_bandwidth_trace.py
-----------------------
Sinh cot "bandwidth" (kbps) cho tung frame, de yolo_metadata_to_rl.py merge
voi YOLO metadata va tinh semantic_score thanh rl_states.jsonl hoan chinh.
encode_grid.py la buoc rieng de tao bitrate/VMAF lookup cho reward,
khong phai nguon tao semantic_score.

Vi sao can file rieng: bandwidth la dieu kien MANG tai thoi diem do, khong
lien quan gi toi noi dung video hay bo ma hoa -- YOLO khong biet, VCU cung
khong tao ra no (VCU chi PHAN UNG voi bandwidth). Co 2 che do:

  1) SYNTHETIC (mac dinh, khong can mang/internet): mo phong bang mot
     Markov chain "dinh" (sticky) qua vai muc bang thong (kem/trung binh/
     tot/rat tot), co xac suat nho de nhay vao trang thai "nghen mang dot
     ngot" (congestion) roi tu hoi phuc dan -- dung ky thuat pho bien trong
     cac paper ABR-RL (vd Pensieve) khi khong co trace do that.

  2) REAL (--from-real-trace): doc 1 file trace mang THAT ban tu tai ve
     (vd FCC Measuring Broadband America, hoac bo trace 4G/5G cua truong
     Oslo -- https://www.uni-goettingen.de forks tren github, hoac dataset
     Cong khai khac ban co san duoi dang CSV 2 cot time_s,bandwidth_kbps),
     roi noi suy (resample) ve dung so frame/fps ban can.

CACH DUNG:
  # Che do synthetic, khop so frame voi video/metadata cua ban:
  python3 gen_bandwidth_trace.py --num-frames 1500 --fps 25 --out outputs/metadata/bandwidth_trace.jsonl --levels 800,2500,5000,8000 --seed 42

  # Che do dung trace mang that ban da co san (CSV: time_s,bandwidth_kbps):
  python3 gen_bandwidth_trace.py --num-frames 1500 --fps 25 --out outputs/metadata/bandwidth_trace.jsonl --from-real-trace my_4g_trace.csv

OUTPUT: file .jsonl, moi dong {"frame_idx": i, "bandwidth": kbps}
        -- dung dinh dang khop voi rl_states.jsonl hien tai de merge theo
        frame_idx (xem ham merge_traces() cuoi file, dung lam vi du).
"""

import argparse
import csv
import json
import os
import random
import sys

BANDWIDTH_TRACE = "outputs/metadata/bandwidth_trace.jsonl"


def gen_synthetic_bandwidth(num_frames, fps, levels, seed=None,
                             min_hold_sec=2.0, max_hold_sec=8.0,
                             congestion_prob=0.03, congestion_hold_sec=3.0,
                             noise_frac=0.05):
    """Markov chain 'dinh' qua cac muc bandwidth (kbps).

    - O moi 'giai doan' (hold), bandwidth giu quanh 1 muc trong `levels`
      trong khoang [min_hold_sec, max_hold_sec] giay roi moi chuyen sang
      muc lang gieng (mo phong client di chuyen / thay doi vi tri song,
      thay vi nhay lung tung giua cac muc rat xa nhau -> giong mang that
      hon la random moi frame).
    - Voi xac suat `congestion_prob` moi giai doan, chen 1 doan "nghen mang"
      ngan (bandwidth tut xuong muc thap nhat) truoc khi quay lai muc binh
      thuong -- mo phong tinh huong stress-test quan trong nhat cho RL
      controller (theo dung muc 5.2/5.4 cua de xuat: "mang on dinh vs gian
      doan").
    - `noise_frac`: nhieu ngau nhien +-X% quanh muc de tranh duong hoan toan
      phang (khong thuc te).
    """
    rng = random.Random(seed)
    levels = sorted(levels)
    n_levels = len(levels)

    bandwidths = []
    level_idx = rng.randrange(n_levels)
    t = 0
    while t < num_frames:
        in_congestion = rng.random() < congestion_prob
        hold_sec = (congestion_hold_sec if in_congestion
                    else rng.uniform(min_hold_sec, max_hold_sec))
        hold_frames = max(1, int(round(hold_sec * fps)))

        base = levels[0] if in_congestion else levels[level_idx]
        for _ in range(hold_frames):
            if t >= num_frames:
                break
            noise = 1.0 + rng.uniform(-noise_frac, noise_frac)
            bandwidths.append(max(50.0, base * noise))
            t += 1

        if not in_congestion:
            # chuyen sang muc lang gieng (khong nhay xa) de mo phong bien
            # thien muot ma hon, giong pattern mang di dong that
            step = rng.choice([-1, 1])
            level_idx = min(max(level_idx + step, 0), n_levels - 1)

    return bandwidths[:num_frames]


def load_real_trace_csv(path):
    """Doc file CSV 2 cot: time_s, bandwidth_kbps (khong header hoac co
    header 'time,bandwidth'). Tra ve 2 list (times, bandwidths)."""
    times, bws = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                t = float(row[0])
                bw = float(row[1])
            except ValueError:
                continue  # bo qua dong header
            times.append(t)
            bws.append(bw)
    if not times:
        raise ValueError(f"Khong doc duoc du lieu so tu {path}")
    return times, bws


def resample_real_trace(times, bws, num_frames, fps):
    """Noi suy tuyen tinh trace mang that (thuong lay mau vai giay/lan)
    ve dung moc thoi gian cua tung frame video (1/fps giay/frame)."""
    out = []
    n = len(times)
    for i in range(num_frames):
        target_t = i / fps
        # tim doan [times[j], times[j+1]] chua target_t (gia dinh times tang dan)
        if target_t <= times[0]:
            out.append(bws[0])
            continue
        if target_t >= times[-1]:
            out.append(bws[-1])
            continue
        # tim kiem nhi phan don gian
        lo, hi = 0, n - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if times[mid] <= target_t:
                lo = mid
            else:
                hi = mid
        t0, t1 = times[lo], times[hi]
        b0, b1 = bws[lo], bws[hi]
        frac = (target_t - t0) / (t1 - t0) if t1 > t0 else 0.0
        out.append(b0 + frac * (b1 - b0))
    return out


def merge_traces(semantic_jsonl_path, bandwidth_jsonl_path, out_path):
    """Vi du ghep semantic_score (tu yolo_metadata_to_rl.py) + bandwidth
    (tu file nay) theo frame_idx thanh 1 file rl_states.jsonl duy nhat.
    current_qp: neu chua co nguon that, tam dat = qp trung binh (min+max)/2
    cho frame dau tien roi de env tu cap nhat qua cac buoc (xem ghi chu
    trong env.py ve gioi han cua current_qp lay tu trace tinh)."""
    bw_by_idx = {}
    with open(bandwidth_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            bw_by_idx[row["frame_idx"]] = row["bandwidth"]

    merged = []
    with open(semantic_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            idx = row["frame_idx"]
            if idx not in bw_by_idx:
                continue
            row["bandwidth"] = bw_by_idx[idx]
            merged.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(merged)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-frames", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", default=BANDWIDTH_TRACE)
    ap.add_argument("--levels", default="800,2500,5000,8000",
                     help="Cac muc bandwidth kbps, cach nhau boi dau phay "
                          "(chi dung khi KHONG co --from-real-trace)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-hold-sec", type=float, default=2.0)
    ap.add_argument("--max-hold-sec", type=float, default=8.0)
    ap.add_argument("--congestion-prob", type=float, default=0.03)
    ap.add_argument("--from-real-trace", default=None,
                     help="Duong dan CSV (time_s,bandwidth_kbps) neu muon "
                          "dung trace mang THAT thay vi synthetic")
    args = ap.parse_args()

    if args.from_real_trace:
        times, bws = load_real_trace_csv(args.from_real_trace)
        bandwidths = resample_real_trace(times, bws, args.num_frames, args.fps)
        print(f"[info] Da noi suy {len(bandwidths)} frame tu trace that "
              f"'{args.from_real_trace}' ({len(times)} diem do goc)",
              file=sys.stderr)
    else:
        levels = [float(x) for x in args.levels.split(",")]
        bandwidths = gen_synthetic_bandwidth(
            args.num_frames, args.fps, levels, seed=args.seed,
            min_hold_sec=args.min_hold_sec, max_hold_sec=args.max_hold_sec,
            congestion_prob=args.congestion_prob,
        )
        print(f"[info] Da sinh synthetic bandwidth: {len(bandwidths)} frame, "
              f"levels={levels}", file=sys.stderr)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, bw in enumerate(bandwidths):
            f.write(json.dumps({"frame_idx": i, "bandwidth": round(bw, 1)},
                                ensure_ascii=False) + "\n")
    print(f"[done] Da luu vao: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

