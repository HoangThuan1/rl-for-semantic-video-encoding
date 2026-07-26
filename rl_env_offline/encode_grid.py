#!/usr/bin/env python3
"""
encode_grid_roi.py
-------------------
Ban mo rong cua encode_grid.py: thay vi 1 QP dong nhat ca khung hinh, dung
QP KHAC NHAU giua vung ROI (object YOLO phat hien) va nen. Day la grid encode
offline; env action hien tai KHONG con ROI strength/qoffset, chi con
{Target Bitrate ratio, Resolution} = 3x3 = 9 hanh dong.

TAI SAO LAM THEO DOAN (SEGMENT), KHONG PHAI TUNG FRAME:
  ffmpeg filter `addroi` KHONG ho tro toa do dong theo frame (da test thuc te:
  dung bien 'n' -> loi "Undefined constant"; filter cung khong co co "T"
  (timeline/enable=) hay "C" (runtime command) trong `ffmpeg -filters`). Nghia
  la 1 lan goi addroi chi ap duoc 1 vung ROI TINH cho ca doan dang encode.

  De ROI thuc su doi theo frame ma van giu nguyen GOP/du doan lien khung That
  (nhu VCU hardware那 lam qua GstVideoRegionOfInterestMeta cua xlnxroivideo1detect
  trong pipeline GStreamer/VVAS de xuat sau nay), can ghi thang side-data vao
  tung AVFrame qua C API -- qua nang cho 1 script offline, va du sao cung
  khong tai tao dung 100% hardware VCU that.

  Vi day CHI la surrogate offline de train RL (khong phai ban trien khai cuoi),
  ta chap nhan xap xi: chia video thanh cac DOAN ngan (vd 25 frame ~ 1s),
  trong moi doan GOM (union) tat ca bbox object thanh 1 vung ROI dai dien,
  encode rieng doan do bang addroi, roi ghep so lieu tung frame lai. Doi
  tuong di chuyen it trong ~1s nen xap xi nay chap nhan duoc cho muc dich
  huan luyen offline.

CACH DUNG:
  python3 encode_grid_roi.py --input dataset/videos/xxx.mp4 \
      --yolo-metadata outputs/metadata/yolo_metadata.jsonl \
      --out outputs/metadata/vcu_encode_grid_roi.json \
      --baseline-qp 30 --segment-frames 25

KET QUA: file JSON, grid["res{i}_qp{j}"][frame_idx] = {bitrate_kbps, vmaf, psnr}
  -- giu NGUYEN cau truc nhu encode_grid.py de load_encode_grid()/env.py
  dung lai duoc, chi khac o CHO "qp{j}" gio la MUC DO ROI (qoffset), khong
  phai QP tuyet doi -- xem "roi_qoffset_levels" va "baseline_qp" trong file
  JSON de biet ro.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
YOLO_METADATA = "outputs/metadata/yolo_metadata.jsonl"
ENCODE_GRID = "outputs/metadata/vcu_encode_grid_roi.json"

# Muc do "ROI QP" cho grid encode offline, gio la ROI qoffset chu khong
# phai QP tuyet doi nua: 0.0 = khong uu tien vung ROI (giong CBR thuong),
# am cang nhieu = vung ROI duoc nen NHE hon (net hon) so voi nen.
# Day khong phai truc action cua env nua.
ROI_QOFFSET_LEVELS = [0.0, -0.3, -0.6]


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe_video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    width, height = int(info["width"]), int(info["height"])
    nb_frames = info.get("nb_frames")
    if nb_frames in (None, "N/A"):
        out2 = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        nb_frames = int(out2)
    else:
        nb_frames = int(nb_frames)
    return fps, width, height, nb_frames


def load_yolo_metadata(path):
    """Tra ve dict: frame_idx -> {"width":.., "height":.., "boxes": [[x1,y1,x2,y2], ...]}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            boxes = [d["bbox"] for d in row.get("detections", [])]
            out[row["frame_idx"]] = {
                "width": row["width"], "height": row["height"], "boxes": boxes,
            }
    return out


def union_roi_for_segment(yolo_by_frame, start_frame, end_frame, src_w, src_h,
                           target_w, target_h):
    """Gom (union) tat ca bbox trong doan [start_frame, end_frame) thanh 1
    vung hinh chu nhat, roi scale toa do ve do phan giai dich (target_w/h).
    Tra ve None neu doan nay khong co object nao (khong ap addroi)."""
    x1s, y1s, x2s, y2s = [], [], [], []
    for fi in range(start_frame, end_frame):
        meta = yolo_by_frame.get(fi)
        if meta is None:
            continue
        for (bx1, by1, bx2, by2) in meta["boxes"]:
            x1s.append(bx1)
            y1s.append(by1)
            x2s.append(bx2)
            y2s.append(by2)
    if not x1s:
        return None

    x1, y1, x2, y2 = min(x1s), min(y1s), max(x2s), max(y2s)
    sx, sy = target_w / src_w, target_h / src_h
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    x1 = max(0, min(int(round(x1)), target_w - 1))
    y1 = max(0, min(int(round(y1)), target_h - 1))
    w = max(1, min(int(round(x2 - x1)), target_w - x1))
    h = max(1, min(int(round(y2 - y1)), target_h - y1))
    return x1, y1, w, h


def encode_segment(input_path, start_frame, end_frame, width, height, roi_box,
                    qoffset, baseline_crf, fps, workdir, tag):
    """Encode 1 doan [start_frame, end_frame) o do phan giai (width,height).

    QUAN TRONG: dung CRF (rate-control chu dong), KHONG dung constant-QP
    (`-qp`). Da TEST THUC NGHIEM va xac nhan: o che do `-qp` co dinh, addroi
    hoan toan KHONG co tac dung (2 file voi qoffset khac nhau ra dung 1 kich
    thuoc byte-for-byte), vi constant-QP ep MOI macroblock dung chung 1 QP,
    khong con cho cho dieu chinh cuc bo theo vung. CRF giu rate-control chu
    dong nen ROI moi thuc su lam bitrate/chat luong vung do khac vung nen
    (da kiem chung: doi qoffset tu 0 -> -0.9 tren noi dung phuc tap lam
    dung luong file tang ~5 lan trong vung ROI)."""
    out_path = os.path.join(workdir, f"seg_{tag}_{start_frame}_{end_frame}.mp4")
    select_expr = f"between(n\\,{start_frame}\\,{end_frame - 1})"
    vf = f"select='{select_expr}',setpts=N/FRAME_RATE/TB,scale={width}:{height}:flags=bicubic"
    if roi_box is not None:
        x, y, w, h = roi_box
        vf += f",addroi=x={x}:y={y}:w={w}:h={h}:qoffset={qoffset}"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(baseline_crf),
        "-bf", "0", "-g", str(max(1, end_frame - start_frame)),
        "-r", f"{fps:.6f}", "-vsync", "cfr", "-pix_fmt", "yuv420p",
        "-loglevel", "error", out_path,
    ]
    run(cmd)
    return out_path


def extract_reference_segment(input_path, start_frame, end_frame, orig_w, orig_h,
                               fps, workdir, tag):
    """Cat dung doan [start_frame,end_frame) TU VIDEO GOC (khong nen), giu
    nguyen do phan giai goc, dung lam reference tinh PSNR cho doan encode
    tuong ung (dam bao 2 ben cung so frame/thu tu)."""
    out_path = os.path.join(workdir, f"ref_{tag}_{start_frame}_{end_frame}.mp4")
    select_expr = f"between(n\\,{start_frame}\\,{end_frame - 1})"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-c:v", "libx264", "-qp", "0",  # nen lossless -- chi dung lam reference, khong tinh bitrate
        "-r", f"{fps:.6f}", "-vsync", "cfr", "-pix_fmt", "yuv420p",
        "-loglevel", "error", out_path,
    ]
    run(cmd)
    return out_path


def extract_bitrate_per_frame(encoded_path, fps):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pkt_size", "-of", "csv=p=0", encoded_path],
        check=True, capture_output=True, text=True,
    ).stdout
    sizes = [int(line.split(",")[0].strip())
             for line in out.strip().splitlines() if line.strip()]
    return [size * 8 * fps / 1000.0 for size in sizes]


def ffmpeg_filter_path(path):
    return path.replace("\\", "/").replace(":", "\\:")


def extract_vmaf_per_frame(encoded_path, reference_path, orig_w, orig_h, workdir, tag):
    log_path = os.path.join(workdir, f"vmaf_{tag}.json")
    cmd = [
        "ffmpeg", "-y", "-i", encoded_path, "-i", reference_path,
        "-lavfi",
        f"[0:v]scale={orig_w}:{orig_h}:flags=bicubic[enc];"
        f"[enc][1:v]libvmaf=log_fmt=json:log_path={ffmpeg_filter_path(log_path)}",
        "-f", "null", "-", "-loglevel", "error",
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Khong tinh duoc VMAF. Hay dung ffmpeg co filter libvmaf "
            "(kiem tra bang: ffmpeg -filters | findstr libvmaf)."
        ) from exc

    with open(log_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    vmafs = []
    for frame in report.get("frames", []):
        metrics = frame.get("metrics", {})
        if "vmaf" in metrics:
            vmafs.append(float(metrics["vmaf"]))
    return vmafs


def extract_psnr_per_frame(encoded_path, reference_path, orig_w, orig_h, workdir, tag):
    log_path = os.path.join(workdir, f"psnr_{tag}.log")
    cmd = [
        "ffmpeg", "-y", "-i", encoded_path, "-i", reference_path,
        "-lavfi",
        f"[0:v]scale={orig_w}:{orig_h}:flags=bicubic[enc];"
        f"[enc][1:v]psnr=stats_file={ffmpeg_filter_path(log_path)}",
        "-f", "null", "-", "-loglevel", "error",
    ]
    run(cmd)
    psnrs = []
    with open(log_path, "r") as f:
        for line in f:
            parts = dict(tok.split(":", 1) for tok in line.strip().split())
            val = parts.get("psnr_avg", "inf")
            psnrs.append(100.0 if val == "inf" else float(val))
    return psnrs


def build_grid(input_path, yolo_metadata_path, baseline_crf, segment_frames,
                keep_files=False, workdir=None):
    fps, orig_w, orig_h, nb_frames = probe_video(input_path)
    yolo_by_frame = load_yolo_metadata(yolo_metadata_path)
    print(f"[info] input: {input_path} | {orig_w}x{orig_h}@{fps:.3f}fps | "
          f"{nb_frames} frames | segment={segment_frames} frames "
          f"(~{segment_frames/fps:.2f}s)", file=sys.stderr)
    print(f"[info] ROI qoffset levels: {ROI_QOFFSET_LEVELS} | baseline_crf={baseline_crf}",
          file=sys.stderr)

    tmp_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    wd = workdir or tmp_ctx.name
    os.makedirs(wd, exist_ok=True)

    boundaries = list(range(0, nb_frames, segment_frames)) + [nb_frames]

    grid = {}
    try:
        for res_idx, (w, h) in enumerate(RESOLUTIONS):
            for qp_idx, qoffset in enumerate(ROI_QOFFSET_LEVELS):
                key = f"res{res_idx}_qp{qp_idx}"
                print(f"[encode] {key}: {w}x{h} | roi_qoffset={qoffset}", file=sys.stderr)
                rows = []
                for si in range(len(boundaries) - 1):
                    start_f, end_f = boundaries[si], boundaries[si + 1]
                    tag = f"r{res_idx}q{qp_idx}s{si}"

                    roi_box = union_roi_for_segment(
                        yolo_by_frame, start_f, end_f, orig_w, orig_h, w, h
                    )
                    enc_path = encode_segment(
                        input_path, start_f, end_f, w, h, roi_box, qoffset,
                        baseline_crf, fps, wd, tag
                    )
                    ref_path = extract_reference_segment(
                        input_path, start_f, end_f, orig_w, orig_h, fps, wd, tag
                    )
                    bitrates = extract_bitrate_per_frame(enc_path, fps)
                    vmafs = extract_vmaf_per_frame(enc_path, ref_path, orig_w, orig_h, wd, tag)
                    psnrs = extract_psnr_per_frame(enc_path, ref_path, orig_w, orig_h, wd, tag)
                    n = min(len(bitrates), len(vmafs), len(psnrs), end_f - start_f)
                    for i in range(n):
                        rows.append({"bitrate_kbps": round(bitrates[i], 2),
                                     "vmaf": round(vmafs[i], 2),
                                     "psnr": round(psnrs[i], 2)})
                    if not keep_files:
                        for p in (enc_path, ref_path):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                grid[key] = rows
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return {
        "source_video": os.path.abspath(input_path),
        "fps": fps,
        "num_frames": nb_frames,
        "segment_frames": segment_frames,
        "resolutions": RESOLUTIONS,
        "roi_qoffset_levels": ROI_QOFFSET_LEVELS,
        "baseline_crf": baseline_crf,
        "min_qp": baseline_crf,   # giu ten khop voi env.py cu (fallback, khong dung de tra QP tuyet doi nua)
        "max_qp": baseline_crf,
        "qp_levels": ROI_QOFFSET_LEVELS,  # ten cu, gio la qoffset -- xem "roi_qoffset_levels" cho ro nghia
        "grid": grid,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--yolo-metadata", default=YOLO_METADATA,
                     help="File jsonl co truong 'detections'[].bbox theo tung frame_idx")
    ap.add_argument("--out", default=ENCODE_GRID)
    ap.add_argument("--baseline-crf", type=int, default=28,
                     help="CRF nen (background) cho ca luoi -- PHAI dung CRF (rate-control "
                          "chu dong), KHONG dung constant-QP, vi da kiem chung constant-QP "
                          "lam addroi/ROI mat tac dung hoan toan. bitrate_ratio trong action "
                          "van dieu chinh tong bitrate qua VBV cap sau khi tra bang.")
    ap.add_argument("--segment-frames", type=int, default=25,
                     help="So frame moi doan (ROI tinh trong 1 doan). 25 ~ 1s o 25fps")
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    input_path = args.input
    yolo_metadata_path = args.yolo_metadata
    out_path = args.out

    result = build_grid(input_path, yolo_metadata_path, args.baseline_crf,
                         args.segment_frames, keep_files=args.keep_files,
                         workdir=args.workdir)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[done] Da luu luoi encode ROI vao: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

