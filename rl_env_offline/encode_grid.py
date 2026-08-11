#!/usr/bin/env python3
"""
encode_grid_roi.py
-------------------
Ban mo rong cua encode_grid.py: do rate-distortion that cho tung segment.
Moi diem grid la:
  segment x resolution x bitrate_level x ROI_level -> actual bitrate, VMAF.
Env action la {Target Bitrate ratio, Resolution, ROI level}; khi tinh reward,
env noi suy tren duong RD cua dung frame/resolution/ROI level.

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
  # Chay tu thu muc cnn_training/rl_env_offline.
  python3 encode_grid.py --input ../dataset/videos/xxx.mp4 \
      --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
      --out ../outputs/metadata/vcu_encode_grid.json \
      --bitrate-levels 600,900,1500,2500,4000,6000 \
      --segment-frames 25 --workers 4 --metrics vmaf

  # Smoke test toc do/pipeline, KHONG nen dung de train reward VMAF:
  python3 encode_grid.py --input ../dataset/videos/xxx.mp4 \
      --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
      --out /tmp/vcu_encode_grid_smoke.json \
      --segment-frames 50 --workers 4 --metrics none --preset ultrafast

KET QUA: file JSON,
  grid["res{i}_br{j}_roi{k}"][frame_idx] =
      {bitrate_kbps, target_bitrate_kbps, vmaf, roi_vmaf}
  Neu chay --metrics vmaf,psnr thi moi row co them field psnr.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
BITRATE_RATIOS = [0.50, 0.75, 0.95]
RESOLUTION_ACTIONS = [0, 1, 2]
YOLO_METADATA = "../outputs/metadata/yolo_metadata.jsonl"
ENCODE_GRID = "../outputs/metadata/vcu_encode_grid.json"
DEFAULT_TRACE_PATH = "../outputs/metadata/rl_states.jsonl"

# Muc do "ROI QP" cho grid encode offline, gio la ROI qoffset chu khong
# phai QP tuyet doi nua: 0.0 = khong uu tien vung ROI (giong CBR thuong),
# am cang nhieu = vung ROI duoc nen NHE hon (net hon) so voi nen.
# Cac index trong list nay la truc roi_idx cua action trong env.py.
ROI_QOFFSET_LEVELS = [0.0, -0.3, -0.6]
ROI_ACTIONS = list(range(len(ROI_QOFFSET_LEVELS)))
ACTIONS = [
    (bitrate_ratio, resolution_idx, roi_idx)
    for bitrate_ratio in BITRATE_RATIOS
    for resolution_idx in RESOLUTION_ACTIONS
    for roi_idx in ROI_ACTIONS
]


def run(cmd, verbose=True):
    if verbose:
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


def load_trace(path):
    trace = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(json.loads(line))
    if not trace:
        raise ValueError(f"Trace rong: {path}")
    return trace


def build_segment_bandwidth(trace, num_frames, segment_frames, default_bandwidth):
    frame_bandwidth = [float(default_bandwidth)] * int(num_frames)
    for i, row in enumerate(trace):
        frame_idx = int(row.get("frame_idx", i))
        if 0 <= frame_idx < num_frames:
            frame_bandwidth[frame_idx] = float(row.get("bandwidth", default_bandwidth))

    segment_bandwidths = []
    for start_f in range(0, num_frames, segment_frames):
        end_f = min(start_f + segment_frames, num_frames)
        chunk = frame_bandwidth[start_f:end_f]
        segment_bandwidths.append(float(sum(chunk) / max(1, len(chunk))))
    return segment_bandwidths


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


def frame_time(frame_idx, fps):
    return f"{frame_idx / fps:.6f}"


def encode_segment(input_path, start_frame, end_frame, width, height, roi_box,
                    qoffset, baseline_crf, fps, workdir, tag, preset="veryfast",
                    seek_mode="fast", verbose=True, target_bitrate_kbps=None):
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
    frame_count = max(1, end_frame - start_frame)
    vf = f"scale={width}:{height}:flags=bicubic"
    if roi_box is not None:
        x, y, w, h = roi_box
        vf += f",addroi=x={x}:y={y}:w={w}:h={h}:qoffset={qoffset}"

    cmd = ["ffmpeg", "-y"]
    if seek_mode == "fast":
        cmd += ["-ss", frame_time(start_frame, fps)]
    cmd += ["-i", input_path]
    if seek_mode == "accurate":
        cmd += ["-ss", frame_time(start_frame, fps)]
    cmd += [
        "-vf", vf,
        "-frames:v", str(frame_count),
        "-an",
        "-c:v", "libx264", "-preset", preset,
    ]
    if target_bitrate_kbps is None:
        cmd += ["-crf", str(baseline_crf)]
    else:
        br = max(1, int(round(float(target_bitrate_kbps))))
        cmd += ["-b:v", f"{br}k", "-maxrate", f"{br}k", "-bufsize", f"{max(2 * br, 1)}k"]
    cmd += [
        "-bf", "0", "-g", str(frame_count),
        "-r", f"{fps:.6f}", "-vsync", "cfr", "-pix_fmt", "yuv420p",
        "-loglevel", "error", out_path,
    ]
    run(cmd, verbose=verbose)
    return out_path


def extract_reference_segment(input_path, start_frame, end_frame, orig_w, orig_h,
                               fps, workdir, tag, preset="ultrafast",
                               seek_mode="fast", verbose=True):
    """Cat dung doan [start_frame,end_frame) TU VIDEO GOC (khong nen), giu
    nguyen do phan giai goc, dung lam reference tinh PSNR cho doan encode
    tuong ung (dam bao 2 ben cung so frame/thu tu)."""
    out_path = os.path.join(workdir, f"ref_{tag}_{start_frame}_{end_frame}.mp4")
    frame_count = max(1, end_frame - start_frame)
    cmd = ["ffmpeg", "-y"]
    if seek_mode == "fast":
        cmd += ["-ss", frame_time(start_frame, fps)]
    cmd += ["-i", input_path]
    if seek_mode == "accurate":
        cmd += ["-ss", frame_time(start_frame, fps)]
    cmd += [
        "-frames:v", str(frame_count),
        "-an",
        "-c:v", "libx264", "-preset", preset, "-qp", "0",  # nen lossless -- chi dung lam reference, khong tinh bitrate
        "-r", f"{fps:.6f}", "-vsync", "cfr", "-pix_fmt", "yuv420p",
        "-loglevel", "error", out_path,
    ]
    run(cmd, verbose=verbose)
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


def extract_vmaf_per_frame(encoded_path, reference_path, orig_w, orig_h, workdir,
                           tag, verbose=True):
    log_path = os.path.join(workdir, f"vmaf_{tag}.json")
    cmd = [
        "ffmpeg", "-y", "-i", encoded_path, "-i", reference_path,
        "-lavfi",
        f"[0:v]scale={orig_w}:{orig_h}:flags=bicubic[enc];"
        f"[enc][1:v]libvmaf=log_fmt=json:log_path={ffmpeg_filter_path(log_path)}",
        "-f", "null", "-", "-loglevel", "error",
    ]
    try:
        run(cmd, verbose=verbose)
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


def extract_roi_vmaf_per_frame(encoded_path, reference_path, roi_box,
                               target_w, target_h, workdir, tag, verbose=True):
    """Tinh VMAF rieng tren union ROI da dung cho addroi.

    Reference duoc scale ve cung resolution encode truoc khi crop. Crop duoc
    can ve toa do/kich thuoc chan de hop voi pixel format yuv420p.
    """
    if roi_box is None:
        return None

    x, y, w, h = roi_box
    x = max(0, min((int(x) // 2) * 2, max(target_w - 2, 0)))
    y = max(0, min((int(y) // 2) * 2, max(target_h - 2, 0)))
    w = max(2, min((int(w) // 2) * 2, target_w - x))
    h = max(2, min((int(h) // 2) * 2, target_h - y))

    log_path = os.path.join(workdir, f"roi_vmaf_{tag}.json")
    crop = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        "ffmpeg", "-y", "-i", encoded_path, "-i", reference_path,
        "-lavfi",
        f"[0:v]{crop}[enc_roi];"
        f"[1:v]scale={target_w}:{target_h}:flags=bicubic,{crop}[ref_roi];"
        f"[enc_roi][ref_roi]libvmaf=log_fmt=json:"
        f"log_path={ffmpeg_filter_path(log_path)}",
        "-f", "null", "-", "-loglevel", "error",
    ]
    try:
        run(cmd, verbose=verbose)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Khong tinh duoc ROI VMAF. Kiem tra libvmaf va ROI crop."
        ) from exc

    with open(log_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return [
        float(frame["metrics"]["vmaf"])
        for frame in report.get("frames", [])
        if "vmaf" in frame.get("metrics", {})
    ]


def extract_psnr_per_frame(encoded_path, reference_path, orig_w, orig_h, workdir,
                           tag, verbose=True):
    log_path = os.path.join(workdir, f"psnr_{tag}.log")
    cmd = [
        "ffmpeg", "-y", "-i", encoded_path, "-i", reference_path,
        "-lavfi",
        f"[0:v]scale={orig_w}:{orig_h}:flags=bicubic[enc];"
        f"[enc][1:v]psnr=stats_file={ffmpeg_filter_path(log_path)}",
        "-f", "null", "-", "-loglevel", "error",
    ]
    run(cmd, verbose=verbose)
    psnrs = []
    with open(log_path, "r") as f:
        for line in f:
            parts = dict(tok.split(":", 1) for tok in line.strip().split())
            val = parts.get("psnr_avg", "inf")
            psnrs.append(100.0 if val == "inf" else float(val))
    return psnrs


def _build_ref_worker(args):
    (
        input_path,
        start_f,
        end_f,
        orig_w,
        orig_h,
        fps,
        workdir,
        si,
        preset,
        seek_mode,
        verbose,
    ) = args
    ref_path = extract_reference_segment(
        input_path, start_f, end_f, orig_w, orig_h, fps, workdir,
        f"ref_s{si}", preset=preset, seek_mode=seek_mode, verbose=verbose
    )
    return si, ref_path


def _grid_segment_worker(args):
    (
        input_path,
        ref_path,
        start_f,
        end_f,
        fps,
        orig_w,
        orig_h,
        width,
        height,
        roi_box,
        qoffset,
        target_bitrate_kbps,
        baseline_crf,
        workdir,
        action_id,
        si,
        res_idx,
        roi_idx,
        metrics,
        preset,
        seek_mode,
        verbose,
        keep_files,
    ) = args

    tag = f"s{si}a{action_id}r{res_idx}roi{roi_idx}"
    enc_path = encode_segment(
        input_path, start_f, end_f, width, height, roi_box, qoffset,
        baseline_crf, fps, workdir, tag, preset=preset,
        seek_mode=seek_mode, verbose=verbose,
        target_bitrate_kbps=target_bitrate_kbps
    )
    try:
        bitrates = extract_bitrate_per_frame(enc_path, fps)

        if "vmaf" in metrics:
            vmafs = extract_vmaf_per_frame(
                enc_path, ref_path, orig_w, orig_h, workdir, tag, verbose=verbose
            )
            roi_vmafs = extract_roi_vmaf_per_frame(
                enc_path, ref_path, roi_box, width, height, workdir, tag,
                verbose=verbose,
            )
            if roi_vmafs is None:
                roi_vmafs = list(vmafs)
        else:
            vmafs = [0.0] * len(bitrates)
            roi_vmafs = [0.0] * len(bitrates)

        psnrs = None
        if "psnr" in metrics:
            psnrs = extract_psnr_per_frame(
                enc_path, ref_path, orig_w, orig_h, workdir, tag, verbose=verbose
            )

        n = min(len(bitrates), len(vmafs), len(roi_vmafs), end_f - start_f)
        if psnrs is not None:
            n = min(n, len(psnrs))

        if n <= 0:
            raise RuntimeError(
                f"Khong thu duoc metric cho segment={si}, action_id={action_id}."
            )

        outcome = {
            "actual_bitrate_kbps": round(float(sum(bitrates[:n]) / n), 2),
            "target_bitrate_kbps": round(float(target_bitrate_kbps), 2),
            "vmaf": round(float(sum(vmafs[:n]) / n), 2),
            "roi_vmaf": round(float(sum(roi_vmafs[:n]) / n), 2),
            "num_frames": int(n),
        }
        if psnrs is not None:
            outcome["psnr"] = round(float(sum(psnrs[:n]) / n), 2)
        return si, action_id, outcome
    finally:
        if not keep_files:
            try:
                os.remove(enc_path)
            except OSError:
                pass


def build_grid(input_path, yolo_metadata_path, trace_path, baseline_crf, segment_frames,
               max_bitrate_kbps=8000.0, keep_files=False, workdir=None, workers=1,
               metrics=("vmaf", "psnr"), preset="veryfast", ref_preset="ultrafast",
               seek_mode="fast", verbose=False):
    fps, orig_w, orig_h, nb_frames = probe_video(input_path)
    trace = load_trace(trace_path)
    yolo_by_frame = load_yolo_metadata(yolo_metadata_path)
    if segment_frames != 5:
        raise ValueError("Grid mapping moi yeu cau segment_frames=5.")
    print(f"[info] input: {input_path} | {orig_w}x{orig_h}@{fps:.3f}fps | "
          f"{nb_frames} frames | segment={segment_frames} frames "
          f"(~{segment_frames/fps:.2f}s)", file=sys.stderr)
    print(f"[info] ROI qoffset levels: {ROI_QOFFSET_LEVELS} | baseline_crf={baseline_crf}",
          file=sys.stderr)
    print(f"[info] so RL actions/segment: {len(ACTIONS)}", file=sys.stderr)
    print(f"[info] workers={workers} | metrics={','.join(metrics) or 'none'} | "
          f"preset={preset} | seek_mode={seek_mode}", file=sys.stderr)

    tmp_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    wd = workdir or tmp_ctx.name
    os.makedirs(wd, exist_ok=True)

    boundaries = list(range(0, nb_frames, segment_frames)) + [nb_frames]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    segment_bandwidths = build_segment_bandwidth(
        trace, nb_frames, segment_frames, default_bandwidth=max_bitrate_kbps * 0.5
    )
    roi_cache = {}
    for res_idx, (w, h) in enumerate(RESOLUTIONS):
        for si, (start_f, end_f) in enumerate(segments):
            roi_cache[(res_idx, si)] = union_roi_for_segment(
                yolo_by_frame, start_f, end_f, orig_w, orig_h, w, h
            )

    segment_action_outcomes = [
        [None for _ in range(len(ACTIONS))]
        for _ in range(len(segments))
    ]
    ref_paths = {}
    try:
        if metrics:
            print(f"[ref] tao {len(segments)} reference segments mot lan de dung lai",
                  file=sys.stderr)
            ref_tasks = [
                (
                    input_path, start_f, end_f, orig_w, orig_h, fps, wd, si,
                    ref_preset, seek_mode, verbose,
                )
                for si, (start_f, end_f) in enumerate(segments)
            ]
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_build_ref_worker, task) for task in ref_tasks]
                    for done, fut in enumerate(as_completed(futures), 1):
                        si, ref_path = fut.result()
                        ref_paths[si] = ref_path
                        if done % 25 == 0 or done == len(futures):
                            print(f"[ref] {done}/{len(futures)}", file=sys.stderr)
            else:
                for done, task in enumerate(ref_tasks, 1):
                    si, ref_path = _build_ref_worker(task)
                    ref_paths[si] = ref_path
                    if done % 25 == 0 or done == len(ref_tasks):
                        print(f"[ref] {done}/{len(ref_tasks)}", file=sys.stderr)

        tasks = []
        copy_from_base = []
        for si, (start_f, end_f) in enumerate(segments):
            segment_bw = float(np.clip(segment_bandwidths[si], 50.0, max_bitrate_kbps))
            for action_id, (bitrate_ratio, res_idx, roi_idx) in enumerate(ACTIONS):
                w, h = RESOLUTIONS[res_idx]
                roi_box = roi_cache[(res_idx, si)]
                qoffset = ROI_QOFFSET_LEVELS[roi_idx]
                target_bitrate_kbps = float(np.clip(bitrate_ratio * segment_bw, 50.0, max_bitrate_kbps))
                if roi_box is None and roi_idx > 0:
                    base_action_id = next(
                        idx for idx, (br, ri, rroi) in enumerate(ACTIONS)
                        if br == bitrate_ratio and ri == res_idx and rroi == 0
                    )
                    copy_from_base.append((si, action_id, base_action_id))
                    continue
                tasks.append(
                    (
                        input_path,
                        ref_paths.get(si),
                        start_f,
                        end_f,
                        fps,
                        orig_w,
                        orig_h,
                        w,
                        h,
                        roi_box,
                        qoffset,
                        target_bitrate_kbps,
                        baseline_crf,
                        wd,
                        action_id,
                        si,
                        res_idx,
                        roi_idx,
                        metrics,
                        preset,
                        seek_mode,
                        verbose,
                        keep_files,
                    )
                )

        print(f"[encode] {len(tasks)} jobs encode/metric "
              f"(bo qua {len(copy_from_base)} jobs trung lap do segment khong co ROI)",
              file=sys.stderr)
        segment_rows = {}
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_grid_segment_worker, task) for task in tasks]
                for done, fut in enumerate(as_completed(futures), 1):
                    si, action_id, outcome = fut.result()
                    segment_rows[(si, action_id)] = outcome
                    if done % 25 == 0 or done == len(futures):
                        print(f"[encode] {done}/{len(futures)}", file=sys.stderr)
        else:
            for done, task in enumerate(tasks, 1):
                si, action_id, outcome = _grid_segment_worker(task)
                segment_rows[(si, action_id)] = outcome
                if done % 25 == 0 or done == len(tasks):
                    print(f"[encode] {done}/{len(tasks)}", file=sys.stderr)

        for si, action_id, base_action_id in copy_from_base:
            segment_rows[(si, action_id)] = dict(segment_rows[(si, base_action_id)])

        for si in range(len(segments)):
            for action_id in range(len(ACTIONS)):
                outcome = segment_rows.get((si, action_id))
                if outcome is None:
                    raise KeyError(
                        f"Thieu outcome segment={si}, action_id={action_id} trong grid."
                    )
                segment_action_outcomes[si][action_id] = outcome
    finally:
        if metrics and not keep_files:
            for ref_path in ref_paths.values():
                try:
                    os.remove(ref_path)
                except OSError:
                    pass
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return {
        "schema_version": 2,
        "source_video": os.path.abspath(input_path),
        "trace_path": os.path.abspath(trace_path),
        "fps": fps,
        "num_frames": nb_frames,
        "segment_frames": segment_frames,
        "num_segments": len(segments),
        "resolutions": RESOLUTIONS,
        "bitrate_ratios": BITRATE_RATIOS,
        "roi_qoffset_levels": ROI_QOFFSET_LEVELS,
        "baseline_crf": baseline_crf,
        "metrics": list(metrics),
        "encode_preset": preset,
        "seek_mode": seek_mode,
        "max_bitrate_kbps": float(max_bitrate_kbps),
        "actions": [
            {
                "action_id": action_id,
                "bitrate_ratio": bitrate_ratio,
                "resolution_idx": resolution_idx,
                "roi_idx": roi_idx,
            }
            for action_id, (bitrate_ratio, resolution_idx, roi_idx) in enumerate(ACTIONS)
        ],
        "segments": [
            {
                "segment_id": si,
                "start_frame": start_f,
                "end_frame": end_f,
                "bandwidth_kbps": round(float(segment_bandwidths[si]), 2),
            }
            for si, (start_f, end_f) in enumerate(segments)
        ],
        "segment_action_outcomes": segment_action_outcomes,
    }


def parse_metrics(value):
    value = value.strip().lower()
    if value in {"none", "off", "no"}:
        return ()
    metrics = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"vmaf", "psnr"}
    unknown = sorted(set(metrics) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"metrics khong hop le: {','.join(unknown)}. Chi ho tro: vmaf,psnr,none"
        )
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--yolo-metadata", default=YOLO_METADATA,
                     help="File jsonl co truong 'detections'[].bbox theo tung frame_idx")
    ap.add_argument("--trace-path", default=DEFAULT_TRACE_PATH,
                     help="RL trace jsonl co frame_idx/bandwidth de tinh target bitrate theo segment.")
    ap.add_argument("--out", default=ENCODE_GRID)
    ap.add_argument("--baseline-crf", type=int, default=28,
                     help="CRF nen (background) cho ca luoi -- PHAI dung CRF (rate-control "
                          "chu dong), KHONG dung constant-QP, vi da kiem chung constant-QP "
                          "lam addroi/ROI mat tac dung hoan toan. bitrate_ratio trong action "
                          "van dieu chinh tong bitrate qua VBV cap sau khi tra bang.")
    ap.add_argument("--segment-frames", type=int, default=5,
                     help="So frame moi doan. Mapping moi bat buoc dung 5 frames/segment.")
    ap.add_argument("--max-bitrate-kbps", type=float, default=8000.0,
                     help="Gioi han bitrate toi da de tinh target bitrate cho action.")
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                     help="So job ffmpeg chay song song. Goi y: 2-4 tren may laptop, "
                          "cao hon neu CPU/SSD manh.")
    ap.add_argument("--metrics", type=parse_metrics, default=("vmaf", "psnr"),
                     help="Metric can tinh: mac dinh 'vmaf,psnr'.")
    ap.add_argument("--preset", default="veryfast",
                     help="libx264 preset cho encode grid. Doi sang ultrafast neu can nhanh hon.")
    ap.add_argument("--ref-preset", default="ultrafast",
                     help="libx264 preset cho reference lossless segment.")
    ap.add_argument("--seek-mode", choices=("fast", "accurate"), default="fast",
                     help="'fast' dung -ss truoc input de xu ly video dai nhanh hon; "
                          "'accurate' cham hon nhung bam frame chinh xac hon.")
    ap.add_argument("--verbose-ffmpeg", action="store_true",
                     help="In tung lenh ffmpeg. Mac dinh tat de log gon khi chay video dai.")
    args = ap.parse_args()

    input_path = args.input
    yolo_metadata_path = args.yolo_metadata
    trace_path = args.trace_path
    out_path = args.out
    workdir = args.workdir

    result = build_grid(input_path, yolo_metadata_path, trace_path, args.baseline_crf,
                         args.segment_frames, keep_files=args.keep_files,
                         workdir=workdir, workers=max(1, args.workers),
                         metrics=args.metrics, preset=args.preset,
                         ref_preset=args.ref_preset, seek_mode=args.seek_mode,
                         verbose=args.verbose_ffmpeg,
                         max_bitrate_kbps=args.max_bitrate_kbps)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[done] Da luu luoi encode ROI vao: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
