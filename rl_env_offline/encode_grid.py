#!/usr/bin/env python3
"""
encode_grid_roi.py
-------------------
Ban mo rong cua encode_grid.py: do rate-distortion that cho tung frame.
Moi diem grid la:
  frame x resolution x bitrate_level x ROI_level -> actual bitrate, VMAF.
Env action la {Target Bitrate ratio, Resolution, ROI level}; khi tinh reward,
env noi suy tren duong RD cua dung frame/resolution/ROI level.

Moi job chi encode mot frame, vi vay danh sach ROI cua frame do duoc gan dung
cho chinh frame do. Moi bbox YOLO tao mot filter `addroi` rieng; khong gom cac
bbox thanh mot hinh chu nhat lon lam mat ranh gioi object.

CACH DUNG:
  # Chay tu thu muc cnn_training/rl_env_offline.
  python3 encode_grid.py --input ../dataset/videos/xxx.mp4 \
      --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
      --out ../outputs/metadata/vcu_encode_grid.json \
      --bitrate-levels 600,900,1500,2500,4000,6000 \
      --workers 4 --metrics vmaf

  # Smoke test toc do/pipeline, KHONG nen dung de train reward VMAF:
  python3 encode_grid.py --input ../dataset/videos/xxx.mp4 \
      --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
      --out /tmp/vcu_encode_grid_smoke.json \
      --workers 4 --metrics none --preset ultrafast

KET QUA: file JSON,
  grid["res{i}_br{j}_roi{k}"][frame_idx] =
      {bitrate_kbps, target_bitrate_kbps, vmaf, roi_vmaf}
  Neu chay --metrics vmaf,psnr thi moi row co them field psnr.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import subprocess
import sys
import tempfile

RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
YOLO_METADATA = "../outputs/metadata/yolo_metadata.jsonl"
ENCODE_GRID = "../outputs/metadata/vcu_encode_grid.json"
BITRATE_LEVELS_KBPS = [600.0, 900.0, 1500.0, 2500.0, 4000.0, 6000.0]

# Muc do "ROI QP" cho grid encode offline, gio la ROI qoffset chu khong
# phai QP tuyet doi nua: 0.0 = khong uu tien vung ROI (giong CBR thuong),
# am cang nhieu = vung ROI duoc nen NHE hon (net hon) so voi nen.
# Cac index trong list nay la truc roi_idx cua action trong env.py.
ROI_QOFFSET_LEVELS = [0.0, -0.3, -0.6]


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
            if "frame_idx" not in row or "width" not in row or "height" not in row:
                raise ValueError("YOLO metadata thieu frame_idx/width/height")
            detections = row.get("detections")
            if not isinstance(detections, list):
                raise ValueError(
                    f"YOLO metadata frame {row['frame_idx']} thieu detections list"
                )
            boxes = []
            for detection in detections:
                bbox = detection.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(
                        f"YOLO metadata frame {row['frame_idx']} co bbox khong hop le"
                    )
                try:
                    values = [float(value) for value in bbox]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"YOLO metadata frame {row['frame_idx']} co bbox khong phai so"
                    ) from exc
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        f"YOLO metadata frame {row['frame_idx']} co bbox khong finite"
                    )
                if values[2] <= values[0] or values[3] <= values[1]:
                    raise ValueError(
                        f"YOLO metadata frame {row['frame_idx']} co bbox rong/am"
                    )
                boxes.append(values)
            out[row["frame_idx"]] = {
                "width": row["width"], "height": row["height"], "boxes": boxes,
            }
    return out


def roi_boxes_for_frames(yolo_by_frame, start_frame, end_frame, src_w, src_h,
                         target_w, target_h):
    """Scale every YOLO bbox in ``[start_frame, end_frame)`` independently.

    The returned rectangles are even-aligned for yuv420p. Repeated rectangles
    are removed while preserving metadata order, which is also FFmpeg ROI
    priority order when rectangles overlap.
    """
    boxes = []
    seen = set()
    sx, sy = target_w / src_w, target_h / src_h
    for fi in range(start_frame, end_frame):
        meta = yolo_by_frame.get(fi)
        if meta is None:
            continue
        for (bx1, by1, bx2, by2) in meta["boxes"]:
            left = max(0, min(int(float(bx1) * sx), target_w - 2))
            top = max(0, min(int(float(by1) * sy), target_h - 2))
            right = max(left + 2, min(int(round(float(bx2) * sx)), target_w))
            bottom = max(top + 2, min(int(round(float(by2) * sy)), target_h))
            left = (left // 2) * 2
            top = (top // 2) * 2
            right = min(target_w, ((right + 1) // 2) * 2)
            bottom = min(target_h, ((bottom + 1) // 2) * 2)
            box = (left, top, right - left, bottom - top)
            if box not in seen:
                seen.add(box)
                boxes.append(box)
    return boxes


def addroi_filter_chain(roi_boxes, qoffset):
    """Return one addroi filter per ROI; the first ROI wins on overlap."""
    return "".join(
        f",addroi=x={x}:y={y}:w={w}:h={h}:qoffset={qoffset}"
        for x, y, w, h in roi_boxes
    )


def frame_time(frame_idx, fps):
    return f"{frame_idx / fps:.6f}"


def encode_frame(input_path, frame_idx, width, height, roi_boxes,
                 qoffset, fps, workdir, tag, target_bitrate_kbps,
                 preset="veryfast", seek_mode="fast", verbose=True):
    """Encode exactly one frame at ``(width, height)``.

    QUAN TRONG: dung ABR/VBV (rate-control chu dong), KHONG dung constant-QP
    (`-qp`). Da TEST THUC NGHIEM va xac nhan: o che do `-qp` co dinh, addroi
    hoan toan KHONG co tac dung (2 file voi qoffset khac nhau ra dung 1 kich
    thuoc byte-for-byte), vi constant-QP ep MOI macroblock dung chung 1 QP,
    khong con cho cho dieu chinh cuc bo theo vung. ABR giu rate-control chu
    dong nen ROI moi thuc su lam bitrate/chat luong vung do khac vung nen
    (da kiem chung: doi qoffset tu 0 -> -0.9 tren noi dung phuc tap lam
    dung luong file tang ~5 lan trong vung ROI)."""
    out_path = os.path.join(workdir, f"frame_{tag}_{frame_idx}.mp4")
    vf = f"scale={width}:{height}:flags=bicubic"
    if roi_boxes and abs(qoffset) > 1e-9:
        vf += addroi_filter_chain(roi_boxes, qoffset)

    cmd = ["ffmpeg", "-y"]
    if seek_mode == "fast":
        cmd += ["-ss", frame_time(frame_idx, fps)]
    cmd += ["-i", input_path]
    if seek_mode == "accurate":
        cmd += ["-ss", frame_time(frame_idx, fps)]
    cmd += [
        "-vf", vf,
        "-frames:v", "1",
        "-an",
        "-c:v", "libx264", "-preset", preset,
    ]
    br = max(1, int(round(float(target_bitrate_kbps))))
    cmd += ["-b:v", f"{br}k", "-maxrate", f"{br}k", "-bufsize", f"{max(2 * br, 1)}k"]
    cmd += [
        "-bf", "0", "-g", "1",
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


def _extract_single_roi_vmaf(encoded_path, reference_path, roi_box,
                               target_w, target_h, workdir, tag, verbose=True):
    """Return per-frame VMAF for one already aligned ROI rectangle."""
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


def extract_roi_vmaf_per_frame(encoded_path, reference_path, roi_boxes,
                               target_w, target_h, workdir, tag, verbose=True):
    """Measure each ROI independently and return its area-weighted VMAF.

    This keeps the quality metric on detected objects instead of including the
    background between objects as the old enclosing-union crop did.
    """
    if not roi_boxes:
        return None

    weighted = None
    total_area = 0.0
    for roi_index, roi_box in enumerate(roi_boxes):
        values = _extract_single_roi_vmaf(
            encoded_path, reference_path, roi_box, target_w, target_h,
            workdir, f"{tag}_roi{roi_index}", verbose=verbose,
        )
        area = float(roi_box[2] * roi_box[3])
        if weighted is None:
            weighted = [0.0] * len(values)
        if len(values) != len(weighted):
            raise RuntimeError("So frame ROI VMAF khong dong nhat")
        for i, value in enumerate(values):
            weighted[i] += value * area
        total_area += area
    return [value / total_area for value in weighted]


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
        frame_idx,
        fps,
        orig_w,
        orig_h,
        width,
        height,
        roi_boxes,
        qoffset,
        target_bitrate_kbps,
        workdir,
        key,
        res_idx,
        br_idx,
        roi_idx,
        metrics,
        preset,
        seek_mode,
        verbose,
        keep_files,
    ) = args

    tag = f"r{res_idx}b{br_idx}roi{roi_idx}f{frame_idx}"
    enc_path = encode_frame(
        input_path, frame_idx, width, height, roi_boxes, qoffset,
        fps, workdir, tag, target_bitrate_kbps, preset=preset,
        seek_mode=seek_mode, verbose=verbose,
    )
    try:
        bitrates = extract_bitrate_per_frame(enc_path, fps)

        if "vmaf" in metrics:
            vmafs = extract_vmaf_per_frame(
                enc_path, ref_path, orig_w, orig_h, workdir, tag, verbose=verbose
            )
            roi_vmafs = extract_roi_vmaf_per_frame(
                enc_path, ref_path, roi_boxes, width, height, workdir, tag,
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

        lengths = [len(bitrates), len(vmafs), len(roi_vmafs)]
        if psnrs is not None:
            lengths.append(len(psnrs))
        if any(length != 1 for length in lengths):
            raise RuntimeError(
                f"Frame {frame_idx}: mong doi dung 1 sample metric, nhan {lengths}"
            )

        rows = []
        for i in range(1):
            row = {
                "bitrate_kbps": round(bitrates[i], 2),
                "target_bitrate_kbps": round(float(target_bitrate_kbps), 2),
                "vmaf": round(vmafs[i], 2),
                "roi_vmaf": round(roi_vmafs[i], 2),
            }
            if psnrs is not None:
                row["psnr"] = round(psnrs[i], 2)
            rows.append(row)
        return key, frame_idx, rows[0]
    finally:
        if not keep_files:
            try:
                os.remove(enc_path)
            except OSError:
                pass


def build_grid(input_path, yolo_metadata_path, keep_files=False, workdir=None,
                workers=1, metrics=("vmaf",),
                preset="veryfast", ref_preset="ultrafast", seek_mode="fast",
                verbose=False, bitrate_levels_kbps=None):
    bitrate_levels_kbps = list(
        BITRATE_LEVELS_KBPS if bitrate_levels_kbps is None else bitrate_levels_kbps
    )
    if not bitrate_levels_kbps:
        raise ValueError("bitrate_levels_kbps khong duoc rong")
    fps, orig_w, orig_h, nb_frames = probe_video(input_path)
    yolo_by_frame = load_yolo_metadata(yolo_metadata_path)
    missing_frames = sorted(set(range(nb_frames)) - set(yolo_by_frame))
    if missing_frames:
        raise ValueError(
            "YOLO metadata thieu frame: "
            + ",".join(str(value) for value in missing_frames[:10])
        )
    for frame_idx in range(nb_frames):
        metadata = yolo_by_frame[frame_idx]
        if (int(metadata["width"]), int(metadata["height"])) != (orig_w, orig_h):
            raise ValueError(
                f"YOLO metadata frame {frame_idx} co kich thuoc "
                f"{metadata['width']}x{metadata['height']}, video la {orig_w}x{orig_h}"
            )
    print(f"[info] input: {input_path} | {orig_w}x{orig_h}@{fps:.3f}fps | "
          f"{nb_frames} frame-level jobs", file=sys.stderr)
    print(f"[info] ROI qoffset levels: {ROI_QOFFSET_LEVELS}", file=sys.stderr)
    print(f"[info] bitrate levels kbps: {bitrate_levels_kbps}", file=sys.stderr)
    print(f"[info] workers={workers} | metrics={','.join(metrics) or 'none'} | "
          f"preset={preset} | seek_mode={seek_mode}", file=sys.stderr)

    tmp_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    wd = workdir or tmp_ctx.name
    os.makedirs(wd, exist_ok=True)

    roi_cache = {}
    for res_idx, (w, h) in enumerate(RESOLUTIONS):
        for frame_idx in range(nb_frames):
            roi_cache[(res_idx, frame_idx)] = roi_boxes_for_frames(
                yolo_by_frame, frame_idx, frame_idx + 1,
                orig_w, orig_h, w, h
            )

    grid = {}
    ref_paths = {}
    try:
        if metrics:
            print(f"[ref] tao {nb_frames} reference frames mot lan de dung lai",
                  file=sys.stderr)
            ref_tasks = [
                (input_path, frame_idx, frame_idx + 1, orig_w, orig_h, fps,
                 wd, frame_idx, ref_preset, seek_mode, verbose)
                for frame_idx in range(nb_frames)
            ]
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_build_ref_worker, task) for task in ref_tasks]
                    for done, fut in enumerate(as_completed(futures), 1):
                        frame_idx, ref_path = fut.result()
                        ref_paths[frame_idx] = ref_path
                        if done % 25 == 0 or done == len(futures):
                            print(f"[ref] {done}/{len(futures)}", file=sys.stderr)
            else:
                for done, task in enumerate(ref_tasks, 1):
                    frame_idx, ref_path = _build_ref_worker(task)
                    ref_paths[frame_idx] = ref_path
                    if done % 25 == 0 or done == len(ref_tasks):
                        print(f"[ref] {done}/{len(ref_tasks)}", file=sys.stderr)

        for res_idx, _ in enumerate(RESOLUTIONS):
            for br_idx, _ in enumerate(bitrate_levels_kbps):
                for roi_idx, _ in enumerate(ROI_QOFFSET_LEVELS):
                    grid[f"res{res_idx}_br{br_idx}_roi{roi_idx}"] = []

        tasks = []
        for res_idx, (w, h) in enumerate(RESOLUTIONS):
            for frame_idx in range(nb_frames):
                roi_boxes = roi_cache[(res_idx, frame_idx)]
                for br_idx, target_bitrate_kbps in enumerate(bitrate_levels_kbps):
                    for roi_idx, qoffset in enumerate(ROI_QOFFSET_LEVELS):
                        key = f"res{res_idx}_br{br_idx}_roi{roi_idx}"
                        tasks.append(
                            (
                                input_path,
                                ref_paths[frame_idx] if metrics else None,
                                frame_idx,
                                fps,
                                orig_w,
                                orig_h,
                                w,
                                h,
                                roi_boxes,
                                qoffset,
                                target_bitrate_kbps,
                                wd,
                                key,
                                res_idx,
                                br_idx,
                                roi_idx,
                                metrics,
                                preset,
                                seek_mode,
                                verbose,
                                keep_files,
                            )
                        )

        print(f"[encode] {len(tasks)} frame-level jobs encode/metric", file=sys.stderr)
        frame_rows = {}
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_grid_segment_worker, task) for task in tasks]
                for done, fut in enumerate(as_completed(futures), 1):
                    key, frame_idx, row = fut.result()
                    frame_rows[(key, frame_idx)] = row
                    if done % 25 == 0 or done == len(futures):
                        print(f"[encode] {done}/{len(futures)}", file=sys.stderr)
        else:
            for done, task in enumerate(tasks, 1):
                key, frame_idx, row = _grid_segment_worker(task)
                frame_rows[(key, frame_idx)] = row
                if done % 25 == 0 or done == len(tasks):
                    print(f"[encode] {done}/{len(tasks)}", file=sys.stderr)

        for res_idx, _ in enumerate(RESOLUTIONS):
            for br_idx, _ in enumerate(bitrate_levels_kbps):
                for roi_idx, _ in enumerate(ROI_QOFFSET_LEVELS):
                    key = f"res{res_idx}_br{br_idx}_roi{roi_idx}"
                    grid[key] = [frame_rows[(key, fi)] for fi in range(nb_frames)]
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
        "source_video": os.path.abspath(input_path),
        "fps": fps,
        "num_frames": nb_frames,
        "grid_unit": "frame",
        "resolutions": RESOLUTIONS,
        "bitrate_levels_kbps": bitrate_levels_kbps,
        "roi_qoffset_levels": ROI_QOFFSET_LEVELS,
        "metrics": list(metrics),
        "encode_preset": preset,
        "seek_mode": seek_mode,
        "grid": grid,
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


def parse_float_list(value):
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Danh sach bitrate phai co dang 600,900,1500"
        ) from exc
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Moi bitrate level phai > 0")
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--yolo-metadata", default=YOLO_METADATA,
                     help="File jsonl co truong 'detections'[].bbox theo tung frame_idx")
    ap.add_argument("--out", default=ENCODE_GRID)
    ap.add_argument("--bitrate-levels", type=parse_float_list,
                     default=BITRATE_LEVELS_KBPS,
                     help="Cac muc bitrate kbps de do duong rate-distortion, vd "
                          "600,900,1500,2500,4000,6000")
    ap.add_argument("--keep-files", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                     help="So job ffmpeg chay song song. Goi y: 2-4 tren may laptop, "
                          "cao hon neu CPU/SSD manh.")
    ap.add_argument("--metrics", type=parse_metrics, default=("vmaf",),
                     help="Metric can tinh: 'vmaf' (mac dinh), 'vmaf,psnr', hoac 'none'. "
                          "Env/reward chi can VMAF, nen bo PSNR se nhanh hon dang ke.")
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
    out_path = args.out
    workdir = args.workdir

    result = build_grid(input_path, yolo_metadata_path,
                         keep_files=args.keep_files,
                         workdir=workdir, workers=max(1, args.workers),
                         metrics=args.metrics, preset=args.preset,
                         ref_preset=args.ref_preset, seek_mode=args.seek_mode,
                         verbose=args.verbose_ffmpeg,
                         bitrate_levels_kbps=args.bitrate_levels)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[done] Da luu luoi encode ROI vao: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
