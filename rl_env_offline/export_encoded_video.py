#!/usr/bin/env python3
"""
export_encoded_video.py
------------------------
Xuat video da duoc encode THAT (khong phai surrogate _simulate_encode()/grid
trong env.py) theo dung quyet dinh cua policy DQN da train (dqn_policy.pt).

KHONG dinh nghia lai bat cu thu gi da co san -- MOI thanh phan duoi day duoc
import/tai su dung TRUC TIEP tu code hien co cua ban, khong doan:

  - QNetwork: import thang tu train.py (dung 100% kien truc da dung de train,
    khong copy-paste rieng de tranh lech).
  - Checkpoint format (dict {"model_state_dict","state_dim","num_actions",...}):
    kiem tra dung nhu load_training_checkpoint() trong train.py (bao loi neu
    state_dim/num_actions khong khop, giong het dieu kien trong train.py).
  - ACTIONS, RESOLUTIONS, BITRATE_RATIOS, ROI_QOFFSET_LEVELS, VCUSimEnv,
    _sanitize_action(), _get_obs(), reset(), segment_len, STATE_DIM,
    SEMANTIC_SCORE_MAX, DEFAULT_TRACE_PATH, DEFAULT_YOLO_METADATA_PATH:
    import thang tu env.py, KHONG viet lai logic sanitize/obs.
  - probe_video(): copy cach doc do phan giai/fps/so frame bang ffprobe.

Exporter encode theo GOP co dinh ``segment_len``. Trong tung GOP, target la
trung binh cua ``bitrate_ratio[t] * bandwidth[t]``; sau khi encode xong, bitrate
do that duoc feedback vao observation cua GOP tiep theo. Resolution va ROI duoc
khoa trong GOP dung theo ``_sanitize_action()``.

Muc ROI cua policy (0/-0.3/-0.6) duoc anh xa mac dinh sang qoffset libx264 an
toan hon (0/-0.05/-0.10). Day la chu y: -0.3/-0.6 qua manh voi VBV bitrate
thap va co the tao frame xam. Report ghi ca muc policy va muc da ap dung.

CACH DUNG:
  python3 export_encoded_video.py \
      --input dataset/videos/xxx.mp4 \
      --policy dqn_policy.pt \
      --trace outputs/metadata/rl_states.jsonl \
      --yolo-metadata outputs/metadata/yolo_metadata.jsonl \
      --out outputs/yolov5_results/exported_policy_video.mp4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (  # noqa: E402  (thu vien that cua ban, khong viet lai)
    ACTIONS,
    DEFAULT_YOLO_METADATA_PATH,
    DEFAULT_TRACE_PATH,
    RESOLUTIONS,
    ROI_QOFFSET_LEVELS,
    VCUSimEnv,
)
from train import QNetwork, _torch_load_checkpoint  # noqa: E402


# qoffset -0.3/-0.6 trong action space duoc dung de phan biet muc ROI khi
# training. Dua truc tiep cac gia tri do vao libx264 + VBV bitrate thap co the
# lam rate-control sap (da quan sat frame xam voi -0.6). Exporter anh xa muc
# action sang cac offset bao thu hon, gan voi vi du -0.1 cua FFmpeg.
DEFAULT_EXPORT_ROI_QOFFSETS = [0.0, -0.05, -0.10]
IMPORTANT_CLASSES = {
    "person": 1.0,
    "car": 0.9,
    "truck": 0.9,
    "bus": 0.9,
    "motorcycle": 0.8,
    "bicycle": 0.7,
}


def probe_video(path, ffprobe_bin="ffprobe"):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Khong tim thay file input: {path} (duong dan tuong doi tinh tu "
            f"thu muc dang chay script: {os.getcwd()})"
        )
    out = run([ffprobe_bin, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
               "-of", "json", path]).stdout
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    width, height = int(info["width"]), int(info["height"])
    nb_frames = info.get("nb_frames")
    if nb_frames in (None, "N/A"):
        out2 = run([ffprobe_bin, "-v", "error", "-select_streams", "v:0",
                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                    "-of", "default=noprint_wrappers=1:nokey=1", path]).stdout.strip()
        nb_frames = int(out2)
    else:
        nb_frames = int(nb_frames)
    return fps, width, height, nb_frames


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    return proc


def extract_bitrate_per_frame(encoded_path, fps, ffprobe_bin="ffprobe"):
    """Copy tu encode_grid.py: doc pkt_size that tu file da encode qua
    ffprobe, quy doi ra kbps that -- dung lam gia tri prev_bitrate THAT
    (khong phai uoc luong) cap lai cho policy o doan tiep theo."""
    out = run([ffprobe_bin, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "frame=pkt_size", "-of", "csv=p=0", encoded_path]).stdout
    sizes = [int(line.split(",")[0].strip())
             for line in out.strip().splitlines() if line.strip()]
    return [size * 8 * fps / 1000.0 for size in sizes]


def load_export_yolo_metadata(path):
    """Giu detection day du de exporter co the loc confidence/class.

    Loader trong encode_grid chi giu bbox, phu hop viec tao grid nhung khong
    du thong tin de tranh union ROI tu false-positive trong export that.
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["frame_idx"])] = row
    return out


def _even(value):
    return max(2, int(round(value)) // 2 * 2)


def fit_resolution(target_w, target_h, src_w, src_h):
    """Fit source vao action resolution ma khong keo meo aspect ratio."""
    scale = min(target_w / src_w, target_h / src_h)
    return _even(src_w * scale), _even(src_h * scale)


def select_roi_boxes(yolo_by_frame, start_frame, end_frame, src_w, src_h,
                     target_w, target_h, min_confidence=0.40, max_rois=3,
                     max_total_fraction=0.35, padding_fraction=0.08):
    """Lay toi da vai ROI tinh tu frame giua GOP thay vi union ca GOP.

    Union moi bbox cua moi frame de lam ROI phong thanh gan toan man hinh.
    Frame giua la xap xi on dinh cho GOP ngan; confidence va class priority
    giup loai bot false-positive/vat the khong quan trong.
    """
    mid = start_frame + max(0, end_frame - start_frame - 1) // 2
    row = yolo_by_frame.get(mid)
    if not row:
        return []

    candidates = []
    for det in row.get("detections", []):
        confidence = float(det.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        x1, y1, x2, y2 = map(float, det.get("bbox", (0, 0, 0, 0)))
        if x2 <= x1 or y2 <= y1:
            continue
        area_fraction = ((x2 - x1) * (y2 - y1)) / max(src_w * src_h, 1)
        priority = IMPORTANT_CLASSES.get(det.get("class_name", ""), 0.25)
        candidates.append((priority * confidence * (1.0 + area_fraction), x1, y1, x2, y2))

    sx, sy = target_w / src_w, target_h / src_h
    selected = []
    used_fraction = 0.0
    for _, x1, y1, x2, y2 in sorted(candidates, reverse=True):
        pad_x = (x2 - x1) * padding_fraction
        pad_y = (y2 - y1) * padding_fraction
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(src_w), x2 + pad_x)
        y2 = min(float(src_h), y2 + pad_y)
        area_fraction = ((x2 - x1) * (y2 - y1)) / max(src_w * src_h, 1)
        if area_fraction > max_total_fraction or used_fraction + area_fraction > max_total_fraction:
            continue

        tx1, ty1 = int(round(x1 * sx)), int(round(y1 * sy))
        tx2, ty2 = int(round(x2 * sx)), int(round(y2 * sy))
        tx1 = max(0, min(tx1, target_w - 2))
        ty1 = max(0, min(ty1, target_h - 2))
        tx2 = max(tx1 + 2, min(tx2, target_w))
        ty2 = max(ty1 + 2, min(ty2, target_h))
        selected.append((tx1, ty1, tx2 - tx1, ty2 - ty1))
        used_fraction += area_fraction
        if len(selected) >= max_rois:
            break
    return selected


# ---------------------------------------------------------------------------
# Encode 1 doan THAT bang che do ABR (-b:v/-maxrate/-bufsize) theo dung
# target_bitrate cua policy, thay vi -crf co dinh nhu encode_grid.py
# (encode_grid.py dung CRF CHI VI muc dich tao grid offline, khong phai vi
# CRF la lua chon dung cho export that -- ban da tu xac nhan dung ABR).
# ---------------------------------------------------------------------------
def encode_segment_abr(input_path, start_frame, end_frame, width, height,
                        roi_boxes, qoffset, target_bitrate_kbps, fps, workdir, tag,
                        ffmpeg_bin="ffmpeg"):
    out_path = os.path.join(workdir, f"seg_{tag}_{start_frame}_{end_frame}.mp4")
    frame_count = end_frame - start_frame
    vf = (
        f"trim=start_frame={start_frame}:end_frame={end_frame},"
        f"setpts=PTS-STARTPTS,scale={width}:{height}:flags=bicubic"
    )
    if abs(qoffset) > 1e-9:
        for x, y, w, h in roi_boxes:
            vf += f",addroi=x={x}:y={y}:w={w}:h={h}:qoffset={qoffset}"

    b_v = max(50.0, target_bitrate_kbps)
    maxrate = b_v * 1.2
    bufsize = b_v * 2.0
    gop = max(1, frame_count)

    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-vf", vf,
        "-an", "-frames:v", str(frame_count),
        "-c:v", "libx264",
        "-b:v", f"{b_v:.0f}k",
        "-maxrate", f"{maxrate:.0f}k",
        "-bufsize", f"{bufsize:.0f}k",
        # libx264 ROI side-data voi B-frame co the tao frame xam/skip. Tat
        # B-frame de ROI quant_offsets duoc ap dung on dinh.
        "-bf", "0", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        # CFR tao timestamp hop le cho MP4. passthrough o day tao packet rong
        # va decoder hien mau xam voi source nay.
        "-r", f"{fps:.6f}", "-fps_mode", "cfr", "-pix_fmt", "yuv420p",
        "-loglevel", "error", out_path,
    ]
    run(cmd)
    return out_path


def load_policy(policy_path, env):
    """Kiem tra checkpoint dung dieu kien y het load_training_checkpoint()
    trong train.py (bao loi neu state_dim/num_actions khong khop env hien
    tai), roi nap trong so vao QNetwork import tu train.py."""
    ckpt = _torch_load_checkpoint(policy_path)
    state_dim = ckpt.get("state_dim")
    num_actions = ckpt.get("num_actions")
    if state_dim != env.STATE_DIM or num_actions != env.NUM_ACTIONS:
        raise ValueError(
            "Checkpoint khong khop env hien tai: "
            f"checkpoint state_dim={state_dim}, num_actions={num_actions}; "
            f"env state_dim={env.STATE_DIM}, num_actions={env.NUM_ACTIONS}."
        )
    qnet = QNetwork(env.STATE_DIM, env.NUM_ACTIONS)
    qnet.load_state_dict(ckpt["model_state_dict"])
    qnet.eval()
    return qnet


def greedy_action(qnet, obs):
    with torch.no_grad():
        q = qnet(torch.from_numpy(obs).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())


def decide_and_encode_segments(input_path, yolo_by_frame, orig_w, orig_h, fps,
                               workdir, env, qnet, export_qoffsets,
                               min_roi_confidence=0.40, max_rois=3,
                               max_roi_fraction=0.35, ffmpeg_bin="ffmpeg",
                               ffprobe_bin="ffprobe"):
    """Quyet dinh va encode online theo GOP co dinh ``env.segment_len``.

    Tat ca frame trong GOP dung cung resolution/ROI (dung sanitize cua env),
    con target bitrate GOP la trung binh cua ratio[t] * bandwidth[t]. Sau khi
    GOP encode xong, bitrate do that duoc dua vao observation GOP ke tiep.
    """
    segments = []
    for seg_start in range(0, env.n, env.segment_len):
        seg_end = min(seg_start + env.segment_len, env.n)
        frame_actions = []
        targets = []
        for t in range(seg_start, seg_end):
            env.t = t
            row = env.trace[t]
            action_idx = greedy_action(qnet, env._get_obs())
            raw_action = ACTIONS[action_idx]
            semantic_score = float(row.get("semantic_score", 0.0))
            bandwidth = float(row.get("bandwidth", env.max_bitrate))
            safe_action = env._sanitize_action(raw_action, semantic_score, bandwidth)
            frame_actions.append(safe_action)
            targets.append(safe_action.bitrate_ratio * bandwidth)
            env.prev_bandwidth = bandwidth
            env.current_resolution_idx = safe_action.resolution_idx
            env.current_roi_idx = safe_action.roi_idx

        segment_action = frame_actions[0]
        requested_w, requested_h = RESOLUTIONS[segment_action.resolution_idx]
        width, height = fit_resolution(requested_w, requested_h, orig_w, orig_h)
        policy_qoffset = ROI_QOFFSET_LEVELS[segment_action.roi_idx]
        applied_qoffset = export_qoffsets[segment_action.roi_idx]
        roi_boxes = (
            select_roi_boxes(
                yolo_by_frame, seg_start, seg_end, orig_w, orig_h, width, height,
                min_confidence=min_roi_confidence, max_rois=max_rois,
                max_total_fraction=max_roi_fraction,
            )
            if abs(applied_qoffset) > 1e-9 else []
        )
        target_bitrate = max(50.0, min(sum(targets) / len(targets), env.max_bitrate))
        enc_path = encode_segment_abr(
            input_path, seg_start, seg_end, width, height, roi_boxes,
            applied_qoffset, target_bitrate, fps, workdir,
            f"gop_{seg_start // env.segment_len}", ffmpeg_bin=ffmpeg_bin,
        )
        real_bitrates = extract_bitrate_per_frame(enc_path, fps, ffprobe_bin=ffprobe_bin)
        if len(real_bitrates) != seg_end - seg_start:
            raise RuntimeError(
                f"Segment [{seg_start},{seg_end}) encode sai so frame: "
                f"expected={seg_end - seg_start}, actual={len(real_bitrates)}"
            )
        avg_real_bitrate = sum(real_bitrates) / len(real_bitrates)
        env.prev_bitrate = avg_real_bitrate

        unique_ratios = sorted({a.bitrate_ratio for a in frame_actions})
        segments.append({
            "start_frame": seg_start,
            "end_frame": seg_end,
            "requested_resolution": [requested_w, requested_h],
            "encoded_resolution": [width, height],
            "bitrate_ratios": unique_ratios,
            "roi_idx": segment_action.roi_idx,
            "policy_roi_qoffset": policy_qoffset,
            "applied_roi_qoffset": applied_qoffset,
            "roi_boxes": [list(box) for box in roi_boxes],
            "roi_applied": bool(roi_boxes) and abs(applied_qoffset) > 1e-9,
            "target_bitrate_kbps": round(target_bitrate, 2),
            "measured_bitrate_kbps": round(avg_real_bitrate, 2),
            "path": enc_path,
        })
        print(
            f"[gop] frames [{seg_start},{seg_end}) | {width}x{height} | "
            f"ratios={unique_ratios} | roi_idx={segment_action.roi_idx} "
            f"qoffset={applied_qoffset} | roi_boxes={len(roi_boxes)} | "
            f"target={target_bitrate:.0f}kbps measured={avg_real_bitrate:.0f}kbps",
            file=sys.stderr,
        )
    return segments


def concat_segments(segments, out_path, target_w, target_h, fps, workdir,
                    input_path=None, ffmpeg_bin="ffmpeg"):
    """Cac doan co the khac do phan giai (resolution la quyet dinh cap
    segment), nen phai scale ve 1 do phan giai chung truoc khi noi bang
    filter_complex concat (khong the dung concat demuxer stream-copy vi do
    phan giai/SAR khac nhau giua cac doan)."""
    inputs = []
    filter_parts = []
    for i, seg in enumerate(segments):
        inputs += ["-i", seg["path"]]
        filter_parts.append(
            f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=bicubic,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(segments)))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(segments)}:v=1:a=0[outv]"

    audio_input_idx = len(segments)
    if input_path is not None:
        inputs += ["-i", input_path]
    total_frames = sum(seg["end_frame"] - seg["start_frame"] for seg in segments)
    cmd = [
        ffmpeg_bin, "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}", "-fps_mode", "cfr",
        "-frames:v", str(total_frames),
    ]
    if input_path is not None:
        cmd += ["-map", f"{audio_input_idx}:a:0?", "-c:a", "aac", "-shortest"]
    cmd += [
        "-loglevel", "error",
        out_path,
    ]
    run(cmd)


def validate_output(path, expected_frames, ffprobe_bin="ffprobe"):
    out = run([
        ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
        "-of", "json", path,
    ]).stdout
    stream = json.loads(out)["streams"][0]
    actual_frames = int(stream["nb_read_frames"])
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"Output sai so frame: expected={expected_frames}, actual={actual_frames}. "
            "File duoc giu lai de debug nhung khong duoc xem la export thanh cong."
        )
    return stream


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Video nguon")
    ap.add_argument("--policy", required=True, help="dqn_policy.pt")
    ap.add_argument("--trace", default=DEFAULT_TRACE_PATH,
                     help="rl_states.jsonl (mac dinh: env.DEFAULT_TRACE_PATH)")
    ap.add_argument("--yolo-metadata", default=DEFAULT_YOLO_METADATA_PATH,
                     help="YOLO JSONL co detections[].bbox de tao ROI box cho tung doan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bitrate-kbps", type=float, default=8000.0,
                     help="Phai khop max_bitrate_kbps da dung khi train (mac dinh env.py: 8000.0)")
    ap.add_argument("--segment-len", type=int, default=8,
                     help="Phai khop segment_len da dung khi train (mac dinh env.py: 8) "
                          "-- cung la kich thuoc GOP/doan ffmpeg vat ly")
    ap.add_argument("--concat-width", type=int, default=1920)
    ap.add_argument("--concat-height", type=int, default=1080)
    ap.add_argument("--export-roi-qoffsets", default="0,-0.05,-0.10",
                    help="qoffset libx264 an toan ung voi roi_idx 0/1/2")
    ap.add_argument("--min-roi-confidence", type=float, default=0.40)
    ap.add_argument("--max-rois", type=int, default=3)
    ap.add_argument("--max-roi-fraction", type=float, default=0.35,
                    help="Tong dien tich ROI toi da tren frame (0..1)")
    ap.add_argument("--keep-segments", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--segments-report", default=None,
                     help="Neu dat, luu JSON mo ta tung doan da encode + quyet dinh cua policy")
    ap.add_argument("--ffmpeg-bin", default="ffmpeg",
                     help="Duong dan binary ffmpeg. Neu 'ffmpeg' khong co trong PATH, tro "
                          "thang toi ban tinh cua ban, vd: "
                          "../ffmpeg-7.0.2-amd64-static/ffmpeg")
    ap.add_argument("--ffprobe-bin", default="ffprobe",
                     help="Duong dan binary ffprobe, vd: "
                          "../ffmpeg-7.0.2-amd64-static/ffprobe")
    args = ap.parse_args()

    export_qoffsets = [float(v.strip()) for v in args.export_roi_qoffsets.split(",")]
    if len(export_qoffsets) != len(ROI_QOFFSET_LEVELS):
        raise ValueError(
            f"--export-roi-qoffsets can {len(ROI_QOFFSET_LEVELS)} gia tri, "
            f"nhan duoc {len(export_qoffsets)}"
        )
    if export_qoffsets[0] != 0.0 or any(not -1.0 <= q <= 1.0 for q in export_qoffsets):
        raise ValueError("ROI qoffset phai nam trong [-1,1] va muc roi_idx=0 phai bang 0")
    if not 0.0 < args.max_roi_fraction <= 1.0:
        raise ValueError("--max-roi-fraction phai nam trong (0,1]")

    fps, orig_w, orig_h, nb_frames = probe_video(args.input, ffprobe_bin=args.ffprobe_bin)
    print(f"[info] input: {args.input} | {orig_w}x{orig_h}@{fps:.3f}fps | {nb_frames} frames",
          file=sys.stderr)

    env = VCUSimEnv(
        trace_path=args.trace,
        encode_grid_path=None,  # KHONG dung grid/simulate -- encode that thay the
        max_bitrate_kbps=args.max_bitrate_kbps,
        segment_len=args.segment_len,
        loop=False,
    )
    if env.n > nb_frames:
        print(f"[warn] trace co {env.n} frame nhung video chi co {nb_frames} frame "
              f"-- chi dung {nb_frames} frame dau cua trace.", file=sys.stderr)
        env.trace = env.trace[:nb_frames]
        env.n = nb_frames
    obs, _ = env.reset()

    qnet = load_policy(args.policy, env)

    if not os.path.isfile(args.yolo_metadata):
        raise FileNotFoundError(f"Khong tim thay YOLO metadata: {args.yolo_metadata}")
    yolo_by_frame = load_export_yolo_metadata(args.yolo_metadata)

    created_temp_workdir = args.workdir is None
    workdir = args.workdir or tempfile.mkdtemp(prefix="policy_export_")
    os.makedirs(workdir, exist_ok=True)

    segments = decide_and_encode_segments(
        args.input, yolo_by_frame, orig_w, orig_h, fps, workdir, env, qnet,
        export_qoffsets=export_qoffsets,
        min_roi_confidence=args.min_roi_confidence,
        max_rois=max(1, args.max_rois),
        max_roi_fraction=args.max_roi_fraction,
        ffmpeg_bin=args.ffmpeg_bin, ffprobe_bin=args.ffprobe_bin,
    )

    print(f"[info] Tong {len(segments)} GOP ffmpeg (moi GOP toi da {env.segment_len} frame)",
          file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    concat_segments(
        segments, args.out, args.concat_width, args.concat_height, fps, workdir,
        input_path=args.input, ffmpeg_bin=args.ffmpeg_bin,
    )
    output_stream = validate_output(args.out, env.n, ffprobe_bin=args.ffprobe_bin)
    print(
        f"[check] output du {env.n} frame | {output_stream['width']}x{output_stream['height']} "
        f"| avg_frame_rate={output_stream['avg_frame_rate']}", file=sys.stderr,
    )
    print(f"[done] Da xuat video: {args.out}", file=sys.stderr)

    if args.segments_report:
        with open(args.segments_report, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"[done] Da luu bao cao doan: {args.segments_report}", file=sys.stderr)

    if not args.keep_segments and created_temp_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    elif args.workdir and not args.keep_segments:
        print(f"[info] Giu lai workdir do nguoi dung chi dinh: {workdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
