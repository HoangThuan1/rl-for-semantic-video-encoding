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

HAI QUYET DINH BAN DA CHON (khong tu suy dien):
  1) Gop doan ffmpeg vat ly: MOI KHI action (bitrate_ratio, resolution_idx
     HOAC roi_idx) sau sanitize THUC SU khac frame truoc, dong doan
     dang mo va bat dau doan moi. (Khong dung segment_len co dinh lam kich
     thuoc vat ly -- segment_len chi con tac dung dung y nghia goc cua no
     trong env.py: khoa resolution va ROI level qua _sanitize_action().)
  2) Rate control: ABR that (-b:v/-maxrate/-bufsize) theo dung
     target_bitrate = bitrate_ratio * bandwidth (env.py dong 279), KHONG
     dung CRF co dinh. Ban tu xac nhan day la lua chon dung y nghia action.

DIEM CAN LUU Y VE NHAN QUA (khong the lam khac di, khong phai loi thiet ke):
  prev_bitrate dung lam input cho quyet dinh cua CAC FRAME SAU chi co the la
  bitrate THAT DA DO DUOC tu doan vua dong (khong the biet bitrate that cua
  1 doan TRUOC KHI encode xong doan do). Vi vay trong luc 1 doan dang mo,
  prev_bitrate duoc GIU NGUYEN bang gia tri that cua doan truoc do (giong
  y het 1 vong feedback control that: hanh dong dua tren ket qua da do
  duoc gan nhat, khong phai ket qua cua chinh no). prev_bandwidth thi
  KHONG co do tre nay vi no la dieu kien mang do duoc real-time, khong phu
  thuoc ket qua encode -- duoc cap nhat tung frame truc tiep tu trace,
  giong het env.step().

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
from encode_grid import load_yolo_metadata, union_roi_for_segment  # noqa: E402
from train import QNetwork, _torch_load_checkpoint  # noqa: E402


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


# ---------------------------------------------------------------------------
# Encode 1 doan THAT bang che do ABR (-b:v/-maxrate/-bufsize) theo dung
# target_bitrate cua policy, thay vi -crf co dinh nhu encode_grid.py
# (encode_grid.py dung CRF CHI VI muc dich tao grid offline, khong phai vi
# CRF la lua chon dung cho export that -- ban da tu xac nhan dung ABR).
# ---------------------------------------------------------------------------
def encode_segment_abr(input_path, start_frame, end_frame, width, height,
                        roi_box, qoffset, target_bitrate_kbps, fps, workdir, tag,
                        ffmpeg_bin="ffmpeg"):
    out_path = os.path.join(workdir, f"seg_{tag}_{start_frame}_{end_frame}.mp4")
    select_expr = f"between(n\\,{start_frame}\\,{end_frame - 1})"
    vf = f"select='{select_expr}',setpts=N/FRAME_RATE/TB,scale={width}:{height}:flags=bicubic"
    if roi_box is not None and abs(qoffset) > 1e-9:
        x, y, w, h = roi_box
        vf += f",addroi=x={x}:y={y}:w={w}:h={h}:qoffset={qoffset}"

    b_v = max(50.0, target_bitrate_kbps)
    maxrate = b_v * 1.2
    bufsize = b_v * 2.0
    gop = max(1, end_frame - start_frame)

    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-b:v", f"{b_v:.0f}k",
        "-maxrate", f"{maxrate:.0f}k",
        "-bufsize", f"{bufsize:.0f}k",
        "-bf", "0", "-g", str(gop),
        "-r", f"{fps:.6f}", "-vsync", "cfr", "-pix_fmt", "yuv420p",
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


def build_decision_stream(env, qnet):
    """Chay qua toan bo trace, MOI FRAME hoi policy 1 lan (giong het vong lap
    trong train.py), qua _sanitize_action() cua env.py de lay action da
    sanitize. Tra ve list [(frame_idx, safe_action, bandwidth, semantic_score)].
    Day CHUA phai buoc encode -- chi la buoc quyet dinh, tach rieng de sau do
    gom doan theo "action thuc su doi" ma khong lam roi logic nhan-qua khi
    encode (xem build_and_encode_segments)."""
    decisions = []
    for t in range(env.n):
        row = env.trace[t]
        env.t = t  # dong bo dung frame hien tai cho _sanitize_action()/_get_obs()
        obs = env._get_obs()
        action_idx = greedy_action(qnet, obs)
        raw_action = ACTIONS[action_idx]
        semantic_score = float(row.get("semantic_score", 0.0))
        bandwidth = float(row.get("bandwidth", env.max_bitrate))
        safe_action = env._sanitize_action(raw_action, semantic_score, bandwidth)
        decisions.append((t, safe_action, bandwidth, semantic_score))

        # Cap nhat cac truong KHONG phu thuoc ket qua encode that (bandwidth
        # la dieu kien mang do real-time, resolution la quyet dinh da chon
        # xong truoc khi encode) -- giong het env.step() lam sau moi buoc.
        env.prev_bandwidth = bandwidth
        env.current_resolution_idx = safe_action.resolution_idx
        env.current_roi_idx = safe_action.roi_idx
        # LUU Y: prev_bitrate KHONG duoc cap nhat o day -- no chi duoc cap
        # nhat trong build_and_encode_segments() SAU KHI mot doan duoc encode
        # that va do duoc bitrate that (xem docstring dau file).
    return decisions


def build_and_encode_segments(input_path, decisions, yolo_by_frame,
                               orig_w, orig_h, fps, workdir, env,
                               ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe"):
    """Gom cac frame lien tiep co CUNG (resolution_idx, bitrate_ratio, roi_idx)
    sau sanitize thanh 1 doan ffmpeg vat ly; MOI KHI action
    doi thi dong doan dang mo, encode that, do bitrate that, cap nhat
    env.prev_bitrate (dung cho quyet dinh cua CAC DOAN SAU -- xem docstring
    dau file ve nhan qua), roi bat dau doan moi."""
    segments = []  # metadata cho concat/bao cao
    seg_start = 0
    seg_action = decisions[0][1]

    def close_segment(seg_start, seg_end, action):
        width, height = RESOLUTIONS[action.resolution_idx]
        qoffset = ROI_QOFFSET_LEVELS[action.roi_idx]
        roi_box = (
            union_roi_for_segment(
                yolo_by_frame, seg_start, seg_end,
                orig_w, orig_h, width, height,
            )
            if abs(qoffset) > 1e-9
            else None
        )

        # target bitrate = trung binh bitrate_ratio * bandwidth tren toan
        # doan (bitrate_ratio co dinh trong doan, bandwidth co the doi nhe
        # frame-to-frame -- lay trung binh de co 1 gia tri -b:v duy nhat
        # cho ca doan, dung dinh nghia target_bitrate trong env.py dong 279).
        bw_values = [decisions[i][2] for i in range(seg_start, seg_end)]
        target_bitrate = action.bitrate_ratio * (sum(bw_values) / len(bw_values))

        tag = f"s{seg_start}_{seg_end}"
        enc_path = encode_segment_abr(
            input_path, seg_start, seg_end, width, height, roi_box, qoffset,
            target_bitrate, fps, workdir, tag, ffmpeg_bin=ffmpeg_bin
        )

        # Do bitrate THAT (khong phai uoc luong) de cap lai prev_bitrate cho
        # cac quyet dinh SAU doan nay -- day la buoc thay the duy nhat cho
        # _simulate_encode()/grid trong env.py.
        real_bitrates = extract_bitrate_per_frame(enc_path, fps, ffprobe_bin=ffprobe_bin)
        avg_real_bitrate = (sum(real_bitrates) / len(real_bitrates)) if real_bitrates else target_bitrate
        env.prev_bitrate = avg_real_bitrate

        segments.append({
            "start_frame": seg_start,
            "end_frame": seg_end,
            "resolution": [width, height],
            "bitrate_ratio": action.bitrate_ratio,
            "roi_idx": action.roi_idx,
            "roi_qoffset": qoffset,
            "roi_box": list(roi_box) if roi_box is not None else None,
            "roi_applied": roi_box is not None and abs(qoffset) > 1e-9,
            "target_bitrate_kbps": round(target_bitrate, 2),
            "measured_bitrate_kbps": round(avg_real_bitrate, 2),
            "path": enc_path,
        })
        print(
            f"[segment] frames [{seg_start},{seg_end}) | {width}x{height} | "
            f"bitrate_ratio={action.bitrate_ratio} | roi_idx={action.roi_idx} "
            f"qoffset={qoffset} | target={target_bitrate:.0f}kbps "
            f"measured={avg_real_bitrate:.0f}kbps",
            file=sys.stderr,
        )

    for idx in range(1, len(decisions) + 1):
        if idx < len(decisions):
            _, action, _, _ = decisions[idx]
            changed = (
                action.resolution_idx != seg_action.resolution_idx
                or action.bitrate_ratio != seg_action.bitrate_ratio
                or action.roi_idx != seg_action.roi_idx
            )
        else:
            changed = True  # het trace -> dong doan cuoi cung

        if changed:
            seg_end = idx
            close_segment(seg_start, seg_end, seg_action)
            if idx < len(decisions):
                seg_start = idx
                seg_action = decisions[idx][1]

    return segments


def concat_segments(segments, out_path, target_w, target_h, fps, workdir, ffmpeg_bin="ffmpeg"):
    """Cac doan co the khac do phan giai (resolution la quyet dinh cap
    segment), nen phai scale ve 1 do phan giai chung truoc khi noi bang
    filter_complex concat (khong the dung concat demuxer stream-copy vi do
    phan giai/SAR khac nhau giua cac doan)."""
    inputs = []
    filter_parts = []
    for i, seg in enumerate(segments):
        inputs += ["-i", seg["path"]]
        filter_parts.append(
            f"[{i}:v]scale={target_w}:{target_h}:flags=bicubic,"
            f"setsar=1,fps={fps:.6f}[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(segments)))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(segments)}:v=1:a=0[outv]"

    cmd = [
        ffmpeg_bin, "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-loglevel", "error",
        out_path,
    ]
    run(cmd)


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
                          "-- dieu khien khi nao resolution/ROI duoc phep doi, KHONG phai "
                          "kich thuoc doan ffmpeg vat ly (xem docstring dau file)")
    ap.add_argument("--concat-width", type=int, default=1920)
    ap.add_argument("--concat-height", type=int, default=1080)
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

    decisions = build_decision_stream(env, qnet)
    if any(action.roi_idx > 0 for _, action, _, _ in decisions):
        if not os.path.isfile(args.yolo_metadata):
            raise FileNotFoundError(
                "Policy da chon ROI encoding nhung khong tim thay YOLO metadata: "
                f"{args.yolo_metadata}"
            )
        yolo_by_frame = load_yolo_metadata(args.yolo_metadata)
    else:
        yolo_by_frame = {}

    workdir = args.workdir or tempfile.mkdtemp(prefix="policy_export_")
    os.makedirs(workdir, exist_ok=True)

    segments = build_and_encode_segments(
        args.input, decisions, yolo_by_frame, orig_w, orig_h, fps, workdir, env,
        ffmpeg_bin=args.ffmpeg_bin, ffprobe_bin=args.ffprobe_bin
    )

    print(f"[info] Tong {len(segments)} doan ffmpeg (moi doan = 1 lan action doi that su)",
          file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    concat_segments(segments, args.out, args.concat_width, args.concat_height, fps, workdir,
                     ffmpeg_bin=args.ffmpeg_bin)
    print(f"[done] Da xuat video: {args.out}", file=sys.stderr)

    if args.segments_report:
        with open(args.segments_report, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"[done] Da luu bao cao doan: {args.segments_report}", file=sys.stderr)

    if not args.keep_segments:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
