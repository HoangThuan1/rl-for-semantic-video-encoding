#!/usr/bin/env python3
"""GStreamer integration runner for the semantic-aware ROI/RL encoder.

This is the *PC simulation* stage of the ZCU106 design.  It uses GStreamer
for decode, frame transfer and H.264 encode, while reusing the trained DQN and
the YOLO/DPU-compatible JSONL metadata already produced by this repository.

The runner deliberately keeps the hardware boundary explicit:

  simulation:  GStreamer decode -> DQN -> ROI GstMeta -> x264enc -> MP4
  ZCU106:      capture -> vvas_xinfer (DPU) -> bridge -> vvas_xvcuenc (VCU)

`x264enc` does not consume the generic ROI metadata.  In simulation it is an
observable contract (also written to --report), not a claim of hardware ROI
rate-control.  The VCU bridge on the board must translate this meta into the
ROI/QP-map API exposed by the installed Xilinx BSP.

Example (small smoke run):
  python3 gstreamer_edge_pipeline.py --backend sim \\
    --input dataset/videos/Vehicle-Dataset-Sample-2_720p.mp4 \\
    --policy rl_env_offline/dqn_policy.pt --max-frames 120 \\
    --out outputs/yolov5_results/gst_policy.mp4

Use --backend zcu106 --print-pipeline on the board to print the integration
template; it never pretends that VVAS plugins are installed on this PC.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from glob import glob
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOLUTIONS = ((640, 480), (1280, 720), (1920, 1080))


def require_gst():
    # Ubuntu installs PyGObject outside a venv, in /usr/lib/python3/dist-packages.
    # Reuse that ABI-compatible binding instead of trying to compile PyGObject
    # inside the project environment.  Also support typelibs extracted locally
    # (useful on machines where apt packages are available but sudo is not).
    local_typelib_dirs = glob(str(
        ROOT / ".local_deps/gstreamer-gir/root/usr/lib/*/girepository-1.0"
    ))
    if local_typelib_dirs:
        existing = os.environ.get("GI_TYPELIB_PATH")
        os.environ["GI_TYPELIB_PATH"] = os.pathsep.join(
            local_typelib_dirs + ([existing] if existing else [])
        )
    try:
        try:
            import gi
        except ImportError:
            system_packages = Path("/usr/lib/python3/dist-packages")
            if system_packages.is_dir():
                sys.path.append(str(system_packages))
            import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Can PyGObject va GStreamer typelibs de chay backend sim. "
            "Ubuntu: cai python3-gi gir1.2-gstreamer-1.0 "
            "gir1.2-gst-plugins-base-1.0"
        ) from exc
    Gst.init(None)
    return Gst, GstVideo


def load_metadata(path):
    """Return detections keyed by frame; schema is shared with DPU output."""
    if not path or not os.path.isfile(path):
        return {}
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row.get("frame_idx", len(rows)))] = row
    return rows


def union_roi(detections, width, height):
    boxes = [d.get("bbox") for d in detections if len(d.get("bbox", [])) == 4]
    if not boxes:
        return None
    x1 = max(0, int(min(b[0] for b in boxes)))
    y1 = max(0, int(min(b[1] for b in boxes)))
    x2 = min(width, int(max(b[2] for b in boxes)))
    y2 = min(height, int(max(b[3] for b in boxes)))
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1)) if x2 > x1 and y2 > y1 else None


def resolve_ffmpeg(requested=None):
    """Return an FFmpeg executable while supporting the bundled static build."""
    if requested:
        candidate = shutil.which(requested)
        if candidate:
            return candidate
        path = Path(requested).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise RuntimeError(f"Khong tim thay FFmpeg executable: {requested}")

    candidate = shutil.which("ffmpeg")
    if candidate:
        return candidate
    bundled = sorted(ROOT.glob("ffmpeg-*-amd64-static/ffmpeg"), reverse=True)
    if bundled:
        return str(bundled[0].resolve())
    raise RuntimeError(
        "Can FFmpeg de ghep cac segment. Cai ffmpeg, dat no trong PATH, "
        "hoac truyen --ffmpeg-bin /duong/dan/toi/ffmpeg."
    )


def ffconcat_quote(path):
    """Quote an absolute path for an ffconcat manifest."""
    return str(Path(path).resolve()).replace("'", "'\\''")


def merge_segments(segments, out_path, fps, expected_frames, work, ffmpeg_bin=None):
    """Re-encode normalized MP4 segments into one timestamp-safe MP4 output."""
    ffmpeg = resolve_ffmpeg(ffmpeg_bin)
    concat_path = work / "segments.ffconcat"
    with open(concat_path, "w", encoding="utf-8") as handle:
        handle.write("ffconcat version 1.0\n")
        for segment in segments:
            handle.write(f"file '{ffconcat_quote(segment)}'\n")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-an", "-vf", f"fps={fps}", "-frames:v", str(expected_frames),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"FFmpeg khong ghep duoc cac segment: {detail}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg khong tao output hop le: {out_path}")
    return concat_path


def zcu106_pipeline_template(args):
    """Print only: element/property names vary with Vitis/VVAS release."""
    return f'''# ZCU106 deployment template (validate names with gst-inspect-1.0)
# DPU output must be converted by semantic_roi_bridge into ROI/QP-map metadata.
v4l2src device={args.camera_device} ! video/x-raw,format=NV12,width={args.width},height={args.height},framerate={args.fps}/1 ! \\
  tee name=t \\
  t. ! queue ! vvas_xinfer config-location=<dpu_infer.json> ! semantic_roi_bridge \\
      policy={args.policy} trace={args.trace} ! queue ! \\
      vvas_xvcuenc bitrate=<RL_TARGET_KBPS> control-rate=constant ! h264parse ! \\
      rtph264pay pt=96 config-interval=1 ! udpsink host=<receiver-ip> port=5004 \\
  t. ! queue ! fakesink

# semantic_roi_bridge is the board-specific adapter to implement.  It receives
# detections {{bbox,class_id,confidence}}, computes semantic score exactly as
# rl_env_offline/env.py, executes the DQN, and applies ROI/QP-map metadata that
# your installed vvas_xvcuenc advertises.  Do not copy this command verbatim
# until gst-inspect confirms the VVAS element/property names for the BSP.'''


class EncoderSegment:
    """One GStreamer appsrc->H.264 segment; rebuilt only when RL action changes."""
    def __init__(self, Gst, path, in_w, in_h, out_w, out_h, action, fps, max_bitrate_kbps):
        self.Gst, self.path, self.frames = Gst, path, 0
        target_w, target_h = RESOLUTIONS[action.resolution_idx]
        bitrate = max(50, round(action.bitrate_ratio * max_bitrate_kbps))
        desc = (
            f'appsrc name=source is-live=false format=time block=true '
            f'caps=video/x-raw,format=BGR,width={in_w},height={in_h},framerate={fps}/1 ! '
            f'videoconvert ! videoscale ! video/x-raw,format=I420,width={target_w},height={target_h} ! '
            f'videoscale ! video/x-raw,format=I420,width={out_w},height={out_h} ! '
            f'x264enc bitrate={bitrate} speed-preset=ultrafast tune=zerolatency key-int-max={fps} bframes=0 ! '
            f'h264parse ! mp4mux faststart=true ! filesink location="{path}"'
        )
        self.pipeline = Gst.parse_launch(desc)
        self.appsrc = self.pipeline.get_by_name("source")
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, data, pts, duration, roi=None):
        buf = self.Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.pts, buf.duration = pts, duration
        if roi:
            # Standard GstVideo ROI metadata. x264enc ignores it; VCU bridge
            # consumes this semantic boundary on the board.
            from gi.repository import GstVideo
            GstVideo.buffer_add_video_region_of_interest_meta(buf, "semantic-roi", *roi)
        result = self.appsrc.emit("push-buffer", buf)
        if result != self.Gst.FlowReturn.OK:
            raise RuntimeError(f"Encoder appsrc loi: {result.value_nick}")
        self.frames += 1

    def close(self):
        self.appsrc.emit("end-of-stream")
        bus = self.pipeline.get_bus()
        message = bus.timed_pop_filtered(20 * self.Gst.SECOND, self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        self.pipeline.set_state(self.Gst.State.NULL)
        if not message or message.type == self.Gst.MessageType.ERROR:
            detail = message.parse_error()[0].message if message else "timeout waiting EOS"
            raise RuntimeError(f"Khong dong duoc segment GStreamer: {detail}")


def run_simulation(args):
    sys.path.insert(0, str(ROOT / "rl_env_offline"))
    from env import VCUSimEnv
    from export_encoded_video import greedy_action, load_policy
    Gst, GstVideo = require_gst()
    env = VCUSimEnv(trace_path=args.trace, loop=False, max_bitrate_kbps=args.max_bitrate_kbps,
                    segment_len=args.segment_len)
    qnet = load_policy(args.policy, env)
    detections = load_metadata(args.dpu_metadata)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / (out_path.stem + "_gst_segments")
    work.mkdir(exist_ok=True)

    decoder = Gst.parse_launch(
        f'filesrc location="{Path(args.input).resolve()}" ! decodebin ! videoconvert ! '
        'video/x-raw,format=BGR ! appsink name=frames sync=false max-buffers=2 drop=false'
    )
    sink = decoder.get_by_name("frames")
    decoder.set_state(Gst.State.PLAYING)
    state, _ = env.reset()
    encoder = None
    current_action = None
    segments, report, frame_idx = [], [], 0
    try:
        while frame_idx < env.n and (not args.max_frames or frame_idx < args.max_frames):
            sample = sink.emit("pull-sample")
            if sample is None:
                break
            caps = sample.get_caps().get_structure(0)
            width, height = caps.get_value("width"), caps.get_value("height")
            buffer = sample.get_buffer()
            ok, mapped = buffer.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("Khong map duoc frame BGR tu GStreamer")
            data = bytes(mapped.data)
            buffer.unmap(mapped)

            action_idx = greedy_action(qnet, state)
            _, _, terminated, truncated, info = env.step(action_idx)
            action = info["safe_action"]
            if action != current_action:
                if encoder:
                    encoder.close()
                seg_path = work / f"segment_{len(segments):04d}.mp4"
                encoder = EncoderSegment(Gst, str(seg_path), width, height, args.width, args.height,
                                         action, args.fps, args.max_bitrate_kbps)
                segments.append(seg_path)
                current_action = action

            row = detections.get(frame_idx, {})
            roi = union_roi(row.get("detections", []), width, height) if action.roi_idx else None
            # Metadata is included in the report.  A production VCU adapter maps
            # this ROI + qoffset to the encoder's hardware QP map.
            encoder.push(data, frame_idx * Gst.SECOND // args.fps, Gst.SECOND // args.fps, roi=roi)
            report.append({"frame_idx": frame_idx, "action": asdict(action), "target_bitrate_kbps": info["target_bitrate"],
                           "semantic_score": float(env.trace[frame_idx].get("semantic_score", 0)),
                           "roi": roi, "roi_qoffset": info["roi_qoffset"]})
            frame_idx += 1
            state = env._get_obs() if not (terminated or truncated) else state
            if terminated or truncated:
                break
    finally:
        decoder.set_state(Gst.State.NULL)
        if encoder:
            encoder.close()

    if not segments:
        raise RuntimeError("Khong doc duoc frame nao tu input")
    manifest_path = work / "manifest.json"
    if len(segments) == 1:
        os.replace(segments[0], out_path)
        segment_outputs = [str(out_path)]
        concat_input = None
    else:
        # Keep the original action-boundary segments for audit/debug, but always
        # create the requested MP4 by decoding and re-encoding them with FFmpeg.
        segment_outputs = [str(p) for p in segments]
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"segments": segment_outputs}, handle, indent=2)
        concat_input = merge_segments(
            segments, out_path, args.fps, frame_idx, work, ffmpeg_bin=args.ffmpeg_bin
        )
    Path(args.report).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump({"backend": "sim", "roi_encoding": "metadata-contract-only (x264enc fallback)",
                   "frames": frame_idx, "output": str(out_path),
                   "segments": segment_outputs,
                   "segment_manifest": str(manifest_path) if len(segments) > 1 else None,
                   "ffconcat_input": str(concat_input) if concat_input else None,
                   "frames_detail": report}, handle,
                  ensure_ascii=False, indent=2)
    print(f"[done] {frame_idx} frames, {len(segments)} segment(s). report={args.report}")
    print(f"[output] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("sim", "zcu106"), default="sim")
    ap.add_argument("--input", help="MP4 input (bat buoc voi sim)")
    ap.add_argument("--policy", default="rl_env_offline/dqn_policy.pt")
    ap.add_argument("--trace", default="outputs/metadata/rl_states.jsonl")
    ap.add_argument("--dpu-metadata", default="outputs/metadata/yolo_metadata.jsonl")
    ap.add_argument("--out", default="outputs/yolov5_results/gst_policy.mp4")
    ap.add_argument("--report", default="outputs/metadata/gstreamer_rl_report.json")
    ap.add_argument("--max-bitrate-kbps", type=float, default=8000.0)
    ap.add_argument("--segment-len", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ffmpeg-bin", default=None,
                    help="FFmpeg executable de ghep segment (mac dinh: tim trong PATH/bundled build)")
    ap.add_argument("--camera-device", default="/dev/video0")
    ap.add_argument("--print-pipeline", action="store_true")
    args = ap.parse_args()
    if args.backend == "zcu106" or args.print_pipeline:
        print(zcu106_pipeline_template(args))
        return
    if not args.input:
        ap.error("--input la bat buoc voi --backend sim")
    run_simulation(args)


if __name__ == "__main__":
    main()
