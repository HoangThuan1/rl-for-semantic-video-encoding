#!/usr/bin/env python3
"""
extract_yolo_metadata.py
-------------------------
BUOC 1 trong pipeline RL cua ban:

    video.mp4 --(script nay)--> yolo_metadata.jsonl --(yolo_metadata_to_rl.py)--> semantic_trace.jsonl
        --(gen_bandwidth_trace.py)--> bandwidth_trace.jsonl --(merge_traces)--> rl_states.jsonl

Script nay CHI chay YOLO tren video va luu RAW detection ra file .jsonl,
KHONG tinh semantic_score (viec do da co san trong yolo_metadata_to_rl.py
cua ban, o BUOC 2 -- lam ca 2 noi la thua/trung lap).

Format moi dong trong yolo_metadata.jsonl (khop chinh xac field ma
yolo_metadata_to_rl.py can: frame_idx, width, height, detections[].class_id/
class_name/centroid/area_ratio):

    {
      "frame_idx": 0, "width": 1280, "height": 720,
      "detections": [
        {"class_id": 2, "class_name": "car", "confidence": 0.91,
         "bbox": [x1,y1,x2,y2], "centroid": [cx,cy], "area_ratio": 0.034},
        ...
      ]
    }

CACH DUNG:
    python3 extract_yolo_metadata.py \
        --source dataset/videos/clip.mp4 \
        --yolo-repo models/yolov5 --weights checkpoints/yolov5s.pt \
        --imgsz 416 --out outputs/metadata/yolo_metadata.jsonl

Sau do chay tiep BUOC 2 (file cua ban, khong doi gi):
    python3 rl_env_offline/yolo_metadata_to_rl.py \
        --input outputs/metadata/yolo_metadata.jsonl \
        --out outputs/metadata/semantic_trace.jsonl \
        --alpha 1.0 --beta 10.0 --gamma 5.0 --delta 2.0
"""
import argparse
import json
import os

import cv2
import torch


def extract_yolo_metadata(model, frame_bgr, frame_idx, imgsz=640):
    """Chay YOLO tren 1 frame, tra ve 1 record dict dung format cho stage 2."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=imgsz)
    df = results.pandas().xyxy[0]  # xmin,ymin,xmax,ymax,confidence,class,name

    frame_area = w * h
    detections = []
    for _, row in df.iterrows():
        bw = row.xmax - row.xmin
        bh = row.ymax - row.ymin
        cx = row.xmin + bw / 2
        cy = row.ymin + bh / 2
        detections.append({
            "class_id": int(row["class"]),
            "class_name": str(row["name"]),
            "confidence": float(row.confidence),
            "bbox": [float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax)],
            "centroid": [float(cx), float(cy)],
            "area_ratio": float((bw * bh) / frame_area) if frame_area > 0 else 0.0,
        })

    return {"frame_idx": frame_idx, "width": w, "height": h, "detections": detections}


def run(video_path, model, out_path, imgsz=640):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    frame_idx = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            record = extract_yolo_metadata(model, frame, frame_idx, imgsz=imgsz)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: {len(record['detections'])} object")
            frame_idx += 1

    cap.release()
    print(f"\nDa luu {frame_idx} frame metadata vao: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="File video")
    ap.add_argument("--yolo-repo", default="models/yolov5")
    ap.add_argument("--weights", default="checkpoints/yolov5s.pt")
    ap.add_argument("--hub-source", default="local", choices=["local", "github"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="outputs/metadata/yolo_metadata.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang dung thiet bi: {device}")

    if args.hub_source == "github":
        model = torch.hub.load("ultralytics/yolov5", "custom", path=args.weights)
    else:
        model = torch.hub.load(args.yolo_repo, "custom", path=args.weights, source="local")
    model.to(device)
    model.conf = args.conf
    if device.type == "cuda":
        model.half()

    run(args.source, model, args.out, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
