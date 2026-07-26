"""
Inference với YOLOv5s — dùng được cả ONLINE và OFFLINE.
Hỗ trợ: 1 ảnh, 1 thư mục ảnh (infer tất cả trong 1 lần), hoặc 1 file video.

Nếu máy CÓ MẠNG (khuyên dùng, đơn giản hơn):
  pip install torch torchvision opencv-python-headless pandas seaborn requests pyyaml tqdm matplotlib
  python infer_yolov5_offline.py --source video.mp4 --hub-source github --weights yolov5s.pt

  (--weights có thể chỉ ghi "yolov5s.pt", torch.hub sẽ tự tải nếu chưa có sẵn file)

Nếu máy KHÔNG CÓ MẠNG, chuẩn bị trước (1 lần, lúc còn mạng):
  1. git clone https://github.com/ultralytics/yolov5   -> models/yolov5
  2. Tải weight yolov5s.pt -> checkpoints/yolov5s.pt
  Sau đó chạy với --hub-source local (mặc định)

Chạy ví dụ:
  # 1 ảnh
  python infer_yolov5_offline.py --source dataset/test_images/anh.jpg

  # cả thư mục ảnh (infer tất cả ảnh trong dataset/test_images cùng lúc)
  python infer_yolov5_offline.py --source dataset/test_images

  # cả thư mục ./dataset, quét luôn các thư mục con
  python infer_yolov5_offline.py --source dataset --recursive

  # video, dùng GPU + FP16 + imgsz nhỏ để đạt fps cao (khuyên dùng khi cần >=30fps)
  python infer_yolov5_offline.py --source dataset/videos/clip.mp4 --hub-source github --imgsz 416

  # CHỈ ghép 1 thư mục ảnh/frame có sẵn thành video, không chạy detect
  python infer_yolov5_offline.py --mode frames2video --frames-dir outputs/yolov5_results/images --fps 30 --output outputs/yolov5_results/videos/merged.mp4

Cấu trúc thư mục output:
  <save-dir>/
    images/     -> ảnh đã vẽ box + detections.csv (khi source là ảnh/thư mục ảnh)
    videos/     -> video đã vẽ box (khi source là video, hoặc dùng mode frames2video)

Để đạt >=30fps khi xử lý video (script sẽ in ra tốc độ xử lý thực tế để bạn kiểm tra):
  - Cần GPU (CUDA). Trên CPU thường không đạt được 30fps với YOLOv5s.
  - Cài torch bản có CUDA: xem https://pytorch.org/get-started/locally/ để lấy đúng lệnh pip
    cho phiên bản CUDA trên máy bạn, ví dụ:
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  - Giảm --imgsz (vd 416 hoặc 320) nếu vẫn chưa đủ nhanh.
  - FP16 (--half) tự bật khi có GPU, không cần chỉnh gì thêm.
"""
import torch
import argparse
import os
import re
import glob
import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def natural_key(path):
    """Sắp xếp 'frame2' trước 'frame10' (sắp theo số thay vì theo ký tự)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def collect_images(source, recursive=False):
    """Trả về list đường dẫn ảnh nếu source là thư mục, hoặc [source] nếu là 1 file ảnh."""
    if os.path.isdir(source):
        pattern = "**/*" if recursive else "*"
        files = glob.glob(os.path.join(source, pattern), recursive=recursive)
        images = [f for f in files if f.lower().endswith(IMG_EXTS)]
        images.sort()
        return images
    else:
        return [source]


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)


def load_model(yolo_repo, weights, conf, hub_source="local", half=None):
    """
    hub_source='local'  -> dùng code YOLOv5 đã git clone sẵn (offline, cần yolo_repo tồn tại).
    hub_source='github' -> tải trực tiếp code YOLOv5 từ GitHub (cần mạng, không cần clone tay).
    """
    if hub_source == "local":
        if not os.path.isdir(yolo_repo):
            raise FileNotFoundError(
                f"Không tìm thấy repo YOLOv5 tại '{yolo_repo}'. "
                f"Chạy: git clone https://github.com/ultralytics/yolov5 {yolo_repo}\n"
                f"(Hoặc nếu máy có mạng, dùng --hub-source github để khỏi cần clone.)"
            )
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"Không tìm thấy weight tại '{weights}'. "
            f"Tải yolov5s.pt từ https://github.com/ultralytics/yolov5/releases và đặt vào đó."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang dùng thiết bị: {device}")
    if device.type == "cpu":
        print("CẢNH BÁO: không có GPU (CUDA). Trên CPU rất khó đạt 30fps với video, "
              "chỉ nên kỳ vọng vài đến ~15fps tuỳ độ phân giải.")

    if hub_source == "github":
        print("Đang tải model từ GitHub (ultralytics/yolov5)...")
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
    else:
        print("Đang load model từ local (offline)...")
        model = torch.hub.load(yolo_repo, "custom", path=weights, source="local")

    model.to(device)
    model.conf = conf

    use_half = half if half is not None else (device.type == "cuda")
    if use_half:
        model.half()  # FP16 -> nhanh gần gấp đôi trên GPU, gần như không mất độ chính xác
        print("Đã bật half-precision (FP16) để tăng tốc.")

    return model, device


def run_on_images(model, image_paths, save_dir, batch_size=16):
    """Infer nhiều ảnh trong 1 (hoặc vài) lần gọi model — không loop từng ảnh gọi model riêng lẻ."""
    if not image_paths:
        print("Không tìm thấy ảnh nào để infer.")
        return

    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu infer...")
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    all_dfs = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        results = model(batch)  # 1 lần gọi model cho cả batch ảnh
        results.print()
        results.save(save_dir=images_dir)  # tự đặt tên theo file gốc, không ghi đè

        dfs = results.pandas().xyxy
        for path, df in zip(batch, dfs):
            df = df.copy()
            df.insert(0, "image", os.path.basename(path))
            all_dfs.append(df)

    if all_dfs:
        import pandas as pd
        full_df = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(images_dir, "detections.csv")
        full_df.to_csv(csv_path, index=False)
        print(f"\nĐã lưu chi tiết detection (tất cả ảnh) vào: {csv_path}")
        print(full_df)

    print(f"\nĐã lưu ảnh kết quả vào: {images_dir}")


def run_on_video(model, video_path, save_dir, conf, skip_frames=0, imgsz=640, output_fps=None):
    """
    YOLOv5 (torch.hub) không đọc video trực tiếp — nó chỉ nhận ảnh (frame).
    Ở đây ta tự tách video thành từng frame bằng OpenCV, infer từng frame,
    vẽ box lên frame rồi ghi lại thành video mới.

    imgsz: kích thước ảnh đưa vào model để infer (không phải kích thước video xuất ra).
           Giảm xuống (vd 416, 320) sẽ NHANH hơn nhưng có thể bỏ sót vật thể nhỏ.
    output_fps: fps ghi vào file video xuất ra. Nếu None thì lấy đúng fps của video gốc.
                Lưu ý: đây chỉ là con số "đóng gói" -- muốn chuyển động mượt ở fps này,
                tốc độ xử lý mỗi frame (in ra cuối chương trình) cũng phải theo kịp.
    """
    import time

    videos_dir = os.path.join(save_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out_fps = output_fps if output_fps else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(videos_dir, "output_" + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

    frame_idx = 0
    processed = 0
    print(f"Đang xử lý video: {video_path} ({width}x{height}, gốc {src_fps:.1f}fps, "
          f"xuất ra {out_fps:.1f}fps, imgsz={imgsz})")

    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if skip_frames and frame_idx % (skip_frames + 1) != 0:
            writer.write(frame)  # ghi frame gốc, không detect (để tăng tốc)
            frame_idx += 1
            continue

        # OpenCV đọc ảnh ở BGR, YOLOv5 cần RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, size=imgsz)

        # results.render() vẽ box trực tiếp lên ảnh (list ảnh RGB có box)
        rendered_rgb = results.render()[0]
        rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        writer.write(rendered_bgr)

        processed += 1
        frame_idx += 1
        if processed % 30 == 0:
            elapsed = time.time() - t_start
            cur_fps = processed / elapsed if elapsed > 0 else 0
            print(f"  Đã xử lý {processed} frame... (tốc độ thực tế: {cur_fps:.1f} fps)")

    cap.release()
    writer.release()

    total_elapsed = time.time() - t_start
    avg_fps = processed / total_elapsed if total_elapsed > 0 else 0
    print(f"\nTốc độ xử lý trung bình: {avg_fps:.1f} fps (đã detect {processed} frame trong {total_elapsed:.1f}s)")
    if avg_fps < out_fps:
        print(f"LƯU Ý: tốc độ xử lý ({avg_fps:.1f}fps) đang CHẬM HƠN fps xuất ra ({out_fps:.1f}fps) "
              f"-> chuyển động trong video xuất ra sẽ không mượt như video gốc. "
              f"Thử giảm --imgsz, dùng --skip-frames, hoặc chạy trên GPU để tăng tốc.")
    print(f"Đã lưu video kết quả vào: {out_path}")


def frames_to_video(frames_dir, output_path, fps=25.0, recursive=False):
    """
    Ghép 1 thư mục ảnh (frame) có sẵn thành 1 file video, KHÔNG chạy detect lại.
    Dùng khi bạn đã có sẵn các ảnh (vd: ảnh đã detect từ trước, hoặc frame trích
    từ nơi khác) và chỉ cần gộp chúng theo đúng thứ tự thành video.

    Ảnh được sắp xếp theo "natural sort" (frame2.jpg đứng trước frame10.jpg),
    nên đặt tên frame có số thứ tự (frame_0001.jpg, frame_0002.jpg, ...) để chắc chắn đúng thứ tự.
    """
    image_paths = collect_images(frames_dir, recursive=recursive)
    image_paths.sort(key=natural_key)

    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong '{frames_dir}'")

    print(f"Tìm thấy {len(image_paths)} frame. Bắt đầu ghép thành video...")

    # Đọc frame đầu tiên để lấy kích thước video
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_paths[0]}")
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, path in enumerate(image_paths, start=1):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  Bỏ qua (không đọc được): {path}")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))  # đảm bảo cùng kích thước
        writer.write(frame)
        if i % 50 == 0:
            print(f"  Đã ghép {i}/{len(image_paths)} frame...")

    writer.release()
    print(f"\nĐã lưu video ghép từ frame vào: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                         help="Đường dẫn ảnh, thư mục ảnh, hoặc file video")
    parser.add_argument("--yolo-repo", type=str, default="models/yolov5",
                         help="Đường dẫn tới thư mục code YOLOv5 đã clone")
    parser.add_argument("--weights", type=str, default="checkpoints/yolov5s.pt",
                         help="Đường dẫn tới file weight .pt")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Ngưỡng confidence")
    parser.add_argument("--save-dir", type=str, default="outputs/yolov5_results",
                         help="Thư mục lưu kết quả")
    parser.add_argument("--recursive", action="store_true",
                         help="Nếu --source là thư mục, quét luôn các thư mục con")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Số ảnh infer trong 1 lần gọi model (chỉ áp dụng khi source là thư mục)")
    parser.add_argument("--skip-frames", type=int, default=0,
                         help="Chỉ dùng cho video: bỏ qua N frame giữa mỗi lần detect để tăng tốc "
                              "(0 = detect mọi frame)")
    parser.add_argument("--hub-source", type=str, default="local", choices=["local", "github"],
                         help="'local' = dùng repo đã git clone (offline). "
                              "'github' = tải trực tiếp từ GitHub (cần mạng, không cần clone tay).")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Kích thước ảnh đưa vào model để infer video (giảm xuống 416/320 "
                              "để tăng tốc, đổi lại có thể bỏ sót vật thể nhỏ)")
    parser.add_argument("--output-fps", type=float, default=None,
                         help="[video] FPS ghi vào file video xuất ra. Mặc định lấy đúng fps video gốc.")
    parser.add_argument("--half", action="store_true", default=None,
                         help="Ép dùng FP16 (half-precision) để tăng tốc trên GPU. "
                              "Mặc định tự bật khi có GPU, tự tắt khi chạy CPU.")
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "frames2video"],
                         help="'infer' = chạy detect (mặc định). "
                              "'frames2video' = chỉ ghép 1 thư mục ảnh có sẵn thành video, không detect.")
    parser.add_argument("--frames-dir", type=str,
                         help="[mode=frames2video] Thư mục chứa các ảnh/frame cần ghép")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="[mode=frames2video] Số khung hình/giây của video xuất ra")
    parser.add_argument("--output", type=str,
                         help="[mode=frames2video] Đường dẫn file video xuất ra, "
                              "mặc định <save-dir>/videos/merged.mp4")
    args = parser.parse_args()

    if args.mode == "frames2video":
        if not args.frames_dir:
            raise ValueError("--mode frames2video cần chỉ định --frames-dir")
        output_path = args.output or os.path.join(args.save_dir, "videos", "merged.mp4")
        frames_to_video(args.frames_dir, output_path, fps=args.fps, recursive=args.recursive)
        return

    model, device = load_model(args.yolo_repo, args.weights, args.conf,
                                hub_source=args.hub_source, half=args.half)

    if is_video(args.source):
        run_on_video(model, args.source, args.save_dir, args.conf,
                     skip_frames=args.skip_frames, imgsz=args.imgsz, output_fps=args.output_fps)
    else:
        image_paths = collect_images(args.source, recursive=args.recursive)
        run_on_images(model, image_paths, args.save_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
