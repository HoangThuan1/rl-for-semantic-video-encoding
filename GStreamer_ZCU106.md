# Pipeline GStreamer: mô phỏng PC → ZCU106

Pipeline mới nằm ở `gstreamer_edge_pipeline.py` và dùng lại nguyên policy DQN,
RL trace và JSONL detection của repository. Luồng dữ liệu là:

```text
MP4 -> GStreamer decode -> DPU/YOLO metadata -> semantic score + DQN
    -> {bitrate, resolution, ROI qoffset} -> ROI metadata/QP-map bridge
    -> GStreamer VCU/x264 encoder -> H.264/MP4 hoặc RTP
```

Trên PC, `--backend sim` chạy decode/encode thật với `x264enc`. Các quyết định
ROI và qoffset được ghi trong report JSON, là hợp đồng input cho VCU bridge.
`x264enc` không hỗ trợ VCU ROI QP-map, vì vậy kết quả này không được diễn giải
như số đo ROI bitrate của phần cứng. Để đo RD/ROI trước khi có board, dùng
`rl_env_offline/encode_grid.py` hoặc `export_encoded_video.py` hiện có.

Backend `sim` cần **cùng một Python environment** có `torch`, `PyGObject`
(`gi`) và GStreamer typelibs `Gst`/`GstVideo`, ngoài binary/plugin `x264enc`.
Trên Ubuntu/Debian, cài các gói hệ thống tương ứng bằng:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

Nếu virtualenv được tạo từ Python hệ thống, runner tự dùng lại `python3-gi`
trong `/usr/lib/python3/dist-packages`.
Kiểm tra tối thiểu:

```bash
gst-inspect-1.0 x264enc
python3 -c "import torch, gi; gi.require_version('Gst', '1.0')"
```

```bash
python3 gstreamer_edge_pipeline.py --backend sim \
  --input dataset/videos/Vehicle-Dataset-Sample-2_720p.mp4 \
  --policy rl_env_offline/dqn_policy.pt --max-frames 120 \
  --out outputs/yolov5_results/gst_policy.mp4 \
  --report outputs/metadata/gstreamer_rl_report.json
```

Nếu action đổi, runner đóng một MP4 segment để bảo toàn cấu hình encoder.
Các segment được chuẩn hóa về `--width` × `--height`; report liệt kê chính xác
mọi ROI/action. Nếu có nhiều segment, runner dùng FFmpeg để decode, nối và
tái mã hóa chúng thành đúng file `--out`, tránh lỗi timestamp/parameter set do
stream-copy. Các segment gốc, `manifest.json` và `segments.ffconcat` vẫn được
giữ trong thư mục `*_gst_segments/` để audit/debug. Có thể chỉ định binary bằng
`--ffmpeg-bin /duong/dan/toi/ffmpeg`; mặc định runner tìm trong `PATH`, sau đó
tìm bản static `ffmpeg-*-amd64-static/ffmpeg` đi kèm repository.

Trên ZCU106, cài image Vitis/VVAS đúng phiên bản BSP và kiểm tra plugin trước:

```bash
gst-inspect-1.0 vvas_xinfer vvas_xvcuenc
python3 gstreamer_edge_pipeline.py --backend zcu106 --print-pipeline
```

`semantic_roi_bridge` là phần adapter cần hiện thực theo BSP: nhận detection
từ `vvas_xinfer`, tạo semantic score giống `env.py`, gọi policy, rồi chuyển
`bbox + ROI_QOFFSET_LEVELS` thành ROI/QP-map metadata mà `vvas_xvcuenc` của
image đó công bố. Tên property VVAS thay đổi theo release, nên template không
hard-code một lệnh được cho là chạy được trên mọi image.
