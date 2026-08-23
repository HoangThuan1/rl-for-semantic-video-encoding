# Pipeline GStreamer: mô phỏng PC → ZCU106

Pipeline mới nằm ở `gstreamer_edge_pipeline.py` và dùng lại nguyên policy DQN,
RL trace và JSONL detection của repository. Luồng dữ liệu là:

```text
                                      /-> DPU -> semantic score + DQN
MP4/camera -> GStreamer decode -> tee
                                      \-> ROI/QP-map apply -> VCU encoder -> RTP
```

Trên ZCU106, hai nhánh ghép kết quả bằng PTS của frame: nhánh DPU công bố
`{bitrate, resolution, ROI qoffset, bbox}`, còn nhánh encoder chờ/nhận quyết
định tương ứng trước khi gắn metadata và đẩy frame vào VCU.

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

## Đo độ trễ end-to-end

Backend `sim` tự đo bằng monotonic wall clock và ghi vào trường `timing` của
report JSON. `stage_totals_ms.end_to_end_until_output_ready` bao phủ khởi tạo,
nạp policy, decode, policy/ROI, encoder drain và ghép output. Thống kê từng
frame có `mean`, `p50`, `p95`, `max` cho các bước:

- `sample_pull`: chờ frame từ decoder;
- `buffer_map_copy`: map và sao chép buffer BGR;
- `policy_env`: DQN inference và cập nhật environment;
- `encoder_reconfigure`: đóng/mở segment khi action thay đổi;
- `roi_lookup`: lấy detection và hợp nhất ROI;
- `encoder_enqueue`: giao buffer cho `appsrc` (không phải latency encode hoàn
  tất vì encoder chạy bất đồng bộ);
- `frame_loop`: tổng thời gian xử lý đồng bộ của frame.

`throughput_fps` được tính trên toàn thời gian đến khi output sẵn sàng.
`realtime_factor = source_duration / wall_time`; giá trị `>= 1` nghĩa là xử lý
kịp hoặc nhanh hơn realtime. Thời gian thực thi plugin VCU trên ZCU106 cần được
đo bổ sung bằng timestamp/probe trên board; số đo PC không đại diện cho VCU.

Trên ZCU106, cài image Vitis/VVAS đúng phiên bản BSP và kiểm tra plugin trước:

```bash
gst-inspect-1.0 vvas_xinfer vvas_xvcuenc
python3 gstreamer_edge_pipeline.py --backend zcu106 --print-pipeline
```

`semantic_roi_bridge` và `semantic_roi_apply` là cặp adapter cần hiện thực theo
BSP. Bridge nhận detection từ `vvas_xinfer`, tạo semantic score giống `env.py`,
gọi policy và công bố quyết định theo PTS. Apply lấy quyết định đúng frame rồi
chuyển `bbox + ROI_QOFFSET_LEVELS` thành ROI/QP-map metadata mà `vvas_xvcuenc`
của image đó công bố. Tên property VVAS thay đổi theo release, nên template
không hard-code một lệnh được cho là chạy được trên mọi image.
