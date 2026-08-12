# Mô hình hóa bài toán RL cho Semantic-Aware Bitrate Control

Thư mục này xây dựng một môi trường Reinforcement Learning offline để huấn
luyện bộ điều khiển bitrate/độ phân giải cho video encoder. Mục tiêu là chọn
cấu hình mã hóa phù hợp với băng thông mạng và độ quan trọng ngữ nghĩa của
từng frame/segment, sao cho chất lượng cảm nhận cao nhưng không dùng bitrate
quá lãng phí.

Bài toán được mô hình hóa dưới dạng Markov Decision Process (MDP) và hiện
được giải bằng Double DQN với replay buffer.

## 1. Bài toán điều khiển

Ở mỗi bước thời gian `t`, môi trường đọc một dòng trong trace
`rl_states.jsonl`, gồm các thông tin như:

- `bandwidth`: băng thông mạng hiện tại, đơn vị kbps.
- `semantic_score`: điểm quan trọng ngữ nghĩa của frame.
- `motion`, `roi_area`, `num_objects`, `priority`: đặc trưng nội dung dùng để
  tạo semantic score hoặc mô phỏng độ phức tạp mã hóa.

Agent quan sát trạng thái, chọn một action mã hóa, môi trường mô phỏng encoder
VCU/surrogate, sau đó trả về reward dựa trên VMAF và chi phí bitrate.

Luồng chính:

```text
YOLO metadata + bandwidth trace
        |
        v
rl_states.jsonl
        |
        v
VCUSimEnv: state -> action
                       |
              encode grid (nếu có)
              hoặc hàm mô phỏng
                       |
                       v
              bitrate/VMAF -> reward
        |
        v
Double DQN policy
```

## 2. Mô hình MDP

### State

State trong `env.py` là vector 3 chiều đã chuẩn hóa về `[0, 1]`:

```text
s_t = [
  bandwidth_t / max_bitrate,
  prev_bitrate_t / max_bitrate,
  semantic_score_t / SEMANTIC_SCORE_MAX
]
```

Ý nghĩa:

- `bandwidth_t`: điều kiện mạng tại thời điểm hiện tại.
- `prev_bitrate_t`: bitrate thực tế của bước trước, giúp agent nhận biết quán
  tính lựa chọn mã hóa.
- `semantic_score_t`: mức độ quan trọng của nội dung hình ảnh.

`prev_bitrate_t` không phải là một cột đọc từ `rl_states.jsonl`. Môi trường
khởi tạo giá trị này khi `reset()`, sau đó cập nhật nó bằng `actual_bitrate`
của action ở bước `t-1`. Tương tự, VMAF không nằm trong state hiện tại. VMAF
là phản hồi sau action và được dùng để tính reward.

Vì vậy encode grid không cung cấp trực tiếp ba phần tử của state. Vai trò của
nó là cung cấp đường rate-distortion để môi trường xác định VMAF sau khi agent
chọn action. Không truyền `--encode-grid-path` vẫn train được, nhưng reward khi
đó dựa trên `_simulate_encode()` thay vì số đo encode offline.

Trong code:

- `STATE_DIM = 3`
- `SEMANTIC_SCORE_MAX = 30.0`
- `max_bitrate_kbps` mặc định là `8000.0`

### Action

Action là rời rạc để phù hợp với DQN. Mỗi action là một tổ hợp:

```text
a_t = (bitrate_ratio, resolution_idx, roi_idx)
```

Trong đó:

- `bitrate_ratio` thuộc `{0.50, 0.75, 0.95}`
- `resolution_idx` thuộc `{0, 1, 2}`
- `resolution_idx = 0`: `640x480`
- `resolution_idx = 1`: `1280x720`
- `resolution_idx = 2`: `1920x1080`
- `roi_idx` thuộc `{0, 1, 2}`:
  - `roi_idx = 0`: `qoffset = 0.0`, không ưu tiên ROI.
  - `roi_idx = 1`: `qoffset = -0.3`, ưu tiên ROI mức vừa.
  - `roi_idx = 2`: `qoffset = -0.6`, ưu tiên ROI mức mạnh.

`qoffset` âm làm vùng ROI được nén nhẹ hơn nền. Bounding box ROI lấy từ
`detections[].bbox` trong YOLO metadata.

Tổng số action:

```text
|A| = 3 x 3 x 3 = 27
```

Bitrate mục tiêu được tính theo:

```text
target_bitrate_t = bitrate_ratio x bandwidth_t
```

Trước khi đưa vào mô phỏng encoder, action được "sanitize" trong
`_sanitize_action()` để tránh lựa chọn không thực tế:

- Nếu băng thông quá thấp, giới hạn độ phân giải và bitrate ratio.
- Nếu semantic score cao và mạng đủ tốt, không hạ xuống độ phân giải quá thấp.
- Độ phân giải và ROI level chỉ đổi theo cấp segment/GOP, không đổi liên tục
  từng frame.

### Transition

Transition không được học bằng mô hình riêng mà được xác định bởi trace và
surrogate encoder:

```text
s_t, a_t -> encoder surrogate -> r_t, s_{t+1}
```

Môi trường tăng `t` sang frame kế tiếp. Khi hết trace, episode kết thúc nếu
`loop=False`.

Nếu có `encode_grid_path`, môi trường dùng đường rate-distortion đo offline
của đúng frame và resolution:

```text
frame x resolution x bitrate_level x ROI_level -> measured bitrate, VMAF
```

Khi action chọn `bitrate_ratio x bandwidth`, môi trường nội suy VMAF trên các
điểm bitrate đo được của frame đó. `actual_bitrate` mà môi trường trả về hiện
được lấy từ target bitrate đã giới hạn, còn các giá trị bitrate đo trong grid
là các trục dùng để nội suy VMAF. Đường RD được chọn bằng `roi_idx` trong
action, nên agent có thể học khi nào nên dành chất lượng cho ROI.

Grid mới lưu hai metric:

- `vmaf`: VMAF toàn frame.
- `roi_vmaf`: trung bình VMAF theo diện tích của từng bounding box YOLO trong
  frame.

QoE dùng cho reward là trung bình có trọng số giữa hai metric. Trọng số ROI
tăng theo `semantic_score` và `roi_area`:

```text
roi_weight = semantic_norm x clip(3 x roi_area, 0, 0.75)
quality_vmaf = (1 - roi_weight) x vmaf + roi_weight x roi_vmaf
```

Nhờ vậy ROI encoding chỉ có lợi khi chất lượng vùng semantic thực sự tăng,
thay vì thưởng cố định chỉ vì agent chọn `roi_idx > 0`.

Vì vậy các frame/segment ít chuyển động và nhiều chuyển động có thể có đường
RD khác nhau dù cùng bitrate. Nếu không có grid, môi trường dùng hàm mô phỏng
`_simulate_encode()` dựa trên bitrate, độ phân giải, motion, ROI area và
semantic score.

### Đồng bộ RL trace với encode grid

`rl_states.jsonl` và encode grid là hai loại file khác nhau:

- `rl_states.jsonl` là JSON Lines, mỗi dòng có `frame_idx`, `bandwidth`,
  `semantic_score` và các đặc trưng nội dung.
- File do `encode_grid.py` tạo là một object JSON duy nhất. Mỗi khóa như
  `res0_br0_roi0` chứa một mảng cell theo thứ tự frame. Cell không cần lưu
  lại `frame_idx`: chỉ số của cell trong mảng chính là frame index.

Ở bước `t`, `env.py` lấy:

```text
frame_idx = rl_states[t]["frame_idx"]
cell = grid[key][frame_idx]
```

`encode_grid.py` encode độc lập từng frame và gán một `addroi` riêng cho mỗi
bbox của đúng frame đó. Do đó việc đồng bộ là theo `frame_idx`, không phải ghép
JSON theo một field nằm trong từng grid cell.

Để đồng bộ đúng, encode grid và semantic trace phải được tạo từ cùng một
video, cùng cách đánh số frame từ `0`, và số frame phải khớp. Môi trường kiểm
tra schema, độ dài các curve và biên `frame_idx`; dữ liệu lệch sẽ báo lỗi thay
vì quay vòng sang frame khác.

### Reward

Reward hiện tại chia trọng số 60/40 giữa QoE và chi phí bitrate. `VMAF_t` ở
đây là VMAF sau khi action đã được áp dụng, tức tương ứng với
`bitrate_ratio x bandwidth`, `resolution_idx` và `roi_idx` của action:

```text
r_t = 0.6 x QoE_t - 0.4 x bitrate_cost_t
```

Trong code:

```python
qoe_norm = clip(vmaf_after_action / 100.0, 0.0, 1.0)
bitrate_cost = actual_bitrate / max_bitrate
reward = 0.6 * qoe_norm - 0.4 * bitrate_cost
```

Ý nghĩa:

- Agent được thưởng khi VMAF cao.
- Agent bị phạt nếu dùng bitrate lớn.
- `semantic_score` không cộng trực tiếp vào reward; nó ảnh hưởng gián tiếp qua
  state, action sanitation và mô phỏng độ phức tạp/chất lượng.

Các đại lượng `latency` và `power` hiện được trả về trong `info` để đánh giá,
nhưng chưa đưa trực tiếp vào reward. Nếu muốn mở rộng, có thể thêm penalty:

```text
r_t = w_q x VMAF_norm
      - w_b x bitrate_cost
      - w_l x max(0, latency - latency_budget)
      - w_p x power_cost
```

## 3. Semantic score

Semantic score được tính từ metadata YOLO theo công thức:

```text
S = alpha x num_obj
    + beta x sum(Area_i)
    + gamma x motion_avg
    + delta x sum(Priority_i)
```

Trong đó:

- `num_obj`: số object được phát hiện.
- `Area_i`: diện tích ROI/bounding box theo tỷ lệ khung hình.
- `motion_avg`: chuyển động trung bình ước lượng bằng nearest-centroid matching.
- `Priority_i`: trọng số ưu tiên theo class, ví dụ person > car/bus/truck >
  object phụ.

Script liên quan:

- `yolo_metadata_to_rl.py`: tạo semantic trace từ `yolo_metadata.jsonl`.
- `gen_bandwidth_trace.py`: tạo hoặc resample bandwidth trace.
- `merge_traces()`: ghép semantic trace và bandwidth trace thành
  `rl_states.jsonl`.

## 4. Thuật toán học

`train.py` dùng Double DQN:

- Mạng Q nhận state 3 chiều và xuất Q-value cho 27 action.
- Agent chọn action bằng epsilon-greedy.
- Replay buffer lưu `(state, action, reward, next_state, done)`.
- Target network được cập nhật định kỳ.
- Loss là Smooth L1 loss giữa `Q(s,a)` và Double DQN target.

Target:

```text
y = r + gamma x Q_target(s', argmax_a Q_online(s', a))
```

Checkpoint lưu:

- `model_state_dict`
- `target_model_state_dict`
- `optimizer_state_dict`
- `state_dim`
- `num_actions`
- danh sách action
- replay buffer
- reward theo episode

## 5. Cài đặt

Từ thư mục repo:

```bash
cd rl_env_offline
python3 -m venv venv
source venv/bin/activate
pip install torch numpy gymnasium
```

Nếu cần vẽ biểu đồ hội tụ khi đánh giá:

```bash
pip install matplotlib
```

Nếu chạy pipeline YOLO/encode grid đầy đủ, cần thêm các gói ngoài như
`opencv-python`, `pandas`, `seaborn`, `pyyaml`, `tqdm` và ffmpeg có `libvmaf`.

## 6. Chạy nhanh với trace mô phỏng

Khi chưa có dữ liệu thật, yêu cầu trace synthetic một cách tường minh:

```bash
python3 train.py --synthetic --episodes 300 --output dqn_policy.pt
```

Đánh giá policy:

```bash
python3 evaluate.py --policy dqn_policy.pt
```

`evaluate.py` so sánh policy đã học với:

- Random policy.
- CBR cố định ở action gần `bitrate_ratio=0.75`, `resolution=720p`,
  `roi_idx=0`.

## 7. Chạy với dữ liệu thật/offline

Ví dụ pipeline khuyến nghị:

```bash
# 1. Tạo semantic trace từ YOLO metadata
python3 yolo_metadata_to_rl.py \
  --input ../outputs/metadata/yolo_metadata.jsonl \
  --out ../outputs/metadata/semantic_trace.jsonl

# 2. Tạo bandwidth trace synthetic hoặc resample trace mạng thật
python3 gen_bandwidth_trace.py \
  --num-frames 1500 \
  --fps 25 \
  --out ../outputs/metadata/bandwidth_trace.jsonl \
  --levels 800,2500,5000,8000 \
  --seed 42

# 3. Ghép semantic + bandwidth thành RL trace
python3 -c "from gen_bandwidth_trace import merge_traces; merge_traces('../outputs/metadata/semantic_trace.jsonl', '../outputs/metadata/bandwidth_trace.jsonl', '../outputs/metadata/rl_states.jsonl')"

# 4. Tạo encode grid có các đường RD cho 3 ROI level
python3 encode_grid.py \
  --input ../dataset/videos/Vehicle-Dataset-Sample-2_720p.mp4 \
  --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
  --out ../outputs/metadata/vcu_encode_grid.json \
  --bitrate-levels 600,900,1500,2500,4000,6000 \
  --workers 4 \
  --metrics vmaf

# 5. Train DQN trên trace và encode grid
python3 train.py \
  --trace-path ../outputs/metadata/rl_states.jsonl \
  --encode-grid-path ../outputs/metadata/vcu_encode_grid.json \
  --episodes 300 \
  --output dqn_policy.pt

# 6. Đánh giá trên cùng trace/grid
python3 evaluate.py \
  --trace ../outputs/metadata/rl_states.jsonl \
  --encode-grid ../outputs/metadata/vcu_encode_grid.json \
  --policy dqn_policy.pt
```

Có thể bỏ `--encode-grid-path` để smoke test bằng `_simulate_encode()`. Chế độ
này có mô hình xấp xỉ ROI quality gain, nhưng nên dùng encode grid có VMAF khi
train kết quả chính thức.

Xuất video thật theo policy:

```bash
python3 export_encoded_video.py \
  --input ../dataset/videos/Vehicle-Dataset-Sample-2_720p.mp4 \
  --policy dqn_policy.pt \
  --trace ../outputs/metadata/rl_states.jsonl \
  --yolo-metadata ../outputs/metadata/yolo_metadata.jsonl \
  --out ../outputs/exported_policy_video.mp4 \
  --segments-report ../outputs/exported_policy_segments.json
```

Khi `roi_idx > 0`, exporter giữ từng bounding box YOLO riêng biệt, scale về
resolution đã chọn và tạo một filter FFmpeg `addroi` cho mỗi ROI với qoffset
tương ứng.
Nếu đoạn không có bounding box thì không thể áp dụng ROI và report sẽ ghi
`"roi_applied": false`.

Vì tọa độ `addroi` của FFmpeg CLI là tĩnh trong một lần gọi filter, exporter
áp dụng toàn bộ danh sách ROI thu thập trong segment cho mọi frame của segment.
Muốn ROI chuyển động chính xác trong một GOP liên-frame cần gắn side-data ROI
vào từng `AVFrame` qua API/GStreamer. Encode grid không dùng xấp xỉ này vì mỗi
job của grid chỉ chứa đúng một frame.

`evaluate.py` cũng in tỷ lệ `roi0/roi1/roi2` thực sự được áp dụng sau bước
sanitize để kiểm tra policy có đang sử dụng ROI hay không.

Checkpoint 9-action cũ không tương thích với môi trường 27-action này; phải
train lại policy. Encode grid cũ không có `roi_vmaf` cũng phải tạo lại bằng
`encode_grid.py`; môi trường sẽ báo lỗi nếu policy chọn ROI nhưng grid thiếu
metric này.

## 8. Cấu trúc file chính

| File | Vai trò |
|---|---|
| `env.py` | Định nghĩa MDP: state, action, transition, reward và surrogate encoder |
| `train.py` | Huấn luyện Double DQN, lưu checkpoint và reward history |
| `evaluate.py` | Kiểm tra hội tụ và so sánh với Random/CBR baseline |
| `export_encoded_video.py` | Encode video thật theo bitrate, resolution và ROI level policy chọn |
| `yolo_metadata_to_rl.py` | Tính semantic score từ YOLO metadata |
| `gen_bandwidth_trace.py` | Sinh/resample bandwidth trace và ghép trace |
| `encode_grid.py` | Tạo encode grid offline có bitrate/VMAF để reward thực tế hơn |
| `dqn_policy.pt` | Checkpoint policy sau huấn luyện |
| `dqn_policy.rewards.json` | Reward theo episode |
| `eval_convergence.png` | Biểu đồ hội tụ khi đánh giá |

## 9. Hướng mở rộng

Các hướng mở rộng tự nhiên cho bài toán:

- Đưa `latency` và `power` vào reward để thành bài toán multi-objective.
- Thêm state như `prev_vmaf`, `prev_latency`, buffer occupancy hoặc packet loss.
- Gắn side-data ROI động theo từng `AVFrame` trong pipeline export để vẫn giữ
  dự đoán liên-frame.
- Dùng trace mạng thật thay vì synthetic bandwidth.
- Dùng encode grid đo từ pipeline VCU thật để giảm sai lệch giữa simulation và
  triển khai trên phần cứng.
