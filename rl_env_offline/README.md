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
VCUSimEnv: state -> action -> encode surrogate -> reward
        |
        v
Double DQN policy
```

## 2. Mô hình MDP

### State

State trong `env.py` là vector 5 chiều đã chuẩn hóa về `[0, 1]`:

```text
s_t = [
  bandwidth_t / max_bitrate,
  previous_vmaf_t / 100,
  semantic_score_t / SEMANTIC_SCORE_MAX,
  previous_bitrate_t / max_bitrate,
  current_resolution_idx / (num_resolutions - 1)
]
```

Ý nghĩa:

- `bandwidth_t`: điều kiện mạng tại thời điểm hiện tại.
- `previous_vmaf_t`: chất lượng đã quan sát được ở bước trước, đóng vai trò
  thành phần `Q` trong đề xuất `[B, Q, S]`.
- `semantic_score_t`: mức độ quan trọng của nội dung hình ảnh.
- `previous_bitrate_t`: bitrate thực tế của bước trước, giúp agent nhận biết
  quán tính lựa chọn mã hóa.
- `current_resolution_idx`: độ phân giải hiện tại, để policy học chi phí đổi
  cấu hình tuần tự.

Trong code:

- `STATE_DIM = 5`
- `SEMANTIC_SCORE_MAX = 30.0`
- `max_bitrate_kbps` mặc định là `8000.0`

### Action

Action là rời rạc để phù hợp với DQN. Mỗi action là một tổ hợp:

```text
a_t = (bitrate_ratio, resolution_idx)
```

Trong đó:

- `bitrate_ratio` thuộc `{0.10, 0.25, 0.50, 0.75, 0.95}`
- `resolution_idx` thuộc `{0, 1, 2}`
- `resolution_idx = 0`: `640x480`
- `resolution_idx = 1`: `1280x720`
- `resolution_idx = 2`: `1920x1080`

Tổng số action:

```text
|A| = 5 x 3 = 15
```

Bitrate mục tiêu được tính theo:

```text
target_bitrate_t = bitrate_ratio x max_bitrate
```

Như vậy `bandwidth_t` không còn tự động nhân vào action. Agent phải tự học:
khi `bandwidth_t` thấp thì chọn target bitrate thấp hơn hoặc resolution thấp
hơn; khi bandwidth cao mới dùng các action chất lượng cao hơn.

Trước khi đưa vào mô phỏng encoder, action chỉ được "sanitize" để giữ quyết
định resolution ở cấp segment/GOP, không đổi liên tục từng frame. Các quan hệ
network/semantic/rate-quality được học qua reward thay vì hard-code bằng luật.

### Transition

Transition không được học bằng mô hình riêng mà được xác định bởi trace và
surrogate encoder:

```text
s_t, a_t -> encoder surrogate -> r_t, s_{t+1}
```

Môi trường tăng `t` sang frame kế tiếp. Khi hết trace, episode kết thúc nếu
`loop=False`.

Nếu có `encode_grid_path`, môi trường lấy bitrate/VMAF từ grid encode offline.
Nếu không có, môi trường dùng hàm mô phỏng `_simulate_encode()` dựa trên độ
phân giải, motion, ROI area và semantic score.

### Reward

Reward không cộng trực tiếp `semantic_score`, vì `S` là thuộc tính nội dung chứ
không phải thành quả do agent tạo ra. Thay vào đó, `S` làm tăng trọng số của
ROI quality:

```text
Q_semantic =
  w_global x VMAF_norm
  + w_roi x semantic_norm x ROI_VMAF_norm

r_t =
  w_q x Q_semantic
  - w_b x actual_bitrate_t / max_bitrate
  - w_o x max(0, (actual_bitrate_t - bandwidth_t) / bandwidth_t)
  - w_s x switch_cost_t
```

Trong code:

```python
global_q = clip(vmaf / 100.0, 0.0, 1.0)
roi_q = clip(roi_vmaf / 100.0, 0.0, 1.0)
semantic_quality = w_global * global_q + w_roi * semantic_norm * roi_q
bitrate_cost = actual_bitrate / max_bitrate
over_bw = max(0, (actual_bitrate - bandwidth) / bandwidth)
reward = w_q * semantic_quality - w_b * bitrate_cost - w_o * over_bw - w_s * switch_cost
```

Ý nghĩa:

- Agent được thưởng khi VMAF/ROI VMAF cao.
- Cảnh semantic cao làm ROI quality đáng giá hơn, nhưng không tự sinh reward.
- Agent bị phạt nếu dùng bitrate lớn.
- Agent bị phạt mạnh nếu bitrate thực tế vượt bandwidth.
- Agent bị phạt nếu đổi bitrate quá mạnh hoặc đổi resolution quá thường xuyên.

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

- Mạng Q nhận state 5 chiều và xuất Q-value cho 15 action.
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

Khi chưa có dữ liệu thật, `VCUSimEnv` tự tạo synthetic trace để smoke test:

```bash
python3 train.py --episodes 300 --output dqn_policy.pt
```

Đánh giá policy:

```bash
python3 evaluate.py --policy dqn_policy.pt
```

`evaluate.py` so sánh policy đã học với:

- Random policy.
- CBR cố định ở action gần `bitrate_ratio=0.75`, `resolution=720p`.

Ngoài bảng baseline, `evaluate.py` còn in các kiểm tra counterfactual để chứng
minh policy đã học đúng 4 quan hệ:

- `Network -> cấu hình`: giữ semantic và history cố định, thay bandwidth
  `{600, 1500, 3500, 6500}` kbps, rồi quan sát target bitrate/resolution mà
  policy chọn.
- `Semantic -> phân bổ tài nguyên`: giữ bandwidth cố định, thay semantic thấp
  và cao theo percentile của trace, rồi kiểm tra bitrate/resolution có tăng cho
  cảnh quan trọng hay không.
- `Rate-quality`: bảng đánh giá in `VMAF TB` và `ROI VMAF` để xem mức bitrate
  policy chọn có tạo chất lượng tốt hơn baseline không.
- `Quyết định tuần tự`: bảng đánh giá in `Vuot BW` và `Doi res`; policy tốt
  phải giảm vượt bandwidth và không đổi resolution liên tục.

## 7. Chạy với dữ liệu thật/offline

Ví dụ pipeline khuyến nghị:

```bash
# 1. Tạo semantic trace từ YOLO metadata
python3 yolo_metadata_to_rl.py \
  --input outputs/metadata/yolo_metadata.jsonl \
  --out outputs/metadata/semantic_trace.jsonl

# 2. Tạo bandwidth trace synthetic hoặc resample trace mạng thật
python3 gen_bandwidth_trace.py \
  --num-frames 1500 \
  --fps 25 \
  --out outputs/metadata/bandwidth_trace.jsonl \
  --levels 800,2500,5000,8000 \
  --seed 42

# 3. Ghép semantic + bandwidth thành RL trace
python3 -c "from gen_bandwidth_trace import merge_traces; merge_traces('outputs/metadata/semantic_trace.jsonl', 'outputs/metadata/bandwidth_trace.jsonl', 'outputs/metadata/rl_states.jsonl')"

# 4. Train DQN trên trace
python3 train.py \
  --trace-path outputs/metadata/rl_states.jsonl \
  --episodes 300 \
  --output dqn_policy.pt

# 5. Đánh giá
python3 evaluate.py \
  --trace outputs/metadata/rl_states.jsonl \
  --policy dqn_policy.pt
```

Nếu đã có encode grid offline chứa bitrate/VMAF:

```bash
python3 train.py \
  --trace-path outputs/metadata/rl_states.jsonl \
  --encode-grid-path outputs/metadata/vcu_encode_grid_roi.json \
  --episodes 300 \
  --output dqn_policy.pt

python3 evaluate.py \
  --trace outputs/metadata/rl_states.jsonl \
  --encode-grid outputs/metadata/vcu_encode_grid_roi.json \
  --policy dqn_policy.pt
```

## 8. Cấu trúc file chính

| File | Vai trò |
|---|---|
| `env.py` | Định nghĩa MDP: state, action, transition, reward và surrogate encoder |
| `train.py` | Huấn luyện Double DQN, lưu checkpoint và reward history |
| `evaluate.py` | Kiểm tra hội tụ và so sánh với Random/CBR baseline |
| `yolo_metadata_to_rl.py` | Tính semantic score từ YOLO metadata |
| `gen_bandwidth_trace.py` | Sinh/resample bandwidth trace và ghép trace |
| `encode_grid.py` | Tạo encode grid offline có bitrate/VMAF để reward thực tế hơn |
| `dqn_policy.pt` | Checkpoint policy sau huấn luyện |
| `dqn_policy.rewards.json` | Reward theo episode |
| `eval_convergence.png` | Biểu đồ hội tụ khi đánh giá |

## 9. Hướng mở rộng

Các hướng mở rộng tự nhiên cho bài toán:

- Đưa `latency` và `power` vào reward để thành bài toán multi-objective.
- Thêm state như `prev_latency`, buffer occupancy hoặc packet loss.
- Mở rộng action sang QP/ROI strength nếu muốn điều khiển ROI encoding trực tiếp.
- Dùng trace mạng thật thay vì synthetic bandwidth.
- Dùng encode grid đo từ pipeline VCU thật để giảm sai lệch giữa simulation và
  triển khai trên phần cứng.
