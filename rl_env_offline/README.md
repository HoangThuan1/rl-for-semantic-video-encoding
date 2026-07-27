# Môi trường RL offline cho Semantic-Aware Bitrate Control (ZCU106)

Bộ code này dùng cho giai đoạn **Tháng 5–6** trong roadmap: xây môi trường RL
mô phỏng, thu thập dữ liệu state-action, huấn luyện offline một policy DQN
ban đầu. Sau này (Tháng 9–10) bạn sẽ port policy này sang C++ (LibTorch) để
chạy trên ARM Cortex-A53 của ZCU106 và tinh chỉnh online.

Không cần biết Python trước — file `env.py` và `train.py` có chú thích so
sánh với C ở những chỗ cú pháp khác biệt.

---

## 1. Cài đặt (chỉ làm 1 lần)

### Bước 1: Cài Python
Kiểm tra đã có Python chưa:
```bash
python3 --version
```
Nếu chưa có (Windows): tải tại https://www.python.org/downloads/ (chọn bản
3.10 hoặc 3.11, nhớ tick "Add Python to PATH" lúc cài).
Nếu dùng Linux/WSL: `sudo apt install python3 python3-pip python3-venv`

### Bước 2: Tạo môi trường ảo (tương đương như tạo một "workspace" riêng,
để các thư viện không đụng nhau giữa các dự án — giống việc bạn tạo folder
project riêng cho mỗi bài tập C):
```bash
python3 -m venv venv
```
Kích hoạt:
- Linux/macOS: `source venv/bin/activate`
- Windows (PowerShell): `venv\Scripts\Activate.ps1`

Sau khi kích hoạt, dòng lệnh sẽ có chữ `(venv)` ở đầu — nghĩa là đang ở
trong workspace riêng.

### Bước 3: Cài thư viện cần thiết
```bash
pip install -r requirements.txt
```
(`pip` giống như một trình quản lý thư viện, tương tự việc bạn tải file
`.h`/`.lib` về nhưng tự động hoá hoàn toàn — không cần build tay.)

---

## 2. Chạy training offline

```bash
python3 train.py
```

Script sẽ:
1. Khởi tạo môi trường mô phỏng (`env.py`)
2. Cho agent DQN tương tác với môi trường hàng nghìn lần (episode)
3. Lưu checkpoint policy vào file `dqn_policy.pt`
4. In ra reward trung bình mỗi 50 episode để bạn theo dõi độ hội tụ

Nếu reward trung bình tăng dần và ổn định theo thời gian → agent đang học
tốt. Nếu dao động mạnh không hội tụ → xem lại phần "RL không hội tụ" trong
mục rủi ro của đề xuất (giảm learning rate, giảm độ phức tạp action space).

---

## 3. Cấu trúc file

| File | Vai trò | Tương đương trong tư duy C |
|---|---|---|
| `env.py` | Định nghĩa state/action/reward, hàm mô phỏng một bước thời gian | Một file `env.c` chứa struct `State` + hàm `step()` |
| `train.py` | Vòng lặp training DQN | File `main.c` gọi `step()` lặp lại, cập nhật "bảng tra cứu" (ở đây là neural network thay cho bảng Q truyền thống) |
| `requirements.txt` | Danh sách thư viện cần cài | Giống Makefile liệt kê thư viện `.so`/`.a` cần link |
| `dqn_policy.pt` | File output — trọng số mạng đã học | Giống một file `.bin`/`.dat` chứa dữ liệu đã huấn luyện, dùng lại được |

## 4. Việc bạn cần chỉnh sửa trước khi dùng thật

Trong `env.py`, các hàm `_simulate_semantic_score()`, `_simulate_network()`,
và `_simulate_qoe()` hiện đang dùng **công thức giả lập đơn giản/ngẫu nhiên**
để bạn chạy thử ngay được. Trước khi dùng cho báo cáo thật, bạn cần thay
bằng dữ liệu thực:
- `_simulate_semantic_score()` → thay bằng số liệu thật lấy từ log DPU chạy
  offline trên video test (num_obj, motion, class) như đã bàn.
- `_simulate_network()` → thay bằng trace băng thông thật (FCC/5G trace, đọc
  từ file CSV).
- `_simulate_qoe()` → thay bằng đường cong rate-distortion đo thật (encode
  thử vài đoạn video ở nhiều mức bitrate/QP, đo PSNR/VMAF, rồi nội suy).
