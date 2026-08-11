# Lý thuyết GStreamer cho pipeline mã hóa video ngữ nghĩa

## 1. GStreamer là gì?

GStreamer là một framework xử lý multimedia theo mô hình **dataflow**. Một ứng
dụng được xây dựng bằng cách nối nhiều phần tử nhỏ thành một pipeline. Mỗi phần
tử đảm nhiệm một công việc cụ thể, ví dụ đọc file, giải mã, biến đổi khung hình,
mã hóa hoặc truyền dữ liệu qua mạng.

Ví dụ tối giản:

```text
filesrc -> demuxer -> decoder -> converter -> encoder -> muxer -> filesink
```

Mô hình này phù hợp với bài toán semantic video encoding vì khâu suy luận,
policy RL và gắn ROI có thể được đặt giữa decoder và encoder mà không phải tự
xây dựng toàn bộ hệ thống đọc, đồng bộ và đóng gói video.

## 2. Các khái niệm nền tảng

### 2.1 Element và plugin

**Element** là đơn vị xử lý cơ bản. Element thường thuộc một trong bốn nhóm:

- **Source** tạo dữ liệu, ví dụ `filesrc` hoặc `rtspsrc`.
- **Filter/transform** biến đổi dữ liệu, ví dụ `videoconvert` và `videoscale`.
- **Codec** nén hoặc giải nén dữ liệu, ví dụ `x264enc` và decoder H.264.
- **Sink** nhận dữ liệu cuối, ví dụ `filesink`, `appsink` hoặc một video sink.

Element được cung cấp bởi các plugin. Vì vậy, việc cài GStreamer không đảm bảo
mọi element đều có sẵn. Có thể kiểm tra một element bằng:

```bash
gst-inspect-1.0 x264enc
```

### 2.2 Pad và liên kết

Element trao đổi dữ liệu qua **pad**:

- `src pad` phát dữ liệu ra;
- `sink pad` nhận dữ liệu vào.

Hai pad chỉ liên kết được khi định dạng mà chúng hỗ trợ tương thích. Một số pad
tồn tại ngay khi element được tạo; một số khác là **dynamic pad**, chỉ xuất hiện
sau khi demuxer hoặc decoder nhận biết stream. Với dynamic pad, ứng dụng phải
xử lý tín hiệu `pad-added` rồi mới nối phần còn lại của pipeline.

### 2.3 Bin và pipeline

**Bin** gom nhiều element thành một khối logic. **Pipeline** là bin cấp cao nhất,
chịu trách nhiệm quản lý clock, trạng thái và luồng message. Việc gom decoder,
inference hoặc encoder vào bin giúp thay backend PC bằng backend ZCU106 mà
không làm thay đổi phần điều khiển ở cấp ứng dụng.

### 2.4 Buffer và metadata

Một `GstBuffer` chứa payload của một đơn vị dữ liệu, chẳng hạn một video frame
đã giải mã hoặc một access unit H.264. Buffer còn mang:

- timestamp (`PTS`, `DTS`);
- thời lượng;
- cờ trạng thái;
- metadata bổ sung.

Bounding box, class, confidence và ROI nên được gắn với đúng buffer bằng
metadata. Cách này giữ quan hệ giữa kết quả inference và frame tốt hơn việc chỉ
trao đổi qua một biến toàn cục hoặc một danh sách không có timestamp.

## 3. Caps và quá trình thương lượng định dạng

**Caps** mô tả kiểu dữ liệu đi qua pad. Đối với raw video, caps thường chứa:

```text
video/x-raw, format=NV12, width=1280, height=720, framerate=30/1
```

Khi pipeline chuyển sang trạng thái chạy, các element thực hiện **caps
negotiation** để thống nhất định dạng. `videoconvert`, `videoscale` và
`capsfilter` thường được dùng để tạo một giao điểm hợp lệ giữa decoder,
inference và encoder.

Trên thiết bị tăng tốc, caps còn có thể biểu diễn loại bộ nhớ. Một pipeline có
thể đúng về pixel format nhưng vẫn thất bại nếu element trước xuất system memory
trong khi element sau chỉ nhận DMA/device memory. Vì vậy cần kiểm tra đồng thời:

- pixel format, kích thước và framerate;
- memory feature mà plugin phần cứng yêu cầu;
- khả năng zero-copy giữa các element.

Mỗi lần thay độ phân giải có thể kích hoạt renegotiation. Encoder phần cứng hoặc
muxer không nhất thiết hỗ trợ thay đổi caps giữa một stream đang chạy. Trong
trường hợp đó, đóng segment hiện tại và tạo segment mới là phương án an toàn.

## 4. Trạng thái của pipeline

GStreamer có bốn trạng thái chính:

```text
NULL -> READY -> PAUSED -> PLAYING
```

- `NULL`: chưa giữ tài nguyên.
- `READY`: đã giữ tài nguyên cơ bản nhưng chưa truyền dữ liệu.
- `PAUSED`: đã preroll; pipeline chuẩn bị frame đầu tiên.
- `PLAYING`: dữ liệu được xử lý theo clock.

Chuyển trạng thái có thể diễn ra bất đồng bộ. Ứng dụng không nên giả định rằng
lời gọi `set_state()` hoàn tất ngay lập tức. Khi kết thúc hoặc gặp lỗi, pipeline
cần trở về `NULL` để giải phóng file handle, buffer và tài nguyên codec/DPU.

## 5. Clock, timestamp và độ trễ

Pipeline dùng một clock chung để các stream chạy đúng thời gian. `PTS` cho biết
khi nào một buffer được trình bày; `DTS` cho biết thứ tự giải mã khi codec có
frame tham chiếu hoặc B-frame.

Đối với file processing, pipeline có thể chạy nhanh hơn thời gian thực nếu sink
không đồng bộ với clock. Đối với camera hoặc RTP, timestamp và latency quyết
định trực tiếp độ trễ end-to-end.

Các nguồn gây trễ quan trọng trong pipeline semantic encoding gồm:

1. decode và chuyển đổi màu;
2. gom batch hoặc chờ DPU inference;
3. hàng đợi giữa inference, policy và encoder;
4. lookahead, B-frame và GOP của encoder;
5. muxing, network jitter buffer và sink synchronization.

Giảm queue hoặc tắt B-frame có thể giảm độ trễ nhưng thường làm giảm hiệu suất
nén. Đây là một đánh đổi cần được đo, không nên suy ra chỉ từ bitrate mục tiêu.

## 6. Threading, queue và backpressure

Nếu các element được nối trực tiếp, chúng thường xử lý trong cùng streaming
thread. `queue` tạo ranh giới thread và cho phép các stage chạy song song:

```text
decode -> queue -> inference -> queue -> encoder
```

Khi stage sau chậm hơn stage trước, buffer tích tụ và tạo **backpressure**. Với
file input, điều này thường chỉ làm pipeline chạy chậm. Với live input, nó làm
latency tăng liên tục nếu không giới hạn queue.

Một thiết kế live cần xác định rõ chính sách quá tải:

- chặn nguồn để không mất frame;
- giới hạn queue rồi bỏ frame cũ;
- bỏ frame inference nhưng vẫn encode video;
- dùng kết quả semantic gần nhất cho một số frame tiếp theo.

Không có chính sách đúng cho mọi bài toán. Điều quan trọng là frame, detection
và quyết định RL vẫn được ghép đúng bằng frame index hoặc timestamp.

## 7. Bus, message và EOS

Pipeline gửi message lên **bus**. Ứng dụng cần xử lý ít nhất:

- `ERROR`: lỗi kèm thông tin debug;
- `WARNING`: cảnh báo có thể ảnh hưởng kết quả;
- `EOS`: dữ liệu đầu vào đã kết thúc;
- `STATE_CHANGED`: hữu ích khi chẩn đoán chuyển trạng thái.

Với file output, nhận `EOS` trước khi đưa pipeline về `NULL` rất quan trọng vì
encoder và muxer cần flush dữ liệu, ghi index và hoàn tất container. Dừng pipeline
đột ngột có thể tạo MP4 không đọc được dù phần lớn frame đã được encode.

## 8. Appsink, appsrc và probe

Có ba cách thường dùng để kết nối thuật toán Python/C++ với pipeline:

### Appsink

`appsink` đưa buffer từ pipeline sang mã ứng dụng. Cách này thuận tiện cho mô
hình AI nhưng có thể phát sinh copy và cần kiểm soát queue để tránh tăng latency.

### Appsrc

`appsrc` đưa dữ liệu do ứng dụng tạo trở lại GStreamer. Khi ghép `appsink` và
`appsrc`, ứng dụng phải bảo toàn caps, timestamp, duration và quy tắc EOS.

### Pad probe hoặc custom element

Pad probe quan sát hoặc bổ sung metadata mà không nhất thiết tách pipeline.
Custom element phù hợp hơn cho production vì có thể mô tả caps, quản lý buffer
và tích hợp scheduler của GStreamer một cách rõ ràng. Trên ZCU106, semantic ROI
bridge nên tiến tới custom element hoặc adapter native nếu mục tiêu là zero-copy
và độ trễ ổn định.

## 9. Mã hóa thích nghi và ROI

Policy trong repository chọn ba thành phần action:

```text
(bitrate ratio, resolution index, ROI qoffset level)
```

Luồng điều khiển khái niệm là:

```text
frame + network state
        |
        v
inference -> semantic score -> RL policy
                              |
                              v
                 bitrate / resolution / ROI
                              |
                              v
                           encoder
```

Bitrate và resolution thường là thuộc tính cấu hình encoder/caps. ROI khác ở
chỗ nó là dữ liệu theo frame hoặc theo vùng: mỗi ROI mô tả tọa độ và mức thay
đổi chất lượng tương đối. Giá trị qoffset âm thường ưu tiên chất lượng cho vùng
quan trọng, nhưng ý nghĩa chính xác và miền giá trị phụ thuộc encoder/plugin.

Metadata ROI của GStreamer không tự đảm bảo encoder sẽ sử dụng nó. Cần kiểm tra:

1. plugin encoder có đọc loại ROI metadata đó hay không;
2. tọa độ ROI đang dùng resolution trước hay sau `videoscale`;
3. encoder yêu cầu alignment theo macroblock/CTU như thế nào;
4. qoffset được ánh xạ sang QP delta hoặc QP-map ra sao;
5. nhiều ROI chồng nhau được hợp nhất theo quy tắc nào.

Backend PC dùng `x264enc` chủ yếu để kiểm chứng luồng decode, policy, segment và
container. Nó không chứng minh hành vi ROI của VCU trên ZCU106. Kết quả phần
cứng cần được xác nhận bằng bitrate thực, chất lượng trong/ngoài ROI, latency và
log của encoder VCU.

## 10. Ánh xạ vào repository

Trong repository này, `gstreamer_edge_pipeline.py` đóng vai trò runner:

- backend `sim` dùng GStreamer và `x264enc` trên PC;
- backend `zcu106` sinh template để tích hợp VVAS/DPU/VCU;
- policy DQN và trace quyết định bitrate, resolution và ROI level;
- report JSON lưu action/ROI để kiểm tra và làm hợp đồng cho bridge phần cứng.

Kiến trúc mục tiêu trên board có thể biểu diễn như sau:

```text
source -> demux/decode -> VVAS inference -> semantic ROI bridge
       -> caps/scale -> VCU encoder -> parser/muxer hoặc RTP payloader -> sink
```

`semantic ROI bridge` cần thực hiện bốn việc: đọc detection gắn với frame, tạo
observation đúng như môi trường huấn luyện, chạy policy, và chuyển action sang
caps/property/ROI metadata mà phiên bản VVAS cụ thể hỗ trợ.

## 11. Nguyên tắc kiểm thử

Nên kiểm thử theo từng lớp để xác định lỗi nhanh:

1. Kiểm tra plugin và caps bằng `gst-inspect-1.0`.
2. Chạy pipeline decode-to-sink không có inference.
3. Thêm inference và xác nhận detection khớp timestamp/frame.
4. Thêm policy nhưng chỉ ghi action ra report.
5. Bật thay đổi bitrate/resolution và kiểm tra từng segment.
6. Bật ROI, so sánh cả chất lượng lẫn bitrate với baseline không ROI.
7. Chạy dài hạn để phát hiện memory leak, queue growth và timestamp drift.

Các chỉ số nên ghi gồm throughput, end-to-end latency, queue level, dropped
frames, bitrate đo được, kích thước segment, VMAF/PSNR và chất lượng riêng trong
ROI. Chỉ khi action, metadata và output được liên kết bằng cùng frame index hoặc
timestamp thì mới có thể đánh giá đúng tác động của policy.

## 12. Kết luận

GStreamer giải quyết phần luân chuyển multimedia, nhưng hiệu quả của pipeline
semantic encoding phụ thuộc vào ba hợp đồng phải nhất quán: hợp đồng định dạng
(caps và memory), hợp đồng thời gian (timestamp/frame identity), và hợp đồng
điều khiển (cách action RL được ánh xạ sang encoder). Thiết kế và đo ba hợp đồng
này riêng rẽ giúp chuyển từ mô phỏng PC sang ZCU106 với ít sai lệch hơn.
