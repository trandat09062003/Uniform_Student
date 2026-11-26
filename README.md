# True Form Student - Hệ Thống Nhận Diện Đồng Phục & Điểm Danh Học Sinh

Hệ thống Edge AI thời gian thực nhận diện đồng phục học sinh (áo trắng, quần xanh, khăn quàng đỏ, bảng tên) từ Webcam máy tính hoặc Camera ngoài, tự động bám vết (tracking) từng học sinh để kiểm tra vi phạm nội quy và đồng bộ sĩ số điểm danh lên **OhStem IoT Dashboard** qua giao thức MQTT.

---

## Sơ Đồ Hoạt Động

```text
Camera (Webcam Laptop / Cam Ngoài USB 1080x1920 / RTSP / Video)
      ↓
  YOLO11m + ByteTrack (Bám vết ID từng học sinh)
      ↓
┌────────────────────────────────────────┐
│   Đánh Giá Theo Chuỗi Thời Gian        │
│   (Temporal Voting + Distance Gate)    │
├───────────────────┬────────────────────┤
│ Điểm danh sĩ số   │ Kiểm tra đồng phục │
│ (OhStem MQTT)     │ (Lưu ảnh vi phạm)  │
└───────────────────┴────────────────────┘
```

---

## Tính Năng Chính

1. **Nhận diện YOLO11m & Bám vết ByteTrack**:
   * Gán ID cố định (`#1`, `#2`, ...) cho từng học sinh khi di chuyển qua camera.
   * Lọc ngưỡng tin cậy riêng cho từng loại đối tượng (`hoc_sinh >= 0.60`, `ao_trang`/`quan_xanh >= 0.50`, `the_ten`/`khan_quang >= 0.35`) giúp hạn chế nhận nhầm phông nền.
2. **Đánh giá theo chuỗi thời gian (Temporal Voting)**:
   * Tích lũy kết quả qua nhiều khung hình liên tiếp thay vì kết luận vội ở 1 frame, tránh báo sai khi học sinh xoay người hoặc bị che khuất tạm thời.
   * Hỗ trợ cổng lọc khoảng cách (`MIN_STUDENT_HEIGHT`), chỉ kiểm tra khi học sinh đi vào vùng đủ gần camera.
   * Tùy chỉnh linh hoạt quy định khăn quàng (`REQUIRE_SCARF=true` cho Cấp 1/2, `false` cho Cấp 3).
3. **Hỗ trợ cả Webcam Máy Tính & Camera Ngoài (`1080x1920`)**:
   * Hỗ trợ quét cổng camera tự động, ép độ phân giải qua DirectShow trên Windows và tự động co giãn cửa sổ hiển thị (`WINDOW_NORMAL`) để không bị tràn màn hình.
4. **Điểm danh OhStem IoT (MQTT)**:
   * Tự động đếm sĩ số và gửi lên feed cấu hình trên OhStem theo chu kỳ 3 giây hoặc khi sĩ số thay đổi.
5. **Hiển thị Tiếng Việt Unicode**:
   * Vẽ nhãn và huy hiệu trạng thái (`ĐỦ ĐỒNG PHỤC`, `VI PHẠM: ...`) bằng tiếng Việt có dấu trực tiếp trên khung hình.

---

## Cấu Trúc Thư Mục

```text
Uniform_Student/
├── main.py                     # Chương trình chạy chính thức (Camera + Tracking + OhStem)
├── .env                        # Cấu hình môi trường (tạo từ .env.example)
├── .env.example                # File cấu hình mẫu
├── requirements.txt            # Danh sách thư viện Python
├── Train_YOLO11_Colab.ipynb    # Notebook huấn luyện YOLO11m trên Google Colab
├── config/
│   └── settings.yaml           # Tham số quy tắc đồng phục và tracking
├── data/
│   ├── dataset.yaml            # Cấu hình đường dẫn dataset
│   ├── violations/             # Tự động lưu ảnh bằng chứng khi có vi phạm
│   └── test_results/           # Lưu ảnh/video kết quả khi chạy script kiểm thử
├── models/
│   └── best_uniform_model.pt   # Trọng số mô hình YOLO11m đã huấn luyện
├── scripts/
│   ├── stream_test.py          # Quét cổng camera & test luồng stream trực tiếp
│   ├── test_image.py           # Kiểm thử nhanh trên 1 file ảnh
│   ├── test_media.py           # Kiểm thử trên ảnh hoặc video
│   ├── run_video.py            # Chạy thử trên file video không kết nối IoT
│   ├── train.py                # Huấn luyện mô hình trên máy có GPU
│   └── export_tensorrt.py      # Xuất model sang TensorRT (.engine) cho Jetson/NVIDIA
└── src/
    ├── camera.py               # Đọc luồng camera đa luồng (USB / RTSP / Video lặp)
    ├── config_loader.py        # Đọc cấu hình từ .env và settings.yaml
    ├── detector.py             # Wrapper YOLO11m hỗ trợ predict & track
    ├── tracker_history.py      # Logic bám vết và đánh giá đồng phục theo thời gian
    ├── violation_checker.py    # Logic kiểm tra vi phạm đồng phục
    ├── ohstem_service.py       # Kết nối MQTT gửi dữ liệu lên OhStem
    ├── debounce.py             # Bộ đếm frame chống nhiễu
    ├── geometry.py             # Tính toán IoU và vị trí bbox
    └── visualize.py            # Vẽ bounding box và chữ Tiếng Việt Unicode
```

---

## Cài Đặt Môi Trường

### 1. Tạo môi trường ảo Python
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
pip install paho-mqtt
```

### 3. Tạo file cấu hình `.env`
Copy từ file mẫu `.env.example` sang `.env` và chỉnh sửa các thông số phù hợp:

```env
# Camera: 0 hoặc 1 tùy thứ tự cổng USB, hoặc URL rtsp:// / http://
CAMERA_SOURCE=0
CAMERA_WIDTH=1080
CAMERA_HEIGHT=1920

# Mô hình & ngưỡng nhận diện
MODEL_PATH=models/best_uniform_model.pt
CONFIDENCE_THRESHOLD=0.35
IOU_THRESHOLD=0.45

# Kích thước suy luận (640 nhanh hơn trên CPU; 960/1280 bắt vật thể nhỏ tốt hơn)
IMGSZ=640

# Khoảng cách tối thiểu (chiều cao bbox học sinh theo px) để kích hoạt kiểm tra
MIN_STUDENT_HEIGHT=250

# Yêu cầu khăn quàng đỏ (true: Cấp 1, 2; false: Cấp 3)
REQUIRE_SCARF=false

# Hiển thị cửa sổ xem trước
SHOW_PREVIEW=true

# Tích hợp OhStem IoT
OHSTEM_USERNAME=your_ohstem_username
OHSTEM_PASSWORD=
OHSTEM_FEED_ID=V1
ENABLE_OHSTEM=false
```

---

## Hướng Dẫn Sử Dụng

### Bước 1: Quét kiểm tra cổng Camera
Chạy lệnh sau để xem danh sách các camera đang kết nối với máy tính (`0`, `1`, `2`...):
```bash
python scripts/stream_test.py --scan
```

### Bước 2: Chạy Stream Test (Webcam máy tính hoặc Camera ngoài)

* **Chạy trên Webcam tích hợp của máy tính** (khi không cắm cam ngoài, mặc định là Index `0`):
  ```bash
  python scripts/stream_test.py --source 0 --ai
  ```

* **Chạy trên Camera ngoài USB (độ phân giải dọc `1080x1920`)**:
  ```bash
  python scripts/stream_test.py --source 1 --width 1080 --height 1920 --ai
  ```
  *(Lưu ý: Trên Windows, khi cắm camera ngoài qua USB, hệ điều hành có thể tự đảo thứ tự cổng — đẩy camera ngoài lên `--source 0` và webcam laptop xuống `--source 1`. Nếu thấy `--source 1` ra webcam máy tính thì đổi lại `--source 0`)*

* **Chạy trên Camera IP WiFi (Yoosee / RTSP)**:
  ```bash
  python scripts/stream_test.py --source "rtsp://admin:MAT_KHAU@192.168.0.12:554/onvif2" --ai
  ```
  *(Lưu ý: Dùng `/onvif2` để luồng nhẹ và mượt hơn; nếu mật khẩu có ký tự `@` thì viết thành `%40`, ví dụ `pass%40123`)*

* **Chạy giả lập từ file video hoặc camera điện thoại (DroidCam/HTTP)**:
  ```bash
  python scripts/stream_test.py --source "duong_dan_video.mp4" --ai
  python scripts/stream_test.py --source "http://192.168.1.X:4747/video" --ai
  ```

**Phím tắt khi đang mở cửa sổ:**
* `Space`: Bật / Tắt nhận diện AI.
* `S`: Chụp và lưu ảnh khung hình hiện tại vào `data/test_results/`.
* `Q`: Thoát.

### Bước 3: Chạy hệ thống chính thức
Sau khi xác định đúng cổng camera (`0` hoặc `1`), cập nhật `CAMERA_SOURCE=` trong `.env` rồi chạy:
```bash
python main.py
```

### Kiểm thử nhanh với ảnh hoặc video có sẵn
```bash
# Test với ảnh
python scripts/test_image.py path/to/image.jpg

# Test với video (tự động lưu kết quả ra data/test_results/)
python scripts/test_media.py path/to/video.mp4
```
