# True Form Student - AI Uniform & Attendance Detection System

Hệ thống AI thông minh thời gian thực (Real-time Edge AI) nhận diện đồng phục học sinh (áo trắng, quần xanh, khăn quàng đỏ, bảng tên) từ Camera, tự động kiểm tra vi phạm nội quy và đồng bộ điểm danh sĩ số học sinh lên **Bảng điều khiển IoT OhStem** qua giao thức MQTT.

---

## 📸 Sơ Đồ Kiến Trúc

```text
Camera (Webcam / RTSP / Video File)
      ↓
  Edge AI (YOLO11m - Medium)
      ↓
┌───────────────────────────────┐
│     Hệ Thống Xử Lý Logic      │
├───────────────┬───────────────┤
│ Điểm danh     │ Vi phạm       │
│ (Đếm sĩ số)   │ (Thiếu đồng   │
│               │  phục,...)    │
└───────┬───────┘───────┬───────┘
        ↓               ↓
┌───────────────┐┌───────────────┐
│  OhStem IoT   ││  Lưu trữ ảnh  │
│ (MQTT Broker) ││   cảnh báo    │
│  Dashboard    ││    local      │
└───────────────┘└───────────────┘
```

---

## ✨ Các Tính Năng Nổi Bật

1. **Nhận diện YOLO11m Mới Nhất**: Sử dụng mô hình YOLO11 phiên bản Medium (`yolo11m.pt`) tối ưu về độ chính xác và khả năng bám vết, hạn chế tối đa hiện tượng nhảy ID đối tượng.
2. **Điểm Danh Thời Gian Thực (OhStem IoT)**: Tự động đếm sĩ số học sinh xuất hiện trên màn hình và gửi dữ liệu lên Dashboard của OhStem.
3. **Phát Hiện Vi Phạm Thông Minh**:
   * Phát hiện mặc quần sai màu, áo sai màu.
   * Phát hiện thiếu khăn quàng đỏ, thiếu bảng tên (chấp nhận cả nhãn `bang_ten` và `the_ten`).
   * Sử dụng cơ chế **Frame Debounce** (xác nhận vi phạm sau 5 khung hình liên tiếp) để tránh nhiễu và báo động sai.
4. **Trực Quan Hóa Đẹp Mắt (Unicode Tiếng Việt)**:
   * Vẽ nhãn trực tiếp bằng tiếng Việt có dấu (`áo trắng`, `bảng tên`, `học sinh`...) nhờ tích hợp công nghệ Pillow.
   * Khi học sinh vi phạm, chỉ hiển thị nhãn đỏ xếp chồng khoa học ngay trên đầu học sinh (không vẽ viền đỏ bao quanh cơ thể để tránh rối mắt).
5. **Bộ Công Cụ Kiểm Thử Tiện Lợi**:
   * Kiểm thử nhanh hình ảnh riêng biệt, hiển thị trực quan và lưu trữ kết quả.
   * Kiểm thử video độc lập để tối ưu hóa góc đặt camera.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
Uniform_Student/
├── main.py                     # Pipeline chạy thời gian thực chính
├── .env                        # File cấu hình môi trường bảo mật (webcam, IoT...)
├── .env.example                # File mẫu để khởi tạo .env
├── Train_YOLO11_Colab.ipynb    # Notebook dùng để train YOLO11m trên Google Colab
├── config/
│   └── settings.yaml           # Các tham số cài đặt tỷ lệ cơ thể và số frame xác nhận
├── data/
│   ├── dataset.yaml            # Cấu hình đường dẫn dataset Roboflow
│   ├── violations/             # Thư mục tự động lưu ảnh khi có vi phạm đồng phục
│   └── test_results/           # Thư mục tự động lưu ảnh/video đầu ra sau khi test
├── models/
│   └── best_uniform_model.pt   # File trọng số (weights) đã được huấn luyện thành công
├── scripts/
│   ├── test_image.py           # Script chuyên dụng kiểm thử nhanh một file hình ảnh
│   ├── test_media.py           # Script kiểm thử đa năng (hỗ trợ cả ảnh và video)
│   ├── run_video.py            # Script chạy thử trên video không tải IoT
│   └── train.py                # Script chạy huấn luyện mô hình local (nếu dùng máy có GPU)
└── src/
    ├── camera.py               # Quản lý luồng đọc Camera đa luồng (Multi-threading)
    ├── detector.py             # Lớp đóng gói YOLO và dự đoán đối tượng
    ├── violation_checker.py    # Logic kiểm tra luật đồng phục học sinh
    ├── ohstem_service.py       # Dịch vụ truyền thông điệp MQTT lên OhStem IoT
    ├── config_loader.py        # Đọc cấu hình từ .env và settings.yaml
    └── visualize.py            # Vẽ nhãn nhận diện và chữ Tiếng Việt Unicode sắc nét
```

---

## 🛠️ Cài Đặt Dưới Local

### 1. Chuẩn bị môi trường Python
```bash
# Kích hoạt môi trường ảo Python
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/MacOS:
source .venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
pip install paho-mqtt
```

### 3. Khởi tạo cấu hình hệ thống
Sao chép file cấu hình mẫu để tạo file `.env`:
```bash
copy .env.example .env
```
Mở file `.env` ra để cập nhật nguồn camera và tài khoản OhStem (nếu sử dụng).

---

## 🤖 Huấn Luyện (Training) Trên Google Colab

Dự án đã chuẩn bị sẵn file [Train_YOLO11_Colab.ipynb](file:///c:/Users/DELL/OneDrive%20-%20Hanoi%20University%20of%20Science%20and%20Technology/Desktop/Uniform_Student/Train_YOLO11_Colab.ipynb) được cấu hình tối ưu để bạn có thể import thẳng lên Google Colab và huấn luyện bằng GPU miễn phí:

1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn tab **Upload** và tải lên file [Train_YOLO11_Colab.ipynb](file:///c:/Users/DELL/OneDrive%20-%20Hanoi%20University%20of%20Science%20and%20Technology/Desktop/Uniform_Student/Train_YOLO11_Colab.ipynb).
3. Đổi loại thời gian chạy thành **T4 GPU** (Runtime -> Change runtime type -> T4 GPU).
4. Nhấn **Run All** (`Ctrl + F9`). Notebook đã được tích hợp mã vá lỗi đường dẫn động (Dynamic Path Scanner) và điền sẵn mã tài khoản Roboflow của bạn, quá trình tải và train sẽ diễn ra tự động 100%.
5. Khi train xong, trình duyệt sẽ tự động tải file `best.pt` về máy tính. Bạn hãy đổi tên nó thành `best_uniform_model.pt` rồi copy vào thư mục `models/` trong dự án.

---

## 📡 Kết Nối Điểm Danh Lên OhStem IoT Panel

Để kết nối điểm danh sĩ số học sinh lên Dashboard của OhStem thời gian thực:

1. Mở file [.env](file:///c:/Users/DELL/OneDrive%20-%20Hanoi%20University%20of%20Science%20and%20Technology/Desktop/Uniform_Student/.env).
2. Kích hoạt tính năng bằng cách đổi `ENABLE_OHSTEM=false` thành `ENABLE_OHSTEM=true`.
3. Điền tên đăng nhập tài khoản của bạn:
   ```env
   OHSTEM_USERNAME=tên_đăng_nhập_ohstem_của_bạn
   OHSTEM_FEED_ID=V1   # Kênh dashboard tương ứng nhận dữ liệu sĩ số
   ```

---

## 🚀 Cách Chạy Ứng Dụng

### 1. Chạy Luồng Chính Real-time (Webcam / Camera trường học):
```bash
python main.py
```
* **Phát hiện vi phạm**: Log sẽ được in gọn gàng ra màn hình console và lưu hình ảnh minh chứng có gán nhãn thời gian vào thư mục `data/violations/`.
* **Điểm danh**: Tự động đếm sĩ số học sinh thực tế trong khung hình và truyền trực tiếp lên Dashboard OhStem.
* Nhấn phím **`q`** để thoát an toàn.

### 2. Kiểm thử nhanh file HÌNH ẢNH:
```bash
python scripts/test_image.py "đường_dẫn_tới_ảnh.jpg"

# Hoặc lưu lại ảnh kết quả sau nhận diện:
python scripts/test_image.py "đường_dẫn_tới_ảnh.jpg" --save
```

### 3. Kiểm thử nhanh file VIDEO:
```bash
python scripts/test_media.py "đường_dẫn_tới_video.mp4"

# Hoặc lưu lại video kết quả sau nhận diện:
python scripts/test_media.py "đường_dẫn_tới_video.mp4" --save
```

---

## 📝 Quy Tắc Phát Hiện Vi Phạm Đồng Phục
* **Vi phạm khăn quàng đỏ**: Nếu phát hiện vùng cơ thể `hoc_sinh` nhưng không tìm thấy `khan_quang` ở vùng thân trên (tỷ lệ 0.55 chiều cao bounding box).
* **Vi phạm bảng tên**: Nếu phát hiện `hoc_sinh` nhưng không tìm thấy `bang_ten` hoặc `the_ten` ở vùng thân trên.
* **Vi phạm màu sắc quần áo**:
  * Phát hiện nhãn `quan_sai_mau` hoặc `ao_sai_mau` chồng đè lên cơ thể học sinh.
  * Hoặc thiếu các nhãn đồng phục cơ bản như `ao_trang` hoặc `quan_xanh` trên vùng cơ thể học sinh.
