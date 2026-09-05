

## 📌 Các Tính Năng Chính

* **Phát hiện đối tượng (Person Detection):** Sử dụng mô hình **YOLOv8** (`ultralytics`) để phát hiện người với độ chính xác cao.
* **Theo dõi hành trình (Multi-Object Tracking):** Tích hợp thuật toán **ByteTrack** (thông qua thư viện `supervision`) để cấp ID cố định cho từng người trong suốt video.
* **Tự động Cắt ảnh (Person Cropping):** Cắt vùng ảnh chứa đối tượng (Bounding Box) có thêm dải lề (padding) để sẵn sàng truyền vào các mô hình nhận diện thuộc tính (như tuổi, giới tính, trang phục...).
* **Làm mượt thuộc tính (Attribute Smoothing):** Áp dụng cơ chế **Majority Voting** trên một khung cửa sổ (Window size) để ổn định kết quả dự đoán thuộc tính, tránh bị giật lag/thay đổi nhãn liên tục giữa các frame.
* **Xử lý Đa luồng (Threaded Pipeline):** Đọc video (`ThreadedVideoReader`) và suy luận mô hình (Inference) trên các luồng độc lập, giúp tối ưu FPS và không bị trễ khung hình khi dùng Webcam.

---

## 🏗️ Cấu Trúc Dự Án

```text
.
├── config.py                 # File cấu hình chung (mô hình, ngưỡng confidence, FPS, hiển thị,...)
├── detector.py               # Lớp PersonDetector (Wrapper cho YOLOv8)
├── tracker_smoother.py       # Các lớp PersonTracker (ByteTrack) & AttributeSmoother (Làm mượt nhãn)
├── cropper.py                # Lớp PersonCropper (Cắt và xử lý ảnh bounding box người)
├── video_reader.py           # Đọc Video/Camera (hỗ trợ cả đệm hàng đợi Queue đa luồng)
├── pipeline.py               # Luồng xử lý tuần tự chính (PersonPipeline)
├── threaded_pipeline.py      # Luồng xử lý đa luồng tăng tốc (ThreadedPersonPipeline)
├── test_threaded.py          # Script chạy thử nghiệm chế độ Đa luồng
└── test_video_reader.py      # Script test đọc file video đơn giản

```

---

## ⚙️ Cài Đặt Môi Trường

### 1. Cài đặt thư viện phụ thuộc

```bash
pip install opencv-python numpy ultralytics supervision

```

---

## ⚙️ Cấu Hinh (config.py)

Bạn có thể tùy chỉnh các thông số trong file `config.py`:

| Cấu hình | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `YOLO_MODEL_PATH` | `"yolov8n.pt"` | Đường dẫn file trọng số YOLOv8 |
| `DETECTION_CONFIDENCE` | `0.40` | Ngưỡng tin cậy tối thiểu để lấy detection |
| `TRACK_ACTIVATION_THRESHOLD` | `0.25` | Ngưỡng kích hoạt track ID mới (ByteTrack) |
| `SMOOTHING_WINDOW_SIZE` | `5` | Số lượng frame lưu lịch sử để bầu chọn thuộc tính |
| `SMOOTHING_MAX_MISSING_FRAMES` | `30` | Số frame mất dấu trước khi xóa lịch sử của ID đó |
| `DISPLAY_MAX_WIDTH` / `HEIGHT` | `1280` / `720` | Kích thước hiển thị màn hình preview |

---



### 1. Chạy Luồng Đa Luồng Tối Ưu (Threaded Pipeline)

Chạy file `test_threaded.py` để kiểm thử xử lý video với luồng đọc khung hình và luồng suy luận riêng biệt:

```python
from pipeline.threaded_pipeline import ThreadedPersonPipeline

# Khởi tạo pipeline đa luồng
pipeline = ThreadedPersonPipeline()

# Chạy xử lý video
pipeline.run_threaded(
    source="path/to/your/video.mp4",  # Hoặc truyền 0 cho Webcam
    show=True,                        # Hiển thị cửa sổ OpenCV Preview
    output_path="results/output.mp4", # Đường dẫn lưu video kết quả (Tùy chọn)
    queue_size=2,                     # Kích thước hàng đợi đệm frame
    drop_frames=False                 # Bỏ qua frame cũ nếu xử lý không kịp (True cho Webcam thời gian thực)
)

```

### 2. Chạy Luồng Tuần Tự Cơ Bản (Single Thread Pipeline)

```python
from pipeline.pipeline import PersonPipeline

pipeline = PersonPipeline(model_path="yolov8n.pt", confidence=0.40)

# Chạy trực tiếp từ file hoặc camera
pipeline.run(
    source="tests/Test5.mp4",
    show=True,
    output_path="tests/results/result.mp4"
)

```

### 3. Tích Hợp Dự Đoán Thuộc Tính (Attribute Predictor)

Nếu bạn có mô hình dự đoán thuộc tính (ví dụ: dự đoán giới tính, áo quần), bạn chỉ cần kế thừa giao diện `AttributePredictorInterface` và truyền vào pipeline:

```python
# Kết quả trả về từ attribute_predictor cần có dạng:
# {
#     "attributes": {"gender": "Female", "clothing": "Shirt"},
#     "confidence_scores": {"gender": 0.95, "clothing": 0.88}
# }

pipeline = PersonPipeline(attribute_predictor=my_attribute_model)

```

---

## ⌨️ Thao Tác Khi Hiển Thị Window

* Nhấn phím **`q`** trên bàn phím hoặc bấm nút đóng cửa sổ (`X`) để dừng chương trình.
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("FILE_CREATED: README.md")

```

```text?code_stdout&code_event_index=1
FILE_CREATED: README.md


```

Your Markdown file is ready


---

