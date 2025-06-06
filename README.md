# Save the full README content to a file named README.md
Tên dự án:
Tạo heatmap vùng hoạt động của các vận động viên cầu lông
****
Tổng quan:
Dự án này triển khai một hệ thống xử lý video để phát hiện sân cầu lông, theo dõi gót chân vận động viên bằng MediaPipe, và xây dựng bản đồ nhiệt (heatmap) thể hiện tần suất di chuyển của người chơi trên sân. Kết quả cho thấy vùng hoạt động ưu tiên, giúp phân tích chiến thuật và thói quen di chuyển của vận động viên.
Dự án này triển khai một hệ thống xử lý video để phát hiện sân cầu lông, theo dõi gót chân vận động viên bằng MediaPipe, và xây dựng bản đồ nhiệt (heatmap) thể hiện tần suất di chuyển của người chơi trên sân. 
****
Chức năng chính:

1. Phát hiện sân cầu lông:
   - Sử dụng không gian màu HSV để tách nền xanh của sân.
   - Làm sạch mask bằng phép biến đổi hình thái.
   - Tìm contour lớn nhất, xấp xỉ thành hình tứ giác hoặc bounding box.

2. Phát hiện và phân loại đường sân:
   - Dùng Canny và HoughLinesP để phát hiện đường.
   - Phân loại theo hướng ngang/dọc, lọc theo độ dài hợp lệ.
   - Tính các giao điểm tạo thành 4 góc sân.

3. Chuẩn hóa phối cảnh sân:
   - Dùng homography để warp sân về góc nhìn từ trên xuống.
   - Kích thước sân theo chuẩn quốc tế: 6.1m x 13.4m.

4. Theo dõi gót chân vận động viên:
   - Áp dụng MediaPipe Pose từ frame thứ 250.
   - Trích xuất vị trí gót chân trái và phải.
   - Chuyển gót chân sang tọa độ sân đã được warp.
     
5. Tạo bản đồ nhiệt:
   - Dùng Gaussian để làm mờ mỗi vị trí xuất hiện của gót chân.
   - Tích lũy theo thời gian thành heatmap.
   - Phủ màu với cv.COLORMAP_JET và overlay lên sân chuẩn.
****
Yêu cầu:

- Python >= 3.7
- Thư viện:
  pip install opencv-python mediapipe numpy matplotlib

Cách sử dụng:

1. Đặt file video vidtest.mp4 vào thư mục chứa mã nguồn(https://www.youtube.com/watch?v=s15HhYHCYxc)
2. Chạy script:
   python badminton_heatmap.py
3. Quan sát các cửa sổ hiển thị:
   - Quá trình phát hiện sân.
   - Gót chân vận động viên được nhận dạng.
   - Gót chân trên sân chuẩn đã được warp.
   - Bản đồ nhiệt trên sân chuẩn.

Đầu ra:

- Ảnh bản đồ nhiệt cuối cùng: final_heatmap_true_scale.png
****
Thông tin nhóm thực hiện:
- Đề tài: Tạo heatmap vùng hoạt động của các vận động viên cầu lông
- Học viên: Trương Vũ Hoàng Anh, Bùi Nguyễn Hoài Thương
- Môn học: Thị giác máy tính
- Năm học: 2025
- Công cụ: Python, OpenCV, MediaPipe, NumPy
