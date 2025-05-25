# --- Import thư viện cần thiết ---
import cv2 as cv 
import numpy as np
import math
from matplotlib import pyplot as plt
import mediapipe as mp
import copy
import matplotlib.pyplot as plt

# --- Phân loại đường Hough thành ngang và dọc dựa trên độ chênh tọa độ ---
def segment_lines(lines, deltaX, deltaY):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2-y1) < deltaY:  # Ngang
                h_lines.append(line)
            elif abs(x2-x1) < deltaX:  # Dọc
                v_lines.append(line)
    return h_lines, v_lines

# --- Lọc đường theo độ dài ---
def filterLines(segments, minLength, maxLength): 
    result = []
    for segment in segments: 
        for x1,y1,x2,y2 in segment:  
            if  minLength < math.dist([x1,y1], [x2,y2]) < maxLength: 
                result.append(segment)
    return result 

# --- Lọc đường sân theo kích thước thực tế (độ dài đặc trưng của sân) ---
def getCourtLines(hsegments, vsegments): 
    h_result = [] 
    v_result = [] 
    for segment in hsegments: 
        for x1,y1,x2,y2 in segment:
            if (50 < math.dist([x1,y1], [x2,y2]) < 449.0) or (50 < math.dist([x1,y1], [x2,y2]) < 885.0): 
                h_result.append(segment)
    for segment in vsegments: 
        for x1,y1,x2,y2 in segment:
            if 50.0 < math.dist([x1,y1], [x2,y2]): 
                v_result.append(segment)
    return h_result, v_result

# --- Tìm điểm giao nhau giữa 2 đường ---
def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py

# --- Sắp xếp 4 góc: trái trên, phải trên, trái dưới, phải dưới ---
def get_four_corners(points):
    pts = np.array(points)
    top_left = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    bottom_right = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    top_right = pts[np.argmax(pts[:, 0] - pts[:, 1])]
    bottom_left = pts[np.argmax(pts[:, 1] - pts[:, 0])]
    return top_left, top_right, bottom_left, bottom_right

# --- Vẽ gaussian làm điểm nóng trên heatmap ---
def draw_gaussian(heatmap, x, y, radius=15, intensity=1.0):
    temp = np.zeros_like(heatmap)
    cv.circle(temp, (int(x), int(y)), radius, intensity, -1)
    temp = cv.GaussianBlur(temp, (0, 0), sigmaX=radius, sigmaY=radius)
    heatmap += temp
    return heatmap

# --- Khởi tạo video input ---
cap = cv.VideoCapture('vidtest.mp4')
frame_count = 0

# --- Khởi tạo MediaPipe Pose để phát hiện cơ thể người ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                    enable_segmentation=False, smooth_segmentation=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Khởi tạo heatmap ---
heatmap = None

# --- Bắt đầu đọc từng frame video ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # --- Phát hiện vùng sân trong 30 frame đầu tiên ---
    if frame_count <= 30:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((7, 7), np.uint8)
        mask_cleaned = cv.morphologyEx(mask_green, cv.MORPH_CLOSE, kernel)

        contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise Exception("Không tìm thấy contour.")
        cnt = max(contours, key=cv.contourArea)
        hull = cv.convexHull(cnt)
        epsilon = 0.01 * cv.arcLength(hull, True)
        approx = cv.approxPolyDP(hull, epsilon, True)

        if len(approx) != 4:
            rect = cv.minAreaRect(hull)
            box = cv.boxPoints(rect)
            box = np.int0(box)
        else:
            box = approx.reshape(-1, 2)

        roi_mask = np.zeros_like(frame)
        cv.fillConvexPoly(roi_mask, box, (255, 255, 255))

    # --- Áp dụng mask sân cho frame ---
    if roi_mask is not None:
        roi = cv.bitwise_and(frame, roi_mask)
        tam = copy.deepcopy(roi)
    else:
        roi = frame

    cv.imshow("COURT DETECTION", tam)

    # --- Tiền xử lý ảnh xám, tạo edge ---
    img = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    ret, th1 = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
    edges = cv.Canny(th1, 120, 200, apertureSize=3)

    # --- Phát hiện đường bằng HoughLinesP ---
    linesP = cv.HoughLinesP(edges, 1, np.pi / 90, 68, None, 30, 255)
    cdstP = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    if linesP is not None:
        h_lines, v_lines = segment_lines(linesP, 280, 0.5)
        filtered_h_lines, filtered_v_lines = getCourtLines(filterLines(h_lines, 200, 400),
                                                           filterLines(v_lines, 100, 510))

        # --- Vẽ các đường hợp lệ ---
        for line in filtered_v_lines + filtered_h_lines:
            l = line[0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        # --- Tính giao điểm giữa các đường ---
        Px, Py = [], []
        for h in filtered_h_lines:
            for v in filtered_v_lines:
                px, py = find_intersection(h, v)
                Px.append(px)
                Py.append(py)

        # --- Vẽ các điểm giao nhau và tìm góc sân ---
        intersectsimg = cv.cvtColor(edges.copy(), cv.COLOR_GRAY2BGR)
        for cx, cy in zip(Px, Py):
            cv.circle(intersectsimg, (int(round(cx)), int(round(cy))), 7, np.random.randint(0,255,3).tolist(), -1)

        points = [(int(round(x)), int(round(y))) for x, y in zip(Px, Py)]
        if len(points) >= 4:
            top_left, top_right, bottom_left, bottom_right = get_four_corners(points)
            for x, y in [top_left, top_right, bottom_left, bottom_right]:
                cv.circle(roi, (x, y), 10, (0, 255, 0), -1)
            cv.imshow("FOUR POINT CORNER", roi)

            # --- Warp ảnh về tọa độ chuẩn của sân ---
            src_pts = np.float32([top_left, top_right, bottom_left, bottom_right])
            scale = 0.75
            court_width_m = 6.1
            court_height_m = 13.4
            court_height_px = int(1000 * scale)
            court_width_px = int(court_height_px * court_width_m / court_height_m)
            dst_pts = np.float32([[0, 0], [court_width_px - 1, 0], [0, court_height_px - 1], [court_width_px - 1, court_height_px - 1]])
            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv.warpPerspective(frame, M, (court_width_px, court_height_px))

            # --- Bắt đầu tracking gót chân sau frame 250 ---
            if frame_count >= 250:
                roi.flags.writeable = False
                image_rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                roi.flags.writeable = True

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(tam, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = tam.shape

                    # --- Gót trái và phải ---
                    x_lh = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x * w)
                    y_lh = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h)
                    x_rh = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x * w)
                    y_rh = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h)
                    cv.circle(tam, (x_lh, y_lh), 5, (0, 0, 255), -1)
                    cv.circle(tam, (x_rh, y_rh), 5, (255, 0, 0), -1)
                    cv.imshow("HEEL COORDINATE", tam)

                    # --- Chuyển đổi gót chân sang tọa độ warp ---
                    points = np.array([[[x_lh, y_lh]], [[x_rh, y_rh]]], dtype=np.float32)
                    warped_points = cv.perspectiveTransform(points, M)

                    # --- Vẽ và cập nhật heatmap ---
                    for (x, y), color in zip(warped_points[:, 0], [(255, 0, 0), (0, 0, 255)]):
                        cv.circle(warped, (int(x), int(y)), 10, color, -1)
                        cv.imshow("PERSPECTIVE TRANSFORM", warped)

                    if heatmap is None:
                        heatmap = np.zeros((warped.shape[0], warped.shape[1]), dtype=np.float32)
                    for (x, y) in warped_points[:, 0]:
                        heatmap = draw_gaussian(heatmap, x, y)

                    # --- Normalize heatmap và áp colormap ---
                    if heatmap.max() > 0:
                        heatmap_norm = np.clip(heatmap / heatmap.max(), 0, 1)
                        heatmap_uint8 = np.uint8(heatmap_norm * 255)
                        heatmap_color = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

    # --- Hiển thị frame video gốc ---
    cv.imshow("VIDEO INPUT", frame)
    if frame_count > 1700:
        break

cap.release()

# --- Vẽ lại sân cầu lông chuẩn trên nền xanh ---
height, width = warped.shape[:2]
background = np.zeros((height, width, 3), dtype=np.uint8)
background[:] = (34, 139, 34)
scale_y = height / 13.4
scale_x = width / 6.1

# Hàm vẽ theo đơn vị mét
def draw_line_m(background, x1_m, y1_m, x2_m, y2_m, color=(255, 255, 255), thickness=2):
    pt1 = (int(round(x1_m * scale_x)), int(round(y1_m * scale_y)))
    pt2 = (int(round(x2_m * scale_x)), int(round(y2_m * scale_y)))
    cv.line(background, pt1, pt2, color, thickness)

# Vẽ các đường sân đúng chuẩn thi đấu
draw_line_m(background, 0, 0, 0, 13.4)
draw_line_m(background, 6.1, 0, 6.1, 13.4)
draw_line_m(background, 0.46, 0, 0.46, 13.4)
draw_line_m(background, 6.1 - 0.46, 0, 6.1 - 0.46, 13.4)
draw_line_m(background, 0, 0, 6.1, 0)
draw_line_m(background, 0, 13.4, 6.1, 13.4)
draw_line_m(background, 0, 6.7, 6.1, 6.7)
draw_line_m(background, 0, 6.7 - 1.98, 6.1, 6.7 - 1.98)
draw_line_m(background, 0, 6.7 + 1.98, 6.1, 6.7 + 1.98)
draw_line_m(background, 3.05, 0, 3.05, 13.4)
draw_line_m(background, 0, 0.76, 6.1, 0.76)
draw_line_m(background, 0, 13.4 - 0.76, 6.1, 13.4 - 0.76)
draw_line_m(background, 0, 0, 6.1, 0, color=(0, 0, 255), thickness=2)
draw_line_m(background, 0, 13.4, 6.1, 13.4, color=(0, 0, 255), thickness=2)

# Overlay heatmap lên sân chuẩn
final_overlay = cv.addWeighted(background, 0.6, heatmap_color, 0.4, 0)

# Hiển thị và lưu kết quả
cv.imshow("Final Heatmap (True Court Scale)", final_overlay)
cv.imwrite("final_heatmap_true_scale.png", final_overlay)
cv.waitKey(0)
cv.destroyAllWindows()
