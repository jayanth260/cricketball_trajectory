from typing import List, Optional
from scipy.signal import savgol_filter
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import argparse
from tqdm import tqdm



def filter_points(init_detections):
    """
    Finds the longest valid sequence of points using Dynamic Programming.
    dp[i] = (length, prev_index) where length is the longest path ending at i
    """
    n = len(init_detections)

    # Filter to only visible detections with their original indices
    visible = [(i, det) for i, det in enumerate(init_detections) if det["visible"] == 1]

    if not visible:
        return [{
            "frame": det["frame"],
            "x": -1.0, "y": -1.0, "visible": 0
        } for det in init_detections]

    # DP table: dp[i] = (max_length_ending_here, previous_index_in_visible_array)
    dp = [(1, -1) for _ in range(len(visible))]

    # Build DP table
    for i in range(1, len(visible)):
        curr_idx, curr_det = visible[i]

        for j in range(i):
            prev_idx, prev_det = visible[j]

            # Check motion constraints
            dx = abs(curr_det["x"] - prev_det["x"])
            dy = abs(curr_det["y"] - prev_det["y"])

            if (1 < dx < 60 or 1 < dy < 60) and (dx < 200 and dy < 200):
                # Valid transition from j to i
                if dp[j][0] + 1 > dp[i][0]:
                    dp[i] = (dp[j][0] + 1, j)

    # Find the ending point of the longest sequence
    max_length = 0
    max_end_idx = 0
    for i, (length, _) in enumerate(dp):
        if length > max_length:
            max_length = length
            max_end_idx = i

    # Backtrack to reconstruct the path
    path_indices = []
    curr = max_end_idx
    while curr != -1:
        path_indices.append(visible[curr][0])  # Original index
        curr = dp[curr][1]

    path_indices.reverse()
    valid_set = set(path_indices)

    # Build filtered output
    filtered_data = []
    for i, det in enumerate(init_detections):
        if i in valid_set:
            filtered_data.append(det)
        else:
            filtered_data.append({
                "frame": det["frame"],
                "x": -1.0, "y": -1.0, "visible": 0
            })

    return filtered_data


def smooth_trajectory(csv_data, window=9, poly=2):
    xs, ys, frames = [], [], []

    for d in csv_data:
        if d["visible"] == 1:
            xs.append(d["x"])
            ys.append(d["y"])
            frames.append(d["frame"])

    if len(xs) < window:
        return csv_data  # Not enough points

    xs_s = savgol_filter(xs, window_length=window, polyorder=poly)
    ys_s = savgol_filter(ys, window_length=window, polyorder=poly)

    idx = 0
    for d in csv_data:
        if d["visible"] == 1:
            d["x"] = float(xs_s[idx])
            d["y"] = float(ys_s[idx])
            idx += 1

    return csv_data

def kalman_smooth(csv_data):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix  = np.array([[1,0,1,0],
                                     [0,1,0,1],
                                     [0,0,1,0],
                                     [0,0,0,1]], np.float32)
    kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03

    for d in csv_data:
        if d["visible"] == 1:
            meas = np.array([[np.float32(d["x"])],
                             [np.float32(d["y"])]])
            kf.correct(meas)

        pred = kf.predict()
        d["x"], d["y"] = float(pred[0]), float(pred[1])
        d["visible"] = 1

    return csv_data

def process_video(
    video_path: str,
    model_path: str,
    output_video_dir: str,
    output_csv_dir: str,
    target_class_id: int = 0,
    conf_threshold: float = 0.1,
    trace_length: Optional[int] = None
):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    filename_base = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    # 2. First Pass: Get all raw detections
    init_detections = []
    frames_buffer = [] 

    print(f"Pass 1: Detecting objects in {filename_base}...")
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        frames_buffer.append(frame)

        results = model.predict(frame, conf=conf_threshold, imgsz=2560, verbose=False, classes=[target_class_id])

        cx, cy, detected = -1.0, -1.0, False
        if results and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            x_raw, y_raw, _, _ = best_box.xywh[0].cpu().numpy()
            cx, cy, detected = float(x_raw), float(y_raw), True

        init_detections.append({
            "frame": frame_idx, "x": round(cx, 1), "y": round(cy, 1), "visible": 1 if detected else 0
        })

    # 3. Filter Pass: Select the "True" Path
    print("Pass 2: Filtering trajectory...")
    csv_data = filter_points(init_detections)
    csv_data = smooth_trajectory(csv_data)
    # csv_data = kalman_smooth(csv_data)

    # 4. Final Pass: Draw and Write Video
    print("Pass 3: Rendering video...")
    save_video_path = os.path.join(output_video_dir, f"{filename_base}_processed.mp4")
    out = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    trajectory_points = []
    for i, det in enumerate(csv_data):
        frame = frames_buffer[i]

        if det["visible"] == 1:
            curr_pt = (int(det["x"]), int(det["y"]))
            trajectory_points.append(curr_pt)

            # Draw Ball and Label
            cv2.circle(frame, curr_pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"F:{det['frame']}", (curr_pt[0]+10, curr_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Trajectory Line
        if len(trajectory_points) > 1:
            pts = np.array(trajectory_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

        out.write(frame)

    # 5. Cleanup
    cap.release()
    out.release()
    pd.DataFrame(csv_data).to_csv(os.path.join(output_csv_dir, f"{filename_base}.csv"), index=False)
    print(f"Finished! Output at {save_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process cricket video to detect and track ball trajectory')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO model')
    parser.add_argument('--output_video_dir', type=str, default='./output_videos',
                        help='Directory to save processed video (default: ./output_videos)')
    parser.add_argument('--output_csv_dir', type=str, default='./output_csv',
                        help='Directory to save trajectory CSV (default: ./output_csv)')
    parser.add_argument('--class_id', type=int, default=0,
                        help='Target class ID to detect (default: 0)')
    parser.add_argument('--conf', type=float, default=0.1,
                        help='Confidence threshold for detection (default: 0.1)')
    
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        model_path=args.model,
        output_video_dir=args.output_video_dir,
        output_csv_dir=args.output_csv_dir,
        target_class_id=args.class_id,
        conf_threshold=args.conf
    )