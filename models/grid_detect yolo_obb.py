import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
import numpy as np


# Set paths
video_folder = '/home/ss/Kirti/lat/video_test'  # Path to the folder with your videos
model_folder = '/home/ss/Kirti/lat/models'  # Path to the folder with your YOLOv8 models
output_folder = '/home/ss/Kirti/lat/output_video'  # Path to save output videos
csv_log_file = os.path.join(output_folder, 'log.csv')

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all videos and models
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
model_files = [f for f in os.listdir(model_folder) if f.endswith(('.pt', '.engine'))]

# Initialize log dataframe
log_columns = ['Model', 'Video', 'Total_Frames', 'Total_Time_Seconds', 'Average_FPS']
log_data = []

print(video_files)
print(model_files)

# Detection image size 
image_size = 640
colors = [(255,0,255), (0,0,255), (0,150,150), (100,255, 0)]
# Target resolution for 720p
start_time = time.time()

def draw_obb_xyxyxyxy(frame, corners8, offset=(0, 0), color=(255, 0, 0), thickness=2):
    """
    corners8: array-like shape (8,) => [x1,y1,x2,y2,x3,y3,x4,y4] in ROI coords
    offset: (ox, oy) ROI top-left in full frame
    """
    ox, oy = offset
    pts = np.array(corners8, dtype=np.float32).reshape(4, 2)
    pts[:, 0] += ox
    pts[:, 1] += oy
    pts = pts.astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return frame


# Loop through each model
for model_file in model_files:
    model_path = os.path.join(model_folder, model_file)
    model = YOLO(model_path)

    # Loop through each video
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_resolution = (width, height)
        N = width // image_size + 1 if width % image_size > 0 else width // image_size
        N_ = width % image_size 
        M = height // image_size + 1 if height % image_size >0 else height //image_size
        M_ = height % image_size 

        print(f'video size = {width}, {height}, N = {N}, M = {M}')

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Prepare output video file
        output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{os.path.splitext(model_file)[0]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_resolution)

        start_time = time.time()
        frame_count = 0

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            num_of_detections = 0

            detected_box = []
            
            for n in range(N):  # N is Width and M is Height
                for m in range(M):
                    
                    print(f'height - {height}, width - {width}, N - {N}, M - {M}, n - {n}, m - {m}')
                    n1 = n*image_size
                    n2 = min ((n+1)*image_size, width)
                    m1 = m*image_size
                    m2 =  min ((m+1)*image_size, height)

                    roi = frame[m1:m2, n1:n2]  # frame[y1:y2, x1:x2]
                                        
                    offset = (n*image_size, m*image_size)

                    # Run YOLOv8 inference on the resized frame
                    results = model(roi, imgsz=image_size, conf=0.25)

                    cv2.rectangle(frame, offset, (n2,m2), colors[3], 1)

                    for result in results:

                        # ---------- OBB MODELS ----------
                        if result.obb is not None and len(result.obb) > 0:
                            obb = result.obb

                            # corners: Nx8 => [x1,y1,x2,y2,x3,y3,x4,y4] in ROI coords
                            corners = obb.xyxyxyxy.cpu().numpy()     # shape (N, 8)
                            confs   = obb.conf.cpu().numpy()         # shape (N,)
                            clss    = obb.cls.cpu().numpy()          # shape (N,)

                            with open('detected_boxes5.txt', 'a') as f:
                                for i in range(len(confs)):
                                    # Ensure corners8 is a 1D numeric array (some outputs can be nested)
                                    corners8 = np.asarray(corners[i]).astype(float).reshape(-1)
                                    confidence = float(confs[i])
                                    class_id   = int(clss[i])

                                    # Draw OBB polygon on full frame with offset
                                    draw_obb_xyxyxyxy(frame, corners8, offset=offset, color=(255, 0, 0), thickness=2)

                                    # Optional: put confidence near first corner
                                    ox, oy = offset
                                    x1, y1 = int(corners8[0] + ox), int(corners8[1] + oy)
                                    cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                                    cv2.putText(frame, f'Class_id: {class_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)

                                    # Log corners (you can change format)
                                    f.write(f"{frame_count},{class_id},{confidence}," + ",".join([f"{float(v):.2f}" for v in corners8]) + "\n")

               
            # Write the frame to the output video       
            out.write(frame)
            frame_count += 1

        # Calculate total time and average FPS
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time

        # Release resources
        cap.release()
        out.release()

        # Log details
        log_data.append([model_file, video_file, frame_count, total_time, avg_fps])

end_time = time.time()

print("TOTAL TIME", end_time-start_time)
# Save logs to CSV
df_log = pd.DataFrame(log_data, columns=log_columns)
df_log.to_csv(csv_log_file, index=False)

print(f"Processing complete. Logs saved to {csv_log_file}")
