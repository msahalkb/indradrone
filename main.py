from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Path to the aerial video
video_path = "./testvideo1.mp4"  
cap = cv2.VideoCapture(video_path)

# Check if the video file is loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow("YOLOv8 Aerial Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()