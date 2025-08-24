from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model (can be 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO("yolov8n.pt")

# Open the webcam (0 = default camera, or give video path like "video.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()