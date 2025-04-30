from ultralytics import YOLO
import cv2
# Load YOLOv8 model - pretrained on COCO dataset
model = YOLO("yolov8n.pt")  # You can also try yolov8s.pt or yolov8m.pt for better accuracy

# Set video source
video_path = "drone_footage.mp4"  # Change this to your drone video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
