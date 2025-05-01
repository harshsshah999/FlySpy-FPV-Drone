from flask import Flask, Response, render_template
import cv2
import numpy as np
import os
from person_detection_streamer import PersonTracker, CLASSES

app = Flask(__name__)

# Initialize your detection components
PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

# Check if model files exist
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model files not found. Please ensure {PROTOTXT_PATH} and {MODEL_PATH} exist in the current directory.")

# Load the Caffe model (only once)
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
except cv2.error as e:
    raise Exception(f"Error loading model: {str(e)}")

CONFIDENCE_THRESHOLD = 0.2
SKIP_FRAMES = 2
tracker = PersonTracker(memory_frames=15, max_distance=100)

def generate_frames():
    # Try different camera indices
    for camera_index in [0, 1]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    if not cap.isOpened():
        print("Error: Could not open any camera")
        # Return a default frame or error message
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera available", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "Please check camera permissions", (50, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    try:
        frame_count = 0
        total_unique_people = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % SKIP_FRAMES != 0:
                continue

            # Process frame
            frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            (h, w) = frame.shape[:2]
            
            # Detect people
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                        0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            boxes = []
            confidences = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] == "person":
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        boxes.append(box.astype("int"))
                        confidences.append(confidence)
            
            # Update tracker
            new_people = tracker.update(boxes, confidences)
            total_unique_people += new_people
            
            # Draw boxes and labels
            for track_id, track in tracker.tracks.items():
                if track['missing_frames'] > 0:
                    continue
                    
                box = track['box']
                confidence = track['confidence']
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                label = f"ID {track_id}: {confidence*100:.1f}%"
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frame, label, (box[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                positions = list(track['positions'])[-5:]
                for i in range(1, len(positions)):
                    cv2.line(frame, positions[i-1], positions[i], (0, 255, 0), 1)
            
            # Add stats to frame
            cv2.putText(frame, f"Active Tracks: {len(tracker.tracks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Unique People: {total_unique_people}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            # Convert frame to bytes for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()  # Ensure camera is released properly

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)