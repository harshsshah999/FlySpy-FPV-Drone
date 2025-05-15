from flask import Flask, Response, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from person_detection_streamer import PersonTracker, CLASSES

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Increase maximum file size to 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

# Adjust these parameters for better tracking
CONFIDENCE_THRESHOLD = 0.01  # Increased from 0.2 for more reliable detections
SKIP_FRAMES = 1  # Reduced from 2 to track more frames
tracker = PersonTracker(memory_frames=30, max_distance=150)  # Increased memory and distance

def generate_frames(video_path=None):
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        # Try different camera indices
        for camera_index in [0, 1]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                break
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        # Return a default frame or error message
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No video source available", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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
            
            # Detect people - adjust blob parameters
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,  # scale factor
                (300, 300),
                127.5   # mean subtraction
            )
            net.setInput(blob)
            detections = net.forward()
            print("[DEBUG] raw detections:", detections.shape[2])  # ← debug print

            boxes = []
            confidences = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] == "person":
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Add boundary checks
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)
                        
                        # Only add valid boxes
                        if startX < endX and startY < endY:
                            boxes.append(box.astype("int"))
                            confidences.append(float(confidence))
            print(f"[DEBUG] proposals: {detections.shape[2]}, persons after filter: {len(boxes)}")

            # Update tracker
            new_people = tracker.update(boxes, confidences)
            total_unique_people += new_people
            
            # Draw boxes and labels
            for track_id, track in tracker.tracks.items():
                if track['missing_frames'] > 0:
                    continue
                    
                box = track['box']
                confidence = track['confidence']
                
                # Draw a more visible box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                
                # Add a more visible label
                label = f"Person {track_id}: {confidence*100:.1f}%"
                y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
                cv2.putText(frame, label, (box[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw motion trail
                positions = list(track['positions'])[-10:]  # Show longer trail
                for i in range(1, len(positions)):
                    cv2.line(frame, positions[i-1], positions[i], (0, 255, 0), 2)
            
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
        cap.release()

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    
    video = request.files['video']
    if video.filename == '':
        return redirect(url_for('index'))
    
    if video:
        # Save the uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        return redirect(url_for('index', video_path=video_path))
    
    return redirect(url_for('index'))


@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    video_path = request.args.get('video_path')
    return render_template('index.html', video_path=video_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)