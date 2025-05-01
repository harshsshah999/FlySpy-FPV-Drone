"""
Person Detection Streamer with Robust Tracking

This script reads from a video source and detects people using MobileNet-SSD model,
with improved tracking to handle drone footage and moving cameras.
"""

import cv2
import numpy as np
import argparse
from collections import deque
import math

# MobileNet SSD class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

class PersonTracker:
    def __init__(self, memory_frames=30, max_distance=100):
        self.tracks = {}
        self.next_track_id = 0
        self.memory_frames = memory_frames
        self.max_distance = max_distance

    def calculate_centroid(self, box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    def calculate_distance(self, centroid1, centroid2):
        return math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)

    def update(self, boxes, confidences):
        current_centroids = [self.calculate_centroid(box) for box in boxes]

        if not self.tracks:
            for centroid, box, conf in zip(current_centroids, boxes, confidences):
                self.tracks[self.next_track_id] = {
                    'centroid': centroid,
                    'box': box,
                    'confidence': conf,
                    'missing_frames': 0,
                    'positions': deque([centroid], maxlen=self.memory_frames)
                }
                self.next_track_id += 1
            return len(current_centroids)

        new_people = 0
        matched_tracks = set()
        matched_detections = set()

        for track_id, track in self.tracks.items():
            if track['missing_frames'] > self.memory_frames:
                continue

            track_centroid = track['centroid']
            min_distance = float('inf')
            best_match = None

            for i, (centroid, box, conf) in enumerate(zip(current_centroids, boxes, confidences)):
                if i in matched_detections:
                    continue

                distance = self.calculate_distance(track_centroid, centroid)
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_match = (i, centroid, box, conf)

            if best_match:
                i, centroid, box, conf = best_match
                matched_detections.add(i)
                matched_tracks.add(track_id)

                track['centroid'] = centroid
                track['box'] = box
                track['confidence'] = conf
                track['missing_frames'] = 0
                track['positions'].append(centroid)

        for i, (centroid, box, conf) in enumerate(zip(current_centroids, boxes, confidences)):
            if i not in matched_detections:
                is_new = True
                for track_id, track in self.tracks.items():
                    if track_id in matched_tracks:
                        continue
                    if self.calculate_distance(centroid, track['centroid']) < self.max_distance:
                        is_new = False
                        break

                if is_new:
                    self.tracks[self.next_track_id] = {
                        'centroid': centroid,
                        'box': box,
                        'confidence': conf,
                        'missing_frames': 0,
                        'positions': deque([centroid], maxlen=self.memory_frames)
                    }
                    self.next_track_id += 1
                    new_people += 1

        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]['missing_frames'] += 1

        self.tracks = {k: v for k, v in self.tracks.items()
                       if v['missing_frames'] <= self.memory_frames}

        return new_people

# --- Only run if script is executed directly ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Person Detection Streamer')
    parser.add_argument('--video', type=str, default='0',
                        help='Path to video file. Use 0 for webcam (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.2,
                        help='Minimum confidence threshold (default: 0.2)')
    parser.add_argument('--skip-frames', type=int, default=2,
                        help='Number of frames to skip between detections (default: 2)')
    args = parser.parse_args()

    PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"
    MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    VIDEO_SOURCE = 0 if args.video == '0' else args.video
    CONFIDENCE_THRESHOLD = args.confidence
    skip_frames = args.skip_frames

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {VIDEO_SOURCE}")
        exit(1)

    tracker = PersonTracker(memory_frames=15, max_distance=100)
    total_unique_people = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        (h, w) = frame.shape[:2]

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

        new_people = tracker.update(boxes, confidences)
        total_unique_people += new_people

        for track_id, track in tracker.tracks.items():
            if track['missing_frames'] > 0:
                continue

            box = track['box']
            confidence = track['confidence']

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"ID {track_id}: {confidence * 100:.1f}%"
            y = box[1] - 15 if box[1] - 15 > 15 else box[1] + 15
            cv2.putText(frame, label, (box[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            positions = list(track['positions'])[-5:]
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i - 1], positions[i], (0, 255, 0), 1)

        cv2.putText(frame, f"Active Tracks: {len(tracker.tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Unique People: {total_unique_people}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Person Detection Streamer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
