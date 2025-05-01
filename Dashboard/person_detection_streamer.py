'''
Person Detection Streamer with Robust Tracking

This script reads from a video source and detects people using MobileNet-SSD model,
with improved tracking to handle drone footage and moving cameras.
'''  
import cv2
import numpy as np
from collections import deque
import math

class PersonTracker:
    def __init__(self, memory_frames=30, max_distance=100):
        self.tracks = {}
        self.next_track_id = 0
        self.memory_frames = memory_frames
        self.max_distance = max_distance
        
    def calculate_centroid(self, box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    
    def calculate_distance(self, centroid1, centroid2):
        return math.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
    
    def update(self, boxes, confidences):
        current_centroids = [self.calculate_centroid(box) for box in boxes]
        
        if not self.tracks:
            for centroid, box, conf in zip(current_centroids, boxes, confidences):
                self.tracks[self.next_track_id] = {
                    'centroid': centroid,
                    'box': box,
                    'confidence': conf,
                    'missing_frames': 0,
                    'positions': deque(maxlen=10)  # Reduced memory frames
                }
                self.tracks[self.next_track_id]['positions'].append(centroid)
                self.next_track_id += 1
            return len(current_centroids)
        
        # Match current detections with existing tracks
        new_people = 0
        matched_tracks = set()
        matched_detections = set()
        
        # First, try to match detections with existing tracks
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
            
            if best_match is not None:
                i, centroid, box, conf = best_match
                matched_detections.add(i)
                matched_tracks.add(track_id)
                
                # Update track
                track['centroid'] = centroid
                track['box'] = box
                track['confidence'] = conf
                track['missing_frames'] = 0
                track['positions'].append(centroid)
        
        # Create new tracks for unmatched detections
        for i, (centroid, box, conf) in enumerate(zip(current_centroids, boxes, confidences)):
            if i not in matched_detections:
                # Check if this might be a new person
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
                        'positions': deque(maxlen=self.memory_frames)
                    }
                    self.tracks[self.next_track_id]['positions'].append(centroid)
                    self.next_track_id += 1
                    new_people += 1
        
        # Update missing frames counter for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]['missing_frames'] += 1
        
        # Remove old tracks
        self.tracks = {k: v for k, v in self.tracks.items() 
                      if v['missing_frames'] <= self.memory_frames}
        
        return new_people

# Only define the class and its methods, remove the execution code
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
