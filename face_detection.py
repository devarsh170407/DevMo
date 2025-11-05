import cv2
import numpy as np
import os
import time
from datetime import datetime

class BasicFaceDetector:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.attendance_log = []
        
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def draw_faces(self, frame, faces):
        """Draw rectangles around detected faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame
    
    def log_attendance(self, face_count):
        """Log attendance with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'face_count': face_count,
            'date': datetime.now().strftime("%Y-%m-%d")
        }
        self.attendance_log.append(log_entry)
        print(f"Attendance logged: {face_count} faces detected at {timestamp}")
    
    def run_detection(self):
        """Run face detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Face detection started. Press 'q' to quit, 's' to save attendance")
        last_save_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw faces on frame
            frame_with_faces = self.draw_faces(frame, faces)
            
            # Display face count
            cv2.putText(frame_with_faces, f'Faces: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Face Detection - DevMo', frame_with_faces)
            
            # Auto-save attendance every 30 seconds
            current_time = time.time()
            if current_time - last_save_time > 30 and len(faces) > 0:
                self.log_attendance(len(faces))
                last_save_time = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.log_attendance(len(faces))
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final attendance log
        print("\n--- Attendance Summary ---")
        for entry in self.attendance_log:
            print(f"{entry['timestamp']}: {entry['face_count']} faces")

if __name__ == "__main__":
    detector = BasicFaceDetector()
    detector.run_detection()