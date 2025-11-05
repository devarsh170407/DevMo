import cv2
import face_recognition
import numpy as np
import os
import json
import csv
from datetime import datetime
import time

class FaceRecognitionAttendance:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_records = {}
        self.load_known_faces()
        self.load_attendance()
        
    def load_known_faces(self):
        """Load known faces from a directory"""
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created {known_faces_dir} directory. Please add face images there.")
            return
            
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])
                    print(f"Loaded face: {filename}")
                else:
                    print(f"No face found in {filename}")
    
    def load_attendance(self):
        """Load existing attendance records"""
        if os.path.exists("attendance.json"):
            with open("attendance.json", "r") as f:
                self.attendance_records = json.load(f)
    
    def save_attendance(self):
        """Save attendance records to JSON"""
        with open("attendance.json", "w") as f:
            json.dump(self.attendance_records, f, indent=2)
    
    def export_attendance_csv(self):
        """Export attendance to CSV"""
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance_{today}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Date', 'Time', 'Status'])
            
            for name, records in self.attendance_records.items():
                for record in records:
                    writer.writerow([name, record['date'], record['time'], record['status']])
        
        print(f"Attendance exported to {filename}")
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized face"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        if name not in self.attendance_records:
            self.attendance_records[name] = []
        
        # Check if already marked today
        today_records = [r for r in self.attendance_records[name] if r['date'] == today]
        if not today_records:
            record = {
                'date': today,
                'time': current_time,
                'status': 'Present'
            }
            self.attendance_records[name].append(record)
            print(f"Attendance marked for {name} at {current_time}")
            self.save_attendance()
            return True
        return False
    
    def run_recognition(self):
        """Run face recognition from webcam"""
        if not self.known_face_encodings:
            print("No known faces loaded. Please add face images to 'known_faces' directory.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Face recognition started. Press 'q' to quit, 'e' to export attendance")
        
        process_this_frame = True
        last_attendance_check = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            if process_this_frame:
                # Find all faces in current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    # Use the known face with the smallest distance
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            
                            # Mark attendance (once per minute per person)
                            current_time = time.time()
                            if name not in last_attendance_check or current_time - last_attendance_check[name] > 60:
                                self.mark_attendance(name)
                                last_attendance_check[name] = current_time
                    
                    face_names.append(name)
            
            process_this_frame = not process_this_frame
            
            # Display results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw box around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label with name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Recognition Attendance - DevMo', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.export_attendance_csv()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Export final attendance
        self.export_attendance_csv()

if __name__ == "__main__":
    attendance_system = FaceRecognitionAttendance()
    attendance_system.run_recognition()