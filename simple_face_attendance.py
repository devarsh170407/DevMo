import cv2
import numpy as np
import os
import json
import csv
from datetime import datetime
import time
import sqlite3

class SimpleFaceAttendance:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.attendance_db = "attendance.db"
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for attendance"""
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT DEFAULT 'Present',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    
    def mark_attendance(self, name):
        """Mark attendance in database"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        
        # Check if already marked today
        cursor.execute(
            "SELECT id FROM attendance WHERE name = ? AND date = ?",
            (name, today)
        )
        
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                (name, today, current_time, 'Present')
            )
            conn.commit()
            print(f"✅ Attendance marked for {name} at {current_time}")
        else:
            print(f"ℹ️ {name} already marked present today")
        
        conn.close()
    
    def export_attendance(self):
        """Export today's attendance to CSV"""
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance_{today}.csv"
        
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name, date, time, status FROM attendance WHERE date = ? ORDER BY time",
            (today,)
        )
        
        records = cursor.fetchall()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Date', 'Time', 'Status'])
            writer.writerows(records)
        
        conn.close()
        print(f"📊 Attendance exported to {filename}")
        return len(records)
    
    def view_attendance(self):
        """View today's attendance records"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name, time FROM attendance WHERE date = ? ORDER BY time",
            (today,)
        )
        
        records = cursor.fetchall()
        
        print(f"\n--- Today's Attendance ({today}) ---")
        if records:
            for name, time in records:
                print(f"👤 {name}: {time}")
        else:
            print("No attendance records for today")
        
        conn.close()
        return records
    
    def manual_attendance(self):
        """Manual attendance entry for testing"""
        print("\n--- Manual Attendance Entry ---")
        name = input("Enter student name: ").strip()
        if name:
            self.mark_attendance(name)
    
    def run_face_detection(self):
        """Run basic face detection with manual naming"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n🎥 Face Detection Started")
        print("Press 'm' to manually mark attendance")
        print("Press 's' to save current face count")
        print("Press 'v' to view attendance")
        print("Press 'e' to export attendance")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display face count
            cv2.putText(frame, f'Faces Detected: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'm' to mark attendance", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('DevMo - Face Detection Attendance', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.manual_attendance()
            elif key == ord('s'):
                if len(faces) > 0:
                    name = f"Unknown_{len(faces)}_faces"
                    self.mark_attendance(name)
            elif key == ord('v'):
                self.view_attendance()
            elif key == ord('e'):
                count = self.export_attendance()
                print(f"Exported {count} records")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = SimpleFaceAttendance()
    
    while True:
        print("\n" + "="*50)
        print("          DevMo - Attendance System")
        print("="*50)
        print("1. Start Face Detection")
        print("2. Manual Attendance Entry")
        print("3. View Today's Attendance")
        print("4. Export Attendance to CSV")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            system.run_face_detection()
        elif choice == '2':
            system.manual_attendance()
        elif choice == '3':
            system.view_attendance()
        elif choice == '4':
            count = system.export_attendance()
            print(f"Exported {count} attendance records")
        elif choice == '5':
            print("Thank you for using DevMo Attendance System!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()