import cv2
import numpy as np
import os
import json
import pickle
import sqlite3
from datetime import datetime
import time
import threading

class InteractiveFaceRecognition:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Database and storage setup
        self.attendance_db = "attendance.db"
        self.known_faces_dir = "known_faces"
        self.face_encodings_file = "face_encodings.pkl"
        
        # Data structures
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Tracking variables
        self.current_unknown_face = None
        self.pending_name_input = False
        self.name_input = ""
        
        # Create directories if they don't exist
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        # Initialize systems
        self.setup_database()
        self.load_known_faces()
        
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
        print("✅ Database initialized successfully")
    
    def load_known_faces(self):
        """Load known faces from directory and generate encodings"""
        print("🔄 Loading known faces...")
        
        # Try to load pre-computed encodings
        if os.path.exists(self.face_encodings_file):
            try:
                with open(self.face_encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"✅ Loaded {len(self.known_face_names)} known faces from cache")
                return
            except:
                print("❌ Failed to load cached encodings, regenerating...")
        
        # Regenerate encodings from images
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.known_faces_dir, filename)
                
                # Load and process image
                image = cv2.imread(image_path)
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) == 1:
                        # Simple face encoding (using face region as feature vector)
                        encoding = self.simple_face_encoding(image, faces[0])
                        if encoding is not None:
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(name)
                            print(f"✅ Loaded face: {name}")
                    else:
                        print(f"⚠️  Skipped {filename}: {len(faces)} faces detected (need exactly 1)")
        
        # Save encodings for future use
        self.save_face_encodings()
        print(f"✅ Loaded {len(self.known_face_names)} known faces")
    
    def save_face_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.face_encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def simple_face_encoding(self, image, face_rect):
        """Create a simple face encoding using face region features"""
        try:
            x, y, w, h = face_rect
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Convert to grayscale
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Flatten and normalize
            encoding = face_gray.flatten().astype(np.float32)
            encoding /= 255.0  # Normalize to [0, 1]
            
            return encoding
        except Exception as e:
            print(f"❌ Error creating face encoding: {e}")
            return None
    
    def recognize_face(self, face_encoding):
        """Recognize a face from known encodings"""
        if not self.known_face_encodings:
            return None
        
        best_match_index = None
        best_distance = float('inf')
        
        for i, known_encoding in enumerate(self.known_face_encodings):
            distance = np.linalg.norm(face_encoding - known_encoding)
            if distance < best_distance:
                best_distance = distance
                best_match_index = i
        
        # Use a threshold to determine if it's a match
        if best_distance < 0.6:  # Adjust this threshold as needed
            return self.known_face_names[best_match_index]
        
        return None
    
    def save_new_face(self, image, face_rect, name):
        """Save a new face image and add to known faces"""
        try:
            x, y, w, h = face_rect
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Create filename (remove special characters)
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}.jpg"
            filepath = os.path.join(self.known_faces_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, face_roi)
            print(f"✅ Saved new face: {name}")
            
            # Add to known faces
            encoding = self.simple_face_encoding(image, face_rect)
            if encoding is not None:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                self.save_face_encodings()
            
            return True
        except Exception as e:
            print(f"❌ Error saving new face: {e}")
            return False
    
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
            result = True
        else:
            print(f"ℹ️ {name} already marked present today")
            result = False
        
        conn.close()
        return result
    
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
        
        # Write to CSV
        import csv
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
            print(f"Total: {len(records)} students")
        else:
            print("No attendance records for today")
        
        conn.close()
        return records
    
    def get_name_input(self):
        """Get name input from user in a separate thread"""
        print("\n🎯 NEW FACE DETECTED!")
        print("Please enter the name for this face:")
        self.name_input = input("Name: ").strip()
        self.pending_name_input = False
    
    def process_unknown_face(self, image, face_rect, face_encoding):
        """Process an unknown face - ask for name and save"""
        if self.pending_name_input:
            return "⏳ Waiting for name input..."
        
        # Store the unknown face data
        self.current_unknown_face = {
            'image': image.copy(),
            'face_rect': face_rect,
            'encoding': face_encoding
        }
        
        # Start name input in a separate thread
        self.pending_name_input = True
        input_thread = threading.Thread(target=self.get_name_input)
        input_thread.daemon = True
        input_thread.start()
        
        return "🆕 Please enter name in console"
    
    def check_name_input(self):
        """Check if name input is complete and process the face"""
        if not self.pending_name_input and self.current_unknown_face is not None and self.name_input:
            # Process the saved face with the provided name
            image = self.current_unknown_face['image']
            face_rect = self.current_unknown_face['face_rect']
            encoding = self.current_unknown_face['encoding']
            name = self.name_input
            
            # Save the new face
            if self.save_new_face(image, face_rect, name):
                # Mark attendance
                self.mark_attendance(name)
                print(f"✅ Successfully registered {name} and marked attendance!")
            
            # Reset
            self.current_unknown_face = None
            self.name_input = ""
            return True
        
        return False
    
    def run_interactive_attendance(self):
        """Run interactive face recognition attendance system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n🎥 Interactive Face Recognition Started")
        print("🔍 System will:")
        print("   - Automatically recognize known faces")
        print("   - Ask for name when new face is detected")
        print("   - Prevent duplicate attendance")
        print("\nControls:")
        print("   - 'q': Quit")
        print("   - 'v': View attendance")
        print("   - 'e': Export attendance")
        print("   - 'c': Clear pending name input")
        
        # Track recently processed faces to avoid duplicates
        recent_faces = {}
        attendance_marked_today = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)  # Larger minimum size for better recognition
            )
            
            current_time = time.time()
            face_info = []
            
            # Check for completed name input
            if self.check_name_input():
                # Refresh known faces after adding new one
                self.load_known_faces()
            
            for (x, y, w, h) in faces:
                # Create face encoding
                face_encoding = self.simple_face_encoding(frame, (x, y, w, h))
                
                if face_encoding is not None:
                    # Try to recognize face
                    name = self.recognize_face(face_encoding)
                    
                    if name:
                        # Known face - check if attendance already marked today
                        if name not in attendance_marked_today:
                            # Mark attendance for known face (only once per day)
                            if self.mark_attendance(name):
                                attendance_marked_today.add(name)
                            status = "✅ Known - Attendance Marked"
                        else:
                            status = "✅ Known - Already Marked"
                        color = (0, 255, 0)  # Green
                    else:
                        # Unknown face
                        face_id = f"unknown_{hash(face_encoding.tobytes()) % 10000:04d}"
                        
                        # Check if we're already processing this face
                        if face_id in recent_faces and current_time - recent_faces[face_id] < 5:
                            status = "⏳ Processing..."
                            color = (255, 165, 0)  # Orange
                            name = "Unknown"
                        else:
                            # New unknown face - ask for name
                            status = self.process_unknown_face(frame, (x, y, w, h), face_encoding)
                            recent_faces[face_id] = current_time
                            color = (255, 255, 0)  # Yellow
                            name = "Unknown"
                    
                    face_info.append((x, y, w, h, name, status, color))
            
            # Draw face rectangles and info
            for (x, y, w, h, name, status, color) in face_info:
                # Draw rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name and status
                cv2.putText(display_frame, f'{name}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(display_frame, status, (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Display stats
            cv2.putText(display_frame, f'Known Faces: {len(self.known_face_names)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f'Faces Detected: {len(faces)}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show pending input status
            if self.pending_name_input:
                status_text = "🟡 WAITING FOR NAME INPUT - Check console!"
                cv2.putText(display_frame, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display controls
            cv2.putText(display_frame, "Press 'q':Quit 'v':View 'e':Export 'c':Clear", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('DevMo - Interactive Face Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                self.view_attendance()
            elif key == ord('e'):
                count = self.export_attendance()
                print(f"Exported {count} records")
            elif key == ord('c'):
                if self.pending_name_input:
                    self.pending_name_input = False
                    self.current_unknown_face = None
                    self.name_input = ""
                    print("✅ Cleared pending name input")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = InteractiveFaceRecognition()
    
    while True:
        print("\n" + "="*60)
        print("           DevMo - Interactive Face Recognition")
        print("="*60)
        print("1. Start Interactive Attendance System")
        print("2. View Today's Attendance")
        print("3. Export Attendance to CSV")
        print("4. View Known Faces")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            system.run_interactive_attendance()
        elif choice == '2':
            system.view_attendance()
        elif choice == '3':
            count = system.export_attendance()
            print(f"Exported {count} attendance records")
        elif choice == '4':
            print(f"\nKnown faces: {len(system.known_face_names)}")
            for name in system.known_face_names:
                print(f"👤 {name}")
        elif choice == '5':
            print("Thank you for using DevMo Interactive Face Recognition System!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()