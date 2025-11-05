import cv2
import numpy as np
import os
import json
import pickle
import sqlite3
from datetime import datetime
import time
import threading
import shutil

class EnhancedFaceRecognition:
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
        self.unknown_face_detected = False
        self.unknown_face_timer = 0
        
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
        
        # Load from person folders
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                # Get all images for this person
                image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    # Use the first image for encoding
                    first_image_path = os.path.join(person_dir, image_files[0])
                    image = cv2.imread(first_image_path)
                    
                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        if len(faces) == 1:
                            encoding = self.simple_face_encoding(image, faces[0])
                            if encoding is not None:
                                self.known_face_encodings.append(encoding)
                                self.known_face_names.append(person_name)
                                print(f"✅ Loaded face: {person_name} ({len(image_files)} photos)")
        
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
    
    def register_new_face(self):
        """Register a new face by taking 3 photos with 5-second gaps"""
        print("\n" + "="*50)
        print("         📸 FACE REGISTRATION PROCESS")
        print("="*50)
        
        # Get person's name
        name = input("Enter the person's name: ").strip()
        if not name:
            print("❌ No name provided. Registration cancelled.")
            return
        
        # Create person's folder
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        person_dir = os.path.join(self.known_faces_dir, safe_name)
        os.makedirs(person_dir, exist_ok=True)
        
        print(f"\n✅ Created folder for: {name}")
        print("📸 Get ready to take 3 photos...")
        print("   - Photos will be taken every 5 seconds")
        print("   - Please look directly at the camera")
        print("   - Make sure your face is clearly visible")
        print("\nPress 's' to start capturing, 'q' to cancel")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        photo_count = 0
        capturing = False
        last_capture_time = 0
        
        while photo_count < 3:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Display instructions
            display_frame = frame.copy()
            
            if not capturing:
                cv2.putText(display_frame, f"Ready to capture: {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 's' to START capturing", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'q' to CANCEL", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Detect face during capture
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                current_time = time.time()
                
                if current_time - last_capture_time >= 5:  # 5 second gap
                    # Take photo
                    photo_count += 1
                    filename = f"photo_{photo_count}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    cv2.imwrite(filepath, frame)
                    last_capture_time = current_time
                    
                    print(f"✅ Photo {photo_count}/3 captured and saved!")
                    
                    # Show countdown for next photo
                    if photo_count < 3:
                        print(f"⏰ Next photo in 5 seconds...")
                
                # Display capture progress
                cv2.putText(display_frame, f"CAPTURING: {name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Photos: {photo_count}/3", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if photo_count < 3:
                    time_remaining = 5 - (current_time - last_capture_time)
                    cv2.putText(display_frame, f"Next photo in: {max(0, int(time_remaining))}s", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "✅ CAPTURE COMPLETE!", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press 'q' to finish", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Registration', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not capturing:
                capturing = True
                last_capture_time = time.time() - 5  # Start immediately
                print("🚀 Starting capture process...")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if photo_count == 3:
            print(f"\n🎉 Successfully registered {name} with 3 photos!")
            print(f"📁 Photos saved in: {person_dir}")
            
            # Reload known faces to include the new person
            self.load_known_faces()
        else:
            print(f"\n❌ Registration incomplete. Only {photo_count}/3 photos captured.")
            # Remove the folder if registration was cancelled
            if os.path.exists(person_dir) and photo_count == 0:
                shutil.rmtree(person_dir)
                print("🗑️ Registration folder removed.")
    
    def save_new_face(self, image, face_rect, name):
        """Save a new face image from automatic detection"""
        try:
            x, y, w, h = face_rect
            
            # Create person's folder
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            person_dir = os.path.join(self.known_faces_dir, safe_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Save single photo for automatic detection
            filename = f"auto_detected.jpg"
            filepath = os.path.join(person_dir, filename)
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
        print("🔴 UNKNOWN PERSON DETECTED!")
        print("Please enter the name for this face:")
        self.name_input = input("Name: ").strip()
        self.pending_name_input = False
    
    def process_unknown_face(self, image, face_rect, face_encoding):
        """Process an unknown face - ask for name and save"""
        if self.pending_name_input:
            return "⏳ Waiting for name input..."
        
        # Set unknown face flag for visual feedback
        self.unknown_face_detected = True
        self.unknown_face_timer = time.time()
        
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
        
        return "🔴 UNKNOWN - Enter name in console"
    
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
            
            # Reset unknown face flag
            self.unknown_face_detected = False
            
            # Reset
            self.current_unknown_face = None
            self.name_input = ""
            return True
        
        return False
    
    def draw_red_alert(self, frame):
        """Draw red alert border and warning message"""
        height, width = frame.shape[:2]
        
        # Draw red border
        border_thickness = 10
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), border_thickness)
        
        # Draw flashing red background for warning text
        current_time = time.time()
        if int(current_time * 2) % 2 == 0:  # Flash every 0.5 seconds
            cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 255), -1)
            cv2.putText(frame, "🚨 UNKNOWN FACE DETECTED!", (width//2 - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Please enter name in console", (width//2 - 180, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_interactive_attendance(self):
        """Run interactive face recognition attendance system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n🎥 Interactive Face Recognition Started")
        print("🔍 System will:")
        print("   - Show 🟢 GREEN for known faces")
        print("   - Show 🔴 RED for unknown faces")
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
                minSize=(50, 50)
            )
            
            current_time = time.time()
            face_info = []
            unknown_faces_detected = False
            
            # Check for completed name input
            if self.check_name_input():
                # Refresh known faces after adding new one
                self.load_known_faces()
            
            # Reset unknown face flag after 3 seconds if no input
            if self.unknown_face_detected and current_time - self.unknown_face_timer > 3:
                self.unknown_face_detected = False
            
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
                        thickness = 2
                    else:
                        # Unknown face
                        unknown_faces_detected = True
                        face_id = f"unknown_{hash(face_encoding.tobytes()) % 10000:04d}"
                        
                        # Check if we're already processing this face
                        if face_id in recent_faces and current_time - recent_faces[face_id] < 5:
                            status = "⏳ Processing..."
                            color = (255, 165, 0)  # Orange
                            name = "Unknown"
                            thickness = 2
                        else:
                            # New unknown face - ask for name
                            status = self.process_unknown_face(frame, (x, y, w, h), face_encoding)
                            recent_faces[face_id] = current_time
                            color = (0, 0, 255)  # RED for unknown
                            name = "UNKNOWN"
                            thickness = 3  # Thicker border for unknown faces
                    
                    face_info.append((x, y, w, h, name, status, color, thickness))
            
            # Draw face rectangles and info
            for (x, y, w, h, name, status, color, thickness) in face_info:
                # Draw rectangle with appropriate thickness
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
                
                # Draw name and status with background for better visibility
                text_color = (255, 255, 255)  # White text
                
                # For unknown faces, use red background
                if color == (0, 0, 255):
                    bg_color = (0, 0, 255)  # Red background
                    text_size_name = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x, y-35), (x+text_size_name[0], y), bg_color, -1)
                    cv2.putText(display_frame, name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    
                    text_size_status = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(display_frame, (x, y+h), (x+text_size_status[0], y+h+25), bg_color, -1)
                    cv2.putText(display_frame, status, (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                else:
                    # For known faces, use semi-transparent background
                    cv2.putText(display_frame, name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(display_frame, status, (x, y+h+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw red alert if unknown faces are detected
            if unknown_faces_detected or self.unknown_face_detected:
                display_frame = self.draw_red_alert(display_frame)
            
            # Display stats with colored backgrounds
            cv2.rectangle(display_frame, (5, 5), (300, 100), (0, 0, 0), -1)  # Black background for stats
            
            cv2.putText(display_frame, f'Known Faces: {len(self.known_face_names)}', (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f'Faces Detected: {len(faces)}', (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show unknown face warning
            if unknown_faces_detected:
                cv2.putText(display_frame, 'UNKNOWN FACE: 🔴', (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, 'Status: 🟢 All Known', (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Show pending input status
            if self.pending_name_input:
                warning_text = "🔴 ENTER NAME IN CONSOLE NOW!"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (5, 105), (5 + text_size[0] + 10, 140), (0, 0, 255), -1)
                cv2.putText(display_frame, warning_text, (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display controls
            cv2.putText(display_frame, "Press 'q':Quit 'v':View 'e':Export 'c':Clear", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('DevMo - Face Recognition (🔴=Unknown)', display_frame)
            
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
                    self.unknown_face_detected = False
                    self.current_unknown_face = None
                    self.name_input = ""
                    print("✅ Cleared pending name input")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = EnhancedFaceRecognition()
    
    while True:
        print("\n" + "="*60)
        print("           DevMo - Enhanced Face Recognition System")
        print("="*60)
        print("1. Start Interactive Attendance System")
        print("2. Add New Face (Register Person)")
        print("3. View Today's Attendance")
        print("4. Export Attendance to CSV")
        print("5. View Known Faces")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            system.run_interactive_attendance()
        elif choice == '2':
            system.register_new_face()
        elif choice == '3':
            system.view_attendance()
        elif choice == '4':
            count = system.export_attendance()
            print(f"Exported {count} attendance records")
        elif choice == '5':
            print(f"\nKnown faces: {len(system.known_face_names)}")
            for name in system.known_face_names:
                person_dir = os.path.join(system.known_faces_dir, name)
                photo_count = len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"👤 {name} ({photo_count} photos)")
        elif choice == '6':
            print("Thank you for using DevMo Enhanced Face Recognition System!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()