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

class ImprovedFaceRecognition:
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
        self.face_encodings_loaded = False
        
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
    
    def improved_face_encoding(self, image, face_rect):
        """Create an improved face encoding with multiple features"""
        try:
            x, y, w, h = face_rect
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to standard size for consistency
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Convert to different color spaces for more features
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
            
            # Extract multiple features
            features = []
            
            # 1. Grayscale histogram (64 bins)
            hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
            features.extend(hist_gray)
            
            # 2. HSV histogram (for skin tone)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            features.extend(hist_h)
            
            # 3. LBP-like features (simplified)
            lbp_features = self.simple_lbp(gray)
            features.extend(lbp_features)
            
            # 4. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Convert to numpy array and normalize
            encoding = np.array(features, dtype=np.float32)
            encoding = cv2.normalize(encoding, None, 0, 1, cv2.NORM_MINMAX)
            
            return encoding
        except Exception as e:
            print(f"❌ Error creating face encoding: {e}")
            return None
    
    def simple_lbp(self, gray_image):
        """Simple Local Binary Pattern implementation"""
        height, width = gray_image.shape
        features = []
        
        # Divide image into 4x4 blocks
        block_h, block_w = height // 4, width // 4
        
        for i in range(4):
            for j in range(4):
                block = gray_image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                if block.size > 0:
                    # Simple texture feature: mean and std of block
                    features.append(np.mean(block))
                    features.append(np.std(block))
        
        return features
    
    def load_known_faces(self):
        """Load known faces from directory and generate improved encodings"""
        print("🔄 Loading known faces with improved recognition...")
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load from person folders
        face_count = 0
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                # Get all images for this person
                image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    successful_encodings = 0
                    # Try multiple images for better recognition
                    for image_file in image_files[:3]:  # Use up to 3 images per person
                        image_path = os.path.join(person_dir, image_file)
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Preprocess image
                            processed_image = self.preprocess_image(image)
                            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                            
                            # Detect faces with different parameters
                            faces = self.face_cascade.detectMultiScale(
                                gray, 
                                scaleFactor=1.05,  # More sensitive
                                minNeighbors=3,    # Less strict
                                minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                            
                            if len(faces) >= 1:
                                # Use the largest face found
                                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                                encoding = self.improved_face_encoding(processed_image, faces[0])
                                
                                if encoding is not None:
                                    self.known_face_encodings.append(encoding)
                                    self.known_face_names.append(person_name)
                                    successful_encodings += 1
                                    face_count += 1
                                    break  # One good encoding per person is enough for now
                    
                    if successful_encodings > 0:
                        print(f"✅ Loaded face: {person_name} ({successful_encodings} encodings)")
                    else:
                        print(f"❌ Failed to load face: {person_name}")
        
        # Save encodings for future use
        self.save_face_encodings()
        print(f"✅ Successfully loaded {face_count} face encodings from {len(set(self.known_face_names))} people")
        self.face_encodings_loaded = True
    
    def preprocess_image(self, image):
        """Preprocess image for better face detection"""
        try:
            # Resize if image is too large
            height, width = image.shape[:2]
            if height > 800 or width > 800:
                scale = min(800/height, 800/width)
                new_height, new_width = int(height * scale), int(width * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return image
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return image
    
    def save_face_encodings(self):
        """Save face encodings to file"""
        if self.known_face_encodings:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.face_encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Saved {len(self.known_face_encodings)} face encodings to cache")
    
    def recognize_face(self, face_encoding, threshold=0.4):
        """Recognize a face from known encodings with improved matching"""
        if not self.known_face_encodings or face_encoding is None:
            return None
        
        best_match_index = None
        best_distance = float('inf')
        
        for i, known_encoding in enumerate(self.known_face_encodings):
            try:
                # Use cosine similarity for better results
                distance = self.cosine_distance(face_encoding, known_encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_index = i
            except Exception as e:
                continue
        
        print(f"🔍 Best match distance: {best_distance:.3f} (threshold: {threshold})")
        
        # Use a threshold to determine if it's a match
        if best_distance < threshold and best_match_index is not None:
            matched_name = self.known_face_names[best_match_index]
            print(f"✅ Matched: {matched_name} with confidence {1-best_distance:.3f}")
            return matched_name
        
        print("❌ No match found")
        return None
    
    def cosine_distance(self, vec1, vec2):
        """Calculate cosine distance between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        
        cosine_similarity = dot_product / (norm1 * norm2)
        return 1 - cosine_similarity  # Convert to distance
    
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
        
        # Set camera resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
                processed_frame = self.preprocess_image(frame)
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                current_time = time.time()
                
                if current_time - last_capture_time >= 5:  # 5 second gap
                    # Take photo
                    photo_count += 1
                    filename = f"photo_{photo_count}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    
                    # Save high-quality image
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
    
    def run_improved_recognition(self):
        """Run improved face recognition attendance system"""
        if not self.face_encodings_loaded:
            print("❌ No face encodings loaded. Please register faces first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n🎥 Improved Face Recognition Started")
        print("🔍 Using enhanced face matching algorithm")
        print("📊 Known faces:", len(self.known_face_names))
        
        attendance_marked_today = set()
        recognition_confidence = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            display_frame = frame.copy()
            processed_frame = self.preprocess_image(frame)
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            current_time = time.time()
            unknown_detected = False
            
            for (x, y, w, h) in faces:
                # Create improved face encoding
                face_encoding = self.improved_face_encoding(processed_frame, (x, y, w, h))
                
                if face_encoding is not None:
                    # Try to recognize face with improved algorithm
                    name = self.recognize_face(face_encoding, threshold=0.3)
                    
                    if name:
                        # Known face
                        color = (0, 255, 0)  # Green
                        thickness = 2
                        
                        if name not in attendance_marked_today:
                            if self.mark_attendance(name):
                                attendance_marked_today.add(name)
                            status = f"✅ {name}"
                        else:
                            status = f"✅ {name} (Marked)"
                    else:
                        # Unknown face
                        unknown_detected = True
                        color = (0, 0, 255)  # Red
                        thickness = 3
                        status = "🔴 UNKNOWN"
                    
                    # Draw rectangle and label
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
                    cv2.putText(display_frame, status, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display statistics
            cv2.putText(display_frame, f"Known Faces: {len(self.known_face_names)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detected: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if unknown_detected:
                cv2.putText(display_frame, "STATUS: 🔴 UNKNOWN FACE", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "STATUS: 🟢 ALL KNOWN", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Improved Face Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = ImprovedFaceRecognition()
    
    while True:
        print("\n" + "="*60)
        print("           DevMo - IMPROVED Face Recognition System")
        print("="*60)
        print("1. Start Improved Face Recognition")
        print("2. Add New Face (Register Person)")
        print("3. View Today's Attendance")
        print("4. Export Attendance to CSV")
        print("5. View Known Faces")
        print("6. Reload Face Database")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == '1':
            system.run_improved_recognition()
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
            print("🔄 Reloading face database...")
            system.load_known_faces()
        elif choice == '7':
            print("Thank you for using DevMo Improved Face Recognition System!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()