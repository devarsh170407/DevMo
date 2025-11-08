import cv2
import os
import sqlite3
import numpy as np
from datetime import datetime
from deepface import DeepFace
import time
import pickle

class FastDeepFaceAttendance:
    def __init__(self):
        self.db_path = "attendance.db"
        self.known_faces_dir = "known_faces"
        os.makedirs(self.known_faces_dir, exist_ok=True)
        self.setup_database()
        self.embeddings = self.load_embeddings()

    # ---------------- DATABASE ----------------
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT,
                name TEXT,
                date TEXT,
                time TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Database ready.")

    # ---------------- LOAD / SAVE EMBEDDINGS ----------------
    def load_embeddings(self):
        emb_file = "face_embeddings.pkl"
        if os.path.exists(emb_file):
            with open(emb_file, "rb") as f:
                return pickle.load(f)
        else:
            return {}

    def save_embeddings(self):
        with open("face_embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)

    # ---------------- REGISTER FACE ----------------
    def register_new_face(self):
        roll_no = input("Enter Roll No: ").strip()
        name = input("Enter Name: ").strip()
        folder = f"{roll_no}_{name}"
        path = os.path.join(self.known_faces_dir, folder)
        os.makedirs(path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        print("📸 Press 's' to save image, 'q' to quit.")
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register Face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                count += 1
                cv2.imwrite(os.path.join(path, f"{count}.jpg"), frame)
                print(f"✅ Saved photo {count}")
                if count >= 3:
                    break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # compute embedding
        print("🔍 Computing embeddings...")
        emb = DeepFace.represent(img_path=os.path.join(path, "1.jpg"), model_name="Facenet512", enforce_detection=False)
        self.embeddings[folder] = emb[0]['embedding']
        self.save_embeddings()
        print(f"✅ Registered {name} ({roll_no}) and stored embedding.")

    # ---------------- MARK ATTENDANCE ----------------
    def mark_attendance(self, roll_no, name):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")

        cur.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (roll_no, today))
        if not cur.fetchone():
            cur.execute("INSERT INTO attendance (roll_no, name, date, time, status) VALUES (?, ?, ?, ?, ?)",
                        (roll_no, name, today, now, "Present"))
            conn.commit()
            print(f"✅ Attendance marked for {name} ({roll_no})")
            print(f"🙏 Thank you, {name}!")
        conn.close()

    # ---------------- RECOGNIZE FACE ----------------
    def recognize_face(self, frame):
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, frame)
        emb = DeepFace.represent(img_path=temp_path, model_name="Facenet512", enforce_detection=False)
        os.remove(temp_path)

        if not emb:
            return None, None
        emb_vec = np.array(emb[0]['embedding'])

        best_match, best_dist = None, 0.4
        for person, known_emb in self.embeddings.items():
            known_vec = np.array(known_emb)
            dist = np.linalg.norm(known_vec - emb_vec)
            if dist < best_dist:
                best_dist = dist
                best_match = person

        if best_match:
            roll_no, name = best_match.split("_", 1)
            return roll_no, name
        return None, None

    # ---------------- START ATTENDANCE ----------------
    def start_attendance(self):
        cap = cv2.VideoCapture(0)
        print("\n🎥 Starting fast attendance system (press 'q' to quit)\n")
        already_marked = set()
        thank_you_time = {}

        frame_skip = 5
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                if frame_count % frame_skip != 0:
                    continue

                face_roi = frame[y:y+h, x:x+w]
                roll_no, name = self.recognize_face(face_roi)

                if roll_no and name:
                    color = (0, 255, 0)
                    cv2.putText(frame, f"{roll_no}-{name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if roll_no not in already_marked:
                        self.mark_attendance(roll_no, name)
                        already_marked.add(roll_no)
                        thank_you_time[name] = time.time()

                else:
                    color = (0, 0, 255)
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # show "Thank You" for 5 sec
            for name, t in list(thank_you_time.items()):
                if time.time() - t < 5:
                    h, w, _ = frame.shape
                    cv2.putText(frame, f"🙏 Thank you, {name}!", (20, h - 30),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    del thank_you_time[name]

            cv2.imshow("Fast DeepFace Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ---------------- MAIN ----------------
def main():
    system = FastDeepFaceAttendance()

    while True:
        print("\n========== DevMo Fast DeepFace Attendance ==========")
        print("1. Start Attendance")
        print("2. Register New Face")
        print("3. Exit")
        choice = input("Select an option: ").strip()

        if choice == '1':
            system.start_attendance()
        elif choice == '2':
            system.register_new_face()
        elif choice == '3':
            print("👋 Exiting system.")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
