import cv2
import mediapipe as mp
import sqlite3
import os
from datetime import datetime
from deepface import DeepFace
import time

# ---------------- SETUP ----------------
KNOWN_FACES_DIR = "known_faces"
DB_FILE = "attendance.db"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ---------------- DATABASE ----------------
def setup_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT,
            name TEXT,
            date TEXT,
            time TEXT,
            status TEXT DEFAULT 'Present'
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database Ready")

def mark_attendance(roll_no, name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    c.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (roll_no, today))
    if not c.fetchone():
        c.execute("INSERT INTO attendance (roll_no, name, date, time, status) VALUES (?, ?, ?, ?, ?)",
                  (roll_no, name, today, now, 'Present'))
        conn.commit()
        print(f"✅ Attendance marked for {name} ({roll_no}) — Thank you, {name}! 🙏")
        conn.close()
        return True
    conn.close()
    return False

def clear_attendance():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    print("🧹 All attendance data cleared successfully!")

# ---------------- REGISTER NEW FACE ----------------
def register_new_face():
    roll_no = input("Enter Roll Number: ").strip()
    name = input("Enter Name: ").strip()
    if not roll_no or not name:
        print("❌ Invalid input.")
        return

    person_dir = os.path.join(KNOWN_FACES_DIR, f"{roll_no}_{name}")
    os.makedirs(person_dir, exist_ok=True)
    print("📸 Capturing 3 photos. Press 's' to save each photo...")

    cap = cv2.VideoCapture(0)
    count = 0
    while count < 3:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Register Face - Press 's' to save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            count += 1
            cv2.imwrite(os.path.join(person_dir, f"photo_{count}.jpg"), frame)
            print(f"✅ Saved photo {count}")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Registration complete for {name} ({roll_no})")

# ---------------- FACE RECOGNITION ----------------
def recognize_face(frame):
    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, frame)
    try:
        result = DeepFace.find(
            img_path=temp_path,
            db_path=KNOWN_FACES_DIR,
            model_name='VGG-Face',
            enforce_detection=False,
            silent=True
        )
        if len(result) > 0 and not result[0].empty:
            matched_path = result[0].iloc[0]['identity']
            folder = os.path.basename(os.path.dirname(matched_path))
            roll_no, name = folder.split("_", 1)
            os.remove(temp_path)
            return roll_no, name
    except Exception as e:
        print("⚠️ Recognition error:", e)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return None, None

# ---------------- VIEW ATTENDANCE ----------------
def view_attendance():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT roll_no, name, time FROM attendance WHERE date=?", (today,))
    rows = c.fetchall()
    print(f"\n🗓️ Attendance for {today}:")
    if rows:
        for r in rows:
            print(f"🎓 {r[0]} - {r[1]} at {r[2]}")
    else:
        print("⚠️ No records yet.")
    conn.close()

# ---------------- HAND DETECTION ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

setup_database()

def start_system():
    cap = cv2.VideoCapture(0)
    print("\n✋ Use One Hand:")
    print("✌️ 2 Fingers → Register | 🖐️ 5 Fingers → Start Attendance | 🤟 3 Fingers → View | 👍 Thumb → Clear Data | ✊ Hold 3s → Exit\n")

    last_gesture = None
    gesture_start_time = None
    thank_you_message = None
    thank_you_time = 0
    THANK_YOU_DURATION = 5  # seconds
    HOLD_TIME = 3  # seconds to confirm gesture

    def count_fingers(hand_landmarks, hand_label):
        tips = [4, 8, 12, 16, 20]
        fingers = []
        if hand_label == "Right":
            fingers.append(1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x else 0)
        else:
            fingers.append(1 if hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[tips[0] - 1].x else 0)
        for tip in tips[1:]:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
        return sum(fingers)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = None
        closed_hand = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for handLms, handLabel in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                label = handLabel.classification[0].label
                finger_count = count_fingers(handLms, label)
                closed_hand = (finger_count == 0)

                if finger_count == 2:
                    gesture = "register"
                elif finger_count == 5:
                    gesture = "start"
                elif finger_count == 3:
                    gesture = "view"
                elif finger_count == 1:
                    gesture = "clear"
                elif finger_count == 0:
                    gesture = "exit"
                else:
                    gesture = None

        # Gesture Hold Detection (3 seconds)
        if gesture:
            if gesture_start_time is None or gesture != last_gesture:
                gesture_start_time = time.time()
                last_gesture = gesture
            else:
                elapsed = time.time() - gesture_start_time
                cv2.putText(frame, f"Holding {gesture}... {elapsed:.1f}s", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if elapsed >= HOLD_TIME:
                    print(f"\n🖐️ Gesture '{gesture}' confirmed after 3 seconds.")
                    gesture_start_time = None  # reset
                    last_gesture = None

                    if gesture == "register":
                        register_new_face()

                    elif gesture == "start":
                        print("\n🎥 Starting attendance...")
                        roll_no, name = recognize_face(frame)
                        if roll_no and name:
                            marked = mark_attendance(roll_no, name)
                            if marked:
                                thank_you_message = f"🙏 Thank you, {name}!"
                                thank_you_time = time.time()
                        else:
                            print("❌ No match found.")

                    elif gesture == "view":
                        view_attendance()

                    elif gesture == "clear":
                        clear_attendance()

                    elif gesture == "exit":
                        print("\n👋 Exiting system... Goodbye!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
        else:
            gesture_start_time = None
            last_gesture = None

        # Show thank-you message for 5 seconds
        if thank_you_message and (time.time() - thank_you_time < THANK_YOU_DURATION):
            h, w, _ = frame.shape
            cv2.putText(frame, thank_you_message, (int(w/4), h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            thank_you_message = None

        cv2.imshow("DevMo Gesture Attendance (Hold 3s)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Exiting system manually...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Start main system
start_system()
