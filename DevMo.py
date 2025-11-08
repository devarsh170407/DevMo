import cv2
import mediapipe as mp
import sqlite3
import os
from datetime import datetime
from deepface import DeepFace
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "devarshbhatt1747@gmail.com"
SENDER_PASSWORD = "fvxhscuxqaljgsjv" 
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587 

KNOWN_FACES_DIR = "known_faces"
DB_FILE = "attendance.db"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
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

def record_attendance(roll_no, name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    cursor.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (roll_no, today))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO attendance (roll_no, name, date, time, status) VALUES (?, ?, ?, ?, ?)",
                      (roll_no, name, today, current_time, 'Present'))
        conn.commit()
        print(f"Attendance recorded for {name} (Roll No: {roll_no})")
        conn.close()
        return True
    
    conn.close()
    return False

def reset_attendance_data():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance") 
    conn.commit()
    conn.close()
    print("All attendance records cleared")

def display_attendance():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT roll_no, name, time FROM attendance WHERE date=?", (today,))
    records = cursor.fetchall()
    
    print(f"\nToday's Attendance Report ({today}):")
    print("-" * 50)
    if records:
        for record in records:
            print(f"Roll No: {record[0]} | Name: {record[1]} | Time: {record[2]}")
    else:
        print("No attendance records found")
    print("-" * 50)
    conn.close()

def generate_attendance_report():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT roll_no, name, time FROM attendance WHERE date=?", (today,))
    records = cursor.fetchall()
    conn.close()

    report = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2> Daily Attendance Report - {today}</h2>
    """
    
    if records:
        report += "<table>"
        report += "<tr><th>Roll No</th><th>Student Name</th><th>Check-in Time</th></tr>"
        for record in records:
            report += f"<tr><td>{record[0]}</td><td>{record[1]}</td><td>{record[2]}</td></tr>"
        report += "</table>"
        report += f"<p><strong>Total Present: {len(records)} students</strong></p>"
    else:
        report += "<p>No attendance records marked yet for today.</p>"
    
    report += "</body></html>"
    return report

def email_attendance_report():
    """Send attendance report via email to teacher"""
    recipient_email = input("Enter teacher's email").strip()
    if not recipient_email:
        print("Please enter a valid email")
        return

    report_html = generate_attendance_report()
    
    message = MIMEMultipart()
    message['From'] = SENDER_EMAIL
    message['To'] = recipient_email
    message['Subject'] = f"Attendance Report - {datetime.now().strftime('%Y-%m-%d')}"
    
    message.attach(MIMEText(report_html, 'html'))
    
    try:
        print(f"Sending attendance...")
        email_server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        email_server.starttls()  
        email_server.login(SENDER_EMAIL, SENDER_PASSWORD)
        email_server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
        email_server.quit()
        print("Attendance report sent")
    except Exception as error:
        print(f"Failed to send email:")

def register_student_face():

    roll_no = input("Enter roll no.: ").strip()
    name = input("Enter Name: ").strip()
    
    if not roll_no or not name:
        print("Please provide both roll no. and name")
        return

    student_folder = os.path.join(KNOWN_FACES_DIR, f"{roll_no}_{name}")
    os.makedirs(student_folder, exist_ok=True)
    
    print("\nCapturing 3 photos for facial registration...")
    print("Press 's' to capture each photo")

    camera = cv2.VideoCapture(0)
    photos_captured = 0
    
    while photos_captured < 3:
        success, video_frame = camera.read()
        if not success:
            break
            
        cv2.putText(video_frame, f"Photo {photos_captured + 1}/3 - Press 's' to capture", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(video_frame, "Ensure face is clearly visible", 
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Student Registration - Face Capture", video_frame)
        key_press = cv2.waitKey(1) & 0xFF
        
        if key_press == ord('s'):
            photos_captured += 1
            cv2.imwrite(os.path.join(student_folder, f"photo_{photos_captured}.jpg"), video_frame)
            print(f"Photo saved")
            
            cv2.putText(video_frame, "CAPTURED!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 3)
            cv2.imshow("Student Registration - Face Capture", video_frame)
            cv2.waitKey(500)
            
        elif key_press == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()
    print(f"Registration done for {name} (Roll No: {roll_no})")

def identify_face(video_frame):
    """Identify a face from the database"""
    temp_image_path = "temp_face.jpg"
    cv2.imwrite(temp_image_path, video_frame)
    
    try:
        recognition_result = DeepFace.find(
            img_path=temp_image_path,
            db_path=KNOWN_FACES_DIR,
            model_name='VGG-Face',
            enforce_detection=False,
            silent=True
        )
        
        if len(recognition_result) > 0 and not recognition_result[0].empty:
            matched_image_path = recognition_result[0].iloc[0]['identity']
            folder_name = os.path.basename(os.path.dirname(matched_image_path))
            roll_no, name = folder_name.split("_", 1)
            os.remove(temp_image_path)
            return roll_no, name
    except Exception as e:
        pass
        
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    return None, None

hand_recognizer = mp.solutions.hands
hands_detector = hand_recognizer.Hands(max_num_hands=1, min_detection_confidence=0.8) 
hand_drawer = mp.solutions.drawing_utils

initialize_database()

def launch_attendance_system():
    camera = cv2.VideoCapture(0)
    
    print("\n" + "="*60)
    print("DevMo Attendance System")
    print("="*60)
    print("1 Finger   -> Email Report")
    print("2 Fingers  -> Register New Student")
    print("3 Fingers  -> View Today's Attendance")
    print("5 Fingers  -> Mark Attendance")
    print("4 Fingers  -> Clear All Data")
    print("Closed Hand-> Exit System")

    current_gesture = None
    gesture_timer = None
    thank_you_display = None
    thank_you_timestamp = 0
    THANK_YOU_DISPLAY_TIME = 5  
    GESTURE_HOLD_TIME = 3  

    def count_raised_fingers(hand_landmarks, hand_side):
        """Count number of fingers raised"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_states = []
        
        if hand_side == "Right":
            finger_states.append(1 if hand_landmarks.landmark[finger_tips[0]].x < 
                                hand_landmarks.landmark[finger_tips[0] - 1].x else 0)
        else: 
            finger_states.append(1 if hand_landmarks.landmark[finger_tips[0]].x > 
                                hand_landmarks.landmark[finger_tips[0] - 1].x else 0)
            
        for tip in finger_tips[1:]:
            finger_states.append(1 if hand_landmarks.landmark[tip].y < 
                                hand_landmarks.landmark[tip - 2].y else 0)
            
        return sum(finger_states)

    while True:
        success, frame = camera.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(rgb_frame)

        detected_gesture = None

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, hand_info in zip(hand_results.multi_hand_landmarks, 
                                               hand_results.multi_handedness):
                hand_drawer.draw_landmarks(frame, hand_landmarks, hand_recognizer.HAND_CONNECTIONS)
                hand_side = hand_info.classification[0].label
                finger_count = count_raised_fingers(hand_landmarks, hand_side)

                gesture_map = {
                    1: "email",
                    2: "register", 
                    3: "view",
                    5: "start",
                    4: "clear",
                    0: "exit"
                }
                
                detected_gesture = gesture_map.get(finger_count)
                        
                cv2.putText(frame, f"Fingers: {finger_count} | Hold for 3 seconds", (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if detected_gesture:
            if gesture_timer is None or detected_gesture != current_gesture:
                gesture_timer = time.time()
                current_gesture = detected_gesture
            else:
                time_held = time.time() - gesture_timer
                
                if time_held >= GESTURE_HOLD_TIME:
                    gesture_timer = None  
                    current_gesture = None

                    if detected_gesture == "email":
                        email_attendance_report()

                    elif detected_gesture == "register":
                        register_student_face()

                    elif detected_gesture == "start":
                        print("🔍 Starting facial recognition...")
                        roll_no, name = identify_face(frame)
                        if roll_no and name:
                            attendance_marked = record_attendance(roll_no, name)
                            if attendance_marked:
                                thank_you_display = f"Thank you, {name}!" 
                                thank_you_timestamp = time.time()
                        else:
                            print("No match")

                    elif detected_gesture == "view": 
                        display_attendance() 

                    elif detected_gesture == "clear":
                        confirm = input("Are you sure you want to clear ALL attendance data? (y/n): ")
                        if confirm.lower() == 'y':
                            reset_attendance_data()
                        else:
                            print("Data clearance cancelled")

                    elif detected_gesture == "exit":
                        print("Shutting down attendance system...")
                        camera.release()
                        cv2.destroyAllWindows()
                        return
        else:
            gesture_timer = None
            current_gesture = None

        if thank_you_display and (time.time() - thank_you_timestamp < THANK_YOU_DISPLAY_TIME):
            frame_height, frame_width, _ = frame.shape
            cv2.putText(frame, thank_you_display, (int(frame_width/4), frame_height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            thank_you_display = None

        cv2.imshow("DevMo", frame)

    camera.release()
    cv2.destroyAllWindows()

# Start the system
if __name__ == "__main__":
    launch_attendance_system()

