import cv2
import os
import sqlite3
from datetime import datetime
from deepface import DeepFace
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import csv


class DeepFaceAttendanceSystem:
    def __init__(self):
        self.known_faces_dir = "known_faces"
        self.attendance_db = "attendance.db"
        os.makedirs(self.known_faces_dir, exist_ok=True)
        self.setup_database()

    # ---------------- DATABASE SETUP ----------------
    def setup_database(self):
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT NOT NULL,
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

    # ---------------- REGISTER NEW FACE ----------------
    def register_new_face(self):
        roll_no = input("Enter Roll Number: ").strip()
        name = input("Enter the person's name: ").strip()
        if not roll_no or not name:
            print("❌ Roll number or name missing. Cancelled.")
            return

        person_dir = os.path.join(self.known_faces_dir, f"{roll_no}_{name}")
        os.makedirs(person_dir, exist_ok=True)
        print(f"\n📸 Capturing 3 photos for registration... (press 's' to save each)")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera.")
            return

        count = 0
        while count < 3:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register Face - Press 's' to save", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                count += 1
                file_path = os.path.join(person_dir, f"photo_{count}.jpg")
                cv2.imwrite(file_path, frame)
                print(f"✅ Saved {file_path}")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Registration completed for {name} ({roll_no})")

    # ---------------- MARK ATTENDANCE ----------------
    def mark_attendance(self, roll_no, name):
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")

        cursor.execute("SELECT * FROM attendance WHERE roll_no=? AND date=?", (roll_no, today))
        result = cursor.fetchone()

        if not result:
            cursor.execute(
                "INSERT INTO attendance (roll_no, name, date, time, status) VALUES (?, ?, ?, ?, ?)",
                (roll_no, name, today, now, 'Present')
            )
            conn.commit()
            print(f"✅ Attendance marked for {name} ({roll_no}) at {now}")
            print(f"🙏 Thank you, {name}! Have a great day! 🌞")
            conn.close()
            return True
        else:
            conn.close()
            return False

    # ---------------- RECOGNIZE FACE ----------------
    def recognize_face(self, face_path):
        try:
            result = DeepFace.find(
                img_path=face_path,
                db_path=self.known_faces_dir,
                model_name='VGG-Face',
                enforce_detection=False,
                silent=True
            )
            if len(result) > 0 and not result[0].empty:
                matched_path = result[0].iloc[0]['identity']
                folder = os.path.basename(os.path.dirname(matched_path))
                roll_no, name = folder.split("_", 1)
                return roll_no, name
            else:
                return None, None
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
            return None, None

    # ---------------- VIEW ATTENDANCE ----------------
    def view_attendance(self):
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT roll_no, name, time FROM attendance WHERE date=?", (today,))
        records = cursor.fetchall()
        print(f"\n🗓️ Attendance for {today}:")
        if records:
            for roll, name, t in records:
                print(f"🎓 {roll} - {name} — {t}")
        else:
            print("No records found.")
        conn.close()

    # ---------------- EXPORT ATTENDANCE ----------------
    def export_attendance(self):
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT roll_no, name, date, time, status FROM attendance WHERE date=?", (today,))
        records = cursor.fetchall()
        if not records:
            print("⚠️ No data to export.")
            return
        filename = f"attendance_{today}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Roll No", "Name", "Date", "Time", "Status"])
            writer.writerows(records)
        conn.close()
        print(f"📁 Exported attendance to {filename}")

    # ---------------- CLEAR ATTENDANCE DATA ----------------
    def clear_attendance_data(self):
        confirm = input("⚠️ Delete ALL attendance records? (y/n): ").lower()
        if confirm == 'y':
            conn = sqlite3.connect(self.attendance_db)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance")
            conn.commit()
            conn.close()
            print("🧹 All attendance data cleared.")
        else:
            print("❌ Cancelled.")

    # ---------------- SEND ATTENDANCE EMAIL ----------------
    def send_attendance_email(self, receiver_email="teacher@gmail.com"):
        conn = sqlite3.connect(self.attendance_db)
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT roll_no, name, time FROM attendance WHERE date=?", (today,))
        records = cursor.fetchall()
        conn.close()

        filename = f"attendance_{today}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Roll No", "Name", "Date", "Time"])
            for roll, name, t in records:
                writer.writerow([roll, name, today, t])

        if not records:
            body = f"No attendance recorded for {today}."
        else:
            body = f"📋 Attendance Summary for {today}:\n\n"
            for roll, name, t in records:
                body += f"🎓 {roll} - {name} at {t}\n"
            body += "\nRegards,\nDevMo Attendance System 🤖"

        sender_email = "devarshbhatt1747@gmail.com"
        sender_password = "fvxhscuxqaljgsjv"  # your Gmail app password
        subject = f"Attendance Report - {today}"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if os.path.exists(filename):
            with open(filename, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={filename}")
            msg.attach(part)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print(f"📧 Attendance report sent to {receiver_email}")
        except Exception as e:
            print(f"⚠️ Email send error: {e}")

    # ---------------- ATTENDANCE SYSTEM ----------------
    def run_attendance_system(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Unable to access camera.")
            return

        print("\n🎥 DeepFace Attendance System Started")
        print("Press 'q' to quit, 'v' to view attendance, 'e' to export CSV.\n")

        already_marked = set()
        thank_you_display = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                temp_path = "temp_face.jpg"
                cv2.imwrite(temp_path, face_roi)
                roll_no, name = self.recognize_face(temp_path)

                if roll_no and name:
                    color = (0, 255, 0)
                    label = f"{roll_no} - {name}"

                    if roll_no not in already_marked:
                        new_entry = self.mark_attendance(roll_no, name)
                        if new_entry:
                            already_marked.add(roll_no)
                            thank_you_display[name] = time.time()

                    # Always show name
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Show thank-you bottom left
                    if name in thank_you_display and time.time() - thank_you_display[name] < 5:
                        h_frame, w_frame, _ = frame.shape
                        cv2.putText(frame, f"🙏 Thank you, {name}!", (20, h_frame - 30),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    color = (0, 0, 255)
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            cv2.imshow("DeepFace Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                self.view_attendance()
            elif key == ord('e'):
                self.export_attendance()

        cap.release()
        cv2.destroyAllWindows()


# ---------------- MAIN MENU ----------------
def main():
    system = DeepFaceAttendanceSystem()

    while True:
        print("\n============================================================")
        print("           DevMo - DeepFace Attendance System")
        print("============================================================")
        print("1. Start Attendance System")
        print("2. Register New Face")
        print("3. View Today's Attendance")
        print("4. Export Attendance")
        print("5. Clear All Attendance Data")
        print("6. Send Attendance Report via Email")
        print("7. Exit")

        choice = input("\nSelect an option (1-7): ").strip()

        if choice == '1':
            system.run_attendance_system()
        elif choice == '2':
            system.register_new_face()
        elif choice == '3':
            system.view_attendance()
        elif choice == '4':
            system.export_attendance()
        elif choice == '5':
            system.clear_attendance_data()
        elif choice == '6':
            receiver = input("Enter teacher's email address: ").strip()
            system.send_attendance_email(receiver)
        elif choice == '7':
            print("👋 Exiting DevMo DeepFace Attendance System. Goodbye!")
            break
        else:
            print("❌ Invalid option, please try again.")


if __name__ == "__main__":
    main()
