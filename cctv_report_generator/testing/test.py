#!/usr/bin/env python3

"""testing the cctv report generator system.
   HANSEL CARZERLET 2024
"""
import cv2
import threading
import numpy as np
import face_recognition
import sqlite3
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from threading import Thread
import time

# Initializing the database connection
conn = sqlite3.connect('office_monitoring.db', check_same_thread=False)
cursor = conn.cursor()

# Creating the table for logging events
cursor.execute('''CREATE TABLE IF NOT EXISTS logs
                  (timestamp TEXT, employees TEXT, visitor_count INTEGER, anomalies TEXT)''')
conn.commit()

# Video Stream Class using laptop camera (index 0)
class VideoStream:
    def __init__(self, camera_index=0):
        self.capture = cv2.VideoCapture(camera_index)
        self.current_frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()
        
    def update_frame(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame

    def get_frame(self):
        return self.current_frame
    
    def stop(self):
        self.running = False
        self.capture.release()

# YOLO Object Detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_people(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    people_count = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Assuming 'person' class_id is 0
            if confidence > 0.5 and class_id == 0:
                people_count += 1
    return people_count

# YOLO Processor Class
class YOLOProcessor(Thread):
    def __init__(self, video_stream):
        Thread.__init__(self)
        self.video_stream = video_stream
        self.people_count = 0
        self.running = True

    def run(self):
        while self.running:
            frame = self.video_stream.get_frame()
            if frame is not None:
                self.people_count = detect_people(frame)

    def stop(self):
        self.running = False

# Face Recognition
known_face_encodings = []  # Employee face encodings will be added here
known_face_names = []      # Corresponding names to the face encodings will be added here

def recognize_faces(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        names.append(name)
    
    return names

# Logging events in the database
def log_event(employees, visitor_count, anomalies):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO logs (timestamp, employees, visitor_count, anomalies) VALUES (?, ?, ?, ?)',
                   (timestamp, ','.join(employees), visitor_count, anomalies))
    conn.commit()

# Generating Weekly Report with Visualizations
def generate_weekly_report():
    # Loading data
    df = pd.read_sql_query("SELECT * FROM logs", conn)

    # Visualization: Number of people each day
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_counts = df.groupby('date').visitor_count.sum()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_counts.index, daily_counts.values)
    plt.title('Visitor Counts per Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Visitors')
    plt.savefig('weekly_report.png')  # Saving the plot as an image

    # Generating PDF Report
    c = canvas.Canvas("Weekly_Report.pdf", pagesize=A4)
    c.drawString(100, 800, "Weekly Office Monitoring Report")
    c.drawImage('weekly_report.png', 50, 500, width=500, height=300)
    c.save()

# Sending Email with the Weekly Report
def send_email():
    # Email setup
    sender = "hanselcarzerlet@gmail.com"
    receiver = "ssemujjuraymond0@gmail.com"
    subject = "Weekly Office Monitoring Report"
    body = "Please find attached the weekly report."

    # Creating message
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attaching report
    filename = "Weekly_Report.pdf"
    attachment = open(filename, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename= {filename}')
    msg.attach(part)

    # Sending email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, "****************")
    server.sendmail(sender, receiver, msg.as_string())
    server.quit()



# Main Execution
video_stream = VideoStream(camera_index=0)  # Using laptop camera (index 0)
yolo_processor = YOLOProcessor(video_stream)
yolo_processor.start()

last_report_time = time.time()  # Tracking the time when the last report was sent

try:
    while True:
        frame = video_stream.get_frame()
        if frame is not None:
            # Recognizing employees
            employees = recognize_faces(frame)
            visitor_count = yolo_processor.people_count - len(employees)
            log_event(employees, visitor_count, "")  

        # Adding a sleep to control the logging frequency
        time.sleep(60)  # Log every minute

        # Checking if five minutes have passed since the last report
        if time.time() - last_report_time >= 300:  # 300 seconds = 5 minutes
            print("Generating and sending report...")  
            generate_weekly_report()  
            send_email()              
            last_report_time = time.time()  # Reseting the timer

except KeyboardInterrupt:
    video_stream.stop()
    yolo_processor.stop()
    conn.close()

