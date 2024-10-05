			#### THE CCTV REPORT ENERATOR SYSTEM ###### HANSEL CARZERLET 2024


1. System Overview
	- Live Feed Processing: Using multithreading to capture and process the video feed in real-time.
	- YOLO for Object Detection: Using YOLO to detect and count people in the video feed.
	- Face Recognition: Identifying employees.
	- Data Storage: Logging events in a database.
	- Report Generation: Compiling a weekly report with visualizations.
	- Email Automation: Send the report to the manager.


2. Required Libraries and Tools
	Installing the necessary libraries: pip install opencv-python numpy cmake dlib tensorflow keras matplotlib seaborn reportlab facial-recognition


3. Notes
	- Make sure to download the yolo3.cfg and yolo3.weights files as they were too big so they were left out. 
	- There is need to generate an App Password specifically for this script, as Gmail won’t allow you to log in using your regular account password.
	- Adjust the RTSP URL (rtsp_url) to match the camera's stream.
	- Add the face encodings for employees in the known_face_encodings and known_face_names lists to use face recognition.
	
	
4. Step-by-Step Guide to Set Up Known Faces
	- Collect Images of Employees: Obtain clear, high-quality images of each employee you want to recognize.
	- Encode Faces: Use the `face_recognition` library to generate face encodings from these images.
	- Store Encodings and Names: Store these encodings in the `known_face_encodings` list and their corresponding names in the `known_face_names` list.
	- Create a Folder for Employee Images: `employee_images` with each image named after the employee (e.g., `john_doe.jpg`, `jane_smith.jpg`).
	- The Code for Creating `known_face_encodings` and `known_face_names` is titled face_encoding_generator.py and once run, it will load images, generate encodings, and populate the lists.

	Example Directory Structure
Assuming the script is in the current directory and the images are in a subdirectory named `employee_images`:
.
├── face_encoding_generator.py
└── employee_images
    ├── john_doe.jpg
    ├── jane_doe.jpg
    └── other_employee.jpg

	Result
known_face_encodings = [<numpy array of encoding for John Doe>, <numpy array of encoding for Jane Smith>, ...]
known_face_names = ['John Doe', 'Jane Smith', ...]


5. Testing	
	- For the purpose of testing, a modified script that uses the laptop camera (instead of an actual CCTV camera) to capture video feed for training the model has been added.
	- To quickly test the system by sending a report every five minutes, the timing logic in the main execution loop has been modified in this testing script. Instead of waiting for a weekly report, we can check the time every minute and send the report when five minutes have passed.

