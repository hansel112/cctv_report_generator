#!/usr/bin/env python3

"""Generating the employee face encodings to be used for facial recognition.
   HANSEL CARZERLET 2024
"""

import face_recognition
import os

# Paths to the folder containing images of employees
employee_images_path = 'employee_images'

# Lists to hold known face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Loading each employee image and extract face encodings
for image_name in os.listdir(employee_images_path):
    # Loading image file
    image_path = os.path.join(employee_images_path, image_name)
    image = face_recognition.load_image_file(image_path)

    # Extracting face encodings
    # Assuming there is only one face in each image
    encodings = face_recognition.face_encodings(image)
    
    # If at least one encoding is found
    if len(encodings) > 0:
        # Adding the first face encoding to the list
        known_face_encodings.append(encodings[0])

        # Extracting the employee's name from the image file name (e.g., 'john_doe.jpg' -> 'John Doe')
        name = os.path.splitext(image_name)[0].replace('_', ' ').title()
        known_face_names.append(name)

print("Known face encodings:", known_face_encodings)
print("Known face names:", known_face_names)

