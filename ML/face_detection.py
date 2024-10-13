import cv2
import os

# Specify the paths for the image and Haar cascade
image_path = 'N:\menuBased\pythontasks\ML\mytest.png'  # Update this path if necessary
cascade_path = 'N:\menuBased\pythontasks\ML\haarcascade_frontalface_alt2.xml'  # Update this path if necessary

# Convert the paths to absolute paths
image_path = os.path.abspath(image_path)
cascade_path = os.path.abspath(cascade_path)

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit()

# Read the input image
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Unable to load image. Check the file path: {image_path}")
    exit()

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Check if the Haar cascade file exists
if not os.path.exists(cascade_path):
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit()

# Load the Haar cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade was loaded successfully
if face_cascade.empty():
    print("Error: Unable to load the Haar cascade file.")
    exit()

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Check if any faces are detected
if len(faces) == 0:
    print("No faces detected.")
else:
    # Draw rectangles around the faces and crop the faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = img[y:y + h, x:x + w]
        face_save_path = os.path.join(os.path.dirname(image_path), f'face_{i + 1}.jpg')
        cv2.imshow(f"Face {i + 1}", face)  # Show each detected face
        cv2.imwrite(face_save_path, face)  # Save each detected face
        print(f"Face {i + 1} saved at {face_save_path}")

# Save and display the output image with rectangles
output_path = os.path.join(os.path.dirname(image_path), 'detected.jpg')
cv2.imwrite(output_path, img)
cv2.imshow('Detected Faces', img)
print(f"Detected faces image saved at {output_path}")

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
