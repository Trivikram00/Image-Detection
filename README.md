import cv2
import time

# Load the cascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Unable to load face cascade classifier")
except IOError as e:
    print(f"Error loading face cascade: {e}")
    exit()

# Read the image
img = cv2.imread('reference_image.jpg')  # Replace with your image file

# Check if image loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

start_time = time.time()  # Start timing 

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

end_time = time.time()  # End timing
detection_time = end_time - start_time
print(f"Face detection took {detection_time:.4f} seconds")

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Show the output
cv2.imshow('Detected Face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
