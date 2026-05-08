import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model('Model.h5')

# Load the labels
labels = {}
with open('label.txt', 'r') as f:
    for line in f:
        key, value = line.strip().split(': ')
        labels[int(key)] = value


def preprocess_frame(frame):
    # Resize frame to the size the model was trained on (64x64)
    frame_resized = cv2.resize(frame, (64, 64))
    # Normalize the image to be in the range [0, 1]
    frame_normalized = frame_resized / 255.0
    # Expand dimensions to match the model input
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

def draw_bounding_box(frame, bbox, label):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def predict_and_draw(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)
    label = labels[predicted_class]

    # Use OpenCV's built-in object detection (Haar Cascades) for simplicity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_objects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_objects:
        draw_bounding_box(frame, (x, y, w, h), label)

    return frame

def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Predict and draw bounding box with label
        frame_with_bbox = predict_and_draw(frame)

        # Display the resulting frame
        cv2.imshow('Webcam Object Detection', frame_with_bbox)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
