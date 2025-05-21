from flask import Flask, request, jsonify, render_template, Response
import os
import cv2
from flask_cors import CORS, cross_origin
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.predictions import PredictionPipeline
from ultralytics import YOLO  # For YOLOv8 model
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)
        self.yolo_model = YOLO("best.pt")  # Load the YOLOv8 model for gesture detection


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    # Get the uploaded image file from the request
    image = request.files['file']
    image.save(clApp.filename)  # Save the image with the same filename
    
    # Run the prediction on the saved image
    result = clApp.classifier.predict()
    
    # Return the result as a JSON response
    return jsonify(result)


def generate_frames():
    """Generate frames for the live video stream."""
    cap = cv2.VideoCapture(0)  # Open webcam
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run YOLOv8 Inference
            # Run YOLOv8 Inference
          
            # YOLOv8 Gesture Detection
            results = clApp.yolo_model(frame, conf=0.3)

                    # Process each detection
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID

                # Label for the bounding box
                label = f"{results[0].names[class_id]}: {conf:.2f}"

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Use MediaPipe for Hand Landmarks
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )
                        # Display the frame
            cv2.imshow('Gesture Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Encode the frame as a JPEG and yield it
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route("/live", methods=['GET'])
@cross_origin()
def live():
    """Stream live video for gesture detection."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)  # Local host
