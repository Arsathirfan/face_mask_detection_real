import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, Response

app = Flask(__name__)

model = load_model("mask_detecton_model.h5")

model_path = "face_detector" \
             "\\res10_300x300_ssd_iter_140000.caffemodel"
config_path = "face_detector" \
              "\\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)
offset = 70


def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype(int)

            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            crop_x = x
            crop_y = y + int(h * 0.25)  # Adjust the crop position as desired

            # Calculate the dimensions of the cropping region
            crop_width = w - crop_x
            crop_height = int(h * 0.2)  # Adjust the crop height as desired

            # Check if the cropping region is valid
            if crop_x >= 0 and crop_y >= 0 and crop_width > 0 and crop_height > 0:
                # Crop the lower part of the face
                crop_img = gray_img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

                # Resize the cropped image to the desired size
                crop_img = cv2.resize(crop_img, (64, 64))

                img = crop_img / 255.0
                img = np.expand_dims(img, axis=0)

                prediction = model.predict(img)

                if prediction > 0.5:
                    predicted = "No Mask Detected"
                    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                    cv2.putText(frame, predicted, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    predicted = 'Mask Detected'
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    cv2.putText(frame, predicted, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Frame', crop_img)

    return frame


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    cap = cv2.VideoCapture(0)

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                break

            frame = detect_faces(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(e)

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
