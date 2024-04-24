from flask import Flask,request,jsonify
import numpy as np
import cv2
import io
from keras.models import model_from_json

emotion_dict = {0: "Confused", 1: "Not Confused"}

# json_file = open('/content/drive/MyDrive/GGH/model/confused_expression_model.json', 'r')
json_file = open('confused_expression_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
confused_expression_model = model_from_json(loaded_model_json)

# emotion_model.load_weights("/content/drive/MyDrive/GGH/model/confused_expression_model.h5")
confused_expression_model.load_weights("confused_expression_model.h5")
print("Loaded model from disk")
app = Flask(__name__)

json_file = open('age_estimation_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
age_estimation_model = model_from_json(loaded_model_json)

age_estimation_model.load_weights("age_estimation_model.hdf5")
print("Loaded model from disk")

@app.route('/', methods=['GET'])
def index():
    return "AI Models API"

@app.route('/isconfused', methods=['POST'])
def isConfused():
    # Get image file from request
    file = request.files['image']
    
    # Read image from file
    image_stream = io.BytesIO(file.read())
    image = cv2.imdecode(np.fromstring(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    original_height, original_width = image.shape[:2]
    ratio = 720 / original_width
    new_height = int(original_height * ratio)
    frame = cv2.resize(image, (720, new_height))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('./haarcascade.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    results = []

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = confused_expression_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        results.append(emotion_dict[maxindex])

    if "Confused" in results:
        return jsonify("Confused")
    
    return jsonify("Not Confused")

@app.route('/predictage', methods=['POST'])
def predictAge():
    file = request.files['image']
    
    image_stream = io.BytesIO(file.read())
    image = cv2.imdecode(np.fromstring(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    
    original_height, original_width = image.shape[:2]
    ratio = 720 / original_width
    new_height = int(original_height * ratio)
    frame = cv2.resize(image, (720, new_height))

    face_detector = cv2.CascadeClassifier('./haarcascade.xml')
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    faces = np.empty((len(num_faces), 224, 224, 3))

    if len(num_faces)>0:
        for (x1, y1, w, h) in num_faces:
            roi_frame = frame[y1:y1 + h, x1:x1 + w]
            faces[0] = cv2.resize(roi_frame, (224, 224))

        results = age_estimation_model.predict(faces)
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        return jsonify(int(predicted_ages[0]))

    return jsonify("No Faces Detected")

if __name__ == '__main__':
    app.run(debug=True)

