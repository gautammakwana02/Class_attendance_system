from flask import Flask, request,  jsonify, render_template
from flask_restful import Api, Resource
import cv2
import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm

app = Flask(__name__)
api = Api(app)


@app.route('/')
def index():
    return render_template('index.html')


# Load images from train_data folder
train_data_folder = "../train_data"
X = []
y = []
target_names = []

for person_name in os.listdir(train_data_folder):
    if os.path.isdir(os.path.join(train_data_folder, person_name)):
        target_names.append(person_name)
        person_folder = os.path.join(train_data_folder, person_name)
        for filename in os.listdir(person_folder)[:3]:  # Considering only first 3 images for each person
            image_path = os.path.join(person_folder, filename)
            image = imread(image_path, as_gray=True)
            image_resized = resize(image, (50, 37))  # Resize images to match LFW dataset
            X.append(image_resized.flatten())
            y.append(len(target_names) - 1)  # Assigning label based on index of target_names

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pca = PCA(n_components=9, whiten=True, random_state=42)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

svm_model = svm.SVC(kernel='rbf', class_weight='balanced', C=10.0, gamma=0.001, random_state=42)
svm_model.fit(X_train_pca, y_train)

class UploadImage(Resource):
    def post(self):
        # Check if image file is present in the request
        if 'image' not in request.files:
            return {'error': 'No image found in the request'}, 400

        file = request.files['image']

        # Save the uploaded image
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        filename = file.filename
        file.save(filename)

        # Detect faces in the uploaded image
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a folder to save detected faces
        if not os.path.exists('../detected_faces'):
            os.makedirs('../detected_faces')

        # Save detected faces
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y + h, x:x + w]
            cv2.imwrite(f'../detected_faces/face_{i}.jpg', face)

        # Pass detected faces through SVM model and generate Excel file
        results = []
        for face_file in os.listdir('../detected_faces'):
            face_image = cv2.imread(os.path.join('../detected_faces', face_file), cv2.IMREAD_GRAYSCALE)
            face_image_resized = cv2.resize(face_image, (50, 37))  # Resize face image
            face_image_flattened = face_image_resized.flatten()
            face_image_pca = pca.transform(face_image_flattened.reshape(1, -1))
            prediction = svm_model.predict(face_image_pca)
            results.append({'Face': face_file, 'Prediction': target_names[prediction[0]]})

        df = pd.DataFrame(results)
        df.to_excel('results.xlsx', index=False)

        response_data = {'message': 'Faces detected and results saved in results.xlsx'}
        return response_data, 200



api.add_resource(UploadImage, '/upload_image')

if __name__ == '__main__':
    app.run(debug=True)
