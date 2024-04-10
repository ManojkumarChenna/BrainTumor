from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')
# Define a function to process the uploaded image
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    return image

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        if file:
            # Read the image file
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            # Process the image
            processed_img = process_image(img)
            # Make prediction
            pred = model.predict(processed_img)
            # Determine the prediction result
            if pred[0][0] > 0.5:
                result = "Tumor"
            else:
                result = "No Tumor"
            return jsonify({'result': result})
    return jsonify({'error': 'No file uploaded'})


if __name__ == '__main__':
    app.run(debug=True)
