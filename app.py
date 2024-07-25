from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained CNN model
model = load_model('model.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size based on your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            img_array = prepare_image(file_path)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            cass_labels=["Acne","Benign_tumors","Eczema"]
            result = class_labels[predicted_class_index]
            result = np.argmax(prediction, axis=1)[0]
            os.remove(file_path)  # Clean up the uploaded file

    return render_template('index.html', result=result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
