
# Skin Disease Classifier

This project implements a Convolutional Neural Network (CNN) for classifying skin diseases.  It uses TensorFlow/Keras and leverages image data augmentation for improved performance.

## Project Structure

## Requirements

- Python 3
- TensorFlow/Keras
- NumPy

You can install the necessary libraries using pip:

```bash
pip install tensorflow numpy
```
#Dataset

The dataset should be organized into two directories: train and test. Each directory should contain subdirectories for each class of skin disease.  Images for each class should be placed within their respective class directories.  For example:
```bash
train/
├── eczema/
│   ├── eczema_image1.jpg
│   ├── eczema_image2.png
│   └── ...
└── psoriasis/
    ├── psoriasis_image1.jpg
    └── ...

test/
├── eczema/
│   ├── eczema_image1.jpg
│   └── ...
└── psoriasis/
    ├── psoriasis_image1.jpg
    └── ...
```
Important: Ensure your image data is appropriately sized or adjust img_height and img_width variables in the code to match your image dimensions.
How to Run

Place your image data in the train and test directories as described above.
Save the provided Python code as skin_disease_classifier.py (or a name of your choice).
Run the script from your terminal:

```Bash

python skin_disease_classifier.py
```
This will:

    Load and preprocess the image data.
    Train the CNN model.
    Evaluate the model on the test data.
    Print the test accuracy.
    Save the trained model as model.h5.

##Model Architecture

The CNN model architecture consists of convolutional layers, max pooling layers, dropout layers, a flatten layer, and fully connected layers.  It's designed for multi-class image classification using the softmax activation function in the final layer.  The code includes data augmentation (rescaling, shearing, zooming, and horizontal flipping) to improve the model's robustness and prevent overfitting.
Training Parameters

The following parameters can be adjusted in the skin_disease_classifier.py script:

    img_height: Height of the input images.
    img_width: Width of the input images.
    batch_size: Number of images processed in each batch during training.
    epochs: Number of training epochs.

##Using the Trained Model

The trained model is saved as model.h5. You can load and use this model for making predictions on new images.  Example code for loading and using the model would look like:
Python

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model.h5')

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class # And potentially the probabilities (predictions)

## Example usage
image_path = "path/to/new_image.jpg"
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")

Remember to replace "path/to/new_image.jpg" with the actual path to your image and include the necessary imports.  You'll also need to map the predicted integer class to the actual disease name based on your dataset's class indices.
