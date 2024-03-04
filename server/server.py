from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the custom metric function
def dice_coefficient(y_true, y_pred):
    smooth = 1.0  # Smoothing factor to avoid division by zero
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

# Register the custom object
custom_objects = {'dice_coefficient': dice_coefficient}

# Load the trained model
model = load_model('fcn_model.h5', custom_objects=custom_objects)

# Define input shape expected by the model
input_shape = (128, 128, 3)  # Update this according to your model's input shape

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Get the file
    image_file = request.files['file']

    # Save the file temporarily
    image_file_path = 'temp.jpg'
    image_file.save(image_file_path)
    
    # Preprocess the image
    img = Image.open(image_file_path)
    img = img.resize(input_shape[:2])  # Resize the image to match input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values

    # Expand dimensions to match the model input shape
    preprocessed_data = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    predictions = model.predict(preprocessed_data)
    binary_predictions = (predictions > 0.5).astype(np.uint8)
    
    # Convert binary predictions to binary mask
    binary_mask = (binary_predictions.squeeze() * 255).astype(np.uint8)
    mask_image = Image.fromarray(binary_mask, mode='L')  # 'L' mode for grayscale image

    # Save the mask image as JPG
    mask_image_path = 'predicted_tumor_mask.jpg'
    mask_image.save(mask_image_path)

    # Remove temporary image file
    os.remove(image_file_path)

    # Send back the predicted tumor segmentation image
    return send_file(mask_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
