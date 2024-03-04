from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/test', methods=['POST'])
def test():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    # Get the file
    image_file = request.files['file']

    # Save the file temporarily (you may skip this step if you're processing the image directly)
    image_file_path = 'temp_image.jpg'
    image_file.save(image_file_path)

    # Send back the image file as the response
    return send_file(image_file_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
