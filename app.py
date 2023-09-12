import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def welcome_page():
    return render_template('welcome_page.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/go_to_index')
def go_to_index():
    return redirect(url_for('index'))

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Load the model and labels outside of the route for better performance
model_path = "retrained_graph.pb"
label_path = "retrained_labels.txt"
graph = None
label_lines = None

# สร้างรายการคำแปลภาษาไทยของระดับเปอร์เซ็นต์
percent_labels = {
    "level of 22 to 26 percent": "ระดับความชื้นอยู่ที่ 22 ถึง 26 เปอร์เซ็นต์",
    "level of 26 to 30 percent": "ระดับความชื้นอยู่ที่ 26 ถึง 30 เปอร์เซ็นต์",
    "level of 18 to 22 percent": "ระดับความชื้นอยู่ที่ 18 ถึง 22 เปอร์เซ็นต์",
    "level of 10 to 14 percent": "ระดับความชื้นอยู่ที่ 10 ถึง 14 เปอร์เซ็นต์",
    "level of 14 to 18 percent": "ระดับความชื้นอยู่ที่ 14 ถึง 18 เปอร์เซ็นต์",
    "level of 6 to 10 percent":  "ระดับความชื้นอยู่ที่ 6 ถึง 10 เปอร์เซ็นต์"
}

def load_model():
    global graph, label_lines
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with open(label_path, 'r') as f:
        label_lines = [line.strip() for line in f.readlines()]

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

load_model()

def classify_image(image_data):
    with tf.compat.v1.Session(graph=graph) as sess:
        # Feed the image_data as input to the graph and get predictions
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        results = []
        for node_id in top_k:
            label = label_lines[node_id]
            score = predictions[0][node_id]

            # แปลงคลาสเป็นคำแปลภาษาไทย
            if label in percent_labels:
                thai_label = percent_labels[label]
            else:
                thai_label = label

            result = {'label': thai_label, 'score': float(score)}
            results.append(result)

        return results

@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'image' in request.files:
            # Get the image file from the form data
            image_file = request.files['image']
            if image_file:
                # Read the image data
                image_data = image_file.read()

                # Perform image classification
                results = classify_image(image_data)

                # Convert results to a human-readable format
                predictions = [f"{result['label']} ({result['score']*100:.2f}%)" for result in results]

                # Save the captured image to a temporary folder (optional)
                image_path = os.path.join("static", "uploads", "captured_image.jpg")
                with open(image_path, 'wb') as f:
                    f.write(image_data)

                return render_template('index.html', predictions=predictions, captured_image_path=image_path)

    return render_template('index.html', predictions=None, captured_image_path=None)

def capture_image():
    # Initialize camera
    camera = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not camera.isOpened():
        raise RuntimeError("Failed to open the camera. Make sure the camera is connected and not in use by other applications.")

    # Capture image
    ret, frame = camera.read()

    # Release camera
    camera.release()

    # Save the captured image to a temporary folder
    image_path = os.path.join("static", "uploads", "captured_image.jpg")
    cv2.imwrite(image_path, frame)

    return image_path

@app.route('/capture', methods=['GET'])
def capture():
    try:
        # Capture image and perform image classification
        image_path = capture_image()
        with open(image_path, 'rb') as f:
            image_data = f.read()
        results = classify_image(image_data)

        # Convert results to a human-readable format
        predictions = [f"{result['label']} ({result['score']*100:.2f}%)" for result in results]

        return render_template('index.html', predictions=predictions, image_path=image_path)
    except Exception as e:
        return render_template('index.html', predictions=None, image_path=None, error_message=str(e))

@app.route('/style.css')
def send_css():
    return app.send_static_file('style.css')

if __name__ == '__main__':
    app.run(port=5001)
