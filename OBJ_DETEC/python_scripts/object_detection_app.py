from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
import io
import threading
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
from queue import Queue, Empty
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import signal

app = Flask(__name__)

# Global variables
picam2 = None
frame = None
frame_lock = threading.Lock()
is_detecting = False
confidence_threshold = 0.5
model_path = "./models/ssd-mobilenet-v1-tflite-default-v1.tflite"
labels_path = "./models/coco_labels.txt"
interpreter = None
detection_queue = Queue(maxsize=1)
latest_detections = []
detections_lock = threading.Lock()

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels(labels_path)

def initialize_camera():
    global picam2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Wait for camera to warm up

def get_frame():
    global frame
    while True:
        stream = io.BytesIO()
        picam2.capture_file(stream, format='jpeg')
        stream.seek(0)
        img = Image.open(stream)
        
        # Draw detections on the image
        img_with_detections = draw_detections(img)
        
        # Convert the image back to bytes
        img_byte_arr = io.BytesIO()
        img_with_detections.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        with frame_lock:
            frame = img_byte_arr
        print("Frame captured and processed")
        time.sleep(0.1)  # Capture frames more frequently

def generate_frames():
    while True:
        with frame_lock:
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # Adjust this value to control frame rate

def load_model():
    global interpreter
    if interpreter is None:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    return interpreter

def detect_objects(img, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the expected input shape
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    # Resize and preprocess the image
    img = img.resize((width, height))
    img = img.convert('RGB')  # Ensure the image is in RGB format
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

    # Check if the model expects float input
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    return boxes, classes, scores, num_detections

def detection_worker():
    global frame, is_detecting, latest_detections
    interpreter = load_model()
    print("Model loaded successfully")

    while True:
        if is_detecting:
            try:
                current_frame = None
                with frame_lock:
                    if frame is not None:
                        current_frame = frame

                if current_frame is not None:
                    img = Image.open(io.BytesIO(current_frame))
                    
                    boxes, classes, scores, num_detections = detect_objects(img, interpreter)
                    
                    new_detections = []
                    for i in range(int(num_detections)):
                        if scores[i] >= confidence_threshold:
                            ymin, xmin, ymax, xmax = boxes[i]
                            (left, right, top, bottom) = (xmin * img.width, xmax * img.width, 
                                                          ymin * img.height, ymax * img.height)
                            
                            class_id = int(classes[i])
                            class_name = labels.get(class_id, f"Class {class_id}")
                            
                            new_detections.append({
                                'class': class_name,
                                'score': float(scores[i]),
                                'box': [left, top, right, bottom]
                            })
                    
                    with detections_lock:
                        latest_detections = new_detections  # Replace instead of append

            except Exception as e:
                print(f"Error in detection worker: {e}")
                import traceback
                traceback.print_exc()
        time.sleep(0.1)  # Process frames more frequently

def draw_detections(img):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    with detections_lock:
        for detection in latest_detections:
            left, top, right, bottom = detection['box']
            draw.rectangle([left, top, right, bottom], outline="red", width=2)
            
            label = f"{detection['class']}: {detection['score']:.2f}"
            draw.text((left, top-15), label, font=font, fill="red")

    return img
        
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            function startDetection() {
                $.post('/start');
                $('#startBtn').prop('disabled', true);
                $('#stopBtn').prop('disabled', false);
            }
        
            function stopDetection() {
                $.post('/stop');
                $('#startBtn').prop('disabled', false);
                $('#stopBtn').prop('disabled', true);
            }
        
            function updateConfidence() {
                var confidence = $('#confidence').val();
                $.post('/update_confidence', {confidence: confidence});
            }
        
            function updateDetections() {
                $.get('/get_detections', function(data) {
                    $('#detections').empty();
                    const stableDetections = data.sort((a, b) => b.score - a.score);
        
                    stableDetections.forEach(detection => {
                        $('#detections').append(`<p>${detection.class}: ${detection.score.toFixed(2)}</p>`);
                    });
                });
            }
        
            function closeApp() {
                if (confirm('Are you sure you want to close the app?')) {
                    $.ajax({
                        url: '/close',
                        type: 'POST',
                        headers: {
                            'X-Requested-With': 'XMLHttpRequest'
                        },
                        success: function(data) {
                            alert(data);
                            window.close();
                        },
                        error: function() {
                            alert('Server has shut down. You can close this window.');
                            window.close();
                        }
                    });
                }
            }
        
            $(document).ready(function() {
                setInterval(updateDetections, 500);  // Update every 500ms
            });
        </script>
            
  </head>
        <body>
            <h1>Object Detection</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480" />
            <br>
            <button id="startBtn" onclick="startDetection()">Start Detection</button>
            <button id="stopBtn" onclick="stopDetection()" disabled>Stop Detection</button>
            <button onclick="closeApp()">Close App</button>
            <br>
            <label for="confidence">Confidence Threshold:</label>
            <input type="number" id="confidence" name="confidence" min="0" max="1" step="0.1" value="0.5" onchange="updateConfidence()">
            <br>
            <div id="detections">Waiting for detections...</div>
        </body>
        </html>
    ''')
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection():
    global is_detecting
    is_detecting = True
    return '', 204

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_detecting
    is_detecting = False
    return '', 204

@app.route('/update_confidence', methods=['POST'])
def update_confidence():
    global confidence_threshold
    confidence_threshold = float(request.form['confidence'])
    return '', 204

@app.route('/get_detections')
def get_detections():
    global latest_detections
    if not is_detecting:
        return jsonify([])
    with detections_lock:
        return jsonify(latest_detections)

@app.route('/close', methods=['POST'])
def close_app():
    global is_detecting
    is_detecting = False
    cleanup()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # This is an AJAX request, so it's an intentional close
        def shutdown():
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=shutdown).start()
        return 'Server shutting down...', 200
    else:
        # This is not an AJAX request, so it's probably a page refresh
        return 'Page refreshed', 200

def cleanup():
    global picam2, is_detecting
    is_detecting = False
    if picam2:
        picam2.stop()
    # You might want to add any additional cleanup code here

if __name__ == '__main__':
    try:
        initialize_camera()
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        frame_thread = threading.Thread(target=get_frame, daemon=True)
        frame_thread.start()
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cleanup()