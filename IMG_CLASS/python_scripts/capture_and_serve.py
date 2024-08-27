from picamera2 import Picamera2
from flask import Flask, send_from_directory, render_template_string
import time
import os
import threading

app = Flask(__name__)

# Global variable to store the latest image filename
latest_image = None

def capture_dataset(num_images, interval, output_dir="dataset"):
    global latest_image
    os.makedirs(output_dir, exist_ok=True)
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)

    try:
        picam2.start()
        time.sleep(2)  # Wait for camera to warm up

        for i in range(num_images):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"image_{timestamp}_{i+1:04d}.jpg"
            full_path = os.path.join(output_dir, filename)

            picam2.capture_file(full_path)
            print(f"Captured image {i+1}/{num_images}: {filename}")

            latest_image = filename  # Update the latest image filename

            if i < num_images - 1:
                time.sleep(interval)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        picam2.stop()
        print("Dataset capture completed.")

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bee Dataset Viewer</title>
            <script>
                function refreshImage() {
                    var img = document.getElementById("latest-image");
                    img.src = "/latest_image?" + new Date().getTime();
                }
                setInterval(refreshImage, 5000);  // Refresh every 5 seconds
            </script>
        </head>
        <body>
            <h1>Latest Captured Image</h1>
            <img id="latest-image" src="/latest_image" alt="Latest captured image" style="max-width: 100%;">
        </body>
        </html>
    ''')

@app.route('/latest_image')
def latest_image():
    if latest_image:
        return send_from_directory('dataset', latest_image)
    else:
        return "No image captured yet", 404

if __name__ == '__main__':
    # Start the image capture in a separate thread
    capture_thread = threading.Thread(target=capture_dataset, args=(100, 5, "dataset"))
    capture_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
