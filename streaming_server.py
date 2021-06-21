import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server
import detect_picamera as detect
from tflite_runtime.interpreter import Interpreter
from io import BytesIO
from PIL import Image, ImageDraw
import argparse

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PAGE="""\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self,camera, objectDetector):
        self.frame = None #Original image
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.camera = camera
        self.image = None
        self.bounded_image = BytesIO() #Image that has the bounding boxes
        self.input_width = 300 #size of image that will go into the Object Detector
        self.input_height = 300
        self.objectDetector = objectDetector
    
    def save_image(self):
        '''
        Gets the image from the camera using the video port and save it as a PIL Image
        '''
        stream = BytesIO()
        self.camera.capture(stream, format="jpeg", use_video_port = True)
        return Image.open(stream)

    def process_od(self):
        '''
        Process the object detection and draw the bounding boxes and the labels
        '''
        pi = self.image.convert('RGB').resize(
            (self.input_width, self.input_height), Image.ANTIALIAS)
        r = self.objectDetector.detect(pi)
        print(r)
        img_draw = ImageDraw.Draw(self.image)
        for obj in r:
            self.draw_obj(obj, img_draw)
        self.bounded_image = BytesIO()
        self.image.save(self.bounded_image, "jpeg")


    def draw_obj(self, obj, image_draw):
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        image_draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0,0,0))
        image_draw.text((xmin,ymin), obj["class"])


    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
            self.image = self.save_image()
            self.process_od()
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
        
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        #frame = output.frame
                        frame = output.bounded_image.getvalue()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    
class ObjectDetector:
    def __init__(self, interpreter, labels):
        self.interpreter =  interpreter
        self.labels = labels
    
    def detect(self, image):
        results = detect.detect_objects(self.interpreter, image, 0.4)
        for obj in results:
            obj['class']=labels[obj['class_id']]
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.4)
    args = parser.parse_args()

    labels = detect.load_labels(args.labels)
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    od = ObjectDetector(interpreter, labels)
    
    
    with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=24) as camera:
        camera.vflip=True
        output = StreamingOutput(camera, od)
        output.input_height = input_height
        output.input_width = input_width
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('', 8000)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.stop_recording()
