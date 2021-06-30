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
import datetime
import os
import logging
from pi_face_detect import FaceDetection, FaceTracker
from gpiozero import Servo
from functools import partial

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

class ObjClassDetectionHook:
    def __init__(self, save_folder, detection_classes):
        '''
        save_folder which folder to save to
        detection_classes - an array of class labels that we want to take a picture when detected
        '''
        self.save_folder = save_folder
        self.detection_classes = detection_classes
    
    def get_filename(self, class_name):
        datetime_filename=datetime.datetime.now().isoformat().replace(":","_")
        return datetime_filename+"_"+class_name+".jpg"

    def process_hook(self, ori_image, json_data):
        print("Process Hook")
        for obj in json_data:
            if obj["class"] in self.detection_classes:
                print("Person detected")
                ori_image.save(os.path.join(self.save_folder, self.get_filename(obj["class"])),"jpeg")
                return

class StreamingOutput(object):
    def __init__(self,camera, objectDetector, output_hook=None):
        self.frame = None #Original image
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.camera = camera
        self.image = None
        self.bounded_image = BytesIO() #Image that has the bounding boxes
        self.input_width = 300 #size of image that will go into the Object Detector
        self.input_height = 300
        self.objectDetector = objectDetector
        self.output_hook = output_hook
    
    def save_image(self):
        '''
        Gets the image from the camera using the video port and save it as a PIL Image
        '''
        stream = BytesIO()
        self.camera.capture(stream, format="jpeg", use_video_port = True)
        return Image.open(stream)

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.image = self.save_image()
                self.condition.notify_all()
            self.buffer.seek(0)
            
            #self.process_od()
            #if self.objectDetector is not None:
            #    self.bounded_image = self.objectDetector.detect(self.image)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def __init__(self, upservo, od, *args, **kwargs):
        self.upservo = upservo
        self.od = od
        # BaseHTTPRequestHandler calls do_GET **inside** __init__ !!!
        # So we have to call super().__init__ after setting attributes.
        super().__init__(*args, **kwargs)

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
                        frame = output.frame
                        #frame = output.bounded_image.getvalue()
                        if self.od is not None:
                            frame=self.od.detect(output.image).getvalue()
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
        elif self.path == "/up":
            self.upservo.value = self.upservo.value - 0.01
            s = "Up Servo "+str(self.upservo.value)
            self.wfile.write(s.encode("utf-8"))
        elif self.path == "/down":
            self.upservo.value = self.upservo.value + 0.01
            s = "Up Servo "+str(self.upservo.value)
            self.wfile.write(s.encode("utf-8"))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    
class ObjectDetector:
    def __init__(self, interpreter, labels, threshold=0.4):
        self.interpreter =  interpreter
        self.labels = labels
        self.threshold = threshold
        self.input_width = 300
        self.input_height = 300

    def draw_obj(self, obj, image_draw):
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        image_draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0,0,0))
        image_draw.text((xmin,ymin), obj["class"])
    
    def detect(self, image):
        '''
        returns an image with the bounding boxes
        '''
        pi = image.convert('RGB').resize(
            (self.input_width, self.input_height), Image.ANTIALIAS)
        results = detect.detect_objects(self.interpreter, pi, self.threshold)
        for obj in results:
            obj['class']=labels[obj['class_id']]

        img_draw = ImageDraw.Draw(image)
        for obj in results:
            self.draw_obj(obj, img_draw)
        bounded_image = BytesIO()
        image.save(bounded_image, "jpeg")
        #if self.output_hook is not None:
        #    self.output_hook.process_hook(self.image, r)
        return bounded_image

class BlankObjectDetector:
    def detect(self, image):
        bounded_image = BytesIO()
        image.save(bounded_image, "jpeg")
        return bounded_image

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

    '''
    labels = detect.load_labels(args.labels)
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    od = ObjectDetector(interpreter, labels)
    od.input_width=input_width
    od.input_height=input_height
    #output_hook = ObjClassDetectionHook("./", ["person"])
    '''

    od = FaceDetection()
    p = Servo(18, min_pulse_width=0.0005, max_pulse_width=0.0025)
    p.value = 0
    #od = FaceTracker(p)
    #od=None
    output_hook=None
    handler = partial(StreamingHandler, p, od)
    with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=2) as camera:
        camera.vflip=True
        output = StreamingOutput(camera, None,output_hook)
        #output.input_height = input_height
        #output.input_width = input_width
        camera.start_recording(output, format='mjpeg')
        try:
            address = ('', 8000)
            server = StreamingServer(address, handler)
            server.serve_forever()
        finally:
            camera.stop_recording()
