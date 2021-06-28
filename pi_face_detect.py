import cv2
import numpy
from io import BytesIO
from PIL import ImageDraw
from gpiozero import Servo

class FaceDetection:
    def draw_faces(self, image, faces):
        image_draw = ImageDraw.Draw(image)
        for (x, y, w, h) in faces:
            image_draw.rectangle([(x, y), (x+w, y+w)], outline=(0,0,0))
        
        bounded_image = BytesIO()
        image.save(bounded_image, "jpeg")
        #if self.output_hook is not None:
        #    self.output_hook.process_hook(self.image, r)
        return bounded_image

    def detect(self, image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return self.draw_faces(image, faces)


class FaceTracker:
    def __init__(self, servo):
        self.servo = servo
        self.center_box_tl = (310,230)
        self.center_box_br = (330,250)
        self.servo_max = 0.75
        self.servo_min = -0.75

    def get_center_point(self, top_left, bottom_right):
        x,y = top_left
        x1, y1 = bottom_right
        c_x = x + ((x1-x)/2)
        c_y = y + ((y1-y)/2)
        return (c_x, c_y)

    def check_limits(self):
        if self.servo.value > self.servo_max:
            self.servo.value = self.servo_max
        elif self.servo.value < self.servo_min:
            self.servo.value = self.servo.min
    
    def move_servo(self, center_points):
        x,y = center_points
        print("target ", center_points)
        if y < self.center_box_br[1]:
            print("Y less")
            self.servo.value = self.servo.value - 0.01 # Tilt Camera backwards/go higher
        elif y >  self.center_box_br[1]:
            print("Y more")
            self.servo.value = self.servo.value + 0.01 # Tilt Camera forwards/go lower

        self.check_limits() 

    def draw_faces(self, image, faces):
        image_draw = ImageDraw.Draw(image)
        first_face = None
        image_draw.rectangle([self.center_box_tl, self.center_box_br], outline = (1,0,0))
        for (x, y, w, h) in faces:
            image_draw.rectangle([(x, y), (x+w, y+h)], outline=(0,0,0))

            print("Face Detected at ", (x,y), (x+w, y+h))
            cp = self.get_center_point((x,y), (x+w, y+h))
            
            if first_face is None:
                first_face = cp
        
        if first_face is not None:
            #print("Center point ", cp)
            self.move_servo(first_face)

        bounded_image = BytesIO()
        image.save(bounded_image, "jpeg")
        #if self.output_hook is not None:
        #    self.output_hook.process_hook(self.image, r)
        return bounded_image
    
    def detect(self, image):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return self.draw_faces(image, faces)