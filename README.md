# Introduction
This is an example project uses Tensorflow Lite and Python on the Raspberry PI to setup a streaming server with object detection.

The original code (detect_camera.py) was from the Tensorflow examples on GITHUB (https://github.com/tensorflow/examples)

The original code needs a monitor in order to view the results but i was mostly access my PI with a web browser and SSH so I needed a way to view the stream via a Web Browser.

I used code from detect_picamera.py in the tensorflow examples
and the code from  [PI Camera - Advance Recipes - Web Steaming](https://picamera.readthedocs.io/en/release-1.13/recipes2.html) to hack together this little example.

Some of the instructions below are from the Tensorflow Lite example's readme.

## Set up your hardware

Before you begin, you need to [set up your Raspberry Pi](
https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up) with
Raspberry Pi OS (preferably updated to Buster).

You also need to [connect and configure the Pi Camera](
https://www.raspberrypi.org/documentation/configuration/camera.md).

You will need to know the IP address of your PI in order to access the Web Stream.

## Install the TensorFlow Lite runtime

In this project, all you need from the TensorFlow Lite API is the `Interpreter`
class. So instead of installing the large `tensorflow` package, we're using the
much smaller `tflite_runtime` package.

To install this on your Raspberry Pi, follow the instructions in the
[Python quickstart](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python).
Return here after you perform the `apt-get install` command.


## Download the example files

First, clone this Git repo onto your Raspberry Pi like this:

```
git clone https://github.com/sljm12/rpi_tf_lite_object_detection_streaming
```

Then use our script to install a couple Python packages, and
download the MobileNet model and labels file:


# The script takes an argument specifying where you want to save the model files
```
bash download.sh /tmp
```


## Run the example

```
python3 streaming_server.py \
  --model /tmp/detect.tflite \
  --labels /tmp/coco_labels.txt
```

Use a web browser to access http://<Your PI IP>:8000
You should see the camera feed appear on the browser attached to your Raspberry
Pi. Put some objects in front of the camera, like a coffee mug or keyboard, and
you'll see boxes drawn around those that the model recognizes, including the
label.

On the console you should see a steady stream of logs that shows the detected objects also.


