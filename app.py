from fileinput import filename
import os
from tkinter.messagebox import YES
import urllib.request
from xml.dom.domreg import well_known_implementations
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
#from tensorflow import keras
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import time
from twilio.rest import Client



# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid=os.environ['TWILIO_ACCOUNT_SID']='AC3037433857b9dfde1d63726cc4e7ac56'
auth_token=os.environ['TWILIO_AUTH_TOKEN']='59466b7faf21019b5986835aa3515a03'
client = Client(account_sid, auth_token)


#### VINO ####
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from argparse import Namespace
from pathlib import Path, WindowsPath
import cv2
from openvino.inference_engine import IECore



#from models import Deblurring
#import monitors
##from pipelines import AsyncPipeline
#from images_capture import open_images_capture
#from performance_metrics import PerformanceMetrics

models = []

def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def main():

    args = Namespace(device='CPU', input='C:\\Program Files (x86)\\Intel\\openvino_2021\\deployment_tools\\inference_engine\\demos\\deblurring_demo\\python\\deblurred_image.png', loop=False, model=WindowsPath('C:/Program Files (x86)/Intel/openvino_2021/deployment_tools/inference_engine/demos/deblurring_demo/python/deblurgan-v2.xml'), no_show=False, num_infer_requests=1, num_streams='', num_threads=None, output=None, output_limit=1000, utilization_monitors='')
    log.info('Initializing Inference Engine...')
    ie = IECore()
    plugin_config = get_plugin_configs(args.device, args.num_streams, args.num_threads)
    cap = open_images_capture(args.input, args.loop)
    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")
    log.info('Loading network...')
    model = Deblurring(ie, args.model, frame.shape)
    pipeline = AsyncPipeline(ie, model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)
    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    next_frame_id = 1
    next_frame_id_to_show = 0
    metrics = PerformanceMetrics()
    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
    video_writer = cv2.VideoWriter()
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (2 * frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    while True:
        if pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break

            # Submit for inference
            pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        else:
            # Wait for empty request
            pipeline.await_any()

        if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]
        # Process all completed requests
        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            result_frame, frame_meta = results
            input_frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if input_frame.shape != result_frame.shape:
                input_frame = cv2.resize(input_frame, (result_frame.shape[1], result_frame.shape[0]))
            final_image = cv2.hconcat([input_frame, result_frame])

            presenter.drawGraphs(final_image)
            metrics.update(start_time, final_image)
            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(final_image)
            if not args.no_show:
                cv2.imshow('Deblurring Results', final_image)
                cv2.imwrite('C:/Users/Kaiwalya/Desktop/deblur/new.png',final_image)
                img = cv2.imread('C:/Users/Kaiwalya/Desktop/deblur/new.png')
                height, width, channels = img.shape
                w1 = int(width/1)
                w2 = int(width/2)
                w4 = int(width/4)
                print(w1,w2)
                crop = img[:, w4:w2]
                cv2.imwrite('C:/Users/Kaiwalya/Desktop/deblur/new_crop.png',crop)
                #image_slicer.slice('C:/Users/Kaiwalya/Desktop/deblur/new.png',4)
                key = cv2.waitKey(1)
                if key == 27 or key == 'q' or key == 'Q':
                    break
                presenter.handleKey(key)
            next_frame_id_to_show += 1

    pipeline.await_all()
    # Process completed requests
    while pipeline.has_completed_request():
        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            result_frame, frame_meta = results
            input_frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if input_frame.shape != result_frame.shape:
                input_frame = cv2.resize(input_frame, (result_frame.shape[1], result_frame.shape[0]))
            final_image = cv2.hconcat([input_frame, result_frame])

            presenter.drawGraphs(final_image)
            metrics.update(start_time, final_image)
            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(final_image)
            if not args.no_show:
                cv2.imshow('Deblurring Results', final_image)
                cv2.imwrite('C:/Users/Kaiwalya/Desktop/deblur/new.png',final_image)
                img = cv2.imread('C:/Users/Kaiwalya/Desktop/deblur/new.png')
                height, width, channels = img.shape
                w1 = int(width/1)
                w2 = int(width/2)
                w4 = int(width/4)
                print(w1,w2)
                crop = img[:, w4:w2]
                cv2.imwrite('C:/Users/Kaiwalya/Desktop/deblur/new_crop.png',crop)
                image_slicer.slice('C:/Users/Kaiwalya/Desktop/deblur/new.png',4)
                key = cv2.waitKey(1)
            next_frame_id_to_show += 1
        else:
            break

    metrics.print_total()
    print(presenter.reportMeans())


###### VINO ENDS ####

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = r'D:\kishan-know-backend\static\uploads'
filename=''
message=''

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 400 * 400 * 400


def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route('/')
def index():
 return render_template('index.html')
 
@app.route('/map')
def map():
 return render_template('map.html')

@app.route('/address')
def address():
 return render_template('address.html')

# @app.route('/filed-health')
# def filed_health():
# 	return render_template('filed-health.html')

# @app.route('/photo')
# def photo():
# 	return render_template('photo.html')

# @app.route('/showMap')
# def show_map():
# 	return render_template('showMap.html')

@app.route('/data')
def show_data():
 return render_template('data.html')
 
@app.route('/upload_form')
def upload_form():
 return render_template('upload.html')

@app.route('/upload_form', methods=['POST'])
def upload_image():
 if 'file' not in request.files:
  flash('No file part')
  return redirect(request.url)
 file = request.files['file']
 if file.filename == '':
  flash('No image selected for uploading')
  return redirect(request.url)
 if file and allowed_file(file.filename):
  filename = secure_filename(file.filename)
  new_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  print(type(new_name))
  file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  #print('upload_image filename: ' + filename)
  flash('Image successfully uploaded and displayed below')



  model = keras.models.load_model(r'keras_model.h5')
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  # Replace this with the path to your image
  #str_path= image_path
  image = Image.open(new_name)
  #resize the image to a 224x224 with the same strategy as in TM2:
  #resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  #turn the image into a numpy array
  image_array = np.asarray(image)
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
  data[0] = normalized_image_array

  # run the inference
  prediction = model.predict(data)
  if prediction[0][1] > prediction[0][0]:
            flash('It may be  infected. Initiating Step 2')
            time.sleep(5)
            flash('CHECKING WITH THE PHYSICAL SETUP DATA')
            time.sleep(5)
            flash('Data Fetched')
            time.sleep(5)
            flash('Infestation Confirmed Recommending solutions')
            time.sleep(5)
            message = client.messages.create(body="It is infected", from_='+17652953355', to='+919003032644')
  else:
      flash('It is safe')
      message1 = client.messages.create(body="It is safe", from_='+17652953355', to='+919003032644')
  print(type(prediction))


  return render_template('upload.html', filename=filename)
 else:
  flash('Allowed image types are -> png, jpg, jpeg, gif')
  return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

'''
model = keras.models.load_model(r'keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
#str_path= image_path
image = Image.open(filename)
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)
print(type(prediction))
'''
# python deblurring_demo.py -i "D:\kishan-know-backend\deblurred_image.png" -d CPU -m "D:\kishan-know-backend\deblurgan-v2.xml"



if __name__ == "__main__":
    app.run()