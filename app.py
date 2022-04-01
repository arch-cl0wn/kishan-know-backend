from fileinput import filename
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
#from tensorflow import keras
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = r'D:\kishan-know-backend\static\uploads'
filename=''

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 400 * 400 * 400


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
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
			flash('It is infected')
		else:
			flash('It is safe')
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

if __name__ == "__main__":
    app.run()