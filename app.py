# app.py
from flask import Flask, render_template, request  # importing the render_template function
from werkzeug.utils import secure_filename

app = Flask(__name__)
# home route
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'