# app.py
from flask import Flask, render_template  # importing the render_template function

app = Flask(__name__)
# home route
@app.route("/")
def hello():
    return render_template('index.html')

app.run(debug = True) 