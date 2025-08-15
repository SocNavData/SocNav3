import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("Started!")

@app.route('/')
def home():
    return send_file('static/index.html')

@app.route('/instructions.mp4')
def instructions_mp4():
    return send_file('static/instructions.mp4')

@app.route('/instructions.webm')
def instructions_webm():
    return send_file('static/instructions.webm')

@app.route('/surveycode.py')
def survey():
    return send_file('static/surveycode.py')

@app.route('/slider.py')
def slider():
    return send_file('static/tasks.py')

@app.route('/tasks.py')
def tasks():
    return send_file('static/tasks.py')

@app.route('/submit', methods=['PUT', 'POST'])
def submit():
    #print("headers", request.headers)
    #print("data", request.data)
    data = str(request.data.decode('UTF-8'))
    print(f"GOT: {len(data)}")
    if len(data) < 10_000_000:
        now = datetime.datetime.now()
        path = "/storage/" + str(now.strftime("%Y-%m-%d_%H:%M:%S"))+".json"
        fd = open(path, "w")
        fd.write(data)
        fd.close()
    return "Received", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

