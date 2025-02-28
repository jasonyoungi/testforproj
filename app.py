import pandas as pd
import numpy as np
import threading
import time
from flask import Flask, render_template, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_buffer = []

def generate_data(n=1):
    np.random.seed()
    joint_temp = np.random.normal(50, 10, n)[0]
    motor_current = np.random.normal(5, 2, n)[0]
    vibration = np.random.normal(2, 1, n)[0]
    position_error = np.random.normal(0.5, 0.2, n)[0]
    failure = (joint_temp > 60) or (motor_current > 8) or (vibration > 3) or (position_error > 1)
    return {
        "joint_temp": joint_temp, "motor_current": motor_current,
        "vibration": vibration, "position_error": position_error,
        "failure": int(failure)
    }

def train_model():
    df = pd.DataFrame([generate_data() for _ in range(1000)])
    X = df[["joint_temp", "motor_current", "vibration", "position_error"]]
    y = df["failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

def data_generator():
    global data_buffer
    while True:
        new_data = generate_data()
        data_buffer.append(new_data)
        if len(data_buffer) > 50:
            data_buffer.pop(0)
        time.sleep(1)

data_thread = threading.Thread(target=data_generator, daemon=True)
data_thread.start()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    joint_temp = float(request.args.get("joint_temp", 50))
    motor_current = float(request.args.get("motor_current", 5))
    vibration = float(request.args.get("vibration", 2))
    position_error = float(request.args.get("position_error", 0.5))
    prediction = model.predict([[joint_temp, motor_current, vibration, position_error]])[0]
    return jsonify({"failure_risk": bool(prediction)})

@app.route('/data')
def get_data():
    return jsonify(data_buffer)

if __name__ == '__main__':
    app.run(debug=True)
